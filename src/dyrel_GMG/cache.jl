# DyrelCache — preallocated storage for the DYREL (Dynamic Relaxation /
# pseudo-transient) Stokes solver. See `src/dyrel/DESIGN.md` for the algorithm
# and the JustRelax.jl `src/DYREL/` source for the reference implementation.
#
# Lives on `FluidState.dyrel`. All fields are sized by `(Nz, Nx)` interior +
# face arrays. None of the W/U cache fields carry x-ghost columns or z-ghost
# rows: that's a deliberate choice to keep per-face autotune arrays compact.
# When residuals/updates need to interact with W (sized (Nz+1, Nx+2)) the
# kernels index into the interior `W[:, 2:end-1]` of `FluidState`.
#
# Sizing convention (matches `DESIGN.md` §6):
#   - Cell-centred fields              : (Nz,   Nx  )
#   - W-direction (z-velocity) face    : (Nz+1, Nx  )   — Nz+1 z-faces, Nx interior x-cols
#   - U-direction (x-velocity) face    : (Nz,   Nx+1)   — Nz interior z-rows, Nx+1 x-faces
#
# These shapes match the "interior" of `FluidState.W` (= W[:, 2:end-1]) and
# `FluidState.U` (= U[2:end-1, :]). The full halo'd W/U fields live on
# `FluidState`; the cache only holds what DYREL needs to update each iter.

mutable struct DyrelCache{T<:AbstractFloat, A<:AbstractMatrix{T}}
    # one-time setup flag; bumped to true once init_dyrel! completes
    initialized::Bool
    # cached grid size — used to detect resize and force re-init
    sig_Nz::Int;  sig_Nx::Int

    # ---- Powell-Hestenes penalty + compressibility (cell-centred) ----------
    γ_eff::A;     η_b::A

    # ---- physical (penalty-free) Gershgorin diagonals D_phys ---------------
    # Snapshotted in `compute_AL_penalty!` before the penalty overwrites D_W/D_U
    # with D_full. The MG pressure-Poisson coefficients κ = ρ/D_phys are built
    # from these so the scalar Schur operator matches the validated single-grid
    # multiplier step (which uses s_P_phys), not the penalised Schur S_γ.
    D_W_phys::A     # = D_W with γ_eff = 0  (size (Nz+1, Nx))
    D_U_phys::A     # = D_U with γ_eff = 0  (size (Nz,   Nx+1))

    # ---- W (z-velocity) per-face autotune state, size (Nz+1, Nx) -----------
    D_W::A          # Gershgorin diagonal (preconditioner)
    λmax_W::A       # per-face max-eigenvalue estimate (from Gershgorin)
    dτ_W::A         # per-face pseudo-time step    = 2·CFL/√λmax_W
    α_W::A          # damping coefficient           = (2 − c·dτ)/(2 + c·dτ)
    β_W::A          # update coefficient            = 2·dτ/(2 + c·dτ)
    dWdτ::A         # accumulated pseudo-rate of W (damping-pong state)
    dW_step::A      # last applied increment = β·dτ·dWdτ  (used by Rayleigh)
    R_W::A          # current momentum residual    (this iter)
    R_W_old::A      # momentum residual snapshot   (previous tuning step)

    # ---- U (x-velocity) per-face autotune state, size (Nz, Nx+1) -----------
    D_U::A;     λmax_U::A
    dτ_U::A;    α_U::A;     β_U::A
    dUdτ::A;    dU_step::A
    R_U::A;     R_U_old::A

    # ---- Continuity residual + Schur-complement penalty (cell-centred) -----
    R_P::A
    P_num::A          # = γ_eff · R_P, injected into the DR momentum residual
    s_P::A            # Schur-complement diagonal for the pressure update (see MATH.md §5)

    # ---- momentum-advection scratch (Navier-Stokes; Re>1) ------------------
    # advn_mz: div(v·ρW) on interior z-faces, size (Nz-1, Nx)
    # advn_mx: div(v·ρU) on all x-faces,      size (Nz,   Nx+1)
    advn_mz::A
    advn_mx::A

    # ---- GMG external RHS (injected restricted residual; zero on fine level) -
    # Shapes match R_W/R_U/R_P. On the fine grid these stay 0 and `fscale=1`
    # (physical forcing) ⇒ identical to the single-grid solve. On coarse levels
    # the smoother solves the error equation A·e = bext with `fscale=0`.
    bext_W::A;  bext_U::A;  bext_P::A

    # ---- Global scalars (boxed in Refs to keep struct fully-typed) ---------
    c::Base.RefValue{T}              # damping; updated by Rayleigh quotient
    last_PH_iters::Base.RefValue{Int}  # diagnostic counter
    last_DR_iters::Base.RefValue{Int}  # diagnostic counter
end

"""
    DyrelCache(::Type{T}, backend, Nz, Nx) -> DyrelCache{T,A}

Allocate an empty `DyrelCache` with all fields zero-initialized. Lazy-init
(`init_dyrel!`) populates `γ_eff`, `η_b`, then `gershgorin!` populates
`D_*`/`λmax_*`/`dτ_*`/`α_*`/`β_*`. Backend-aware via `xcore_zeros`.
"""
function DyrelCache(::Type{T}, backend, Nz::Integer, Nx::Integer) where {T<:AbstractFloat}
    z(args...) = xcore_zeros(backend, T, args...)
    AT  = typeof(z(1, 1))
    nzf = Nz + 1   # number of z-faces (W has Nz+1 rows)
    nxf = Nx + 1   # number of x-faces (U has Nx+1 cols)
    return DyrelCache{T, AT}(
        false, 0, 0,
        z(Nz, Nx),  z(Nz, Nx),                                          # γ_eff, η_b
        z(nzf, Nx), z(Nz, nxf),                                         # D_W_phys, D_U_phys
        # W cache: (Nz+1, Nx)
        z(nzf, Nx), z(nzf, Nx), z(nzf, Nx), z(nzf, Nx), z(nzf, Nx),
        z(nzf, Nx), z(nzf, Nx), z(nzf, Nx), z(nzf, Nx),
        # U cache: (Nz, Nx+1)
        z(Nz, nxf), z(Nz, nxf), z(Nz, nxf), z(Nz, nxf), z(Nz, nxf),
        z(Nz, nxf), z(Nz, nxf), z(Nz, nxf), z(Nz, nxf),
        z(Nz, Nx),  z(Nz, Nx),  z(Nz, Nx),                              # R_P, P_num, s_P
        z(max(Nz - 1, 1), Nx),  z(Nz, nxf),                             # advn_mz, advn_mx
        z(nzf, Nx), z(Nz, nxf), z(Nz, Nx),                              # bext_W, bext_U, bext_P
        Ref(zero(T)),  Ref(0),  Ref(0),                                 # c, counters
    )
end

"""
    DyrelCache(::Type{T}, backend, grid::Grid) -> DyrelCache{T,A}

Convenience constructor sizing the cache from a `Grid`. Equivalent to
`DyrelCache(T, backend, grid.Nz, grid.Nx)`.
"""
DyrelCache(::Type{T}, backend, grid::Grid) where {T<:AbstractFloat} =
    DyrelCache(T, backend, grid.Nz, grid.Nx)
