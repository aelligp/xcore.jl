# DyrelCache3D — preallocated DYREL (pseudo-transient) scratch for the 3-D GMG
# smoother. The dimension lift of `src/dyrel/cache.jl::DyrelCache`, with the
# extra V (y-velocity) face block. Field names match the `c.` accesses in
# `src/dyrel3D/schur.jl` so the residual/Schur/PH kernel wrappers operate on it
# unchanged.
#
# Face/cell shapes (Nz×Nx×Ny interior):
#   W-face : (Nz+1, Nx,   Ny  )      U-face : (Nz, Nx+1, Ny  )
#   V-face : (Nz,   Nx,   Ny+1)      cell   : (Nz, Nx,   Ny  )
#
# As in 2-D none of the face autotune arrays carry ghosts — the kernels index
# into the ghosted `FluidState3D.W/U/V/P` interiors directly.

mutable struct DyrelCache3D{T<:AbstractFloat, A<:AbstractArray{T,3}}
    initialized::Bool
    sig_Nz::Int;  sig_Nx::Int;  sig_Ny::Int

    # Powell-Hestenes penalty + compressibility (cell-centred)
    γ_eff::A;     η_b::A

    # physical (penalty-free) Gershgorin diagonals D_phys (snapshot before the
    # penalty overwrites D_* with D_full); the Schur diagonal is built from these.
    D_W_phys::A;  D_U_phys::A;  D_V_phys::A

    # W (z-velocity) per-face autotune state, (Nz+1, Nx, Ny)
    D_W::A;  λmax_W::A;  dτ_W::A;  α_W::A;  β_W::A
    dWdτ::A; dW_step::A; R_W::A;   R_W_old::A

    # U (x-velocity) per-face autotune state, (Nz, Nx+1, Ny)
    D_U::A;  λmax_U::A;  dτ_U::A;  α_U::A;  β_U::A
    dUdτ::A; dU_step::A; R_U::A;   R_U_old::A

    # V (y-velocity) per-face autotune state, (Nz, Nx, Ny+1)
    D_V::A;  λmax_V::A;  dτ_V::A;  α_V::A;  β_V::A
    dVdτ::A; dV_step::A; R_V::A;   R_V_old::A

    # continuity residual + Schur penalty (cell-centred)
    R_P::A;  P_num::A;  s_P::A

    # momentum-advection scratch (Navier-Stokes; Re>1)
    advn_mz::A;  advn_mx::A;  advn_my::A

    # GMG external RHS (restricted residual; zero on the fine level)
    bext_W::A;  bext_U::A;  bext_V::A;  bext_P::A

    # global scalars
    c::Base.RefValue{T}
    last_PH_iters::Base.RefValue{Int}
    last_DR_iters::Base.RefValue{Int}
end

"""
    DyrelCache3D(::Type{T}, backend, Nz, Nx, Ny) -> DyrelCache3D

Allocate a zero-initialised 3-D DYREL cache on `backend`. Lazy-init (the 3-D
`init_dyrel!`, Phase 2) populates `γ_eff`/`η_b`, then `gershgorin3d!` fills the
`D_*`/`λmax_*`/`dτ_*`/`α_*`/`β_*` autotune arrays.
"""
function DyrelCache3D(::Type{T}, backend, Nz::Integer, Nx::Integer, Ny::Integer) where {T<:AbstractFloat}
    z(dims...) = xcore_zeros(backend, T, dims...)
    AT  = typeof(z(1, 1, 1))
    nzf = Nz + 1;  nxf = Nx + 1;  nyf = Ny + 1
    W(dims...) = z(nzf, Nx, Ny)     # W-face block helper
    U()        = z(Nz, nxf, Ny)     # U-face block helper
    Vf()       = z(Nz, Nx, nyf)     # V-face block helper
    C()        = z(Nz, Nx, Ny)      # cell block helper
    return DyrelCache3D{T, AT}(
        false, 0, 0, 0,
        C(), C(),                                          # γ_eff, η_b
        z(nzf, Nx, Ny), U(), Vf(),                         # D_W_phys, D_U_phys, D_V_phys
        # W block (9 arrays)
        z(nzf, Nx, Ny), z(nzf, Nx, Ny), z(nzf, Nx, Ny), z(nzf, Nx, Ny), z(nzf, Nx, Ny),
        z(nzf, Nx, Ny), z(nzf, Nx, Ny), z(nzf, Nx, Ny), z(nzf, Nx, Ny),
        # U block (9 arrays)
        U(), U(), U(), U(), U(), U(), U(), U(), U(),
        # V block (9 arrays)
        Vf(), Vf(), Vf(), Vf(), Vf(), Vf(), Vf(), Vf(), Vf(),
        C(), C(), C(),                                     # R_P, P_num, s_P
        z(max(Nz - 1, 1), Nx, Ny), U(), Vf(),              # advn_mz, advn_mx, advn_my
        z(nzf, Nx, Ny), U(), Vf(), C(),                    # bext_W, bext_U, bext_V, bext_P
        Ref(zero(T)), Ref(0), Ref(0),                      # c, counters
    )
end

DyrelCache3D(::Type{T}, backend, grid::Grid3D) where {T<:AbstractFloat} =
    DyrelCache3D(T, backend, grid.Nz, grid.Nx, grid.Ny)
