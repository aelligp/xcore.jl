using SparseArrays

# ----------------------------------------------------------------------------
# FluidmechCache — preallocated storage for the Stokes solve. Sparsity of the
# four block matrices (KV, GG, DM, KP) is fully determined by (Nz, Nx, bnchm);
# only the entry VALUES change each Picard sweep.
#
# Strategy: the first call assembles the (I, J, V) triplets locally and builds
# each block once with `sparse(...)`, then caches `nz_*[k]` = the position in
# `M.nzval` where triplet k lands. Subsequent calls skip the sort+dedup
# `sparse(...)` step and scatter the freshly computed values into `M.nzval` via
# the cached maps (`_scatter!`). The UMFPACK numeric-refactor (`lu!`) reuses the
# symbolic factorization stored in `lu_F`.
#
# Lives on `FluidState.solver`; automatically invalidated if `(Nz, Nx, bnchm)`
# differs from the cached signature.
# ----------------------------------------------------------------------------

mutable struct FluidmechCache{T<:AbstractFloat}
    initialized::Bool
    sig_Nz::Int;  sig_Nx::Int;  sig_bnchm::Bool

    # cached block matrices (built once, scattered into thereafter)
    KV::SparseMatrixCSC{T,Int}
    GG::SparseMatrixCSC{T,Int}
    DM::SparseMatrixCSC{T,Int}
    KP::SparseMatrixCSC{T,Int}

    # nz_*[k] = linear index in M.nzval where triplet k contributes
    nz_KV::Vector{Int}
    nz_GG::Vector{Int}
    nz_DM::Vector{Int}
    nz_KP::Vector{Int}

    # UMFPACK numeric-refactor cache (built lazily on first solve).
    # Untyped on the matrix-element side because `SparseArrays.UMFPACK.UmfpackLU`
    # only admits {Float64, ComplexF64}; using the bare UnionAll keeps the field
    # type valid for `FluidmechCache{Float32}` too, even though Float32 sims
    # can't actually invoke fluidmech! (UMFPACK won't accept Float32 input).
    lu_F::Union{Nothing, SparseArrays.UMFPACK.UmfpackLU}
end

function FluidmechCache(::Type{T}) where {T<:AbstractFloat}
    FluidmechCache{T}(
        false, 0, 0, false,
        spzeros(T, 0, 0), spzeros(T, 0, 0), spzeros(T, 0, 0), spzeros(T, 0, 0),
        Int[], Int[], Int[], Int[],
        nothing,
    )
end

"""
    FluidState{T,A}

Fields touched by the fluid-mechanics solver. Sizing follows `src/init.m`:

* `W`        `(Nz+1, Nx+2)` — z-velocity on z-faces (with x-ghost columns)
* `U`        `(Nz+2, Nx+1)` — x-velocity on x-faces (with z-ghost rows)
* `P`        `(Nz+2, Nx+2)` — pressure on cells (with ghost rings on both sides)
* `eta`      `(Nz,   Nx  )` — cell-centred shear viscosity
* `etaco`    `(Nz+1, Nx+1)` — corner-interpolated shear viscosity
* `rho`      `(Nz,   Nx  )` — cell-centred bulk density
* `rhow`     `(Nz+1, Nx  )` — z-face density (W-points, x-interior)
* `rhou`     `(Nz,   Nx+1)` — x-face density (U-points, z-interior)
* `Drho`     `(Nz+1, Nx  )` — density anomaly `rhow .- rowmean(rhow)`
* `MFS`      `(Nz,   Nx  )` — mass-flux source term
* `rhoWo/rhoWoo` `(Nz+1, Nx)`  — previous time-step momentum-z flux
* `rhoUo/rhoUoo` `(Nz,   Nx+1)` — previous time-step momentum-x flux

This struct intentionally holds only what `fluidmech!` consumes. The
constitutive `update!` path (permission weights, phase fractions, noise) will
add its own fields as we implement those steps.
"""
struct FluidState{T<:AbstractFloat, A<:AbstractMatrix{T}}
    # primary unknowns
    W::A;  U::A;  P::A
    # viscosities
    eta::A;     etaco::A
    # densities (current + 1-step and 2-step lagged for BD2)
    rho::A;     rhow::A;    rhou::A;    Drho::A
    rhoo::A;    rhooo::A
    # density-evolution rates (current + lagged)
    drhodt::A;  drhodto::A;  drhodtoo::A
    # source terms
    MFS::A
    # momentum-flux history for the inertial RHS in fluidmech!
    rhoWo::A;   rhoWoo::A
    rhoUo::A;   rhoUoo::A
    # preallocated Stokes-solve cache (sparse blocks + LU). Built lazily on
    # the first `fluidmech!` call; reused thereafter via in-place nzval
    # scatter and UMFPACK numeric refactor.
    solver::FluidmechCache{T}
end

"""
    FluidState(::Type{T}, backend, Nz, Nx) -> FluidState{T,A}

Allocate a `FluidState` with all fields initialised to zero, using the given
KA backend so GPU backends drop in without changes to call sites.
"""
function FluidState(::Type{T}, backend, Nz::Integer, Nx::Integer) where {T<:AbstractFloat}
    z(args...) = xcore_zeros(backend, T, args...)
    return FluidState{T, typeof(z(1, 1))}(
        z(Nz + 1, Nx + 2),  # W
        z(Nz + 2, Nx + 1),  # U
        z(Nz + 2, Nx + 2),  # P
        z(Nz, Nx),          # eta
        z(Nz + 1, Nx + 1),  # etaco
        z(Nz, Nx),          # rho
        z(Nz + 1, Nx),      # rhow
        z(Nz, Nx + 1),      # rhou
        z(Nz + 1, Nx),      # Drho
        z(Nz, Nx), z(Nz, Nx),                     # rhoo, rhooo
        z(Nz, Nx), z(Nz, Nx), z(Nz, Nx),          # drhodt, drhodto, drhodtoo
        z(Nz, Nx),          # MFS
        z(Nz + 1, Nx),      # rhoWo
        z(Nz + 1, Nx),      # rhoWoo
        z(Nz, Nx + 1),      # rhoUo
        z(Nz, Nx + 1),      # rhoUoo
        FluidmechCache(T),  # solver cache (lazy-init on first fluidmech! call)
    )
end

"""
    BnchmData{T,A}

Bundle of MMS source terms (`src_*`) and exact-solution arrays (`*_exact`)
needed by the `bnchm = true` branch of `fluidmech!`. Sizes match the
corresponding primary unknowns in `FluidState`.
"""
struct BnchmData{T<:AbstractFloat, A<:AbstractMatrix{T}}
    src_W::A;  src_U::A;  src_P::A
    W_exact::A;  U_exact::A;  P_exact::A
end

"""
    PhaseState{T,A,AB}

Constitutive workspace consumed and produced by `update!` — everything the
MATLAB `src/update.m` reads/writes apart from the primary fluid-mechanics
unknowns. Pairs with a `FluidState` of the same `(Nz, Nx)`.

* `x`, `m`, `X`, `M` — phase fractions (mass) and phase densities, cell-centred.
* `chi`, `mu` — volume fractions in `[ε, 1-ε]`.
* `chiw`, `muw` `(Nz+1, Nx+2)`, `chiu`, `muu` `(Nz+2, Nx+1)` — face-interpolated
  volume fractions (with ghost columns/rows, following `update.m`).
* `x_w`, `m_w` `(Nz+1, Nx)`, `x_u`, `m_u` `(Nz, Nx+1)` — face-interpolated mass fractions.
* `Pl`, `Pt` `(Nz, Nx)` — lithostatic and total pressure.
* `etamix` `(Nz, Nx)` — permission-weighted mixture viscosity.
* `etae`   `(Nz, Nx)` — eddy viscosity contribution `fReL * ke * rho`.
* `ke`, `kx` `(Nz, Nx)` — eddy diffusivity and effective particle diffusivity.
  (`ks` ≡ 0 until segregation lands; `kx` falls back to `ke`.)
* `V` `(Nz, Nx)` — convection-speed magnitude `|<W,U>|`.
* `Div_V` `(Nz, Nx)` — velocity divergence.
* `exx`, `ezz` `(Nz, Nx)`, `exz` `(Nz+1, Nx+1)`, `eII` `(Nz, Nx)` — strain rates.
* `txx`, `tzz` `(Nz, Nx)`, `txz` `(Nz+1, Nx+1)`, `tII` `(Nz, Nx)` — stresses.
* `fReL`, `ReL`, `ReD`, `Ra` `(Nz, Nx)` — Re-dependent ramp factor and
  dimensionless numbers.
* `hasx`, `hasm` `(Nz, Nx)` — Bool phase indicators.

Segregation/noise-dependent quantities (`vx`, `vm`, `xie`, `xix`, `xis`,
`Wx`, `Ux`, `Wm`, `Um`, `etas`, `etasw`) are not allocated yet — they'll be
added when `phsevo!` and `noise!` come online.
"""
struct PhaseState{T<:AbstractFloat, A<:AbstractMatrix{T}, AB<:AbstractMatrix{Bool}}
    # phase fractions (current + history)
    x::A;   m::A;   X::A;   M::A
    Xo::A;  Xoo::A; Mo::A;  Moo::A
    chi::A; mu::A
    hasx::AB;  hasm::AB
    # face interpolations
    chiw::A;  muw::A;  chiu::A;  muu::A
    x_w::A;   m_w::A;  x_u::A;   m_u::A
    # density contrasts (face-staggered with rhow)
    Drhom::A;  Drhox::A
    # pressure
    Pl::A;    Pt::A
    # rheology — eddy regularisation
    etamix::A;  etae::A
    ke::A;      kx::A
    # rheology — segregation/drag regularisation
    etat::A;    etas::A;    etasw::A
    ks::A
    # kinematics
    V::A;       Div_V::A
    exx::A;     ezz::A;   exz::A;   eII::A
    vx::A;      vm::A                       # segregation speed magnitudes
    # stresses
    txx::A;     tzz::A;   txz::A;   tII::A
    # dimensionless
    fReL::A;    ReL::A;   ReD::A;   Ra::A
    fRel::A;    Rel::A;   Red::A;   Rc::A
    # phase velocities (W,U + segregation + noise; noise still pending)
    Wx::A;  Wm::A;  Ux::A;  Um::A
    wx::A;  wm::A                            # segregation speeds
    # boundary taper (zeroes wx at closed top/bot); `bndshape` is the top-
    # localised exponential profile used by the boundary crystallisation
    # reaction Gx = G0·(1-x)·bndshape
    bndtaperw::A
    bndshape::A
    # phase evolution rates (current + history for BD2)
    advn_X::A;  advn_M::A;  advn_rho::A
    # face-centred fluxes from advect/diffus (MATLAB sizing per src/advect.m, diffus.m):
    #   qx_* : (Nz+2, Nx+1)   qz_* : (Nz+1, Nx+2)
    qx_advn_X::A;  qz_advn_X::A;  qx_advn_M::A;  qz_advn_M::A
    qx_dffn_X::A;  qz_dffn_X::A;  qx_dffn_M::A;  qz_dffn_M::A
    dffn_X::A;  Gx::A
    dXdt::A;    dXdto::A;   dXdtoo::A
end

"""
    PhaseState(::Type{T}, backend, Nz, Nx) -> PhaseState

Allocate a `PhaseState` with all fields zero-initialised, sizing per the
docstring. `hasx`/`hasm` are `Matrix{Bool}` so they fit alongside `Array{T}`
without forcing a per-element conversion.
"""
function PhaseState(::Type{T}, backend, Nz::Integer, Nx::Integer) where {T<:AbstractFloat}
    z(args...) = xcore_zeros(backend, T, args...)
    AT = typeof(z(1, 1))
    ABT = typeof(KernelAbstractions.zeros(backend, Bool, 1, 1))
    return PhaseState{T, AT, ABT}(
        z(Nz, Nx), z(Nz, Nx), z(Nz, Nx), z(Nz, Nx),   # x, m, X, M
        z(Nz, Nx), z(Nz, Nx), z(Nz, Nx), z(Nz, Nx),   # Xo, Xoo, Mo, Moo
        z(Nz, Nx), z(Nz, Nx),                          # chi, mu
        KernelAbstractions.zeros(backend, Bool, Nz, Nx),
        KernelAbstractions.zeros(backend, Bool, Nz, Nx),
        z(Nz + 1, Nx + 2), z(Nz + 1, Nx + 2),          # chiw, muw
        z(Nz + 2, Nx + 1), z(Nz + 2, Nx + 1),          # chiu, muu
        z(Nz + 1, Nx),     z(Nz + 1, Nx),              # x_w, m_w
        z(Nz, Nx + 1),     z(Nz, Nx + 1),              # x_u, m_u
        z(Nz + 1, Nx),     z(Nz + 1, Nx),              # Drhom, Drhox
        z(Nz, Nx), z(Nz, Nx),                          # Pl, Pt
        z(Nz, Nx), z(Nz, Nx),                          # etamix, etae
        z(Nz, Nx), z(Nz, Nx),                          # ke, kx
        z(Nz, Nx), z(Nz, Nx), z(Nz + 1, Nx),           # etat, etas, etasw
        z(Nz, Nx),                                     # ks
        z(Nz, Nx), z(Nz, Nx),                          # V, Div_V
        z(Nz, Nx), z(Nz, Nx),                          # exx, ezz
        z(Nz + 1, Nx + 1), z(Nz, Nx),                  # exz, eII
        z(Nz, Nx), z(Nz, Nx),                          # vx, vm
        z(Nz, Nx), z(Nz, Nx),                          # txx, tzz
        z(Nz + 1, Nx + 1), z(Nz, Nx),                  # txz, tII
        z(Nz, Nx), z(Nz, Nx), z(Nz, Nx), z(Nz, Nx),    # fReL, ReL, ReD, Ra
        z(Nz, Nx), z(Nz, Nx), z(Nz, Nx), z(Nz, Nx),    # fRel, Rel, Red, Rc
        z(Nz + 1, Nx + 2), z(Nz + 1, Nx + 2),          # Wx, Wm
        z(Nz + 2, Nx + 1), z(Nz + 2, Nx + 1),          # Ux, Um
        z(Nz + 1, Nx + 2), z(Nz + 1, Nx + 2),          # wx, wm
        z(Nz + 1, Nx + 2), z(Nz, Nx),                  # bndtaperw, bndshape
        z(Nz, Nx), z(Nz, Nx), z(Nz, Nx),               # advn_X, advn_M, advn_rho
        # face fluxes (MATLAB sizing): qx (Nz+2, Nx+1), qz (Nz+1, Nx+2)
        z(Nz + 2, Nx + 1), z(Nz + 1, Nx + 2),          # qx_advn_X, qz_advn_X
        z(Nz + 2, Nx + 1), z(Nz + 1, Nx + 2),          # qx_advn_M, qz_advn_M
        z(Nz + 2, Nx + 1), z(Nz + 1, Nx + 2),          # qx_dffn_X, qz_dffn_X
        z(Nz + 2, Nx + 1), z(Nz + 1, Nx + 2),          # qx_dffn_M, qz_dffn_M (kept zero — no melt diffusion in MATLAB)
        z(Nz, Nx), z(Nz, Nx),                          # dffn_X, Gx
        z(Nz, Nx), z(Nz, Nx), z(Nz, Nx),               # dXdt, dXdto, dXdtoo
    )
end
