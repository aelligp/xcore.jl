using SparseArrays

# ----------------------------------------------------------------------------
# FluidmechCache: preallocated storage for the `:direct` Stokes solve. The
# sparsity of the four block matrices (KV, GG, DM, KP) is fully determined by
# (Nz, Nx, bnchm, xBC); only the entry values change each Picard sweep.
#
# The first call assembles the (I, J, V) triplets and builds each block once
# with `sparse(...)`, then caches `nz_*[k]`, the position in `M.nzval` where
# triplet k lands. Later calls skip the sort and dedup of `sparse(...)` and
# scatter the new values into `M.nzval` through the cached maps
# (`scatter_nzval!`). The UMFPACK numeric refactorization (`lu!`) reuses the
# symbolic factorization stored in `lu_F`.
#
# Lives on `FluidState.solver`; rebuilt when `(Nz, Nx, bnchm, xBC)` differs
# from the cached signature `sig_*`.
# ----------------------------------------------------------------------------

mutable struct FluidmechCache{T<:AbstractFloat}
    initialized::Bool
    sig_Nz::Int;  sig_Nx::Int;  sig_bnchm::Bool;  sig_xBC::Symbol

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

    # Global system `LL = [KV GG; DM KP]`, assembled once. Later sweeps scatter
    # the block values into `LL.nzval` (via map_*) and write the symmetric-Jacobi
    # scaling in place into `LLs`, which shares LL's structure. `Lcol[k]` is the
    # column of `LL.nzval[k]`. Built lazily; `initialized` is reset on a
    # signature change.
    LL_initialized::Bool
    LL::SparseMatrixCSC{T,Int}
    LLs::SparseMatrixCSC{T,Int}
    map_KV::Vector{Int};  map_GG::Vector{Int};  map_DM::Vector{Int};  map_KP::Vector{Int}
    Lcol::Vector{Int}

    # Advection output buffers for the momentum RHS, sized on first use and
    # reused across sweeps.
    advn_mz::Matrix{T};  advn_mx::Matrix{T}

    # UMFPACK numeric-refactor cache, built on the first solve. The bare
    # `UmfpackLU` UnionAll keeps the field type valid for `FluidmechCache{Float32}`
    # even though UMFPACK itself only admits Float64 and ComplexF64.
    lu_F::Union{Nothing, SparseArrays.UMFPACK.UmfpackLU}
end

function FluidmechCache(::Type{T}) where {T<:AbstractFloat}
    FluidmechCache{T}(
        false, 0, 0, false, :unset,
        spzeros(T, 0, 0), spzeros(T, 0, 0), spzeros(T, 0, 0), spzeros(T, 0, 0),
        Int[], Int[], Int[], Int[],
        false, spzeros(T, 0, 0), spzeros(T, 0, 0),
        Int[], Int[], Int[], Int[], Int[],
        Matrix{T}(undef, 0, 0), Matrix{T}(undef, 0, 0),
        nothing,
    )
end

# The 2-D field names are kept in 3-D; fields that exist only in 3-D are
# `nothing` in 2-D. Their type is the parameter `B` (the array type in 3-D,
# `Nothing` in 2-D), so field access stays inferable. Shapes are given in terms
# of the interior cell counts N = (Nz, Nx[, Ny]):
#   cells               N
#   ghosted cells       N .+ 2
#   compact face d      `face_shape(N, d)`: N + 1 along d
#   ghosted face d      `ghost_face_shape(N, d)`: N + 1 along d, N + 2 across
#   edge (a, b)         `edge_shape(N, a, b)`: N + 1 along a and b

"""
    FluidState{T,D,A,B}

Fields touched by the fluid-mechanics solvers, for `D = 2` or `3`. Sizing
follows MATLAB `init.m`; shapes are described above the definition. Axes 1, 2,
3 are z, x, y.

* `W`, `U`, `V`  : z-, x-, y-velocity on ghosted faces 1, 2, 3
* `P`            : pressure on ghosted cells
* `eta`          : shear viscosity on cells
* `etaco`        : shear viscosity on the z-x edges (cell corners in 2-D)
* `etayz`, `etaxy` : shear viscosity on the z-y and x-y edges
* `rho`, `rhoo`, `rhooo` : bulk density on cells, one and two steps back
* `rhow`, `rhou`, `rhov` : density on compact faces 1, 2, 3
* `Drho`         : density anomaly `rhow .- mean` on compact z-faces
* `drhodt`, `drhodto`, `drhodtoo` : density rate on cells, one and two steps back
* `MFS`          : mass-flux source term on cells
* `rhoWo/rhoWoo`, `rhoUo/rhoUoo`, `rhoVo/rhoVoo` : momentum fluxes one and two
  steps back on compact faces 1, 2, 3

`V`, `etayz`, `etaxy`, `rhov`, `rhoVo` and `rhoVoo` are `nothing` in 2-D.
`solver` caches the sparse system of the 2-D `:direct` solver.
"""
struct FluidState{T<:AbstractFloat, D, A<:AbstractArray{T, D}, B<:Union{A, Nothing}}
    # primary unknowns
    W::A;  U::A;  V::B;  P::A
    # viscosities
    eta::A;     etaco::A;   etayz::B;   etaxy::B
    # densities (current, then one and two steps back for BD2)
    rho::A;     rhow::A;    rhou::A;    rhov::B;    Drho::A
    rhoo::A;    rhooo::A
    # density rates (current and lagged)
    drhodt::A;  drhodto::A;  drhodtoo::A
    # source terms
    MFS::A
    # momentum-flux history for the inertial RHS
    rhoWo::A;   rhoWoo::A
    rhoUo::A;   rhoUoo::A
    rhoVo::B;   rhoVoo::B
    # `:direct` Stokes-solve cache, built on the first `fluidmech!` call
    solver::FluidmechCache{T}
end

"""
    FluidState(::Type{T}, backend, Nz, Nx[, Ny]) -> FluidState
    FluidState(::Type{T}, backend, grid) -> FluidState

Allocate a 2-D or 3-D `FluidState` with all fields zero, on the given
KernelAbstractions backend.
"""
function FluidState(::Type{T}, backend, N::Vararg{Integer, D}) where {T<:AbstractFloat, D}
    N = Int.(N)
    z(dims) = xcore_zeros(backend, T, dims...)
    y(dims) = D == 3 ? z(dims) : nothing   # fields that exist only in 3-D
    A = typeof(z(N))
    return FluidState{T, D, A, D == 3 ? A : Nothing}(
        z(ghost_face_shape(N, 1)), z(ghost_face_shape(N, 2)), y(ghost_face_shape(N, 3)),  # W, U, V
        z(N .+ 2),                                                                        # P
        z(N),                                                                             # eta
        z(edge_shape(N, 1, 2)), y(edge_shape(N, 1, 3)), y(edge_shape(N, 2, 3)),           # etaco, etayz, etaxy
        z(N),                                                                             # rho
        z(face_shape(N, 1)), z(face_shape(N, 2)), y(face_shape(N, 3)),                    # rhow, rhou, rhov
        z(face_shape(N, 1)),                                                              # Drho
        z(N), z(N),                                                                       # rhoo, rhooo
        z(N), z(N), z(N),                                                                 # drhodt, drhodto, drhodtoo
        z(N),                                                                             # MFS
        z(face_shape(N, 1)), z(face_shape(N, 1)),                                         # rhoWo, rhoWoo
        z(face_shape(N, 2)), z(face_shape(N, 2)),                                         # rhoUo, rhoUoo
        y(face_shape(N, 3)), y(face_shape(N, 3)),                                         # rhoVo, rhoVoo
        FluidmechCache(T),
    )
end
FluidState(::Type{T}, backend, grid::Grid) where {T<:AbstractFloat} = FluidState(T, backend, cells(grid)...)

"""
    velocity(fluid) -> NTuple{D}

Ghosted face velocities `(W, U)` (2-D) or `(W, U, V)` (3-D), one per axis.
"""
velocity(f::FluidState{<:Any, D}) where {D} = per_axis(Val(D), f.W, f.U, f.V)

"""
    edge_viscosity(fluid) -> tuple

Edge viscosities in the pair order (1,2), (1,3), (2,3): `(etaco,)` in 2-D,
`(etaco, etayz, etaxy)` in 3-D.
"""
edge_viscosity(f::FluidState{<:Any, D}) where {D} = per_pair(Val(D), f.etaco, f.etayz, f.etaxy)

"""
    face_density(fluid) -> NTuple{D}

Face densities `(rhow, rhou[, rhov])` on compact faces.
"""
face_density(f::FluidState{<:Any, D}) where {D} = per_axis(Val(D), f.rhow, f.rhou, f.rhov)

# Momentum-flux history (BD2), one and two steps back, compact faces.
momentum_lag1(f::FluidState{<:Any, D}) where {D} = per_axis(Val(D), f.rhoWo, f.rhoUo, f.rhoVo)
momentum_lag2(f::FluidState{<:Any, D}) where {D} = per_axis(Val(D), f.rhoWoo, f.rhoUoo, f.rhoVoo)

"""
    BnchmData{T,A}

Method-of-manufactured-solutions source terms (`src_*`) and exact solutions
(`*_exact`) used by the `bnchm = true` branch of `fluidmech!`. Each array has
the shape of the matching unknown `W`, `U`, `P` in `FluidState`.
"""
struct BnchmData{T<:AbstractFloat, A<:AbstractMatrix{T}}
    src_W::A;  src_U::A;  src_P::A
    W_exact::A;  U_exact::A;  P_exact::A
end

"""
    PhaseState{T,D,A,AB,B}

Constitutive workspace read and written by `update!`: everything MATLAB
`update.m` touches apart from the primary fluid-mechanics unknowns. Pairs with
a `FluidState` of the same cell counts `N`; shapes are described above the
`FluidState` definition. Axes 1, 2, 3 are z, x, y.

* `x`, `m`, `X`, `M` : phase mass fractions and phase densities, cells
* `Xo`, `Xoo`, `Mo`, `Moo` : `X` and `M` one and two steps back, cells
* `chi`, `mu` : volume fractions in `[ε, 1-ε]`, cells
* `hasx`, `hasm` : Bool phase indicators, cells
* `chiw`/`muw`, `chiu`/`muu`, `chiv`/`muv` : volume fractions on ghosted faces 1, 2, 3
* `x_w`/`m_w`, `x_u`/`m_u`, `x_v`/`m_v` : mass fractions on compact faces 1, 2, 3
* `Drhom`, `Drhox` : phase density contrasts on compact z-faces
* `Pl`, `Pt` : lithostatic and total pressure, cells
* `etamix` : permission-weighted mixture viscosity; `etae` : eddy viscosity
  `fReL * ke * rho`; `ke`, `kx` : eddy and effective particle diffusivity; cells
* `etat`, `etas`, `ks` : segregation/drag regularization, cells; `etasw` : on compact z-faces
* `V`, `Div_V` : convection-speed magnitude and velocity divergence, cells
* `exx`, `ezz`, `eyy` : normal strain rates, cells; `exz`, `eyz`, `exy` : shear strain
  rates on the z-x, z-y, x-y edges; `eII` : second invariant, cells
* `txx`, `tzz`, `tyy`, `txz`, `tyz`, `txy`, `tII` : stresses, located like the strain rates
* `vx`, `vm` : segregation-speed magnitudes, cells
* `fReL`, `ReL`, `ReD`, `Ra`, `fRel`, `Rel`, `Red`, `Rc` : Re-dependent ramp
  factors and dimensionless numbers, cells
* `Wx`/`Wm`, `Ux`/`Um`, `Vx`/`Vm` : crystal/melt velocities on ghosted faces 1, 2, 3
* `wx`, `wm` : segregation speeds on ghosted z-faces
* `bndtaperw` : boundary taper on ghosted z-faces (zeroes `wx` at closed top and
  bottom); `bndshape` : top-localized exponential profile of the boundary
  crystallization rate `Gx = G0 (1 - x) bndshape`, cells
* `advn_X`, `advn_M`, `advn_rho`, `dffn_X`, `Gx` : advection, diffusion and
  reaction rates, cells; `dXdt`, `dXdto`, `dXdtoo` : `X` rate and its history, cells
* `qz_*`, `qx_*`, `qy_*` : advective (`advn`) and diffusive (`dffn`) face fluxes of
  `X` and `M`, on ghosted faces 1, 2, 3, as returned by `advect_with_flux!` and
  `diffus_with_flux!`; `qz/qx/qy_dffn_M` stay zero because MATLAB has no melt diffusion

The y-fields (`chiv`, `muv`, `x_v`, `m_v`, `eyy`, `eyz`, `exy`, `tyy`, `tyz`,
`txy`, `Vx`, `Vm`, `qy_*`) are `nothing` in 2-D.
"""
struct PhaseState{T<:AbstractFloat, D, A<:AbstractArray{T, D}, AB<:AbstractArray{Bool, D},
                  B<:Union{A, Nothing}}
    # phase fractions (current + history)
    x::A;   m::A;   X::A;   M::A
    Xo::A;  Xoo::A; Mo::A;  Moo::A
    chi::A; mu::A
    hasx::AB;  hasm::AB
    # face interpolations
    chiw::A;  muw::A;  chiu::A;  muu::A;  chiv::B;  muv::B
    x_w::A;   m_w::A;  x_u::A;   m_u::A;  x_v::B;   m_v::B
    # density contrasts (face-staggered with rhow)
    Drhom::A;  Drhox::A
    # pressure
    Pl::A;    Pt::A
    # rheology — eddy regularization
    etamix::A;  etae::A
    ke::A;      kx::A
    # rheology — segregation/drag regularization
    etat::A;    etas::A;    etasw::A
    ks::A
    # kinematics
    V::A;       Div_V::A
    exx::A;     ezz::A;   eyy::B
    exz::A;     eyz::B;   exy::B
    eII::A
    vx::A;      vm::A                       # segregation speed magnitudes
    # stresses
    txx::A;     tzz::A;   tyy::B
    txz::A;     tyz::B;   txy::B
    tII::A
    # dimensionless
    fReL::A;    ReL::A;   ReD::A;   Ra::A
    fRel::A;    Rel::A;   Red::A;   Rc::A
    # phase velocities (W,U[,V] + segregation + noise)
    Wx::A;  Wm::A;  Ux::A;  Um::A;  Vx::B;  Vm::B
    wx::A;  wm::A                            # segregation speeds
    # boundary taper and boundary-crystallization profile
    bndtaperw::A
    bndshape::A
    # phase evolution rates (current + history for BD2)
    advn_X::A;  advn_M::A;  advn_rho::A
    # padded face fluxes from advect_with_flux!/diffus_with_flux! (advect.m, diffus.m)
    qz_advn_X::A;  qx_advn_X::A;  qy_advn_X::B
    qz_advn_M::A;  qx_advn_M::A;  qy_advn_M::B
    qz_dffn_X::A;  qx_dffn_X::A;  qy_dffn_X::B
    qz_dffn_M::A;  qx_dffn_M::A;  qy_dffn_M::B
    dffn_X::A;  Gx::A
    dXdt::A;    dXdto::A;   dXdtoo::A
end

"""
    PhaseState(::Type{T}, backend, Nz, Nx[, Ny]) -> PhaseState
    PhaseState(::Type{T}, backend, grid) -> PhaseState

Allocate a 2-D or 3-D `PhaseState` with all fields zero (`false` for the Bool
indicators), on the given KernelAbstractions backend.
"""
function PhaseState(::Type{T}, backend, N::Vararg{Integer, D}) where {T<:AbstractFloat, D}
    N = Int.(N)
    z(dims) = xcore_zeros(backend, T, dims...)
    y(dims) = D == 3 ? z(dims) : nothing   # fields that exist only in 3-D
    b() = KernelAbstractions.zeros(backend, Bool, N...)
    c() = z(N)
    gfaces() = (z(ghost_face_shape(N, 1)), z(ghost_face_shape(N, 2)), y(ghost_face_shape(N, 3)))
    A = typeof(c())
    return PhaseState{T, D, A, typeof(b()), D == 3 ? A : Nothing}(
        c(), c(), c(), c(),                                                               # x, m, X, M
        c(), c(), c(), c(),                                                               # Xo, Xoo, Mo, Moo
        c(), c(),                                                                         # chi, mu
        b(), b(),                                                                         # hasx, hasm
        z(ghost_face_shape(N, 1)), z(ghost_face_shape(N, 1)),                             # chiw, muw
        z(ghost_face_shape(N, 2)), z(ghost_face_shape(N, 2)),                             # chiu, muu
        y(ghost_face_shape(N, 3)), y(ghost_face_shape(N, 3)),                             # chiv, muv
        z(face_shape(N, 1)), z(face_shape(N, 1)),                                         # x_w, m_w
        z(face_shape(N, 2)), z(face_shape(N, 2)),                                         # x_u, m_u
        y(face_shape(N, 3)), y(face_shape(N, 3)),                                         # x_v, m_v
        z(face_shape(N, 1)), z(face_shape(N, 1)),                                         # Drhom, Drhox
        c(), c(),                                                                         # Pl, Pt
        c(), c(),                                                                         # etamix, etae
        c(), c(),                                                                         # ke, kx
        c(), c(), z(face_shape(N, 1)),                                                    # etat, etas, etasw
        c(),                                                                              # ks
        c(), c(),                                                                         # V, Div_V
        c(), c(), y(N),                                                                   # exx, ezz, eyy
        z(edge_shape(N, 1, 2)), y(edge_shape(N, 1, 3)), y(edge_shape(N, 2, 3)),           # exz, eyz, exy
        c(),                                                                              # eII
        c(), c(),                                                                         # vx, vm
        c(), c(), y(N),                                                                   # txx, tzz, tyy
        z(edge_shape(N, 1, 2)), y(edge_shape(N, 1, 3)), y(edge_shape(N, 2, 3)),           # txz, tyz, txy
        c(),                                                                              # tII
        c(), c(), c(), c(),                                                               # fReL, ReL, ReD, Ra
        c(), c(), c(), c(),                                                               # fRel, Rel, Red, Rc
        z(ghost_face_shape(N, 1)), z(ghost_face_shape(N, 1)),                             # Wx, Wm
        z(ghost_face_shape(N, 2)), z(ghost_face_shape(N, 2)),                             # Ux, Um
        y(ghost_face_shape(N, 3)), y(ghost_face_shape(N, 3)),                             # Vx, Vm
        z(ghost_face_shape(N, 1)), z(ghost_face_shape(N, 1)),                             # wx, wm
        z(ghost_face_shape(N, 1)), c(),                                                   # bndtaperw, bndshape
        c(), c(), c(),                                                                    # advn_X, advn_M, advn_rho
        gfaces()...,                                                                      # qz, qx, qy_advn_X
        gfaces()...,                                                                      # qz, qx, qy_advn_M
        gfaces()...,                                                                      # qz, qx, qy_dffn_X
        gfaces()...,                                                                      # qz, qx, qy_dffn_M
        c(), c(),                                                                         # dffn_X, Gx
        c(), c(), c(),                                                                    # dXdt, dXdto, dXdtoo
    )
end
PhaseState(::Type{T}, backend, grid::Grid) where {T<:AbstractFloat} = PhaseState(T, backend, cells(grid)...)
