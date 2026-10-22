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
# Lives on `FluidState.solver`; automatically invalidated if `(Nz, Nx, bnchm, xBC)`
# differs from the cached signature.
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

    # global system `LL = [KV GG; DM KP]` assembled once; later sweeps scatter
    # block values into `LL.nzval` (via map_*) and write the symmetric-Jacobi
    # scaling into `LLs` (shares LL's structure) in place. `Lcol[k]` = column of
    # LL.nzval[k]. Built lazily; reset with `initialized` on a signature change.
    LL_initialized::Bool
    LL::SparseMatrixCSC{T,Int}
    LLs::SparseMatrixCSC{T,Int}
    map_KV::Vector{Int};  map_GG::Vector{Int};  map_DM::Vector{Int};  map_KP::Vector{Int}
    Lcol::Vector{Int}

    # advection out-buffers for the momentum RHS, sized lazily and reused across
    # sweeps (the value bands are recomputed every sweep; only the index triplets
    # are first-call-only).
    advn_mz::Matrix{T};  advn_mx::Matrix{T}

    # UMFPACK numeric-refactor cache (built lazily on first solve).
    # Untyped on the matrix-element side because `SparseArrays.UMFPACK.UmfpackLU`
    # only admits {Float64, ComplexF64}; using the bare UnionAll keeps the field
    # type valid for `FluidmechCache{Float32}` too, even though Float32 sims
    # can't actually invoke fluidmech! (UMFPACK won't accept Float32 input).
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
# of the interior cell counts N = (Nz, Nx[, Ny]): cells N, ghosted cells N .+ 2,
# and the face and edge shapes of `face_shape`, `ghost_face_shape`, `edge_shape`.

"""
    FluidState{T,D,A,B}

Fields touched by the fluid-mechanics solvers, for `D = 2` or `3`. Sizing
follows `src/init.m` (see the shape notes above):

* `W`, `U`, `V`  — z-, x-, y-velocity on ghosted faces 1, 2, 3
* `P`            — pressure on ghosted cells
* `eta`          — cell-centered shear viscosity
* `etaco`        — shear viscosity on the z–x edges (cell corners in 2-D)
* `etayz`, `etaxy` — shear viscosity on the z–y and x–y edges
* `rho`          — cell-centered bulk density
* `rhow`, `rhou`, `rhov` — densities on compact faces 1, 2, 3
* `Drho`         — density anomaly on compact z-faces, `rhow .- mean`
* `MFS`          — mass-flux source term (cells)
* `rhoWo/rhoWoo`, `rhoUo/rhoUoo`, `rhoVo/rhoVoo` — previous time-step momentum
  fluxes on compact faces 1, 2, 3

`V`, `etayz`, `etaxy`, `rhov`, `rhoVo` and `rhoVoo` are `nothing` in 2-D.
`solver` caches the sparse system of the 2-D `:direct` solver.
"""
struct FluidState{T<:AbstractFloat, D, A<:AbstractArray{T, D}, B<:Union{A, Nothing}}
    # primary unknowns
    W::A;  U::A;  V::B;  P::A
    # viscosities
    eta::A;     etaco::A;   etayz::B;   etaxy::B
    # densities (current + 1-step and 2-step lagged for BD2)
    rho::A;     rhow::A;    rhou::A;    rhov::B;    Drho::A
    rhoo::A;    rhooo::A
    # density-evolution rates (current + lagged)
    drhodt::A;  drhodto::A;  drhodtoo::A
    # source terms
    MFS::A
    # momentum-flux history for the inertial RHS
    rhoWo::A;   rhoWoo::A
    rhoUo::A;   rhoUoo::A
    rhoVo::B;   rhoVoo::B
    # preallocated `:direct` Stokes-solve cache (sparse blocks + LU). Built
    # lazily on the first `fluidmech!` call; reused thereafter via in-place
    # nzval scatter and UMFPACK numeric refactor.
    solver::FluidmechCache{T}
end

"""
    FluidState(::Type{T}, backend, Nz, Nx[, Ny]) -> FluidState
    FluidState(::Type{T}, backend, grid) -> FluidState

Allocate a 2-D or 3-D `FluidState` with all fields initialized to zero, using
the given KA backend so GPU backends drop in without changes to call sites.
"""
function FluidState(::Type{T}, backend, N::Vararg{Integer, D}) where {T<:AbstractFloat, D}
    N = Int.(N)
    z(dims) = xcore_zeros(backend, T, dims...)
    y(dims) = D == 3 ? z(dims) : nothing
    A = typeof(z(N))
    return FluidState{T, D, A, D == 3 ? A : Nothing}(
        z(ghost_face_shape(N, 1)), z(ghost_face_shape(N, 2)), y(ghost_face_shape(N, 3)),   # W, U, V
        z(N .+ 2),                                                 # P
        z(N),                                                       # eta
        z(edge_shape(N, 1, 2)), y(edge_shape(N, 1, 3)), y(edge_shape(N, 2, 3)),            # etaco, etayz, etaxy
        z(N),                                                       # rho
        z(face_shape(N, 1)), z(face_shape(N, 2)), y(face_shape(N, 3)),                     # rhow, rhou, rhov
        z(face_shape(N, 1)),                                                    # Drho
        z(N), z(N),                                           # rhoo, rhooo
        z(N), z(N), z(N),                               # drhodt, drhodto, drhodtoo
        z(N),                                                       # MFS
        z(face_shape(N, 1)), z(face_shape(N, 1)),                                     # rhoWo, rhoWoo
        z(face_shape(N, 2)), z(face_shape(N, 2)),                                     # rhoUo, rhoUoo
        y(face_shape(N, 3)), y(face_shape(N, 3)),                                     # rhoVo, rhoVoo
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

Face densities `(rhow, rhou[, rhov])`, compact face shape.
"""
face_density(f::FluidState{<:Any, D}) where {D} = per_axis(Val(D), f.rhow, f.rhou, f.rhov)

# BD2 momentum-flux history, one and two steps back
momentum_lag1(f::FluidState{<:Any, D}) where {D} = per_axis(Val(D), f.rhoWo, f.rhoUo, f.rhoVo)
momentum_lag2(f::FluidState{<:Any, D}) where {D} = per_axis(Val(D), f.rhoWoo, f.rhoUoo, f.rhoVoo)

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
    PhaseState{T,D,A,AB,B}

Constitutive workspace consumed and produced by `update!` — everything the
MATLAB `src/update.m` reads/writes apart from the primary fluid-mechanics
unknowns. Pairs with a `FluidState` of the same cell counts `N`; shapes follow
the notes above the `FluidState` definition.

* `x`, `m`, `X`, `M` — phase fractions (mass) and phase densities, cells.
* `chi`, `mu` — volume fractions in `[ε, 1-ε]`.
* `chiw`/`muw`, `chiu`/`muu`, `chiv`/`muv` — volume fractions on ghosted faces 1, 2, 3.
* `x_w`/`m_w`, `x_u`/`m_u`, `x_v`/`m_v` — mass fractions on compact faces 1, 2, 3.
* `Drhom`, `Drhox` — phase density contrasts on compact z-faces.
* `Pl`, `Pt` — lithostatic and total pressure.
* `etamix` — permission-weighted mixture viscosity; `etae` the eddy viscosity
  `fReL * ke * rho`; `ke`, `kx` eddy and effective particle diffusivity.
* `V` — convection-speed magnitude; `Div_V` — velocity divergence.
* `ezz`, `exx`, `eyy` (cells), `exz`, `eyz`, `exy` (z–x, z–y, x–y edges), `eII` — strain rates.
* `tzz`, `txx`, `tyy`, `txz`, `tyz`, `txy`, `tII` — stresses, located as the strain rates.
* `fReL`, `ReL`, `ReD`, `Ra`, … — Re-dependent ramp factors and dimensionless numbers.
* `hasx`, `hasm` — Bool phase indicators.
* `Wx`/`Wm`, `Ux`/`Um`, `Vx`/`Vm` — crystal/melt velocities on ghosted faces 1, 2, 3;
  `wx`, `wm` — segregation speeds on ghosted z-faces.
* `qz_*`, `qx_*`, `qy_*` — advective and diffusive face fluxes of `X` and `M`,
  padded as in `advect_with_flux!` (ghosted faces 1, 2, 3).

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
    # boundary taper (zeroes wx at closed top/bot); `bndshape` is the top-
    # localized exponential profile used by the boundary crystallization
    # reaction Gx = G0·(1-x)·bndshape
    bndtaperw::A
    bndshape::A
    # phase evolution rates (current + history for BD2)
    advn_X::A;  advn_M::A;  advn_rho::A
    # padded face fluxes from advect_with_flux!/diffus_with_flux! (src/advect.m, diffus.m)
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

Allocate a 2-D or 3-D `PhaseState` with all fields zero-initialized, sizing per
the docstring.
"""
function PhaseState(::Type{T}, backend, N::Vararg{Integer, D}) where {T<:AbstractFloat, D}
    N = Int.(N)
    z(dims) = xcore_zeros(backend, T, dims...)
    y(dims) = D == 3 ? z(dims) : nothing
    b() = KernelAbstractions.zeros(backend, Bool, N...)
    c() = z(N)
    A = typeof(c())
    return PhaseState{T, D, A, typeof(b()), D == 3 ? A : Nothing}(
        c(), c(), c(), c(),                                                  # x, m, X, M
        c(), c(), c(), c(),                                                  # Xo, Xoo, Mo, Moo
        c(), c(),                                                            # chi, mu
        b(), b(),                                                            # hasx, hasm
        z(ghost_face_shape(N, 1)), z(ghost_face_shape(N, 1)),                            # chiw, muw
        z(ghost_face_shape(N, 2)), z(ghost_face_shape(N, 2)),                            # chiu, muu
        y(ghost_face_shape(N, 3)), y(ghost_face_shape(N, 3)),                            # chiv, muv
        z(face_shape(N, 1)), z(face_shape(N, 1)),                                        # x_w, m_w
        z(face_shape(N, 2)), z(face_shape(N, 2)),                                        # x_u, m_u
        y(face_shape(N, 3)), y(face_shape(N, 3)),                                        # x_v, m_v
        z(face_shape(N, 1)), z(face_shape(N, 1)),                                        # Drhom, Drhox
        c(), c(),                                                            # Pl, Pt
        c(), c(),                                                            # etamix, etae
        c(), c(),                                                            # ke, kx
        c(), c(), z(face_shape(N, 1)),                                             # etat, etas, etasw
        c(),                                                                 # ks
        c(), c(),                                                            # V, Div_V
        c(), c(), y(N),                                                # exx, ezz, eyy
        z(edge_shape(N, 1, 2)), y(edge_shape(N, 1, 3)), y(edge_shape(N, 2, 3)),               # exz, eyz, exy
        c(),                                                                 # eII
        c(), c(),                                                            # vx, vm
        c(), c(), y(N),                                                # txx, tzz, tyy
        z(edge_shape(N, 1, 2)), y(edge_shape(N, 1, 3)), y(edge_shape(N, 2, 3)),               # txz, tyz, txy
        c(),                                                                 # tII
        c(), c(), c(), c(),                                                  # fReL, ReL, ReD, Ra
        c(), c(), c(), c(),                                                  # fRel, Rel, Red, Rc
        z(ghost_face_shape(N, 1)), z(ghost_face_shape(N, 1)),                            # Wx, Wm
        z(ghost_face_shape(N, 2)), z(ghost_face_shape(N, 2)),                            # Ux, Um
        y(ghost_face_shape(N, 3)), y(ghost_face_shape(N, 3)),                            # Vx, Vm
        z(ghost_face_shape(N, 1)), z(ghost_face_shape(N, 1)),                            # wx, wm
        z(ghost_face_shape(N, 1)), c(),                                            # bndtaperw, bndshape
        c(), c(), c(),                                                       # advn_X, advn_M, advn_rho
        # padded fluxes; qz/qx/qy_dffn_M stay zero — no melt diffusion in MATLAB
        (z(ghost_face_shape(N, 1)), z(ghost_face_shape(N, 2)), y(ghost_face_shape(N, 3)))...,  # q*_advn_X
        (z(ghost_face_shape(N, 1)), z(ghost_face_shape(N, 2)), y(ghost_face_shape(N, 3)))...,  # q*_advn_M
        (z(ghost_face_shape(N, 1)), z(ghost_face_shape(N, 2)), y(ghost_face_shape(N, 3)))...,  # q*_dffn_X
        (z(ghost_face_shape(N, 1)), z(ghost_face_shape(N, 2)), y(ghost_face_shape(N, 3)))...,  # q*_dffn_M
        c(), c(),                                                            # dffn_X, Gx
        c(), c(), c(),                                                       # dXdt, dXdto, dXdtoo
    )
end
PhaseState(::Type{T}, backend, grid::Grid) where {T<:AbstractFloat} = PhaseState(T, backend, cells(grid)...)
