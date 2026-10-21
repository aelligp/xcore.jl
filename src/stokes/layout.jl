# Dimension-generic index arithmetic, the accessors that present the 2-D and 3-D
# state types to the solver as tuples, and the frozen operator coefficients.
#
# Axis 1 is z (gravity; closed: Dirichlet normal velocity, free/no-slip tangential
# via `top_cnv`/`bot_cnv`). Axes 2..D are horizontal (x, then y in 3-D), each
# periodic or bounded by solid walls whose slip is set by `sds`. Velocity
# component d lives on the faces normal to axis d. With N the interior cell
# counts and e_d the unit vector along d:
#
#   ghosted state    v[d]: N .+ 2 .- e_d        P: N .+ 2
#   compact cache    R[d]: N .+ e_d             cell fields: N
#
# Compact face index J of component d ↔ velocity index J + 1 - e_d; cell c ↔
# P[c + 1]. The edge viscosity of the axis pair (a, b), a < b, has shape
# N .+ e_a .+ e_b and the pairs are stored in lexicographic order (1,2), (1,3),
# (2,3): 2-D `(etaco,)`, 3-D `(etaxz, etayz, etaxy)`.
#
# Sums over axes run horizontal axes first, then z: the order the stencils of
# the divergence, the Schur diagonal and the advection flux balance use.

# axes other than d, in increasing order
@inline _others(::Val{d}, ::Val{D}) where {d, D} = ntuple(i -> i < d ? i : i + 1, Val(D - 1))

# position of the axis pair (a, b) in the lexicographic edge-viscosity tuple
@inline function _pair(a::Int, b::Int, D::Int)
    a > b && return _pair(b, a, D)
    return (a - 1) * D - ((a - 1) * a) ÷ 2 + b - a
end

# Σ over the tuple in horizontal-then-z order
@inline _hsum(g::NTuple{D}) where {D} = _tfoldl(+, g[2], (ntuple(k -> g[k + 2], Val(D - 2))..., g[1]))

# compile-time loop over the components 1..D
@inline _each_axis(f, ::Val{D}) where {D} = (ntuple(d -> (f(Val(d)); nothing), Val(D)); nothing)

# ---------------------------------------------------------------------------
# State accessors
# ---------------------------------------------------------------------------

"""
    velocity(fluid) -> NTuple{D}

Ghosted face velocities `(W, U)` (2-D) or `(W, U, V)` (3-D), one per axis.
"""
velocity(f::FluidState) = (f.W, f.U)
velocity(f::FluidState3D) = (f.W, f.U, f.V)

"""
    edge_viscosity(fluid) -> tuple

Edge viscosities in the pair order (1,2), (1,3), (2,3): `(etaco,)` in 2-D,
`(etaxz, etayz, etaxy)` in 3-D.
"""
edge_viscosity(f::FluidState) = (f.etaco,)
edge_viscosity(f::FluidState3D) = (f.etaxz, f.etayz, f.etaxy)

"""
    face_density(fluid) -> NTuple{D}

Face densities `(rhow, rhou[, rhov])`, compact face shape.
"""
face_density(f::FluidState) = (f.rhow, f.rhou)
face_density(f::FluidState3D) = (f.rhow, f.rhou, f.rhov)

# BD2 momentum-flux history, one and two steps back
momentum_lag1(f::FluidState) = (f.rhoWo, f.rhoUo)
momentum_lag1(f::FluidState3D) = (f.rhoWo, f.rhoUo, f.rhoVo)
momentum_lag2(f::FluidState) = (f.rhoWoo, f.rhoUoo)
momentum_lag2(f::FluidState3D) = (f.rhoWoo, f.rhoUoo, f.rhoVoo)

"""
    cells(grid) -> NTuple{D,Int}

Interior cell counts in axis order: `(Nz, Nx)` or `(Nz, Nx, Ny)`.
"""
cells(g::Grid) = (g.Nz, g.Nx)
cells(g::Grid3D) = (g.Nz, g.Nx, g.Ny)

_ndims(::NTuple{D}) where {D} = D

# zero-initialized state of the same kind at another resolution
_similar_fluid(::FluidState{T}, backend, N::NTuple{2,Int}) where {T} = FluidState(T, backend, N...)
_similar_fluid(::FluidState3D{T}, backend, N::NTuple{3,Int}) where {T} = FluidState3D(T, backend, N...)

# interior (ghost ring removed) of a ghosted cell field
_interior(P::AbstractArray{<:Any,D}) where {D} = view(P, ntuple(k -> 2:size(P, k) - 1, Val(D))...)

# ---------------------------------------------------------------------------
# Frozen operator coefficients and boundary conditions of one solve
# ---------------------------------------------------------------------------

"""
    StokesOp{D,T}

Coefficients of the momentum/continuity operator that stay fixed during one
solve: BD2 weights `a1, a2, a3`, time step `dt` (`inv_dt = 0` for `dt = Inf`),
gravity `g0`, the inertia coefficient of each component `cin` (`a1`, plus the
horizontal-inertia factor `gamma` on periodic horizontal axes), the periodicity
of each axis (axis 1 never), the side-wall slip `sds`, and the top/bottom
tangential reflections `top_cnv`, `bot_cnv`.
"""
struct StokesOp{D, T}
    a1::T;  a2::T;  a3::T
    dt::T;  inv_dt::T
    g0::T
    cin::NTuple{D, T}
    periodic::NTuple{D, Bool}
    sds::T
    top_cnv::T;  bot_cnv::T
end

"""
    StokesOp(T, Val(D), par; a1, a2, a3, gamma, dt, xBC, yBC, zBC, sds,
             top_cnv, bot_cnv) -> StokesOp{D,T}

Validate the boundary conditions the matrix-free kernels implement and collect
the operator coefficients. `xBC`/`yBC` ∈ (`:periodic`, `:closed`); `yBC` is read
only for D = 3. `zBC` must be `:closed`.
"""
function StokesOp(::Type{T}, ::Val{D}, par::Parameters;
                  a1::Real, a2::Real, a3::Real, gamma::Real, dt::Real,
                  xBC::Symbol = :periodic, yBC::Symbol = :periodic,
                  zBC::Symbol = :closed, sds::Integer = -1,
                  top_cnv::Integer = 1, bot_cnv::Integer = 1) where {T, D}
    zBC === :closed || throw(ArgumentError(
        "matrix-free Stokes solvers implement only zBC = :closed, got $zBC"))
    hbc = D == 2 ? (xBC,) : (xBC, yBC)
    for bc in hbc
        bc in (:periodic, :closed) || throw(ArgumentError(
            "matrix-free Stokes solvers implement horizontal BCs :periodic or :closed, got $bc"))
    end
    periodic = (false, map(==(:periodic), hbc)...)
    inv_dt = isfinite(dt) ? T(inv(dt)) : zero(T)
    cin = ntuple(d -> T(a1) + (periodic[d] ? T(gamma) : zero(T)), Val(D))
    return StokesOp{D, T}(T(a1), T(a2), T(a3), T(dt), inv_dt, T(par.g0), cin,
                          periodic, T(sds), T(top_cnv), T(bot_cnv))
end
