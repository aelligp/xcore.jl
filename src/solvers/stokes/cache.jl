# Preallocated storage of the matrix-free Stokes/Navier–Stokes solvers. Every
# per-component field is an `NTuple{D}` indexed by velocity component, compact
# face shape `N .+ e_d` (no ghosts; the kernels index the ghosted velocities of
# the fluid state directly). Cell-centered fields have shape `N`.

# Anderson-acceleration history of the Powell-Hestenes multiplier iteration
# (see `anderson_pressure_update!`). Cell-centered arrays and the host-side
# m×m least-squares scratch, allocated on first use.
mutable struct AndersonPH{T<:AbstractFloat, A<:AbstractArray{T}}
    prev::Vector{A}     # [pressure, update] of the previous sweep
    dX::Vector{A}       # pressure differences, ring buffer of depth m
    dF::Vector{A}       # update differences, same slots as dX
    G::Matrix{T}        # Gram matrix ΔFᵀΔF
    b::Vector{T}        # ΔFᵀf
    n::Int              # valid history columns
    slot::Int           # last column written
    have_prev::Bool     # `prev` holds the previous sweep
end
AndersonPH{T, A}() where {T, A} =
    AndersonPH{T, A}(A[], A[], A[], zeros(T, 0, 0), T[], 0, 0, false)

"""
    StokesCache{D,T,A}

Scratch and autotune state of the dynamic-relaxation (DR) / Powell-Hestenes
solve on one grid:

* `γ_eff`, `η_b` — per-cell augmented-Lagrangian penalty and the coefficient of
  the pseudo-transient continuity term (multigrid smoother only).
* per component: Gershgorin diagonal `Dv`, `λmax`, pseudo-rate `dvdτ`, residual `Rv`
  and its snapshot `Rv_old`, frozen momentum advection `advn` (zero on Dirichlet
  faces), and the injected multigrid right-hand side `bext`.
* continuity residual `R_P`, Schur penalty `P_num = γ_eff·R_P`, injected
  `bext_P`.
* `P0` (ghosted pressure shape): the reference pressure of a solve (gauge level,
  pseudo-transient term); zero on multigrid coarse levels.
* `dP`: the interior pressure change of the last multiplier update or V-cycle.
* `c` (global damping from the Rayleigh quotient) and `γfact` (penalty factor
  learned by the last solve; 0 = start from `par.γfact_PT`).
"""
mutable struct StokesCache{D, T<:AbstractFloat, A<:AbstractArray{T, D}}
    const γ_eff::A
    const η_b::A
    const Dv::NTuple{D, A}
    const λmax::NTuple{D, A}
    const dvdτ::NTuple{D, A}
    const Rv::NTuple{D, A}
    const Rv_old::NTuple{D, A}
    const advn::NTuple{D, A}
    const bext::NTuple{D, A}
    const R_P::A
    const P_num::A
    const bext_P::A
    const P0::A
    const dP::A
    const anderson::AndersonPH{T, A}
    c::T
    γfact::T
end

"""
    StokesCache(T, backend, N::NTuple{D,Int}) -> StokesCache{D,T}
    StokesCache(T, backend, grid) -> StokesCache

Allocate a zero-initialized cache for `N` interior cells on `backend`.
"""
function StokesCache(::Type{T}, backend, N::NTuple{D, Int}) where {T<:AbstractFloat, D}
    z(dims) = xcore_zeros(backend, T, dims...)
    face(d) = N .+ ntuple(k -> Int(k == d), Val(D))
    fv() = ntuple(d -> z(face(d)), Val(D))
    A = typeof(z(N))
    return StokesCache{D, T, A}(z(N), z(N),
                                fv(), fv(), fv(), fv(), fv(), fv(), fv(),
                                z(N), z(N), z(N), z(N .+ 2), z(N),
                                AndersonPH{T, A}(), zero(T), zero(T))
end
StokesCache(::Type{T}, backend, grid::Grid) where {T<:AbstractFloat} =
    StokesCache(T, backend, cells(grid))

cells(c::StokesCache) = size(c.R_P)

"""
    DyrelCache

Alias of [`StokesCache`](@ref): `DyrelCache(T, backend, grid)` builds the
cache `fluidmech_dyrel!` takes.
"""
const DyrelCache = StokesCache
