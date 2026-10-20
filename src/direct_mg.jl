# Assembled-operator multigrid for the `:direct` system (`solver = :gmg_direct`).
#
# `fluidmech!` assembles and Jacobi-scales LL = [A G; D K]. Instead of an LU
# factorization, the update is found by FGMRES, right-preconditioned with the
# block upper-triangular factor of LL,
#     P = [Â G; 0 Ŝ],   Â⁻¹ = one AMG V-cycle on A,
# and a Cahouet–Chabard approximation of the Schur complement S = K − D·A⁻¹·G:
#     Ŝ⁻¹ = diag(K − D·diag(A_v)⁻¹·G)⁻¹ + (K − D·M⁻¹·G)⁻¹,
# where M is the inertial part of diag(A) (a1·ρ/dt, plus the artificial
# horizontal inertia on U) and A_v = diag(A) − M. The first term is spectrally
# equivalent to the 1/η-weighted pressure mass matrix (viscous limit); the
# second is the density-cancelling pressure Poisson operator (inertial limit),
# inverted by one AMG V-cycle. Host-CPU only, like the rest of `fluidmech!`.
#
# The AMG hierarchies are built on interior operators: the ghost-copy and
# Dirichlet boundary rows are eliminated exactly first (`_ReducedAMG`). Those
# rows make the full blocks strongly non-symmetric, and Ruge–Stüben coarsening
# of the full blocks diverges. The velocity block is then coarsened unknown by
# unknown (separate W and U hierarchies, cross-coupling dropped from the
# preconditioner): scalar AMG across the U–W cross terms loses its
# strength-of-connection and needs O(N) iterations.

using AlgebraicMultigrid: ruge_stuben, aspreconditioner
using SparseArrays: spdiagm, dropzeros, nnz
using Krylov: fgmres

mutable struct DirectMGStats
    converged::Bool
    niter::Int
end

# Boundary block M[b,b] = Db + E with E nilpotent: a boundary row couples at
# most to other boundary rows that end the chain (a ghost corner points at a
# Dirichlet row). `x = M[b,b]⁻¹·r` by the terminating Neumann series.
function _bb_solve(invdb, E, r, nsweep)
    x = invdb .* r
    for _ in 1:nsweep
        x = invdb .* (r - E * x)
    end
    return x
end

# Preconditioner `ldiv!(y, R, x)` for M: the boundary rows (all rows outside
# `groups`) are eliminated exactly, and the interior operator is approximated
# by one AMG V-cycle per group, the coupling between groups dropped.
struct _ReducedAMG{T,P,MT}
    i::Vector{Int}
    b::Vector{Int}
    invdb::Vector{T}
    E::MT
    nsweep::Int
    Mib::MT
    Mbi::MT
    gpos::Vector{UnitRange{Int}}   # positions of each group within `i`
    Pg::Vector{P}
    xi::Vector{T}
    yi::Vector{T}
end

function _ReducedAMG(M::SparseMatrixCSC{T}, groups::AbstractVector{<:AbstractVector{Int}}) where {T}
    i = reduce(vcat, groups)
    allunique(i) || throw(ArgumentError("_ReducedAMG: groups overlap"))
    ends = cumsum(length.(groups))
    gpos = [(e - length(g) + 1):e for (g, e) in zip(groups, ends)]
    b = setdiff(axes(M, 1), i)
    Mbb = M[b, b]
    db = Vector(diag(Mbb))
    all(!iszero, db) || error("_ReducedAMG: zero diagonal in a boundary row")
    invdb = inv.(db)
    E = Mbb - spdiagm(0 => db)
    # E is nilpotent iff (Db⁻¹E)^k vanishes for some k ≤ length(b); in practice k ≤ 2.
    nsweep = 0
    Nk = Diagonal(invdb) * E
    while nnz(dropzeros(Nk)) > 0
        nsweep += 1
        nsweep <= 8 || error("_ReducedAMG: boundary rows couple in a chain longer than 8; " *
                             "the boundary block is not a terminating Neumann series")
        Nk = Diagonal(invdb) * E * Nk
    end
    Mib = M[i, b];  Mbi = M[b, i]
    Mr = M[i, i] - Mib * _bb_solve(invdb, E, Mbi, nsweep)
    Pg = [aspreconditioner(ruge_stuben(dropzeros(Mr[r, r]))) for r in gpos]
    return _ReducedAMG(i, b, invdb, E, nsweep, Mib, Mbi, gpos, Pg,
                       zeros(T, length(i)), zeros(T, length(i)))
end

function LinearAlgebra.ldiv!(y::AbstractVector, R::_ReducedAMG, x::AbstractVector)
    xb = x[R.b]
    yb = _bb_solve(R.invdb, R.E, xb, R.nsweep)
    R.xi .= view(x, R.i)
    mul!(R.xi, R.Mib, yb, -1, 1)
    for (r, P) in zip(R.gpos, R.Pg)
        ldiv!(view(R.yi, r), P, view(R.xi, r))
    end
    y[R.i] .= R.yi
    y[R.b] .= _bb_solve(R.invdb, R.E, xb - R.Mbi * R.yi, R.nsweep)
    return y
end

struct _StokesBlockPrec{T,PA,PS,MG}
    nv::Int
    Pa::PA                 # reduced AMG V-cycle on A
    Ps::PS                 # reduced AMG V-cycle on the inertial Poisson operator, or nothing
    G::MG
    invdS::Vector{T}       # viscous Schur diagonal, inverted
    tmp_v::Vector{T}
    tmp_p::Vector{T}
end

function LinearAlgebra.mul!(y::AbstractVector, P::_StokesBlockPrec, x::AbstractVector)
    nv = P.nv
    xv = view(x, 1:nv);  xp = view(x, nv+1:length(x))
    yv = view(y, 1:nv);  yp = view(y, nv+1:length(y))
    yp .= P.invdS .* xp
    if P.Ps !== nothing
        ldiv!(P.tmp_p, P.Ps, xp)
        yp .+= P.tmp_p
    end
    mul!(P.tmp_v, P.G, yp)
    P.tmp_v .= xv .- P.tmp_v
    ldiv!(yv, P.Pa, P.tmp_v)
    return y
end

"""
    direct_mg_solve(LLs, FF, nv, minert, iv, ip; rtol, itmax) -> (x, stats::DirectMGStats)

Solve the Jacobi-scaled `:direct` system `LLs·x = FF` (velocity unknowns
`1:nv`, pressure after) by AMG-preconditioned FGMRES. `minert` is the inertial
part of the velocity diagonal of `LLs`, in the same scaling. `iv` lists the
interior unknowns of each velocity component, `ip` the interior pressure
unknowns (counted from 1); all other rows are boundary rows.
"""
function direct_mg_solve(LLs::SparseMatrixCSC{T}, FF::AbstractVector{T}, nv::Int,
                         minert::AbstractVector{T}, iv::AbstractVector{<:AbstractVector{Int}},
                         ip::AbstractVector{Int}; rtol::Real, itmax::Int) where {T}
    n = size(LLs, 1)
    length(minert) == nv || throw(DimensionMismatch("minert has length $(length(minert)), expected $nv"))
    A = LLs[1:nv, 1:nv];      G = LLs[1:nv, nv+1:n]
    D = LLs[nv+1:n, 1:nv];    K = LLs[nv+1:n, nv+1:n]

    dv = Vector(diag(A)) .- minert
    all(>(0), dv) || error("direct_mg_solve: viscous part of the momentum diagonal is not " *
                           "positive (min = $(minimum(dv))); the Schur approximation assumes it is")
    dS = Vector(diag(K) .- diag(D * Diagonal(inv.(dv)) * G))
    all(>(0), dS) || error("direct_mg_solve: viscous Schur diagonal is not positive (min = $(minimum(dS)))")

    Ps = if any(>(0), minert)
        minv = map(m -> m > 0 ? inv(m) : zero(T), minert)
        _ReducedAMG(K - D * Diagonal(minv) * G, [ip])
    else
        nothing
    end
    P = _StokesBlockPrec(nv, _ReducedAMG(A, iv), Ps, G,
                         inv.(dS), zeros(T, nv), zeros(T, n - nv))

    x, st = fgmres(LLs, FF; N = P, rtol = T(rtol), atol = zero(T),
                   memory = itmax, restart = false, itmax)
    return x, DirectMGStats(st.solved, st.niter)
end
