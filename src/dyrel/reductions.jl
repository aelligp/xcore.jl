using KernelAbstractions
using LinearAlgebra: norm

# Global reductions for xcore-DYREL: the Rayleigh-quotient λ_min estimate and
# the residual norms used for convergence checks. These are the only non-local
# operations in the solver; everything else is embarrassingly parallel.
#
# On CPU these are plain `sum`/`norm` over the arrays. On GPU backends, KA
# arrays dispatch through their own `sum`/`mapreduce`, so no special-casing is
# needed (the reductions just run on-device).

"""
    dyrel_rayleigh_λmin(cache) -> λmin

Estimate the minimum eigenvalue of the (preconditioned) velocity operator via
the Rayleigh quotient (DESIGN §5.2, JustRelax src/DYREL/solver.jl `compute_λminV!`):

    λmin = |Σ ΔW·(R_W − R_W_old) + Σ ΔU·(R_U − R_U_old)|
           / (Σ ΔW² + Σ ΔU²)

where `ΔW = cache.dW_step` is the velocity increment applied last step. Uses
the residual snapshots `cache.R_W_old`, `cache.R_U_old` captured before the
current DR sweep. Returns a positive scalar (caller guards against zero).
"""
function dyrel_rayleigh_λmin(cache::DyrelCache{T}) where {T<:AbstractFloat}
    dW = cache.dW_step;  dU = cache.dU_step
    # numerator: dot(ΔV, ΔR) summed across both velocity components
    num = abs(_dyrel_dot_diff(dW, cache.R_W, cache.R_W_old) +
              _dyrel_dot_diff(dU, cache.R_U, cache.R_U_old))
    # denominator: ΣΔV²
    den = _dyrel_sumsq(dW) + _dyrel_sumsq(dU)
    return den > zero(T) ? T(num / den) : zero(T)
end

# Fused, allocation-free, GPU-friendly reduction: reduce over a LAZY `Broadcasted`
# so the elementwise expression is never materialised into a temporary array. A
# multi-array `mapreduce(f, +, A, B, C)` zips on the host → forces scalar indexing
# on GPU arrays (and allocates ~N²); an eager `sum(@. f(A,B))` allocates an N²
# temp. `sum(_bc(f, A, B, C))` does neither: GPUArrays specialises
# `mapreduce(::Any, +, ::Broadcasted)` to a single on-device pass (≈96 B host
# overhead, no scalar indexing). Mirrors JustRelax's `norm_mpi(D .* R)` pattern.
@inline _bc(f, args...) = Broadcast.instantiate(Broadcast.broadcasted(f, args...))

# Σ a·(b − c) — fused over the three arrays in lockstep, no temporary.
_dyrel_dot_diff(a::AbstractMatrix{T}, b::AbstractMatrix{T}, c::AbstractMatrix{T}) where {T} =
    sum(_bc((ai, bi, ci) -> ai * (bi - ci), a, b, c))

_dyrel_sumsq(a::AbstractMatrix{T}) where {T} = sum(abs2, a)   # already temp-free

"""
    dyrel_residual_norms(cache, rho) -> (normW, normU, normP)

L2 norms of the momentum residuals and the continuity residual, each normalised
by √N (RMS) so the magnitudes are grid-independent.

`cache.R_P` is the `1/ρ`-scaled continuity residual `(MFS−∇·ρv)/ρ` (the scaling
is needed inside the DR step so the Schur penalty matches the Gershgorin λmax).
For the convergence gate we want the *true* mass-flux imbalance, so `normP`
multiplies `R_P` back by the cell density `rho`. Reporting the scaled residual
overstates conservation by a factor ρ (≈2700) — it is the actual `∇·(ρv)−MFS`
that must be small for mass to be conserved.
"""
function dyrel_residual_norms(cache::DyrelCache{T}, rho::AbstractMatrix{T}) where {T<:AbstractFloat}
    # norm(R_W) = √Σ R_W² is already temp-free (sum(abs2) is a mapreduce). For
    # the ρ-weighted P-norm, mapreduce avoids the `@. R_P*rho` temporary.
    normW = T(sqrt(sum(abs2, cache.R_W)) / sqrt(length(cache.R_W)))
    normU = T(sqrt(sum(abs2, cache.R_U)) / sqrt(length(cache.R_U)))
    normP = T(sqrt(sum(_bc((r, ρ) -> abs2(r * ρ), cache.R_P, rho))) /
              sqrt(length(cache.R_P)))
    return normW, normU, normP
end

"""
    dyrel_pressure_update_norm(cache, fluid, h) -> RMS(δP) / max(RMS(P̃), RMS(τ_II))

Size of the pending Powell-Hestenes multiplier update `δP = γ_eff·R_P`, measured
against the pressure scale of the current flow. `≤ atol` is the mixed criterion

    RMS(δP) ≤ atol · max(RMS(P̃), RMS(τ_II))

where `P̃ = P − mean(P)` is the interior pressure without its free constant mode
(relative part) and `τ_II` the second invariant of the deviatoric stress
`τ = η(ε̇ − ∇·v/3)` (absolute part: the stress the pressure balances in the
momentum equation, independent of the pressure field). Both are invariant under
a constant pressure offset, and the stress keeps the scale nonzero when the
pressure is uniform. With neither (fluid at rest under a uniform pressure) the
update is returned unscaled.

The continuity residual alone does not bound the pressure error: the multiplier
step is γ_eff·R_P, so at a fixed `R_P` the pressure a solve returns is off by an
amount that grows with the penalty. Gating on this instead makes the accepted
pressure independent of how strongly the constraint is penalised.
"""
function dyrel_pressure_update_norm(cache::DyrelCache{T}, fluid::FluidState{T},
                                    h::Real) where {T<:AbstractFloat}
    dP    = sqrt(sum(_bc((g, r) -> abs2(g * r), cache.γ_eff, cache.R_P)) / length(cache.R_P))
    scale = max(_pressure_deviation_rms(fluid.P), _deviatoric_stress_rms(fluid, T(inv(h))))
    return scale > zero(T) ? T(dP / scale) : T(dP)
end

# RMS of the interior pressure about its mean (two passes: the mean can dominate).
function _pressure_deviation_rms(P::AbstractMatrix{T}) where {T}
    Pint = @view P[2:end-1, 2:end-1]
    Pm   = sum(Pint) / length(Pint)
    return sqrt(sum(_bc(p -> abs2(p - Pm), Pint)) / length(Pint))
end

# RMS of τ_II = √(½τxx² + ½τzz² + τxz²), with the stencils of the residual kernels:
# normal stresses at cell centres, shear at corners (boundary ghosts carry the BCs).
function _deviatoric_stress_rms(fluid::FluidState{T}, invh::T) where {T}
    W, U = fluid.W, fluid.U
    third = T(1//3);  half = T(1//2)
    normal = sum(_bc(@view(W[2:end, 2:end-1]), @view(W[1:end-1, 2:end-1]),
                     @view(U[2:end-1, 2:end]), @view(U[2:end-1, 1:end-1]), fluid.eta) do w1, w0, u1, u0, η
        Wz = (w1 - w0) * invh;  Ux = (u1 - u0) * invh;  div = (Wz + Ux) * third
        half * (abs2(η * (Wz - div)) + abs2(η * (Ux - div)))
    end) / length(fluid.eta)
    shear = sum(_bc(@view(U[2:end, :]), @view(U[1:end-1, :]),
                    @view(W[:, 2:end]), @view(W[:, 1:end-1]), fluid.etaco) do u1, u0, w1, w0, η
        abs2(η * ((u1 - u0) + (w1 - w0)) * invh * half)
    end) / length(fluid.etaco)
    return sqrt(normal + shear)
end

"""
    dyrel_velocity_residual_norm(cache) -> err

Preconditioner-weighted velocity-residual RMS used for the inner-DR
convergence test (JustRelax weights by the Jacobi diagonal `D` so the
inner-loop error is dimensionally consistent across viscosity contrasts):

    err = max( RMS(D_W·R_W), RMS(D_U·R_U) )
"""
function dyrel_velocity_residual_norm(cache::DyrelCache{T}) where {T<:AbstractFloat}
    # lazy-broadcast reduce avoids the `@. D·R` temporaries (two N²-sized allocs
    # per call) without the scalar-indexing of a multi-array mapreduce.
    errW = T(sqrt(sum(_bc((d, r) -> abs2(d * r), cache.D_W, cache.R_W))) /
             sqrt(length(cache.R_W)))
    errU = T(sqrt(sum(_bc((d, r) -> abs2(d * r), cache.D_U, cache.R_U))) /
             sqrt(length(cache.R_U)))
    return max(errW, errU)
end
