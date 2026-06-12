# Operator-derived augmented-Lagrangian penalty setup for the DYREL solver.
#
# The Powell-Hestenes pressure penalty γ_eff must approximate the inverse Schur
# complement S⁻¹ (S = GᵀK⁻¹G) for the multiplier iteration to converge
# geometrically. Rather than a hand-tuned constant γfact·⟨η⟩ (correct only for
# creeping Stokes, S~1/η; needs ρ/dt-retuning at Re>1 where K~a₁ρ/dt), we build
# γ_eff per-cell from the Schur DIAGONAL of the actual frozen operator:
#
#     γ_eff[i] = γfact / s_P_phys[i]
#
# where s_P_phys is `compute_schur_diag!` evaluated on the PENALTY-FREE momentum
# diagonal D_phys (viscous η-convention + BD2 inertia, computed by `gershgorin!`
# with γ_eff = 0, avoiding self-reference). Units: s_P is [s·m/kg], so 1/s_P is a
# viscosity [Pa·s] and γfact is DIMENSIONLESS — the same fixed value converges
# creeping-Stokes and Re>1/dense runs without per-setup tuning (the operator
# scaling lives entirely in s_P). At dt→∞ (steady/creeping) D_phys is
# viscous-dominated and the creeping penalty is recovered.
#
# The same field γ_eff is used CONSISTENTLY as (i) the momentum penalty injected
# via P_num = γ_eff·R_P, (ii) the multiplier update P += ω·γ_eff·R_P, and
# (iii) the artificial-compressibility coefficient η_b in comp = (P−P0)/(η_b·dt).
# Consistency across all three is what makes the Schur-scaled scheme stable
# (the earlier scalar-η_b vs per-cell-s_P mismatch caused divergence).

"""
    compute_AL_penalty!(cache, fluid, h; γfact, a1, gamma, dt) -> nothing

Build the per-cell augmented-Lagrangian penalty `cache.γ_eff = γfact / s_P_phys`
(and set `cache.η_b = cache.γ_eff`), then leave `cache.D_W/D_U` holding the
penalty-laden diagonal `D_full` (Gershgorin with the new γ_eff) ready for the
velocity Jacobi preconditioner and λmax. Reuses `gershgorin!` and
`compute_schur_diag!` verbatim. Call once per `fluidmech_dyrel!` entry.

`γfact` is the dimensionless over-penalisation factor (`par.γfact_PT`); `a1`,
`gamma`, `dt` are the BD2 / horizontal-inertia / time-step coefficients passed to
`gershgorin!`.
"""
function compute_AL_penalty!(cache::DyrelCache{T}, fluid::FluidState{T}, h::Real;
                             γfact::Real, a1::Real, gamma::Real, dt::Real) where {T<:AbstractFloat}
    # 1. penalty-free momentum diagonal D_phys (viscous + inertia, γ_eff = 0)
    fill!(cache.γ_eff, zero(T))
    gershgorin!(cache, fluid, h; a1, gamma, dt)
    # 2. physical Schur diagonal from D_phys
    compute_schur_diag!(cache, fluid, h)
    # 3. operator-derived per-cell penalty γ_eff = γfact / s_P_phys; η_b matched
    γ = T(γfact)
    @. cache.γ_eff = γ / cache.s_P
    @. cache.η_b   = cache.γ_eff
    # 4. recompute the diagonal WITH the penalty (D_full) for velocity precond/λmax
    gershgorin!(cache, fluid, h; a1, gamma, dt)
    return nothing
end
