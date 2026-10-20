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
# viscosity [Pa·s] and γfact is DIMENSIONLESS. At dt→∞ (steady/creeping) D_phys
# is viscous-dominated and the creeping penalty is recovered.
#
# s_P fixes the SCALE of the Schur complement, not its SPECTRUM. The
# Powell-Hestenes contraction is ρ = 1/(1 + γfact·λmin(diag(s_P)⁻¹·S)), and that
# λmin is O(1) only in the viscous-dominated limit where S is mass-matrix-like.
# Once a1·ρ/dt ≫ η/h² the momentum block is the mass matrix, S becomes a Poisson
# operator and λmin falls with h², so no fixed γfact holds the contraction across
# resolutions or time steps. `fluidmech_dyrel!` therefore treats `par.γfact_PT`
# as a starting value and re-derives it from the contraction it measures.
#
# The same field γ_eff is used CONSISTENTLY as (i) the momentum penalty injected
# via P_num = γ_eff·R_P, (ii) the multiplier update P += ω·γ_eff·R_P, and
# (iii) η_b, the coefficient of the pseudo-transient term comp = (P−P0)/(η_b·dt)
# that the multigrid smoother adds to the continuity residual (η_b = γ_eff
# wherever ρ·dt ≥ 1; floored above γ_eff below that). The single-grid
# solver passes dt = Inf to `dyrel_residual_P!` and so carries no comp term: the
# assembled constraint is ∇·(ρv) = MFS with no pressure entry, and at a fixed P0
# the term would not vanish at the fixed point (see MATH.md §2).
# Consistency across all three is what makes the Schur-scaled scheme stable
# (the earlier scalar-η_b vs per-cell-s_P mismatch caused divergence).

"""
    compute_AL_penalty!(cache, fluid, h; γfact, a1, gamma, dt, g0) -> nothing

Build the per-cell augmented-Lagrangian penalty `cache.γ_eff = γfact / s_P_phys`
(and set `cache.η_b = cache.γ_eff`), then leave `cache.D_W/D_U` holding the
penalty-laden diagonal `D_full` (Gershgorin with the new γ_eff) ready for the
velocity Jacobi preconditioner and λmax. Reuses `gershgorin!` and
`compute_schur_diag!` verbatim. Call once per `fluidmech_dyrel!` entry.

`γfact` is the dimensionless over-penalisation factor (`par.γfact_PT`); `a1`,
`gamma`, `dt`, `g0` are the BD2 / horizontal-inertia / time-step / gravity
coefficients passed to `gershgorin!`.
"""
function compute_AL_penalty!(cache::DyrelCache{T}, fluid::FluidState{T}, h::Real;
                             γfact::Real, a1::Real, gamma::Real, dt::Real,
                             g0::Real, xBC::Symbol = :periodic) where {T<:AbstractFloat}
    # 1. penalty-free momentum diagonal D_phys (viscous + inertia, γ_eff = 0)
    fill!(cache.γ_eff, zero(T))
    gershgorin!(cache, fluid, h; a1, gamma, dt, g0, xBC)
    # 2. physical Schur diagonal from D_phys
    compute_schur_diag!(cache, fluid, h; xBC)
    # 3. operator-derived per-cell penalty γ_eff = γfact / s_P_phys; η_b matched
    γ = T(γfact)
    @. cache.γ_eff = γ / cache.s_P
    # The multiplier update P += ω·γ_eff·R_P sees comp/ρ as the diagonal
    # ω·γ_eff/(η_b·ρ·dt): amplification 1 − that, unstable above 2. Flooring
    # η_b at γ_eff/(ρ·dt) caps the diagonal at ω (deadbeat for ω = 1).
    dtT = T(dt)
    @. cache.η_b   = cache.γ_eff * max(one(T), inv(fluid.rho * dtT))
    # 4. recompute the diagonal WITH the penalty (D_full) for velocity precond/λmax
    gershgorin!(cache, fluid, h; a1, gamma, dt, g0, xBC)
    return nothing
end
