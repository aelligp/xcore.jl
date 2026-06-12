using Statistics

# Penalty parameter + bulk-viscosity setup for the DYREL Stokes solver.
# Direct adaptation of JustRelax.jl's `compute_bulk_viscosity_and_penalty!`.
#
# In xcore the bulk modulus K is effectively infinite (incompressible),
# so the "physical" penalty `γ_phy = K·dt` is treated as `γfact·⟨η⟩` and the
# harmonic mean collapses to the numerical penalty alone. The same field
# `γ_eff` is used as both the Powell-Hestenes Schur-complement penalty (in
# the velocity residual) and the per-iter pressure-update gain.
#
# Mirrors `DESIGN.md` §5.4.

@kernel function _dyrel_init_penalty_kernel!(γ_eff, η_b, γ_num, K_b_dt)
    iz, ix = @index(Global, NTuple)
    @inbounds begin
        # When K_b·dt < ∞ (compressible), the physical penalty is K_b·dt.
        # When K_b·dt = ∞ (incompressible), we substitute γ_num.
        γ_phy = isfinite(K_b_dt) ? K_b_dt : γ_num
        # Harmonic mean of physical and numerical — picks the smaller; avoids
        # blow-up at the incompressible limit.
        γ_eff[iz, ix] = γ_phy * γ_num / (γ_phy + γ_num)
        # Bulk viscosity for the (P - P_o)/(η_b·dt) compressibility term in
        # R_P. Set to γ_num at K → ∞ so the term vanishes (P - P_o stays
        # bounded as ∇·v → 0).
        η_b[iz, ix]   = γ_phy
    end
end

"""
    init_dyrel_penalty!(cache, η; γfact, K_b = Inf, dt = Inf) -> nothing

Populate `cache.γ_eff` and `cache.η_b` from the current viscosity field `η`
and the user's penalty factor `γfact` (= `par.γfact_PT`).

xcore's default is incompressible (`K_b = Inf`), so the physical penalty
defaults to the numerical one. For compressible runs pass `K_b` from
`par.Kb` (when that field exists) and the BD2 time-step `dt`.

The penalty `γ_num = γfact·⟨η⟩` is in viscosity units. xcore's continuity
residual is scaled by `1/ρ` (in `dyrel_residual_P!`) so the Schur-complement
term `γ_eff·R_P` injected into momentum is `~γ_eff·∇v`, balanced against the
viscous operator exactly as the Gershgorin λmax assumes (which carries γ_eff
without any ρ factor). This matches JustRelax's ρ=1 formulation.
"""
function init_dyrel_penalty!(cache::DyrelCache{T}, η::AbstractMatrix{T};
                             γfact::Real,
                             K_b::Real = T(Inf),
                             dt::Real  = T(Inf)) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(η)
    η_mean  = T(mean(η))
    γ_num   = T(γfact) * η_mean
    K_b_dt  = T(K_b) * T(dt)
    _dyrel_init_penalty_kernel!(backend)(cache.γ_eff, cache.η_b,
                                                   γ_num, K_b_dt;
                                                   ndrange = size(cache.γ_eff))
    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    compute_AL_penalty!(cache, fluid, h; γfact, a1, gamma, dt) -> nothing

Operator-derived per-cell augmented-Lagrangian penalty (ported from
`src/dyrel/init.jl`): `γ_eff[i] = γfact / s_P_phys[i]`, with `s_P_phys` the Schur
diagonal of the PENALTY-FREE momentum operator (`gershgorin!` with `γ_eff=0` →
D_phys; `compute_schur_diag!`). `1/s_P` is a viscosity `[Pa·s]` carrying the full
operator scaling (viscous + inertia), so `γfact` is a dimensionless, scale-
invariant over-penalisation factor. Sets `η_b=γ_eff` (matched comp coefficient)
and leaves `cache.D_W/D_U = D_full` (penalty-laden) for the velocity
preconditioner/λmax. Also snapshots the penalty-free faces into
`cache.D_W_phys/D_U_phys` (for the MG pressure-Poisson κ = ρ/D_phys, matching the
physical Schur s_P_phys). Replaces `init_dyrel_penalty!` in the GMG smoother setup.
"""
function compute_AL_penalty!(cache::DyrelCache{T}, fluid::FluidState{T}, h::Real;
                             γfact::Real, a1::Real, gamma::Real, dt::Real) where {T<:AbstractFloat}
    fill!(cache.γ_eff, zero(T))
    gershgorin!(cache, fluid, h; a1, gamma, dt)        # D_phys (γ_eff = 0)
    # snapshot the penalty-free face diagonals before step-4 overwrites D_W/D_U
    # with D_full — the MG pressure-Poisson κ = ρ/D_phys is built from these.
    copyto!(cache.D_W_phys, cache.D_W)
    copyto!(cache.D_U_phys, cache.D_U)
    compute_schur_diag!(cache, fluid, h)               # s_P_phys
    γ = T(γfact)
    @. cache.γ_eff = γ / cache.s_P
    @. cache.η_b   = cache.γ_eff
    gershgorin!(cache, fluid, h; a1, gamma, dt)        # D_full (with γ_eff)
    return nothing
end
