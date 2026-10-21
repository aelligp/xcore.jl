# Powell-Hestenes / dynamic-relaxation solve of one frozen-coefficient
# Navier–Stokes system (the system one `fluidmech!` call solves):
#
#   frozen momentum advection, operator-derived penalty, then per PH sweep:
#     PH residuals → convergence gate → penalty self-tuning →
#     inner DR loop at fixed P (continuity → penalty P_num → preconditioned
#     momentum residual → damped velocity update → BCs; every n_tune_PT
#     iterations a Rayleigh λmin / Gershgorin λmax retune) →
#     multiplier update P += ω·γ_eff·R_P (optionally Anderson-accelerated)

# Powell-Hestenes penalty self-tuning (see `fluidmech_dyrel!`). A contraction of
# ρ_want per sweep reaches atol_PH = 1e-6 in ~20 sweeps; asking for less costs
# more inner DR work per sweep than it saves in sweeps.
const ρ_want        = 0.5
const γ_win_width   = 4     # PH sweeps the contraction is measured over
const γ_max_retunes = 6
const γ_max_growth  = 50    # per retune, bounding the response to a noisy ρ

_anderson_reset!(aa::AndersonPH) = (aa.n = 0; aa.slot = 0; aa.have_prev = false; nothing)

"""
    _anderson_pressure_update!(aa, P, f, m) -> nothing

Anderson-accelerated (type II, depth `m`) step of the multiplier fixed-point
iteration `P ← P + f`, where `P` is the interior pressure the sweep's velocity
solve ran under and `f` the plain update `ω·γ_eff·R_P` it produced. With the
history `ΔX = [ΔP]`, `ΔF = [Δf]` of earlier sweeps,

    θ = argmin ‖f − ΔF·θ‖₂,    P ← P + f − (ΔX + ΔF)·θ.

Only the m×m least-squares problem runs on the host; every array operation is a
device broadcast or reduction.
"""
function _anderson_pressure_update!(aa::AndersonPH{T}, P, f, m::Int) where {T}
    if length(aa.dX) != m
        aa.prev = [similar(f) for _ in 1:2]
        aa.dX = [similar(f) for _ in 1:m];  aa.dF = [similar(f) for _ in 1:m]
        aa.G = zeros(T, m, m);  aa.b = zeros(T, m)
        _anderson_reset!(aa)
    end
    if aa.have_prev
        aa.slot = aa.slot % m + 1
        aa.dX[aa.slot] .= P .- aa.prev[1]
        aa.dF[aa.slot] .= f .- aa.prev[2]
        aa.n = min(aa.n + 1, m)
    end
    aa.prev[1] .= P;  aa.prev[2] .= f;  aa.have_prev = true
    P .+= f
    aa.n == 0 && return nothing
    n = aa.n;  G = aa.G;  b = aa.b
    for j in 1:n
        for i in 1:j
            G[i, j] = G[j, i] = sum(_bc(*, aa.dF[i], aa.dF[j]))
        end
        b[j] = sum(_bc(*, aa.dF[j], f))
    end
    # The history columns turn nearly collinear as the iteration converges; the
    # pseudo-inverse drops the directions the residual differences cannot resolve.
    θ = pinv(@view(G[1:n, 1:n]); rtol = sqrt(eps(T))) * @view(b[1:n])
    for i in 1:n
        P .-= θ[i] .* (aa.dX[i] .+ aa.dF[i])
    end
    return nothing
end

# Mass-flux source from the density evolution (phsevo!'s `advn_rho`), as the
# direct solver forms it; the outer Picard loop converges it.
function _update_mass_source!(fluid, phase, par, dt; a1, a2, a3, b1, b2, b3)
    T = eltype(fluid.MFS)
    inv_dt = isfinite(dt) ? T(inv(dt)) : zero(T)
    @. fluid.drhodt = -phase.advn_rho
    @. fluid.MFS = fluid.MFS - par.alpha * (
        (T(a1) * fluid.rho - T(a2) * fluid.rhoo - T(a3) * fluid.rhooo) * inv_dt -
        (T(b1) * fluid.drhodt + T(b2) * fluid.drhodto + T(b3) * fluid.drhodtoo))
    return nothing
end

_dims(grid) = Val(length(cells(grid)))

"""
    fluidmech_dyrel!(fluid, phase, cache, grid, par, scales;
                     dt, P0 = nothing, a1 = 1, a2 = 1, a3 = 0, b1 = 1, b2 = 0, b3 = 0,
                     gamma = par.gamma, rtol_PH = 0, bnchm = false,
                     xBC = :periodic, yBC = :periodic, zBC = :closed, sds = -1,
                     top_cnv = 1, bot_cnv = 1, open_cnv = false,
                     require_convergence = false, verbose = false)
        -> (; converged, relaxed, total_iter, PH_iters)

Solve one momentum + continuity system with dynamic relaxation inside a
Powell-Hestenes iteration, writing the velocity and pressure into `fluid`
(D = 2: `FluidState`/`Grid`; D = 3: `FluidState3D`/`Grid3D`; `yBC` is read only
in 3-D). Full compressible Navier–Stokes terms: BD2 inertia (`a1, a2, a3` and
the lagged momentum flux on `fluid`), momentum advection frozen at the entry
velocity (scheme `par.ADVN`), drunken-sailor stabilization, continuity
`∇·(ρv) = MFS` with MFS updated once from `phase.advn_rho` (skipped with
`bnchm`). `dt` must be finite: the drunken-sailor term scales with it.

The continuity residual carries no pressure term: the assembled constraint has
an empty pressure block, so an artificial-compressibility term with a fixed
reference would not vanish at the fixed point. `P0` (default: the entry
pressure, copied into `cache.P0`) only sets the level the pressure gauge keeps.

`converged`: the error reached `par.atol_PH`. With `rtol_PH > 0` the solve may
stop once the error has dropped by `rtol_PH` relative to its entry value
(`relaxed`), the inexact-solve forcing of the outer Picard loop.
`par.anderson_PH = m > 0` Anderson-accelerates the multiplier update over the
last `m` sweeps. An exhausted budget returns `converged = false`, or throws with
`require_convergence = true`.
"""
function fluidmech_dyrel!(fluid, phase, cache::StokesCache{D, T}, grid, par::Parameters,
                          scales::Scales;
                          dt::Real, P0 = nothing,
                          a1::Real = 1, a2::Real = 1, a3::Real = 0,
                          b1::Real = 1, b2::Real = 0, b3::Real = 0,
                          gamma::Real = par.gamma, rtol_PH::Real = 0, bnchm::Bool = false,
                          xBC::Symbol = :periodic, yBC::Symbol = :periodic,
                          zBC::Symbol = :closed, sds::Integer = -1,
                          top_cnv::Integer = 1, bot_cnv::Integer = 1, open_cnv::Bool = false,
                          require_convergence::Bool = false,
                          verbose::Bool = false) where {D, T}
    isfinite(dt) || throw(ArgumentError(
        "fluidmech_dyrel!: dt must be finite (got $dt); the drunken-sailor term ∂ρ·g0·dt scales with dt"))
    m_aa = par.anderson_PH
    m_aa >= 0 || throw(ArgumentError("fluidmech_dyrel!: anderson_PH must be ≥ 0 (got $m_aa)"))
    par.rheology_in_PT && D != 2 && throw(ArgumentError(
        "fluidmech_dyrel!: rheology_in_PT is implemented for D = 2 only"))
    op = StokesOp(T, Val(D), par; a1, a2, a3, gamma, dt, xBC, yBC, zBC, sds, top_cnv, bot_cnv)
    h = grid.h
    _anderson_reset!(cache.anderson)
    P0arr = P0 === nothing ? copyto!(cache.P0, fluid.P) : P0

    @timeit_debug TO "setup" begin
    bnchm || _update_mass_source!(fluid, phase, par, dt; a1, a2, a3, b1, b2, b3)
    # Mean mass-flux source: drains the net volume source through the bottom
    # (zero in MMS mode, where the bottom stays closed).
    mfs_mean = bnchm ? zero(T) : T(sum(fluid.MFS) / length(fluid.MFS))
    Ddom = T(grid.D)
    momentum_advection!(cache, fluid, grid, par, op)
    # The penalty factor learned by the previous solve is reused: the operator
    # barely moves between Picard sweeps (a fresh cache starts from γfact_PT).
    γfact = cache.γfact > zero(T) ? cache.γfact : T(par.γfact_PT)
    penalty!(cache, fluid, grid, op; γfact)
    check_preconditioner(cache)
    # `cache.c` carries the damping learned by the previous solve (0 = undamped).
    foreach(A -> fill!(A, zero(T)), cache.dvdτ)
    apply_velocity_bcs!(fluid, grid, op; mfs_mean, Ddom, open_cnv)
    bnchm || pressure_gauge!(fluid, P0arr)
    apply_pressure_bcs!(fluid, op)
    end

    err0 = ntuple(_ -> one(T), Val(D));  errP0 = one(T)
    err_min  = T(Inf)
    rel_drop = T(par.rel_drop_PT)
    total_iter = 0;  PH_iters = 0
    converged = false;  relaxed = false
    err_first = zero(T)
    γ_win_from = 2                           # skip sweep 1 (errP0 re-baselines at 2)
    γ_retunes  = 0
    err_ref    = zero(T)

    for itPH in 1:par.maxit_PH
        PH_iters = itPH
        @timeit_debug TO "PH residual" begin
        residual_momentum!(cache, fluid, grid, op; dr = false)
        residual_continuity!(cache, fluid, grid, P0arr, Inf)
        normv, normP = residual_norms(cache, fluid.rho)
        all(isfinite, (normv..., normP)) ||
            error("fluidmech_dyrel! produced a non-finite residual at PH iteration $itPH")
        if itPH == 1
            err0 = normv .+ eps(T);  errP0 = normP + eps(T)
        elseif itPH == 2
            # At entry P is the previous Picard sweep's field; normalizing by its
            # residual would hide the multiplier iteration's progress.
            errP0 = normP + eps(T)
        end
        # The returned pressure is off by O(γ_eff·R_P), so the multiplier step
        # must be small too, not just R_P.
        normdP = pressure_update_norm(cache, fluid, h)
        end
        err = max(maximum(min.(normv ./ err0, normv)), min(normP / errP0, normP), normdP)
        verbose && @printf("  [PH %02d] iter=%07d  err=%.3e  (v=%s P=%.2e dP=%.2e)\n",
                           itPH, total_iter, err, join((@sprintf("%.2e", x) for x in normv), "/"),
                           normP, normdP)
        if err < par.atol_PH
            converged = true
            break
        end
        itPH == 1 && (err_first = err)
        if itPH > 1 && err < rtol_PH * err_first
            relaxed = true
            break
        end

        # A stalling outer error tightens the inner velocity solve.
        if err > err_min * T(1.05)
            rel_drop = max(rel_drop * T(0.1), T(1e-3))
        end
        err_min = min(err_min, err)

        # Penalty self-tuning. The PH contraction is ρ = 1/(1 + γfact·λ),
        # λ = λmin(diag(s_P)⁻¹·S), O(1) only while the momentum block is viscous-
        # dominated; once a1·ρ/dt ≫ η/h², S is a Poisson operator and λ falls with
        # h². λ is a property of the frozen operator, so the contraction measured
        # over a window of sweeps gives the γfact reaching ρ_want in closed form.
        # The rate is read off the continuity residual: it does not itself scale
        # with γ_eff.
        if γ_retunes < γ_max_retunes && itPH == γ_win_from
            err_ref = normP
        elseif γ_retunes < γ_max_retunes && itPH == γ_win_from + γ_win_width
            ρ_obs = (normP / err_ref)^(one(T) / γ_win_width)
            if ρ_obs > ρ_want
                # ρ_obs ≥ 1 leaves λ unmeasurable; fall back to the growth cap.
                λ = max((inv(ρ_obs) - one(T)) / γfact, zero(T))
                γfact = λ > zero(T) ?
                        min((inv(ρ_want) - one(T)) / λ, γfact * γ_max_growth) :
                        γfact * γ_max_growth
                # P is the Lagrange multiplier and stays valid: γ_eff only sets the
                # step length of its update.
                @timeit_debug TO "PH penalty retune" begin
                penalty!(cache, fluid, grid, op; γfact)
                check_preconditioner(cache)
                end
                _anderson_reset!(cache.anderson)    # history from a different map
                verbose && @printf("  [PH %02d] rho=%.3f over %d sweeps -> gfact=%.4g\n",
                                   itPH, ρ_obs, γ_win_width, γfact)
                # Only an applied correction spends budget.
                γ_retunes += 1
            end
            γ_win_from = itPH
            err_ref    = normP
        end

        # Inner-DR target: reduce the velocity residual by `rel_drop` within this
        # sweep; the baseline is the sweep's first poll.
        errVel0 = zero(T)
        itPT = 0
        @timeit_debug TO "DR iterations" while itPT < par.maxit_PT
            itPT += 1;  total_iter += 1
            # Rv_old feeds only the Rayleigh quotient at the retune iterations
            if itPT % par.n_tune_PT == 0
                foreach(copyto!, cache.Rv_old, cache.Rv)
            end
            @timeit_debug TO "residual_P" residual_continuity!(cache, fluid, grid, P0arr, Inf)
            @timeit_debug TO "residual_V" residual_momentum!(cache, fluid, grid, op; dr = true)
            @timeit_debug TO "update velocity + BCs" begin
            update_velocity!(cache, fluid, par.CFL_PT)
            apply_velocity_bcs!(fluid, grid, op; mfs_mean, Ddom, open_cnv)
            end
            if itPT % par.n_tune_PT == 0
                errVel = @timeit_debug TO "poll norm" velocity_residual_norm(cache)
                isfinite(errVel) ||
                    error("fluidmech_dyrel! produced a non-finite velocity residual at DR iteration $total_iter")
                if errVel0 == zero(T)
                    errVel0 = errVel + eps(T)
                else
                    relerr = errVel / errVel0
                    verbose && @printf("    [DR %07d] errVel=%.3e  rel=%.3e\n", itPT, errVel, relerr)
                    relerr < rel_drop && break
                end
                # Optional in-solve rheology refresh (`par.rheology_in_PT`), before
                # the retune so the preconditioner sees the residual's η.
                if par.rheology_in_PT
                    update_kinematics!(phase, fluid, grid)
                    update_viscosity!(phase, fluid, grid, par, scales; xBC, zBC,
                                      relaxation = par.visc_relax_PT)
                end
                @timeit_debug TO "retune (Rayleigh + Gershgorin)" begin
                λmin = rayleigh_λmin(cache, par.CFL_PT)
                cache.c = 2 * sqrt(λmin) * par.c_fact_PT
                gershgorin!(cache, fluid, grid, op)
                check_preconditioner(cache)
                end
            end
        end

        # Multiplier update P += ω·γ_eff·R_P on a refreshed R_P (the last velocity
        # update landed after the last continuity evaluation).
        @timeit_debug TO "PH multiplier update" begin
        residual_continuity!(cache, fluid, grid, P0arr, Inf)
        Pint = _interior(fluid.P)
        if m_aa > 0
            # P_num = γ_eff·R_P serves as the update scratch: the next dr = true
            # residual rebuilds it first.
            f = cache.P_num
            f .*= par.pres_relax_PT
            # the gauge removes the constant mode, so the history should not fit it
            bnchm || (f .-= sum(f) / length(f))
            _anderson_pressure_update!(cache.anderson, Pint, f, m_aa)
        else
            @. Pint += par.pres_relax_PT * cache.γ_eff * cache.R_P
        end
        bnchm || pressure_gauge!(fluid, P0arr)
        apply_pressure_bcs!(fluid, op)
        end
        total_iter > par.total_iterMax_PT && break
    end
    cache.γfact = γfact

    !converged && require_convergence && error(
        "fluidmech_dyrel! failed to meet atol_PH=$(par.atol_PH) after " *
        "$PH_iters Powell-Hestenes iterations and $total_iter DR iterations")
    return (; converged, relaxed, total_iter, PH_iters)
end
