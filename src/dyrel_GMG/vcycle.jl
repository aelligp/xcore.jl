# V-cycle recursion + outer driver for GMG: pre-smooth → restrict residual →
# recurse (coarse error solve) → prolong + mean-remove + damped-apply correction
# → post-smooth. The DR `smooth!` is the per-level relaxation; correction-scheme
# (linear, frozen-coefficient) MG.

# zero a coarse level's solution (correction scheme starts the coarse error at 0)
function _zero_solution!(lvl)
    fill!(lvl.fluid.W, 0);  fill!(lvl.fluid.U, 0);  fill!(lvl.fluid.P, 0)
    return nothing
end

# subtract the mean of an array's interior (null-space gauge fix)
_remove_mean!(a) = (a .-= sum(a) / length(a); nothing)

# Inertia-to-viscous ratio a1·ρ·h²/(η·dt) above which the coarsest level retunes
# its penalty (`coarse_solve!`). The Powell-Hestenes rate law ρ = 1/(1 + γfact·λ)
# assumes the Schur DIAGONAL is what limits λ, which holds once the momentum
# block is mass-matrix-like. Below this the coarse grid is viscous-dominated: λ
# is already O(1), a larger penalty buys nothing, and where a strong viscosity
# contrast is what limits the coarse solve instead, raising γfact on the measured
# stall degrades the V-cycle it is supposed to help.
# Measured at the coarsest level: no gain at 1.3e3 (256², production dt), clear
# gains from 3.6e3 up (256² at dt/4, 512², 64² at dt/64).
const γ_tune_ratio = 2e3

"""
    coarse_solve!(lvl, hier, par; fscale, a1,a2,a3,gamma,dt,P0,
                  top_cnv,bot_cnv,mfs_mean,Ddom) -> nothing

Solve the coarsest level: iterate single PH steps until the level residual drops
by `hier.coarse_rtol` or `hier.ncoarse` steps are spent. Each PH step's inner DR is
BOUNDED at `coarse_inner = 20·(Nz+Nx)` iterations (NOT `par.maxit_PT`) — the
coarsest grid is small, so this resolves it while making the per-V-cycle cost a
hard `O(ncoarse · coarse_inner)`. An unbounded `maxit_PT` here is what made the
V-cycle appear to hang (a non-converging coarse error-equation grinds the full
budget every PH step, every cycle). Returns the residual drop achieved.
"""
function coarse_solve!(lvl, hier::MGHierarchy{T}, par::Parameters{T};
                       fscale, a1, a2, a3, gamma, dt, P0,
                       top_cnv, bot_cnv, mfs_mean, Ddom,
                       xBC::Symbol = :periodic, sds::Integer = -1) where {T}
    cache = lvl.cache
    coarse_inner = 20 * (lvl.grid.Nz + lvl.grid.Nx)         # bounded inner DR budget
    level_residual!(lvl, par; a1, a2, a3, gamma, dt, P0, fscale, xBC)
    nW, nU, nP = dyrel_residual_norms(cache, lvl.fluid.rho)
    r0 = max(nW, nU, nP) + eps(T)
    # Penalty self-tuning, as in `fluidmech_dyrel!`: the Powell-Hestenes
    # contraction is ρ = 1/(1 + γfact·λ), and λ = λmin(diag(s_P)⁻¹·S) is O(1)
    # only while the momentum block is viscous-dominated. The coarsest grid has
    # the hierarchy's largest h, so it is the first level where a1·ρ/dt ≫ η/h²
    # turns S into a Poisson operator and λ falls with h² — no fixed γfact holds
    # its contraction across resolutions or time steps. Measuring ρ over a window
    # of PH steps gives the γfact that reaches `par.mg_coarse_rho` in closed form. The value
    # persists in `cache.γfact` across V-cycles and Picard sweeps.
    γfact = cache.γfact[] > zero(T) ? cache.γfact[] : T(par.γfact_PT)
    γ_target  = T(par.mg_coarse_rho)
    inertia   = T(a1) * (sum(lvl.fluid.rho) / length(lvl.fluid.rho)) * T(lvl.grid.h)^2 /
                ((sum(lvl.fluid.eta) / length(lvl.fluid.eta)) * T(dt))
    tune      = inertia > T(γ_tune_ratio)
    γ_retunes = (tune && γ_target > zero(T)) ? 0 : γ_max_retunes
    rtol      = hier.coarse_rtol
    win_from  = 0
    err_ref   = nP
    ρ_best    = T(Inf)                  # best window contraction, and the γfact that gave it
    γ_best    = γfact
    for k in 1:hier.ncoarse
        # adaptive: the coarsest is the only level that solves to tolerance, so
        # it keeps the early-stop poll + Rayleigh retune (smooth modes need the
        # light global damping, not the smoother's fixed high-frequency one).
        smooth!(lvl, par; nsweeps = 1, ninner = coarse_inner, adaptive = true,
                fscale, a1, a2, a3, gamma, dt, P0,
                top_cnv, bot_cnv, mfs_mean, Ddom, xBC, sds)
        level_residual!(lvl, par; a1, a2, a3, gamma, dt, P0, fscale, xBC)
        nW, nU, nP = dyrel_residual_norms(cache, lvl.fluid.rho)
        # The rate is read off the continuity residual: it is the quantity the
        # rate law describes and the one term that does not itself scale with γ_eff.
        if γ_retunes < γ_max_retunes && k == win_from + γ_win_width
            ρ_obs = (nP / (err_ref + eps(T)))^(one(T) / γ_win_width)
            γ_window = γfact                # the penalty this window ran under
            # A raised penalty that made the contraction WORSE was the wrong
            # remedy: the rate law assumes the Schur diagonal is what limits λ,
            # which a strong viscosity contrast breaks. Restore the best γfact
            # seen and stop, instead of compounding the growth cap into divergence.
            if ρ_obs > ρ_best
                γfact = γ_best
                compute_AL_penalty!(cache, lvl.fluid, lvl.grid.h; γfact, a1, gamma, dt,
                                    g0 = par.g0, xBC)
                _check_dyrel_preconditioner(cache)
                update_dτ_α_β!(cache, par.CFL_PT)
                cache.γfact[] = γfact
                γ_retunes = γ_max_retunes
            elseif ρ_obs > γ_target
                # ρ_obs ≥ 1 leaves λ unmeasurable; fall back to the growth cap.
                λ = max((inv(ρ_obs) - one(T)) / γfact, zero(T))
                γfact = λ > zero(T) ?
                        min((inv(γ_target) - one(T)) / λ, γfact * γ_max_growth) :
                        γfact * γ_max_growth
                # P is the Lagrange multiplier and stays valid: γ_eff only sets
                # the step length of its update, and compute_AL_penalty! keeps
                # γ_eff, η_b and the preconditioner diagonals mutually consistent.
                compute_AL_penalty!(cache, lvl.fluid, lvl.grid.h; γfact, a1, gamma, dt,
                                    g0 = par.g0, xBC)
                _check_dyrel_preconditioner(cache)
                update_dτ_α_β!(cache, par.CFL_PT)
                cache.γfact[] = γfact
                γ_retunes += 1
            end
            if ρ_obs < ρ_best
                ρ_best = ρ_obs;  γ_best = γ_window
            end
            win_from = k
            err_ref  = nP
        end
        max(nW, nU, nP) / r0 < rtol && break
    end
    return nothing
end

"""
    vcycle!(hier, l, par; a1,a2,a3,gamma,dt, P0, top_cnv,bot_cnv, mfs_mean0) -> nothing

One V-cycle starting at level `l` (1 = finest). Level 1 carries the physical
forcing (`fscale=1`) + bottom-drain (`mfs_mean0`); coarse levels solve the
homogeneous error equation (`fscale=0`, no drain) with the restricted residual
in `cache.bext_*`.
"""
function vcycle!(hier::MGHierarchy{T}, l::Int, par::Parameters{T};
                 a1, a2, a3, gamma, dt, P0,
                 top_cnv::Integer, bot_cnv::Integer, mfs_mean0::Real,
                 xBC::Symbol = :periodic, sds::Integer = -1) where {T}
    lvl   = hier.levels[l]
    fine  = (l == 1)
    fscale   = fine ? one(T) : zero(T)
    mfs_mean = fine ? T(mfs_mean0) : zero(T)
    Ddom     = T(lvl.grid.D)
    # Per-level reference pressure for comp=(P−P0)/(η_b·dt). The fine level uses
    # the physical snapshot the caller tracks per cycle. Coarse levels use their
    # own permanently-zero `P0`: their `P` is the pressure CORRECTION, so
    # comp = P/(η_b·dt) is the same operator term the fine level applies to its
    # per-cycle increment — keeping the coarse operator a discretisation of the
    # fine one. (Aliasing P0 to the level's own P instead kills comp entirely and
    # drops that diagonal from the coarse operator; the fine-sized snapshot cannot
    # be used at all, its shape does not match.)
    P0_lvl = fine ? P0 : lvl.P0
    # minimal smoothing: each smoothing PH step does EXACTLY `hier.inner` DR
    # iterations with the fixed per-level damping (no reductions, no polling —
    # a true smoother does a few damped sweeps, not a sub-solve).
    sm(n) = smooth!(lvl, par; nsweeps = n, ninner = hier.inner, fscale,
                    a1, a2, a3, gamma, dt, P0 = P0_lvl,
                    top_cnv, bot_cnv, mfs_mean, Ddom, xBC, sds)

    if l == length(hier.levels)
        # coarsest level: the ONLY level solved to tolerance (full inner DR budget,
        # iterate PH steps until the level residual drops by `hier.coarse_rtol`).
        @timeit_debug TO "L$l coarse solve" coarse_solve!(lvl, hier, par; fscale, a1, a2, a3, gamma, dt, P0 = P0_lvl,
                      top_cnv, bot_cnv, mfs_mean, Ddom, xBC, sds)
        return nothing
    end

    @timeit_debug TO "L$l pre-smooth" sm(hier.npre)
    @timeit_debug TO "L$l residual" level_residual!(lvl, par; a1, a2, a3, gamma, dt, P0 = P0_lvl, fscale, xBC)   # r = b − A·v → cache.R_*

    # restrict residual → next-level injected RHS; start coarse error at 0
    cl = hier.levels[l + 1]
    @timeit_debug TO "L$l restrict" begin
    restrict_w!(cl.cache.bext_W, lvl.cache.R_W)
    restrict_u!(cl.cache.bext_U, lvl.cache.R_U)
    restrict_p!(cl.cache.bext_P, lvl.cache.R_P)
    _zero_solution!(cl)
    end

    vcycle!(hier, l + 1, par; a1, a2, a3, gamma, dt, P0,
            top_cnv, bot_cnv, mfs_mean0, xBC, sds)    # coarse error solve

    # prolong correction into fine-sized scratch (reuse cache.R_*; free post-restrict)
    @timeit_debug TO "L$l prolong + correct" begin
    eW = lvl.cache.R_W;  eU = lvl.cache.R_U;  eP = lvl.cache.R_P
    @views prolong_w!(eW, cl.fluid.W[:, 2:end - 1]; xBC, sds)
    @views prolong_u!(eU, cl.fluid.U[2:end - 1, :]; top_cnv, bot_cnv)
    @views prolong_p!(eP, cl.fluid.P[2:end - 1, 2:end - 1]; xBC)
    # Null-space gauge on the pressure correction only. W is Dirichlet-0 on both
    # z-boundaries and U reflects as `-cnv·U`, so neither has a constant null
    # mode to gauge away — subtracting their mean deletes a real part of the
    # correction (and shifts W off its Dirichlet value, which the BC pass below
    # then clips back, leaving a kink at the boundary rows).
    _remove_mean!(eP)
    # apply damped correction  v ← v + γ_mg·e. The sign follows xcore's residual
    # convention R = b − A·v: the coarse solve returns e ≈ A⁻¹·R (the error
    # estimate), so A·(v+e) ≈ b. Wrong sign ⇒ geometric blow-up.
    γ = hier.γ_mg
    @views @. lvl.fluid.W[:, 2:end - 1]        += γ * eW
    @views @. lvl.fluid.U[2:end - 1, :]        += γ * eU
    @views @. lvl.fluid.P[2:end - 1, 2:end - 1] += γ * eP
    dyrel_apply_velocity_bcs!(lvl.fluid; top_cnv, bot_cnv, xBC, sds,
                              mfs_mean = mfs_mean, Ddom = Ddom)
    dyrel_apply_pressure_bcs!(lvl.fluid; xBC)
    end

    @timeit_debug TO "L$l post-smooth" sm(hier.npost)
    return nothing
end

"""
    fluidmech_gmg!(fluid, phase, hier, grid, par, scales; dt, P0=nothing,
                   a1,a2,a3,b1,b2,b3, gamma, bnchm=false, xBC,zBC,
                   top_cnv,bot_cnv, rtol_PH=0, verbose=false)
        -> (; converged, relaxed, ncyc, errs)

Geometric-multigrid Stokes solve: MFS source + frozen momentum advection on the
fine grid (as the single-grid solver), coefficient coarsening + autotune, then
V-cycles to `par.atol_PH`. `hier` is built once via `build_hierarchy`.

With `rtol_PH > 0` the solve may also stop, as `relaxed` rather than
`converged`, once the residual has dropped by `rtol_PH` below its value after the
first V-cycle: the inexact-solve forcing of the outer Picard loop (see
`par.forcing_PH`), which keeps each solve from running far below the nonlinear
error it serves.

`errs[k]` is the fine-level residual after V-cycle `k`; the geometric mean of its
successive ratios is the cycle contraction factor.
"""
function fluidmech_gmg!(fluid::FluidState{T}, phase::PhaseState{T},
                        hier::MGHierarchy{T}, grid::Grid{T}, par::Parameters{T},
                        scales::Scales{T};
                        dt::Real, P0 = nothing,
                        a1::Real = 1, a2::Real = 1, a3::Real = 0,
                        b1::Real = 1, b2::Real = 0, b3::Real = 0,
                        gamma::Real = par.gamma, rtol_PH::Real = 0,
                        bnchm::Bool = false,
                        xBC::Symbol = :periodic, zBC::Symbol = :closed,
                        sds::Integer = -1,
                        top_cnv::Integer = 1, bot_cnv::Integer = 1,
                        verbose::Bool = false) where {T}
    # Snapshot the entry pressure (NOT an alias) so the artificial-compressibility
    # term comp=(P−P0)/(η_b·dt) is active in the smoother — essential for the AL
    # iteration to converge geometrically (matches src/dyrel/solve.jl:67). At large
    # dt comp→0 at the fixed point, so it does not bias ∇·ρv=MFS.
    isfinite(dt) || throw(ArgumentError(
        "fluidmech_gmg!: dt must be finite (got $dt); the drunken-sailor term ∂ρ·g0·dt scales with dt"))
    P0arr = P0 === nothing ? copy(fluid.P) : P0
    fcache = hier.levels[1].cache
    @timeit_debug TO "setup" begin
    # mass-flux source (fine grid; mirrors single-grid solve)
    if !bnchm
        inv_dt = isfinite(dt) ? T(inv(dt)) : zero(T)
        @. fluid.drhodt = -phase.advn_rho
        @. fluid.MFS = fluid.MFS - par.alpha * (
            (T(a1) * fluid.rho - T(a2) * fluid.rhoo - T(a3) * fluid.rhooo) * inv_dt -
            (T(b1) * fluid.drhodt + T(b2) * fluid.drhodto + T(b3) * fluid.drhodtoo))
    end

    # frozen momentum advection (fine grid only)
    dyrel_momentum_advection!(fcache, fluid, grid, par; xBC, zBC)
    mfs_mean = bnchm ? zero(T) : T(sum(fluid.MFS) / length(fluid.MFS))

    # coefficient hierarchy + per-level autotune
    refresh_coefficients!(hier, par; a1, gamma, dt, xBC)

    dyrel_apply_velocity_bcs!(fluid; top_cnv, bot_cnv, xBC, sds, mfs_mean, Ddom = T(grid.D))
    # gauge-fix the free constant pressure mode (skipped in MMS/bnchm, where the
    # manufactured solution pins the level explicitly)
    bnchm || dyrel_apply_pressure_gauge!(fluid, P0arr, fcache.η_b)
    dyrel_apply_pressure_bcs!(fluid; xBC)
    end

    err0 = one(T);  converged = false;  relaxed = false;  ncyc = 0
    errs = T[]
    for cyc in 1:par.maxit_PH                          # reuse maxit_PH as max V-cycles
        ncyc = cyc
        @timeit_debug TO "V-cycle" vcycle!(hier, 1, par; a1, a2, a3, gamma, dt, P0 = P0arr,
                top_cnv, bot_cnv, mfs_mean0 = mfs_mean, xBC, sds)
        # re-pin the pressure gauge once per cycle (the smoother's multiplier
        # updates have nonzero mean mid-solve), then refresh ghosts before the
        # convergence-gate residual.
        @timeit_debug TO "gauge + convergence residual" begin
        bnchm || dyrel_apply_pressure_gauge!(fluid, P0arr, fcache.η_b)
        dyrel_apply_pressure_bcs!(fluid; xBC)
        # Gate on the system being solved, ∇·(ρv) = MFS: `P0 = fluid.P` zeroes the
        # artificial-compressibility term, which would otherwise absorb the very
        # mass imbalance the gate must bound.
        level_residual!(hier.levels[1], par; a1, a2, a3, gamma, dt, P0 = fluid.P, fscale = 1, xBC)
        normW, normU, normP = dyrel_residual_norms(fcache, fluid.rho)
        end
        err = max(normW, normU, normP)
        isfinite(err) || error("fluidmech_gmg! produced a non-finite residual at V-cycle $cyc")
        push!(errs, err)
        cyc == 1 && (err0 = err + eps(T))
        verbose && @printf("  [GMG V%02d] err=%.3e  (W=%.2e U=%.2e P=%.2e)\n",
                           cyc, err, normW, normU, normP)
        if err < par.atol_PH
            converged = true;  break
        end
        if cyc > 1 && err < rtol_PH * err0
            relaxed = true;  break
        end
        # PSEUDO-TRANSIENT artificial-compressibility reference update: advance
        # P0 to the just-computed pressure so the comp = (P−P0)/(η_b·dt) term
        # measures the PER-CYCLE increment, not the deviation from the entry
        # pressure. With a frozen entry-P0, the converged solve satisfies
        # ∇·ρv = MFS − comp (comp = (P−P0_entry)/(η_b·dt) ≠ 0), biasing the
        # solution off the exact (direct) answer by O(1/γfact); that bias exceeds
        # the outer Picard rtol at the default γfact and stalls the nonlinear loop.
        # Tracking P0 each cycle makes comp → 0 as the V-cycle converges (P→P0),
        # so the fixed point is ∇·ρv = MFS EXACTLY at any γfact — the standard PT
        # artificial-compressibility formulation (P0 = previous PT iterate). The
        # comp term still regularises the smoother (nonzero on the increment while
        # iterating). Skipped in MMS (bnchm), where the manufactured P is pinned.
        bnchm || (P0arr .= fluid.P)
    end
    converged || relaxed || error("fluidmech_gmg! failed to meet atol_PH=$(par.atol_PH) after $ncyc V-cycles")
    return (; converged, relaxed, ncyc, errs)
end
