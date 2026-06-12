using KernelAbstractions
using Printf

# Powell-Hestenes / Dynamic-Relaxation solver orchestrator for xcore-DYREL.
# Full compressible two-phase Navier-Stokes (Re>1): BD2 inertia, momentum
# advection, drunken-sailor stabilization, and the MFS mass-flux source are all
# included. Mirrors the outer/inner structure of JustRelax's `_solve_DYREL!`
# (src/DYREL/DYREL2D.jl), extended with the xcore-specific terms.
#
#   compute momentum advection once (frozen RHS), autotune setup, then:
#   Powell-Hestenes outer loop:
#     refresh stress → PH residuals → convergence check
#     inner Dynamic-Relaxation loop:
#       snapshot residuals → refresh stress → continuity residual ∇·(ρv)
#       → Schur penalty P_num = γ_eff·R_P → preconditioned DR residual
#       → damped velocity update → BCs
#       → every n_tune: Rayleigh λ_min + Gershgorin λ_max retune
#     pressure penalty update  P += γ_eff·R_P
#
# `dyrel::DyrelCache` holds all autotune/scratch state. `phase.txx/tzz/txz` are
# used as stress scratch (refreshed each iteration by `dyrel_update_stress!`).

"""
    fluidmech_dyrel!(fluid, phase, dyrel, grid, par, scales;
                     dt = Inf, P0 = nothing, a1=1, a2=1, a3=0, gamma=par.gamma,
                     xBC=:periodic, zBC=:closed, top_cnv = 1, bot_cnv = 1,
                     verbose = false) -> (; converged, total_iter, PH_iters)

Solve one momentum + continuity linear system (the same system one
`fluidmech!` call solves) with the DYREL pseudo-transient scheme, writing the
converged `(W, U, P)` back into `fluid` in place. `dyrel::DyrelCache` is a
standalone object (build once via `DyrelCache(T, backend, Nz, Nx)`), mirroring
JustRelax's `DYREL` + `solve_DYREL!`.

Includes the full compressible Navier-Stokes terms: BD2 inertia
(`a1,a2,a3` + lagged momentum-flux history on `fluid`), momentum advection
(frozen at the entry velocity, scheme `par.ADVN`), drunken-sailor
stabilization, and the compressible continuity `∇·(ρv) = MFS`. The outer
Picard loop (in `run!`) updates the frozen/lagged terms by re-calling this.

Buoyancy enters via `fluid.Drho`; BCs: periodic-x, closed-z for W, free/no-slip
for U (`top_cnv`/`bot_cnv` = ±1). Use a large finite `dt` (not `Inf`) so the
dt-scaled drunken-sailor term stays finite.
"""
function fluidmech_dyrel!(fluid::FluidState{T}, phase::PhaseState{T},
                          dyrel::DyrelCache{T},
                          grid::Grid{T}, par::Parameters{T}, scales::Scales{T};
                          dt::Real = Inf, P0 = nothing,
                          a1::Real = 1, a2::Real = 1, a3::Real = 0,
                          b1::Real = 1, b2::Real = 0, b3::Real = 0,
                          gamma::Real = par.gamma,
                          bnchm::Bool = false,
                          xBC::Symbol = :periodic, zBC::Symbol = :closed,
                          top_cnv::Integer = 1, bot_cnv::Integer = 1,
                          open_cnv::Bool = false,
                          verbose::Bool = false) where {T<:AbstractFloat}
    cache  = dyrel
    h      = grid.h
    # Reference pressure P0 for the artificial-compressibility term
    # comp = (P − P0)/(η_b·dt) in the continuity residual. Snapshot the entry
    # pressure (JustRelax's `@copy stokes.P0 stokes.P`): comp then gives the
    # pressure equation a diagonal "restoring" coupling so the augmented-
    # Lagrangian multiplier iteration converges GEOMETRICALLY. `copy` (not alias)
    # is essential. The outer Picard loop re-enters with updated P each sweep, so
    # comp = (P_k − P_{k-1})/(η_b·dt) → 0 at Picard convergence ⇒ the fixed point
    # still satisfies ∇·ρv = MFS exactly.
    P0arr = P0 === nothing ? copy(fluid.P) : P0
    # Pressure gauge: the free constant P mode is pinned each PH update with the
    # γ_eff-WEIGHTED gauge Σ(P−P0)/γ_eff = 0 (see dyrel_apply_pressure_gauge!),
    # which makes the comp term integrate to zero — the global mass solvability
    # constraint — instead of the unweighted mean (whose mismatch left a constant
    # P-residual floor at high N).

    # --- mass-flux source update (mirrors fluidmech.jl:71-82). DYREL does NOT
    # update MFS inside the solve, so do it here once per call from the
    # phsevo!-provided `advn_rho`, exactly as the direct solver does. The outer
    # Picard loop converges it. Skipped in MMS mode. ---
    if !bnchm
        inv_dt = isfinite(dt) ? T(inv(dt)) : zero(T)
        @. fluid.drhodt = -phase.advn_rho
        @. fluid.MFS = fluid.MFS - par.alpha * (
            (T(a1) * fluid.rho - T(a2) * fluid.rhoo - T(a3) * fluid.rhooo) * inv_dt -
            (T(b1) * fluid.drhodt + T(b2) * fluid.drhodto + T(b3) * fluid.drhodtoo))
    end

    # Mean mass-flux source — drives the bottom-boundary drain W_bot =
    # MFSmean·D/ρw so the net volume source (∫MFS≠0 from the evolving nonlinear
    # density) leaves through the bottom, exactly as the direct solver does.
    # Without it the closed W_bot=0 BC leaves an irreducible continuity floor =
    # mean(MFS). Zero in MMS mode (bnchm) so the bottom stays closed.
    mfs_mean = bnchm ? zero(T) : T(sum(fluid.MFS) / length(fluid.MFS))
    Ddom     = T(grid.D)

    # --- momentum advection: computed ONCE from the entry velocity, frozen as
    # a constant RHS term throughout the PH/DR iteration (matches fluidmech!;
    # the outer Picard loop updates it on the next call). ---
    dyrel_momentum_advection!(cache, fluid, grid, par; xBC, zBC)

    # --- one-time autotune setup for the current operator ---
    # Operator-derived augmented-Lagrangian penalty γ_eff = γfact / s_P_phys
    # (per-cell, from the penalty-free Schur diagonal) — a single dimensionless
    # γfact converges all Re/scales (see init.jl). Leaves cache.D_W/D_U = D_full.
    compute_AL_penalty!(cache, fluid, h; γfact = par.γfact_PT, a1, gamma, dt)
    cache.c[] = zero(T)                      # undamped start (α=1); refined by Rayleigh
    update_dτ_α_β!(cache, par.CFL_PT)
    cache.initialized = true
    cache.sig_Nz = grid.Nz;  cache.sig_Nx = grid.Nx

    fill!(cache.dWdτ, zero(T));  fill!(cache.dUdτ, zero(T))

    # initial BC enforcement
    dyrel_apply_velocity_bcs!(fluid; top_cnv, bot_cnv, open_cnv, h, mfs_mean, Ddom)
    # gauge-fix the free constant pressure mode (skipped in MMS/bnchm, where the
    # manufactured solution pins the level explicitly)
    bnchm || dyrel_apply_pressure_gauge!(fluid, P0arr, cache.γ_eff)
    dyrel_apply_pressure_bcs!(fluid)

    errW0 = errU0 = errP0 = one(T)
    errVel0 = one(T)
    err_min = T(Inf)                         # best outer err so far (JustRelax)
    rel_drop = T(par.rel_drop_PT)            # adaptive inner-DR tolerance factor
    total_iter = 0
    PH_iters   = 0
    converged  = false

    for itPH in 1:par.maxit_PH
        PH_iters = itPH

        # PH residuals (stress is recomputed inline inside dyrel_residual_V!)
        dyrel_residual_V!(cache, phase, fluid, grid, par; dr = false, a1, a2, a3, gamma, dt)
        dyrel_residual_P!(cache, fluid, grid, par, P0arr, dt)

        normW, normU, normP = dyrel_residual_norms(cache, fluid.rho)
        if itPH == 1
            errW0 = normW + eps(T);  errU0 = normU + eps(T);  errP0 = normP + eps(T)
        end
        err = max(min(normW / errW0, normW),
                  min(normU / errU0, normU),
                  min(normP / errP0, normP))
        verbose && @printf("  [PH %02d] iter=%07d  err=%.3e  (W=%.2e U=%.2e P=%.2e)\n",
                           itPH, total_iter, err, normW, normU, normP)
        if err < par.atol_PH
            converged = true
            break
        end

        # inner-DR tolerance scales with the current outer residual. When the
        # outer error stalls (no longer dropping), tighten rel_drop so the inner
        # velocity solve is more accurate — matches JustRelax and prevents the
        # outer Powell-Hestenes loop from stagnating on an under-solved velocity.
        if err > err_min * T(1.05)
            rel_drop = max(rel_drop * T(0.1), T(1e-3))
        end
        err_min = min(err_min, err)
        ϵ_vel = err * rel_drop

        itPT = 0
        while itPT < par.maxit_PT
            itPT += 1;  total_iter += 1

            # snapshot residuals for the Rayleigh-quotient λ_min. R_W is rewritten
            # once per iter by dyrel_residual_V! below, so this snapshot captures
            # the *previous* iter's residual regardless of intervening iters —
            # hence we only need it on the iters that actually retune (every
            # n_tune_PT). Skipping it elsewhere removes 2 device copies/iter.
            if itPT % par.n_tune_PT == 0
                copyto!(cache.R_W_old, cache.R_W)
                copyto!(cache.R_U_old, cache.R_U)
            end

            # continuity residual + fused Schur penalty P_num (momentum stress is
            # recomputed inline inside dyrel_residual_V! below)
            dyrel_residual_P!(cache, fluid, grid, par, P0arr, dt)

            # preconditioned DR momentum residual + damped velocity update
            dyrel_residual_V!(cache, phase, fluid, grid, par; dr = true, a1, a2, a3, gamma, dt)
            dyrel_update_velocity!(cache, fluid)
            dyrel_apply_velocity_bcs!(fluid; top_cnv, bot_cnv, open_cnv, h, mfs_mean, Ddom)

            # periodic re-tune of λ_max (Gershgorin) and λ_min (Rayleigh)
            if itPT % par.n_tune_PT == 0
                errVel = dyrel_velocity_residual_norm(cache)
                if total_iter == par.n_tune_PT    # set once, globally (matches JustRelax `iter == nout`)
                    errVel0 = errVel + eps(T)
                end
                relerr = errVel / errVel0
                verbose && @printf("    [DR %07d] errVel=%.3e  rel=%.3e\n",
                                   itPT, errVel, relerr)
                relerr < ϵ_vel && break

                λmin = dyrel_rayleigh_λmin(cache)
                cache.c[] = 2 * sqrt(λmin) * par.c_fact_PT
                gershgorin!(cache, fluid, h; a1, gamma, dt)   # D_full retune (γ_eff fixed)
                update_dτ_α_β!(cache, par.CFL_PT)
            end
        end
        cache.last_DR_iters[] = itPT

        # Powell-Hestenes augmented-Lagrangian multiplier update on interior
        # cells: P += ω·γ_eff·R_P, with the SAME per-cell γ_eff = γfact/s_P_phys
        # injected into the momentum residual (P_num) and the comp term (η_b).
        # ω = par.pres_relax_PT is the under-relaxation (1 = Newton-Uzawa step).
        @views @. fluid.P[2:end-1, 2:end-1] += par.pres_relax_PT * cache.γ_eff * cache.R_P
        # pin the pressure gauge (remove the free constant mode) then refresh ghosts
        bnchm || dyrel_apply_pressure_gauge!(fluid, P0arr, cache.γ_eff)
        dyrel_apply_pressure_bcs!(fluid)
        # NB: P0 is deliberately NOT advanced per PH step here (unlike the GMG
        # driver, which tracks P0 each V-cycle). The single-grid PH multiplier
        # iteration relies on the strong restoring force of comp=(P−P0_entry)/(η_b·dt)
        # against a FIXED reference; tracking P0 weakens it and the high-N solve
        # caps (measured: itPH→100, Picard wobble worsens). The frozen-P0 comp
        # bias is instead removed by the outer Picard loop (P0=prev sweep), or by a
        # larger γfact_PT. Use :gmg for high-N production (its V-cycle PT-tracks P0).

        # Global iteration cap (JustRelax `total_iterMax`): stop grinding and
        # accept the current solve — the outer Picard loop in `run!` re-solves
        # next sweep. Prevents the multi-million-iteration stalls seen when a
        # single solve cannot reach `atol_PH` within budget.
        total_iter > par.total_iterMax_PT && break
    end
    cache.last_PH_iters[] = PH_iters

    return (; converged, total_iter, PH_iters)
end
