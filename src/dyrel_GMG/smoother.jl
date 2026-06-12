# DR smoother in correction form for GMG. `smooth!` does `nsweeps` Powell-
# Hestenes steps on one level; each PH step = a FIXED, SMALL number of damped
# DR velocity iterations at frozen P (`ninner`) followed by ONE augmented-
# Lagrangian multiplier update `P += ω·γ_eff·R_P`. The system A·v = b is
# encoded by `fscale` (physical forcing on/off) + `cache.bext_*` (injected
# restricted residual). `level_residual!` computes the un-preconditioned
# r = b − A·v into `cache.R_{W,U,P}` for restriction to the next level.
#
# SMOOTHER ECONOMY (the point of MG): each level only has to damp the UPPER
# HALF of its frequency spectrum — the coarser levels handle the rest. So the
# per-level DR runs a minimal fixed iteration count with a FIXED per-level
# damping c = mg_cfact·√(mean λmax) (critical damping for the λ ≳ λmax/4 band;
# set per level in `refresh_coefficients!`), instead of the single-grid
# solver's Rayleigh-tuned light damping — which targets the GLOBAL λmin,
# leaves the iteration effectively undamped (c = 0 until the first retune at
# n_tune_PT iters, which a short smoother never reaches), and therefore
# smooths high frequencies poorly. That undamped smoothing is what made small
# inner counts amplify the coarse correction (the old empirical mg_inner=32
# "stability floor"). Fixed counts ⇒ ZERO reductions / host syncs inside the
# smoother and a deterministic per-V-cycle cost — both essential on GPU.
#
# The coarsest level must actually SOLVE its (tiny) system including the
# smooth modes only it sees, so `adaptive = true` restores the single-grid
# behaviour there: early-stop polling on the preconditioned velocity norm +
# periodic Rayleigh-λmin / Gershgorin retune.
#
# CRITICAL (kept from the first working version): P stays FROZEN during the
# inner DR iterations — DR is a damped 2nd-order iteration that only converges
# against a static operator; updating P inside the velocity loop made the
# standalone smoother diverge (~1e30 in 600 sweeps). Velocity sub-iteration
# and multiplier update stay strictly separated, exactly as src/dyrel/solve.jl.

"""
    level_residual!(lvl, par; a1,a2,a3,gamma,dt, P0, fscale) -> nothing

Fill `lvl.cache.R_{W,U,P}` with the PH-form residual `b − A·v` at the current
`(W,U,P)` (un-preconditioned, `dr=false`). Stress is recomputed inline inside
`dyrel_residual_V!`.
"""
function level_residual!(lvl, par::Parameters; a1, a2, a3, gamma, dt, P0, fscale)
    dyrel_residual_P!(lvl.cache, lvl.fluid, lvl.grid, par, P0, dt; fscale)
    dyrel_residual_V!(lvl.cache, lvl.phase, lvl.fluid, lvl.grid, par;
                      dr = false, a1, a2, a3, gamma, dt, fscale)
    return nothing
end

"""
    smooth!(lvl, par; nsweeps, ninner, fscale, a1,a2,a3,gamma,dt, P0,
            top_cnv, bot_cnv, mfs_mean, Ddom, adaptive = false) -> nothing

`nsweeps` Powell-Hestenes steps on level `lvl`: per step, `ninner` damped DR
velocity iterations at frozen P (accumulators reset each step), then ONE AL
multiplier update `P += ω·γ_eff·R_P`. With `adaptive = false` (smoothing
levels) `ninner` is an EXACT count and the loop body contains no reductions.
With `adaptive = true` (coarsest level) `ninner` is a cap, with the
single-grid early-stop poll (`rel_drop_PT` drop, every `n_tune_PT÷8` iters)
and the periodic Rayleigh-λmin / Gershgorin retune.
"""
function smooth!(lvl, par::Parameters{T}; nsweeps::Int, ninner::Int, fscale,
                 a1, a2, a3, gamma, dt, P0,
                 top_cnv::Integer, bot_cnv::Integer,
                 mfs_mean::Real, Ddom::Real,
                 adaptive::Bool = false) where {T}
    cache = lvl.cache;  fluid = lvl.fluid;  phase = lvl.phase;  grid = lvl.grid
    h = grid.h;  fs = T(fscale)
    dyrel_apply_velocity_bcs!(fluid; top_cnv, bot_cnv, mfs_mean, Ddom)
    dyrel_apply_pressure_bcs!(fluid)

    rel_drop = T(par.rel_drop_PT)
    ncheck   = max(2, par.n_tune_PT ÷ 8)   # adaptive-mode early-stop poll cadence

    for itPH in 1:nsweeps
        # inner DR velocity sub-iteration at FIXED P, fresh accumulators
        fill!(cache.dWdτ, zero(T));  fill!(cache.dUdτ, zero(T))
        errVel0 = zero(T);  itPT = 0
        while itPT < ninner
            itPT += 1
            # R_W_old only feeds the Rayleigh λmin at adaptive retunes
            if adaptive && itPT % par.n_tune_PT == 0
                copyto!(cache.R_W_old, cache.R_W)
                copyto!(cache.R_U_old, cache.R_U)
            end
            dyrel_residual_P!(cache, fluid, grid, par, P0, dt; fscale = fs)   # P_num fused in
            dyrel_residual_V!(cache, phase, fluid, grid, par;
                              dr = true, a1, a2, a3, gamma, dt, fscale = fs)  # stress inline
            dyrel_update_velocity!(cache, fluid)
            dyrel_apply_velocity_bcs!(fluid; top_cnv, bot_cnv, mfs_mean, Ddom)
            adaptive || continue                      # fixed-count smoothing: no reductions
            if itPT % ncheck == 0
                errVel = dyrel_velocity_residual_norm(cache)
                if errVel0 == zero(T)
                    errVel0 = errVel + eps(T)         # first poll = baseline (no break)
                elseif errVel / errVel0 < rel_drop
                    break
                end
            end
            if itPT % par.n_tune_PT == 0              # expensive retune: rarely
                λmin = dyrel_rayleigh_λmin(cache)
                cache.c[] = 2 * sqrt(λmin) * par.c_fact_PT
                gershgorin!(cache, fluid, h; a1, gamma, dt)
                update_dτ_α_β!(cache, par.CFL_PT)
            end
        end

        # Powell-Hestenes AL multiplier update (ONCE per sweep, after the
        # velocity sub-iteration): same update as src/dyrel/solve.jl.
        dyrel_residual_P!(cache, fluid, grid, par, P0, dt; fscale = fs)
        @views @. fluid.P[2:end-1, 2:end-1] += par.pres_relax_PT * cache.γ_eff * cache.R_P
        dyrel_apply_pressure_bcs!(fluid)
    end
    return nothing
end
