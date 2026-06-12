# V-cycle recursion + outer driver for GMG. Mirrors the MATLAB `multigrid_solve`:
# pre-smooth → restrict residual → recurse (coarse error solve) → prolong +
# mean-remove + damped-apply correction → post-smooth. The DR `smooth!` is the
# per-level relaxation; correction-scheme (linear, frozen-coefficient) MG.

# zero a coarse level's solution (correction scheme starts the coarse error at 0)
function _zero_solution!(lvl)
    fill!(lvl.fluid.W, 0);  fill!(lvl.fluid.U, 0);  fill!(lvl.fluid.P, 0)
    return nothing
end

# subtract the mean of an array's interior (null-space gauge fix)
_remove_mean!(a) = (a .-= sum(a) / length(a); nothing)

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
                       top_cnv, bot_cnv, mfs_mean, Ddom) where {T}
    coarse_inner = 20 * (lvl.grid.Nz + lvl.grid.Nx)         # bounded inner DR budget
    level_residual!(lvl, par; a1, a2, a3, gamma, dt, P0, fscale)
    nW, nU, nP = dyrel_residual_norms(lvl.cache, lvl.fluid.rho)
    r0 = max(nW, nU, nP) + eps(T)
    for _ in 1:hier.ncoarse
        # adaptive: the coarsest is the only level that solves to tolerance, so
        # it keeps the early-stop poll + Rayleigh retune (smooth modes need the
        # light global damping, not the smoother's fixed high-frequency one).
        smooth!(lvl, par; nsweeps = 1, ninner = coarse_inner, adaptive = true,
                fscale, a1, a2, a3, gamma, dt, P0,
                top_cnv, bot_cnv, mfs_mean, Ddom)
        level_residual!(lvl, par; a1, a2, a3, gamma, dt, P0, fscale)
        nW, nU, nP = dyrel_residual_norms(lvl.cache, lvl.fluid.rho)
        max(nW, nU, nP) / r0 < hier.coarse_rtol && break
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
                 top_cnv::Integer, bot_cnv::Integer, mfs_mean0::Real) where {T}
    lvl   = hier.levels[l]
    fine  = (l == 1)
    fscale   = fine ? one(T) : zero(T)
    mfs_mean = fine ? T(mfs_mean0) : zero(T)
    Ddom     = T(lvl.grid.D)
    # Per-level reference pressure for comp=(P−P0)/(η_b·dt). ONLY the fine level
    # carries the physical snapshot (comp active). Coarse levels solve the
    # homogeneous error equation, where P is a pressure-CORRECTION: alias P0 to the
    # level's own (coarse-sized) P ⇒ comp≡0. Passing the fine-sized snapshot to a
    # coarse level would index a mismatched array ⇒ garbage forcing ⇒ the coarse
    # DR solve never converges (grinds to maxit_PT). This is the per-level fix.
    P0_lvl = fine ? P0 : lvl.fluid.P
    # minimal smoothing: each smoothing PH step does EXACTLY `hier.inner` DR
    # iterations with the fixed per-level damping (no reductions, no polling —
    # a true smoother does a few damped sweeps, not a sub-solve).
    sm(n) = smooth!(lvl, par; nsweeps = n, ninner = hier.inner, fscale,
                    a1, a2, a3, gamma, dt, P0 = P0_lvl,
                    top_cnv, bot_cnv, mfs_mean, Ddom)

    if l == length(hier.levels)
        # coarsest level: the ONLY level solved to tolerance (full inner DR budget,
        # iterate PH steps until the level residual drops by `hier.coarse_rtol`).
        coarse_solve!(lvl, hier, par; fscale, a1, a2, a3, gamma, dt, P0 = P0_lvl,
                      top_cnv, bot_cnv, mfs_mean, Ddom)
        return nothing
    end

    sm(hier.npre)                                     # pre-smooth
    # NB: P0_lvl, not the fine-grid P0 — on intermediate levels the fine snapshot
    # is a mismatched (larger) array, and indexing it puts a spurious comp term
    # into the residual restricted to the next level.
    level_residual!(lvl, par; a1, a2, a3, gamma, dt, P0 = P0_lvl, fscale)   # r = b − A·v → cache.R_*

    # restrict residual → next-level injected RHS; start coarse error at 0
    cl = hier.levels[l + 1]
    restrict_w!(cl.cache.bext_W, lvl.cache.R_W)
    restrict_u!(cl.cache.bext_U, lvl.cache.R_U)
    restrict_p!(cl.cache.bext_P, lvl.cache.R_P)
    _zero_solution!(cl)

    vcycle!(hier, l + 1, par; a1, a2, a3, gamma, dt, P0,
            top_cnv, bot_cnv, mfs_mean0)              # coarse error solve

    # prolong correction into fine-sized scratch (reuse cache.R_*; free post-restrict)
    eW = lvl.cache.R_W;  eU = lvl.cache.R_U;  eP = lvl.cache.R_P
    @views prolong_w!(eW, cl.fluid.W[:, 2:end - 1])
    @views prolong_u!(eU, cl.fluid.U[2:end - 1, :])
    @views prolong_p!(eP, cl.fluid.P[2:end - 1, 2:end - 1])
    # null-space gauge: pressure always; periodic-x ⇒ velocity too
    _remove_mean!(eP);  _remove_mean!(eW);  _remove_mean!(eU)
    # apply damped correction  v ← v + γ_mg·e.  PLUS, not the MATLAB minus:
    # xcore's residual is R = b − A·v, so the coarse solve returns e ≈ A⁻¹·R (the
    # error estimate) and A·(v+e) ≈ b. (MATLAB uses the opposite residual sign
    # A·v − b, hence its `u -= γ·upd`.) Wrong sign ⇒ geometric blow-up.
    γ = hier.γ_mg
    @views @. lvl.fluid.W[:, 2:end - 1]        += γ * eW
    @views @. lvl.fluid.U[2:end - 1, :]        += γ * eU
    @views @. lvl.fluid.P[2:end - 1, 2:end - 1] += γ * eP
    dyrel_apply_velocity_bcs!(lvl.fluid; top_cnv, bot_cnv,
                              mfs_mean = mfs_mean, Ddom = Ddom)
    dyrel_apply_pressure_bcs!(lvl.fluid)

    sm(hier.npost)                                    # post-smooth
    return nothing
end

"""
    fluidmech_gmg!(fluid, phase, hier, grid, par, scales; dt, P0=nothing,
                   a1,a2,a3,b1,b2,b3, gamma, bnchm=false, xBC,zBC,
                   top_cnv,bot_cnv, verbose=false) -> (; converged, ncyc)

Geometric-multigrid Stokes solve: MFS source + frozen momentum advection on the
fine grid (as the single-grid solver), coefficient coarsening + autotune, then
V-cycles to `par.atol_PH`. `hier` is built once via `build_hierarchy`.
"""
function fluidmech_gmg!(fluid::FluidState{T}, phase::PhaseState{T},
                        hier::MGHierarchy{T}, grid::Grid{T}, par::Parameters{T},
                        scales::Scales{T};
                        dt::Real, P0 = nothing,
                        a1::Real = 1, a2::Real = 1, a3::Real = 0,
                        b1::Real = 1, b2::Real = 0, b3::Real = 0,
                        gamma::Real = par.gamma, bnchm::Bool = false,
                        xBC::Symbol = :periodic, zBC::Symbol = :closed,
                        top_cnv::Integer = 1, bot_cnv::Integer = 1,
                        verbose::Bool = false) where {T}
    # Snapshot the entry pressure (NOT an alias) so the artificial-compressibility
    # term comp=(P−P0)/(η_b·dt) is active in the smoother — essential for the AL
    # iteration to converge geometrically (matches src/dyrel/solve.jl:67). At large
    # dt comp→0 at the fixed point, so it does not bias ∇·ρv=MFS.
    P0arr = P0 === nothing ? copy(fluid.P) : P0
    fcache = hier.levels[1].cache
    # pressure-gauge target: keep the interior-mean of P pinned to the previous
    # step's mean so the (P−P0)/(η_b·dt) compressibility anchor and the gauge
    # don't fight. Removes the free constant pressure mode (it is only weakly
    # anchored by comp and otherwise drifts across V-cycles, stalling the
    # continuity residual — same fix as src/dyrel/solve.jl).
    P0int = @view P0arr[2:end-1, 2:end-1]
    Pgauge_target = T(sum(P0int) / length(P0int))

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
    refresh_coefficients!(hier, par; a1, gamma, dt)

    dyrel_apply_velocity_bcs!(fluid; top_cnv, bot_cnv, mfs_mean, Ddom = T(grid.D))
    # gauge-fix the free constant pressure mode (skipped in MMS/bnchm, where the
    # manufactured solution pins the level explicitly)
    bnchm || dyrel_apply_pressure_gauge!(fluid, P0arr, fcache.γ_eff)
    dyrel_apply_pressure_bcs!(fluid)

    err0 = one(T);  converged = false;  ncyc = 0
    for cyc in 1:par.maxit_PH                          # reuse maxit_PH as max V-cycles
        ncyc = cyc
        vcycle!(hier, 1, par; a1, a2, a3, gamma, dt, P0 = P0arr,
                top_cnv, bot_cnv, mfs_mean0 = mfs_mean)
        # re-pin the pressure gauge once per cycle (the smoother's multiplier
        # updates have nonzero mean mid-solve), then refresh ghosts before the
        # convergence-gate residual.
        bnchm || dyrel_apply_pressure_gauge!(fluid, P0arr, fcache.γ_eff)
        dyrel_apply_pressure_bcs!(fluid)
        level_residual!(hier.levels[1], par; a1, a2, a3, gamma, dt, P0 = P0arr, fscale = 1)
        normW, normU, normP = dyrel_residual_norms(fcache, fluid.rho)
        err = max(normW, normU, normP)
        cyc == 1 && (err0 = err + eps(T))
        verbose && @printf("  [GMG V%02d] err=%.3e  (W=%.2e U=%.2e P=%.2e)\n",
                           cyc, err, normW, normU, normP)
        if err < par.atol_PH || err / err0 < par.rtol
            converged = true;  break
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
    return (; converged, ncyc)
end
