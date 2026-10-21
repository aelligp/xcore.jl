# Coupled geometric multigrid: correction-scheme V-cycles with the DR sweep as
# the per-level smoother. Each smoothing PH step runs a fixed, small number of
# damped DR iterations at frozen P with a fixed per-level damping (no
# reductions, deterministic cost), then one multiplier update. The coarsest
# level solves its system to tolerance with the single-grid early-stop poll and
# Rayleigh retune. P stays frozen during the DR iterations: DR is a damped
# second-order iteration that converges only against a static operator.
#
# Every level carries the pseudo-transient continuity term (P − P0)/(η_b·dt).
# On the fine level P0 tracks the previous V-cycle iterate, so the term vanishes
# at convergence and the fixed point is ∇·(ρv) = MFS; coarse levels use P0 = 0
# (their P is the correction).

"""
    level_residual!(lvl, op; P0, fscale) -> nothing

Un-preconditioned residual `b − A·v` of a level (continuity with the
pseudo-transient term at `op.dt`) into `lvl.cache.Rv`/`R_P`.
"""
function level_residual!(lvl, op::StokesOp; P0, fscale)
    residual_continuity!(lvl.cache, lvl.fluid, lvl.grid, P0, op.dt; fscale)
    residual_momentum!(lvl.cache, lvl.fluid, lvl.grid, op; dr = false, fscale)
    return nothing
end

"""
    smooth!(lvl, par, op; nsweeps, ninner, fscale, P0, mfs_mean, Ddom,
            adaptive = false, update_P = true) -> nothing

`nsweeps` Powell-Hestenes steps on one level: `ninner` damped DR iterations at
fixed P (accumulators reset per step), then one multiplier update
`P += ω·γ_eff·R_P` (skipped with `update_P = false`). `adaptive = false`:
`ninner` is an exact count and the loop has no reductions. `adaptive = true`:
`ninner` is a cap, with the early-stop poll (`rel_drop_PT`, every
`n_tune_PT ÷ 8` iterations) and the periodic Rayleigh/Gershgorin retune.
"""
function smooth!(lvl, par::Parameters, op::StokesOp{D, T}; nsweeps::Int, ninner::Int,
                 fscale, P0, mfs_mean::Real, Ddom::Real,
                 adaptive::Bool = false, update_P::Bool = true) where {D, T}
    cache = lvl.cache;  fluid = lvl.fluid;  grid = lvl.grid
    fs = T(fscale)
    apply_velocity_bcs!(fluid, grid, op; mfs_mean, Ddom)
    apply_pressure_bcs!(fluid, op)
    rel_drop = T(par.rel_drop_PT)
    ncheck = max(2, par.n_tune_PT ÷ 8)
    for _ in 1:nsweeps
        foreach(A -> fill!(A, zero(T)), cache.dvdτ)
        errVel0 = zero(T);  itPT = 0
        while itPT < ninner
            itPT += 1
            if adaptive && itPT % par.n_tune_PT == 0
                foreach(copyto!, cache.Rv_old, cache.Rv)
            end
            @timeit_debug TO "residual_P" residual_continuity!(cache, fluid, grid, P0, op.dt; fscale = fs)
            @timeit_debug TO "residual_V" residual_momentum!(cache, fluid, grid, op; dr = true, fscale = fs)
            @timeit_debug TO "update velocity + BCs" begin
            update_velocity!(cache, fluid, par.CFL_PT)
            apply_velocity_bcs!(fluid, grid, op; mfs_mean, Ddom)
            end
            adaptive || continue
            if itPT % ncheck == 0
                errVel = @timeit_debug TO "poll norm" velocity_residual_norm(cache)
                if errVel0 == zero(T)
                    errVel0 = errVel + eps(T)
                elseif errVel / errVel0 < rel_drop
                    break
                end
            end
            if itPT % par.n_tune_PT == 0
                @timeit_debug TO "retune (Rayleigh + Gershgorin)" begin
                λmin = rayleigh_λmin(cache, par.CFL_PT)
                cache.c = 2 * sqrt(λmin) * par.c_fact_PT
                gershgorin!(cache, fluid, grid, op)
                check_preconditioner(cache)
                end
            end
        end
        update_P || continue
        @timeit_debug TO "PH multiplier update" begin
        residual_continuity!(cache, fluid, grid, P0, op.dt; fscale = fs)
        Pint = _interior(fluid.P)
        @. Pint += par.pres_relax_PT * cache.γ_eff * cache.R_P
        apply_pressure_bcs!(fluid, op)
        end
    end
    return nothing
end

# zero a coarse level's solution (the correction starts at 0)
function _zero_solution!(lvl)
    foreach(A -> fill!(A, 0), velocity(lvl.fluid));  fill!(lvl.fluid.P, 0)
    return nothing
end

_remove_mean!(a) = (a .-= sum(a) / length(a); nothing)

# Inertia-to-viscous ratio a1·ρ·h²/(η·dt) above which the coarsest level retunes
# its penalty. The rate law ρ = 1/(1 + γfact·λ) assumes the Schur diagonal
# limits λ, which holds once the momentum block is mass-matrix-like; below this
# the coarse grid is viscous-dominated, λ is already O(1), and under a strong
# viscosity contrast raising γfact degrades the V-cycle. Measured on the
# coarsest level: no gain at 1.3e3, clear gains from 3.6e3 up.
const γ_tune_ratio = 2e3

"""
    coarse_solve!(lvl, hier, par, op; fscale, P0, mfs_mean, Ddom) -> nothing

Solve the coarsest level: single PH steps (inner DR bounded at 20·ΣN) until the
residual drops by `hier.coarse_rtol` or `hier.ncoarse` steps are spent. When the
level is inertia-dominated its penalty is re-derived from the contraction
measured over windows of `γ_win_width` steps (target `par.mg_coarse_rho`); a
retune that worsens the contraction is undone. The value persists in
`lvl.cache.γfact`.
"""
function coarse_solve!(lvl, hier::MGHierarchy{T}, par::Parameters, op::StokesOp;
                       fscale, P0, mfs_mean, Ddom) where {T}
    cache = lvl.cache
    coarse_inner = 20 * sum(cells(lvl.grid))
    level_residual!(lvl, op; P0, fscale)
    nv, nP = residual_norms(cache, lvl.fluid.rho)
    r0 = max(nv..., nP) + eps(T)
    γfact = cache.γfact > zero(T) ? cache.γfact : T(par.γfact_PT)
    γ_target = T(par.mg_coarse_rho)
    inertia = op.a1 * (sum(lvl.fluid.rho) / length(lvl.fluid.rho)) * T(lvl.grid.h)^2 /
              ((sum(lvl.fluid.eta) / length(lvl.fluid.eta)) * op.dt)
    tune = inertia > T(γ_tune_ratio)
    γ_retunes = (tune && γ_target > zero(T)) ? 0 : γ_max_retunes
    win_from = 0;  err_ref = nP
    ρ_best = T(Inf);  γ_best = γfact
    retune!(γ) = (penalty!(cache, lvl.fluid, lvl.grid, op; γfact = γ);
                  check_preconditioner(cache); cache.γfact = γ)
    for k in 1:hier.ncoarse
        smooth!(lvl, par, op; nsweeps = 1, ninner = coarse_inner, adaptive = true,
                fscale, P0, mfs_mean, Ddom)
        level_residual!(lvl, op; P0, fscale)
        nv, nP = residual_norms(cache, lvl.fluid.rho)
        if γ_retunes < γ_max_retunes && k == win_from + γ_win_width
            ρ_obs = (nP / (err_ref + eps(T)))^(one(T) / γ_win_width)
            γ_window = γfact
            if ρ_obs > ρ_best
                γfact = γ_best
                retune!(γfact)
                γ_retunes = γ_max_retunes
            elseif ρ_obs > γ_target
                λ = max((inv(ρ_obs) - one(T)) / γfact, zero(T))
                γfact = λ > zero(T) ?
                        min((inv(γ_target) - one(T)) / λ, γfact * γ_max_growth) :
                        γfact * γ_max_growth
                retune!(γfact)
                γ_retunes += 1
            end
            if ρ_obs < ρ_best
                ρ_best = ρ_obs;  γ_best = γ_window
            end
            win_from = k;  err_ref = nP
        end
        max(nv..., nP) / r0 < hier.coarse_rtol && break
    end
    return nothing
end

"""
    vcycle!(hier, l, par, op; P0, mfs_mean0) -> nothing

One V-cycle from level `l`: pre-smooth, restrict the residual into the next
level's `bext`, recurse on the homogeneous error equation (`fscale = 0`, no
drain), prolong and add the correction `γ_mg·e` (pressure correction mean-free),
post-smooth. Level 1 carries the physical forcing and the bottom drain.
"""
function vcycle!(hier::MGHierarchy{T}, l::Int, par::Parameters, op::StokesOp{D};
                 P0, mfs_mean0::Real) where {T, D}
    lvl = hier.levels[l];  fine = l == 1
    fscale   = fine ? one(T) : zero(T)
    mfs_mean = fine ? T(mfs_mean0) : zero(T)
    Ddom     = T(lvl.grid.D)
    P0_lvl   = fine ? P0 : lvl.cache.P0
    sm(n) = smooth!(lvl, par, op; nsweeps = n, ninner = hier.inner, fscale,
                    P0 = P0_lvl, mfs_mean, Ddom)
    if l == length(hier.levels)
        @timeit_debug TO "L$l coarse solve" coarse_solve!(lvl, hier, par, op; fscale,
                                                          P0 = P0_lvl, mfs_mean, Ddom)
        return nothing
    end
    @timeit_debug TO "L$l pre-smooth" sm(hier.npre)
    @timeit_debug TO "L$l residual" level_residual!(lvl, op; P0 = P0_lvl, fscale)
    cl = hier.levels[l + 1]
    @timeit_debug TO "L$l restrict" begin
    _each_axis(Val(D)) do comp
        d = _val(comp)
        restrict_face!(cl.cache.bext[d], lvl.cache.Rv[d], comp)
    end
    restrict_cell!(cl.cache.bext_P, lvl.cache.R_P)
    _zero_solution!(cl)
    end
    vcycle!(hier, l + 1, par, op; P0, mfs_mean0)
    @timeit_debug TO "L$l prolong + correct" begin
    v = velocity(lvl.fluid);  vc = velocity(cl.fluid)
    _each_axis(Val(D)) do comp
        d = _val(comp)
        prolong_face!(lvl.cache.Rv[d], _compact(vc[d], d), comp, op)
    end
    eP = lvl.cache.R_P
    prolong_cell!(eP, _interior(cl.fluid.P), op.periodic)
    # The pressure correction carries the null mode; no velocity component has
    # one (Dirichlet or reflected at z), so only P is made mean-free.
    _remove_mean!(eP)
    # residual convention b − A·v: the coarse solve returns e ≈ A⁻¹·r, v += γ·e
    γ = hier.γ_mg
    for d in 1:D
        vd = _compact(v[d], d)
        @. vd += γ * lvl.cache.Rv[d]
    end
    Pint = _interior(lvl.fluid.P)
    @. Pint += γ * eP
    apply_velocity_bcs!(lvl.fluid, lvl.grid, op; mfs_mean, Ddom)
    apply_pressure_bcs!(lvl.fluid, op)
    end
    @timeit_debug TO "L$l post-smooth" sm(hier.npost)
    return nothing
end

"""
    fluidmech_gmg!(fluid, phase, hier, grid, par, scales; dt, P0 = nothing,
                   a1, a2, a3, b1, b2, b3, gamma = par.gamma, rtol_PH = 0,
                   bnchm = false, xBC, yBC, zBC, sds, top_cnv, bot_cnv, verbose = false)
        -> (; converged, relaxed, ncyc, errs)

Coupled geometric-multigrid solve: MFS source and frozen momentum advection on
the fine grid, coefficient coarsening and per-level autotune, then V-cycles
until the fine residual (continuity without the pseudo-transient term) and the
cycle's pressure change (`pressure_change_norm`) meet `par.atol_PH`, or — with
`rtol_PH > 0` — once the residual alone drops by `rtol_PH` below its value after
the first cycle (`relaxed`). `errs[k]` is the larger of the residual and the
pressure change after cycle `k`.
Throws if neither is reached within `par.maxit_PH` cycles.
"""
function fluidmech_gmg!(fluid, phase, hier::MGHierarchy{T}, grid, par::Parameters,
                        scales::Scales;
                        dt::Real, P0 = nothing,
                        a1::Real = 1, a2::Real = 1, a3::Real = 0,
                        b1::Real = 1, b2::Real = 0, b3::Real = 0,
                        gamma::Real = par.gamma, rtol_PH::Real = 0, bnchm::Bool = false,
                        xBC::Symbol = :periodic, yBC::Symbol = :periodic,
                        zBC::Symbol = :closed, sds::Integer = -1,
                        top_cnv::Integer = 1, bot_cnv::Integer = 1,
                        verbose::Bool = false) where {T}
    isfinite(dt) || throw(ArgumentError(
        "fluidmech_gmg!: dt must be finite (got $dt); the drunken-sailor term ∂ρ·g0·dt scales with dt"))
    op = StokesOp(T, _dims(grid), par; a1, a2, a3, gamma, dt, xBC, yBC, zBC, sds, top_cnv, bot_cnv)
    P0arr = P0 === nothing ? copyto!(hier.levels[1].cache.P0, fluid.P) : P0
    lvl1 = hier.levels[1];  fcache = lvl1.cache
    @timeit_debug TO "setup" begin
    bnchm || _update_mass_source!(fluid, phase, par, dt; a1, a2, a3, b1, b2, b3)
    momentum_advection!(fcache, fluid, grid, par, op)
    mfs_mean = bnchm ? zero(T) : T(sum(fluid.MFS) / length(fluid.MFS))
    refresh_coefficients!(hier, par, op)
    apply_velocity_bcs!(fluid, grid, op; mfs_mean, Ddom = T(grid.D))
    bnchm || pressure_gauge!(fluid, P0arr, fcache.η_b)
    apply_pressure_bcs!(fluid, op)
    end
    err0 = one(T);  converged = false;  relaxed = false;  ncyc = 0
    errs = T[]
    for cyc in 1:par.maxit_PH
        ncyc = cyc
        fcache.dP .= _interior(fluid.P)
        @timeit_debug TO "V-cycle" vcycle!(hier, 1, par, op; P0 = P0arr, mfs_mean0 = mfs_mean)
        @timeit_debug TO "gauge + convergence residual" begin
        bnchm || pressure_gauge!(fluid, P0arr, fcache.η_b)
        apply_pressure_bcs!(fluid, op)
        # gate on ∇·(ρv) = MFS: P0 = P removes the pseudo-transient term, which
        # would otherwise absorb the mass imbalance the gate must bound
        level_residual!(lvl1, op; P0 = fluid.P, fscale = 1)
        nv, nP = residual_norms(fcache, fluid.rho)
        fcache.dP .= _interior(fluid.P) .- fcache.dP
        ndP = pressure_change_norm(fcache.dP, fluid, grid.h)
        end
        # The residuals bound the pressure error only up to the conditioning of
        # the Schur complement; the cycle's pressure change bounds it directly.
        res = max(nv..., nP)
        err = max(res, ndP)
        isfinite(err) || error("fluidmech_gmg! produced a non-finite residual at V-cycle $cyc")
        push!(errs, err)
        cyc == 1 && (err0 = res + eps(T))
        verbose && @printf("  [GMG V%02d] err=%.3e  (v=%s P=%.2e dP=%.2e)\n", cyc, err,
                           join((@sprintf("%.2e", x) for x in nv), "/"), nP, ndP)
        if err < par.atol_PH
            converged = true;  break
        end
        if cyc > 1 && res < rtol_PH * err0
            relaxed = true;  break
        end
        # Pseudo-transient reference: P0 follows the latest iterate, so the term
        # measures the per-cycle increment and vanishes at convergence.
        bnchm || (P0arr .= fluid.P)
    end
    converged || relaxed ||
        error("fluidmech_gmg! failed to meet atol_PH=$(par.atol_PH) after $ncyc V-cycles")
    return (; converged, relaxed, ncyc, errs)
end
