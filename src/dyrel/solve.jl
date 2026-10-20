using KernelAbstractions
using Printf

# Powell-Hestenes / Dynamic-Relaxation solver orchestrator for xcore-DYREL.
# Full compressible two-phase Navier-Stokes (Re>1): BD2 inertia, momentum
# advection, drunken-sailor stabilization, and the MFS mass-flux source are all
# included. Mirrors the outer/inner structure of JustRelax's `_solve_DYREL!`
# (src/DYREL/solver.jl), extended with the xcore-specific terms.
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

# Powell-Hestenes penalty self-tuning (see the block in `fluidmech_dyrel!`).
# A contraction of ρ_want per sweep reaches atol_PH = 1e-6 in ~20 sweeps; asking
# for less than that costs more inner DR work per sweep than it saves in sweeps.
const ρ_want         = 0.5
const γ_win_width    = 4     # PH sweeps the contraction is measured over
const γ_max_retunes  = 6
const γ_max_growth   = 50    # per retune, bounding the response to a noisy ρ

_anderson_reset!(aa::AndersonPH) = (aa.n = 0; aa.slot = 0; aa.have_prev = false; nothing)

"""
    _anderson_pressure_update!(aa, P, f, m) -> nothing

Anderson-accelerated (type II, depth `m`) step of the multiplier fixed-point
iteration `P ← P + f`, where `P` is the interior pressure the sweep's velocity
solve ran under and `f` the plain update `ω·γ_eff·R_P` it produced. With the
history `ΔX = [ΔP]`, `ΔF = [Δf]` of earlier sweeps,

    θ = argmin ‖f − ΔF·θ‖₂,    P ← P + f − (ΔX + ΔF)·θ,

i.e. the update is extrapolated to where the linear model of the last `m`
residuals vanishes. Only the m×m least-squares problem runs on the host; every
array operation is a device broadcast or reduction.
"""
function _anderson_pressure_update!(aa::AndersonPH{T}, P::AbstractMatrix{T},
                                    f::AbstractMatrix{T}, m::Int) where {T}
    if length(aa.dX) != m
        aa.prev = [similar(f) for _ in 1:2]
        aa.dX = [similar(f) for _ in 1:m];  aa.dF = [similar(f) for _ in 1:m]
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
    n = aa.n
    G = [sum(_bc(*, aa.dF[i], aa.dF[j])) for i in 1:n, j in 1:n]
    b = [sum(_bc(*, aa.dF[i], f)) for i in 1:n]
    # The history columns turn nearly collinear as the iteration converges; the
    # pseudo-inverse drops the directions the residual differences cannot resolve.
    θ = pinv(G; rtol = sqrt(eps(T))) * b
    for i in 1:n
        P .-= T(θ[i]) .* (aa.dX[i] .+ aa.dF[i])
    end
    return nothing
end

function _check_dyrel_preconditioner(cache::DyrelCache{T}) where {T}
    for A in (cache.γ_eff, cache.η_b, cache.D_W, cache.D_U, cache.λmax_W, cache.λmax_U)
        all(isfinite, A) && all(>(zero(T)), A) ||
            throw(DomainError(A, "DYREL produced a non-finite or non-positive preconditioner entry"))
    end
    return nothing
end

"""
    fluidmech_dyrel!(fluid, phase, dyrel, grid, par, scales;
                     dt, P0 = nothing, a1=1, a2=1, a3=0, gamma=par.gamma,
                     require_convergence=false,
                     xBC=:periodic, zBC=:closed, top_cnv = 1, bot_cnv = 1,
                     rtol_PH = 0, verbose = false)
        -> (; converged, relaxed, total_iter, PH_iters)

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

`converged` means the PH error reached `par.atol_PH`. With `rtol_PH > 0` the
solve may instead stop once that error has dropped by `rtol_PH` relative to its
value at entry (`relaxed`), after at least one DR pass. At entry the error is
the nonlinear residual of the current Picard iterate, so this is the
inexact-solve forcing of the outer loop: each solve reduces it by a fixed factor
rather than driving it far below what the next buoyancy update will change.

`par.anderson_PH = m > 0` replaces the plain multiplier update with an
Anderson-accelerated one over the last `m` sweeps (`_anderson_pressure_update!`).

Buoyancy enters via `fluid.Drho`; BCs: periodic-x, closed-z for W, free/no-slip
for U (`top_cnv`/`bot_cnv` = ±1). `dt` must be finite: the drunken-sailor term
`∂ρ·g0·dt` scales with it, in both the residual and the Gershgorin bound.
"""
function fluidmech_dyrel!(fluid::FluidState{T}, phase::PhaseState{T},
                          dyrel::DyrelCache{T},
                          grid::Grid{T}, par::Parameters{T}, scales::Scales{T};
                          dt::Real, P0 = nothing,
                          a1::Real = 1, a2::Real = 1, a3::Real = 0,
                          b1::Real = 1, b2::Real = 0, b3::Real = 0,
                          gamma::Real = par.gamma,
                          rtol_PH::Real = 0,
                          bnchm::Bool = false,
                          xBC::Symbol = :periodic, zBC::Symbol = :closed,
                          sds::Integer = -1,
                          top_cnv::Integer = 1, bot_cnv::Integer = 1,
                          open_cnv::Bool = false,
                          require_convergence::Bool = false,
                          verbose::Bool = false) where {T<:AbstractFloat}
    isfinite(dt) || throw(ArgumentError(
        "fluidmech_dyrel!: dt must be finite (got $dt); the drunken-sailor term ∂ρ·g0·dt scales with dt"))
    m_aa = par.anderson_PH
    m_aa >= 0 || throw(ArgumentError("fluidmech_dyrel!: anderson_PH must be ≥ 0 (got $m_aa)"))
    cache  = dyrel
    h      = grid.h
    # P0, MFS and the frozen advection change between calls: a new map each solve
    _anderson_reset!(cache.anderson)
    # Reference pressure: P0 only sets the level the pressure gauge re-references
    # to, so the free constant mode is removed without the absolute level (read
    # back by `update_pressure!` as Pt = Pl + P) drifting between solves.
    #
    # The continuity residual carries NO pressure term here. `fluidmech.jl`
    # assembles the constraint as ∇·(ρv) = MFS with an empty pressure block, and
    # the physical compressibility already reaches it through MFS, so an
    # artificial-compressibility term comp = (P−P0)/(η_b·dt) has no counterpart in
    # the system being solved: with P0 held fixed it would not vanish at the DYREL
    # fixed point, which would then solve ∇·(ρv) = MFS − comp, biased off the
    # direct solution by O(1/γfact). `dt = Inf` in the `dyrel_residual_P!` calls
    # below is what keeps it out. The multigrid path does carry the term, as a
    # pseudo-transient regulariser whose P0 tracks the previous V-cycle iterate,
    # so there it vanishes at convergence.
    P0arr = P0 === nothing ? copy(fluid.P) : P0

    # --- mass-flux source update (mirrors fluidmech.jl:71-82). DYREL does NOT
    # update MFS inside the solve, so do it here once per call from the
    # phsevo!-provided `advn_rho`, exactly as the direct solver does. The outer
    # Picard loop converges it. Skipped in MMS mode. ---
    @timeit_debug TO "setup" begin
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

    # --- autotune setup for the current operator ---
    # Operator-derived augmented-Lagrangian penalty γ_eff = γfact / s_P_phys
    # (per-cell, from the penalty-free Schur diagonal); see init.jl. Leaves
    # cache.D_W/D_U = D_full. The PH loop below re-derives γfact from the
    # contraction it measures and leaves it in the cache: the operator barely
    # moves between Picard sweeps, so re-entering at `par.γfact_PT` would spend
    # the first sweeps of every solve rediscovering a value already known. The
    # cache constructor zeroes it, so the first solve of a run starts from the
    # parameter.
    γfact = cache.γfact[] > zero(T) ? cache.γfact[] : T(par.γfact_PT)
    compute_AL_penalty!(cache, fluid, h; γfact, a1, gamma, dt,
                        g0 = par.g0, xBC)
    _check_dyrel_preconditioner(cache)
    # `cache.c` carries the damping learned by the previous solve. The operator
    # barely moves between Picard sweeps, so re-entering undamped (α=1) would
    # discard a usable estimate and spend the first n_tune_PT iterations
    # rediscovering it. The cache constructor zeroes it, so the first solve of a
    # run is undamped.
    update_dτ_α_β!(cache, par.CFL_PT)
    cache.initialized = true
    cache.sig_Nz = grid.Nz;  cache.sig_Nx = grid.Nx

    fill!(cache.dWdτ, zero(T));  fill!(cache.dUdτ, zero(T))

    # initial BC enforcement
    dyrel_apply_velocity_bcs!(fluid; top_cnv, bot_cnv, open_cnv, xBC, sds, h, mfs_mean, Ddom)
    # gauge-fix the free constant pressure mode (skipped in MMS/bnchm, where the
    # manufactured solution pins the level explicitly)
    bnchm || dyrel_apply_pressure_gauge!(fluid, P0arr)
    dyrel_apply_pressure_bcs!(fluid; xBC)
    end

    errW0 = errU0 = errP0 = one(T)
    err_min = T(Inf)                         # best outer err so far (JustRelax)
    rel_drop = T(par.rel_drop_PT)            # adaptive inner-DR tolerance factor
    total_iter = 0
    PH_iters   = 0
    converged  = false
    relaxed    = false
    err_first  = zero(T)
    # penalty self-tuning state: window over which the PH contraction is measured
    γ_win_from = 2                           # skip sweep 1 (errP0 re-baselines at 2)
    γ_retunes  = 0
    err_ref    = zero(T)

    for itPH in 1:par.maxit_PH
        PH_iters = itPH

        # PH residuals (stress is recomputed inline inside dyrel_residual_V!)
        @timeit_debug TO "PH residual" begin
        dyrel_residual_V!(cache, phase, fluid, grid, par; dr = false, a1, a2, a3, gamma, dt, xBC)
        dyrel_residual_P!(cache, fluid, grid, par, P0arr, Inf)

        normW, normU, normP = dyrel_residual_norms(cache, fluid.rho)
        all(isfinite, (normW, normU, normP)) ||
            error("fluidmech_dyrel! produced a non-finite residual at PH iteration $itPH")
        if itPH == 1
            errW0 = normW + eps(T);  errU0 = normU + eps(T);  errP0 = normP + eps(T)
        elseif itPH == 2
            # Re-baseline the pressure scale on the second iterate. At entry P is
            # the previous Picard sweep's field, so normP measures a stale
            # pressure and normalising by it hides the multiplier iteration's
            # actual progress. Matches JustRelax's `itPH == 2` errPt0 reset.
            errP0 = normP + eps(T)
        end
        # The multiplier step γ_eff·R_P must be small too, not just R_P: the
        # pressure a solve returns is off by O(γ_eff·R_P), so a residual-only
        # gate accepts a less accurate pressure the harder the constraint is
        # penalised, and the Picard loop above cannot see the difference.
        normdP = dyrel_pressure_update_norm(cache, fluid, h)
        end
        err = max(min(normW / errW0, normW),
                  min(normU / errU0, normU),
                  min(normP / errP0, normP),
                  normdP)
        verbose && @printf("  [PH %02d] iter=%07d  err=%.3e  (W=%.2e U=%.2e P=%.2e dP=%.2e)\n",
                           itPH, total_iter, err, normW, normU, normP, normdP)
        if err < par.atol_PH
            converged = true
            break
        end
        itPH == 1 && (err_first = err)
        if itPH > 1 && err < rtol_PH * err_first
            relaxed = true
            break
        end

        # When the outer error stalls (no longer dropping), tighten rel_drop so
        # the inner velocity solve is more accurate — this prevents the outer
        # Powell-Hestenes loop from stagnating on an under-solved velocity.
        if err > err_min * T(1.05)
            rel_drop = max(rel_drop * T(0.1), T(1e-3))
        end
        err_min = min(err_min, err)

        # --- augmented-Lagrangian penalty self-tuning ------------------------
        # The Powell-Hestenes contraction is ρ = 1/(1 + γfact·λ), where
        # λ = λmin(diag(s_P)⁻¹·S) measures how well the Schur DIAGONAL
        # preconditions the Schur complement S. λ is O(1) only while the
        # momentum block is viscous-dominated, where S is mass-matrix-like. Once
        # a1·ρ/dt ≫ η/h² the momentum block is the mass matrix, S becomes a
        # Poisson operator, and λ falls with h² — so no fixed γfact holds the
        # contraction across resolutions or time steps. λ IS a property of the
        # frozen operator, so measuring ρ over a window of sweeps gives the
        # γfact that reaches `ρ_want` in closed form.
        #
        # The rate is read off the continuity residual, the quantity the rate law
        # describes and the one term of the gate that does not itself scale with
        # γ_eff — reading it off `err` would let each correction inflate the
        # multiplier step it is judged by and drive the penalty up without bound.
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
                # P is the Lagrange multiplier and stays valid: γ_eff only sets
                # the step length of its update, and compute_AL_penalty! keeps
                # γ_eff, η_b and the preconditioner diagonals mutually consistent.
                @timeit_debug TO "PH penalty retune" begin
                compute_AL_penalty!(cache, fluid, h; γfact, a1, gamma, dt,
                                    g0 = par.g0, xBC)
                _check_dyrel_preconditioner(cache)
                update_dτ_α_β!(cache, par.CFL_PT)
                end
                # the history was gathered under the old γ_eff, a different map
                _anderson_reset!(cache.anderson)
                verbose && @printf("  [PH %02d] rho=%.3f over %d sweeps -> gfact=%.4g\n",
                                   itPH, ρ_obs, γ_win_width, γfact)
                # Only an applied correction spends budget: the early sweeps
                # contract well on any penalty, so windows that find nothing to
                # fix must not exhaust it before the loop reaches its asymptote.
                γ_retunes += 1
            end
            γ_win_from = itPH                 # next window starts here
            err_ref    = normP
        end

        # Inner-DR target: reduce the velocity residual by `rel_drop` WITHIN this
        # PH sweep. The baseline is re-established at each sweep's first poll
        # (`errVel0 = 0` here), matching the multigrid smoother. Measuring the
        # drop against a baseline frozen at one earlier sweep instead makes the
        # break point depend on how that one sample compares to the present
        # sweep, which is unrelated to how well the current velocity is solved:
        # at a strong penalty every sweep then exits at the first poll with the
        # velocity far from solved, and the PH loop stalls at ρ ≈ 1.
        errVel0 = zero(T)

        itPT = 0
        @timeit_debug TO "DR iterations" while itPT < par.maxit_PT
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
            @timeit_debug TO "residual_P" dyrel_residual_P!(cache, fluid, grid, par, P0arr, Inf)

            # preconditioned DR momentum residual + damped velocity update
            @timeit_debug TO "residual_V" dyrel_residual_V!(cache, phase, fluid, grid, par; dr = true, a1, a2, a3, gamma, dt, xBC)
            @timeit_debug TO "update velocity + BCs" begin
            dyrel_update_velocity!(cache, fluid)
            dyrel_apply_velocity_bcs!(fluid; top_cnv, bot_cnv, open_cnv, xBC, sds, h, mfs_mean, Ddom)
            end

            # periodic re-tune of λ_max (Gershgorin) and λ_min (Rayleigh)
            if itPT % par.n_tune_PT == 0
                errVel = @timeit_debug TO "poll norm" dyrel_velocity_residual_norm(cache)
                isfinite(errVel) ||
                    error("fluidmech_dyrel! produced a non-finite velocity residual at DR iteration $total_iter")
                if errVel0 == zero(T)
                    errVel0 = errVel + eps(T)     # this sweep's baseline: never breaks
                else
                    relerr = errVel / errVel0
                    verbose && @printf("    [DR %07d] errVel=%.3e  rel=%.3e\n",
                                       itPT, errVel, relerr)
                    relerr < rel_drop && break
                end

                # Optional rheology refresh: η(eII) is a function of the very
                # velocity this loop solves for, so it can be resolved here
                # instead of costing the outer Picard loop a linear solve per
                # rheology update. Refreshed before the Gershgorin retune below
                # so the preconditioner and λmax the next iterations run on see
                # the same η the residual does. Off by default — it redefines
                # what this function solves; see `par.rheology_in_PT`.
                if par.rheology_in_PT
                    update_kinematics!(phase, fluid, grid)
                    update_viscosity!(phase, fluid, grid, par, scales; xBC, zBC,
                                      relaxation = par.visc_relax_PT,
                                      eta_prev = cache.eta_prev)
                end

                @timeit_debug TO "retune (Rayleigh + Gershgorin)" begin
                λmin = dyrel_rayleigh_λmin(cache)
                cache.c[] = 2 * sqrt(λmin) * par.c_fact_PT
                gershgorin!(cache, fluid, h; a1, gamma, dt, g0 = par.g0, xBC)  # D_full retune (γ_eff fixed)
                _check_dyrel_preconditioner(cache)
                update_dτ_α_β!(cache, par.CFL_PT)
                end
            end
        end
        cache.last_DR_iters[] = itPT

        # Powell-Hestenes augmented-Lagrangian multiplier update on interior
        # cells: P += ω·γ_eff·R_P, with the SAME per-cell γ_eff = γfact/s_P_phys
        # injected into the momentum residual (P_num) and the comp term (η_b).
        # ω = par.pres_relax_PT is the under-relaxation (1 = Newton-Uzawa step).
        # R_P is refreshed first: the inner loop's last velocity update landed
        # after its last continuity evaluation, so the cached R_P is one step
        # stale. Matches JustRelax's `compute_∇V_strain_rate_RP!` immediately
        # before `stokes.P += γ_eff·RP`.
        @timeit_debug TO "PH multiplier update" begin
        dyrel_residual_P!(cache, fluid, grid, par, P0arr, Inf)
        if m_aa > 0
            # P_num = γ_eff·R_P from the call above serves as the update scratch:
            # it is next read by a dr = true residual, which rebuilds it first.
            f = cache.P_num
            f .*= par.pres_relax_PT
            # the gauge below removes the constant mode, so it is no part of the
            # fixed-point residual the history should fit
            bnchm || (f .-= sum(f) / length(f))
            _anderson_pressure_update!(cache.anderson, @view(fluid.P[2:end-1, 2:end-1]), f, m_aa)
        else
            @views @. fluid.P[2:end-1, 2:end-1] += par.pres_relax_PT * cache.γ_eff * cache.R_P
        end
        # pin the pressure gauge (remove the free constant mode) then refresh ghosts
        bnchm || dyrel_apply_pressure_gauge!(fluid, P0arr)
        dyrel_apply_pressure_bcs!(fluid; xBC)
        end
        # The production path returns an inexact update at the cap; validation
        # can make a missed tolerance a hard failure.
        total_iter > par.total_iterMax_PT && break
    end
    cache.last_PH_iters[] = PH_iters
    cache.γfact[] = γfact

    !converged && require_convergence && error(
        "fluidmech_dyrel! failed to meet atol_PH=$(par.atol_PH) after " *
        "$PH_iters Powell-Hestenes iterations and $total_iter DR iterations")
    return (; converged, relaxed, total_iter, PH_iters)
end
