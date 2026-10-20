# Velocity-block V-cycle: multigrid for the momentum equations at FIXED
# pressure, A_γ·v = b − ∇P, where A_γ carries the augmented-Lagrangian term
# γ_eff·∇(∇·v) through `P_num = γ_eff·R_P`. The DR sweep at fixed P is the
# smoother (`smooth!` with `update_P = false`). Coarse levels hold P = 0 and
# bext_P = 0, so their residual kernels evaluate A_γ·e − bext_{W,U}: the
# pressure never enters the hierarchy. `comp` is off on every level — the fine
# level passes its own P as the reference and coarse levels have P = P0 = 0.

# Momentum residual r = b − A_γ·v at the level's current (W, U, P) into
# cache.R_{W,U}; continuity is evaluated only to form P_num (dt = Inf ⇒ no comp).
# The `dr = false` kernels return b − A·v WITHOUT the penalty; the `dr = true`
# form is (b − A_γ·v)/D, so it is evaluated and multiplied back by D.
function velocity_residual!(lvl, par::Parameters; a1, a2, a3, gamma, dt, fscale,
                            xBC::Symbol = :periodic)
    c = lvl.cache
    dyrel_residual_P!(c, lvl.fluid, lvl.grid, par, lvl.fluid.P, Inf; fscale)
    dyrel_residual_V!(c, lvl.phase, lvl.fluid, lvl.grid, par;
                      dr = true, a1, a2, a3, gamma, dt, fscale, xBC)
    c.R_W .*= c.D_W;  c.R_U .*= c.D_U
    return nothing
end

_velocity_norm(lvl) = (n = dyrel_residual_norms(lvl.cache, lvl.fluid.rho); max(n[1], n[2]))

"""
    velocity_vcycle!(hier, l, par; a1,a2,a3,gamma,dt, top_cnv,bot_cnv, mfs_mean0,
                     xBC, sds) -> nothing

One velocity V-cycle from level `l` at fixed pressure. Smoothing levels do
`hier.npre`/`hier.npost` smoother calls of exactly `hier.inner` DR iterations;
the coarsest level runs adaptive DR until its velocity residual drops by
`hier.coarse_rtol` (at most `hier.ncoarse` sub-solves).
"""
function velocity_vcycle!(hier::MGHierarchy{T}, l::Int, par::Parameters{T};
                          a1, a2, a3, gamma, dt,
                          top_cnv::Integer, bot_cnv::Integer, mfs_mean0::Real,
                          xBC::Symbol = :periodic, sds::Integer = -1) where {T}
    lvl      = hier.levels[l]
    fine     = (l == 1)
    fscale   = fine ? one(T) : zero(T)
    mfs_mean = fine ? T(mfs_mean0) : zero(T)
    Ddom     = T(lvl.grid.D)
    P0       = fine ? lvl.fluid.P : lvl.P0
    sm(n, inner, adaptive) = smooth!(lvl, par; nsweeps = n, ninner = inner, fscale,
                                     a1, a2, a3, gamma, dt, P0, top_cnv, bot_cnv,
                                     mfs_mean, Ddom, xBC, sds, adaptive, update_P = false)
    res!() = velocity_residual!(lvl, par; a1, a2, a3, gamma, dt, fscale, xBC)

    if l == length(hier.levels)
        @timeit_debug TO "L$l coarse solve" begin
        res!();  r0 = _velocity_norm(lvl) + eps(T)
        for _ in 1:hier.ncoarse
            sm(1, 20 * (lvl.grid.Nz + lvl.grid.Nx), true)
            res!()
            _velocity_norm(lvl) / r0 < hier.coarse_rtol && break
        end
        end
        return nothing
    end

    @timeit_debug TO "L$l pre-smooth" sm(hier.npre, hier.inner, false)
    @timeit_debug TO "L$l residual" res!()
    cl = hier.levels[l + 1]
    @timeit_debug TO "L$l restrict" begin
    restrict_w!(cl.cache.bext_W, lvl.cache.R_W)
    restrict_u!(cl.cache.bext_U, lvl.cache.R_U)
    fill!(cl.cache.bext_P, 0)
    _zero_solution!(cl)
    end

    velocity_vcycle!(hier, l + 1, par; a1, a2, a3, gamma, dt,
                     top_cnv, bot_cnv, mfs_mean0, xBC, sds)

    @timeit_debug TO "L$l prolong + correct" begin
    eW = lvl.cache.R_W;  eU = lvl.cache.R_U
    @views prolong_w!(eW, cl.fluid.W[:, 2:end - 1]; xBC, sds)
    @views prolong_u!(eU, cl.fluid.U[2:end - 1, :]; top_cnv, bot_cnv)
    @views @. lvl.fluid.W[:, 2:end - 1] += eW
    @views @. lvl.fluid.U[2:end - 1, :] += eU
    dyrel_apply_velocity_bcs!(lvl.fluid; top_cnv, bot_cnv, xBC, sds, mfs_mean, Ddom)
    end

    @timeit_debug TO "L$l post-smooth" sm(hier.npost, hier.inner, false)
    return nothing
end

# Fine-level forcing (MFS source, frozen momentum advection), coefficient
# hierarchy with the AL penalty γ_eff = γfact/s_P on every level, and BCs.
# Returns the mean MFS that sets the bottom drain.
function _velocity_mg_setup!(fluid::FluidState{T}, phase, hier, grid, par;
                             dt, a1, a2, a3, b1, b2, b3, gamma, γfact, bnchm,
                             xBC, zBC, sds, top_cnv, bot_cnv) where {T}
    fcache = hier.levels[1].cache
    if !bnchm
        inv_dt = isfinite(dt) ? T(inv(dt)) : zero(T)
        @. fluid.drhodt = -phase.advn_rho
        @. fluid.MFS = fluid.MFS - par.alpha * (
            (T(a1) * fluid.rho - T(a2) * fluid.rhoo - T(a3) * fluid.rhooo) * inv_dt -
            (T(b1) * fluid.drhodt + T(b2) * fluid.drhodto + T(b3) * fluid.drhodtoo))
    end
    dyrel_momentum_advection!(fcache, fluid, grid, par; xBC, zBC)
    mfs_mean = bnchm ? zero(T) : T(sum(fluid.MFS) / length(fluid.MFS))
    refresh_coefficients!(hier, par; a1, gamma, dt, xBC, γfact)
    dyrel_apply_velocity_bcs!(fluid; top_cnv, bot_cnv, xBC, sds, mfs_mean, Ddom = T(grid.D))
    dyrel_apply_pressure_bcs!(fluid; xBC)
    return mfs_mean
end

"""
    velocity_mg_solve!(fluid, phase, hier, grid, par; dt, a1,a2,a3,b1,b2,b3,
                       gamma, γfact, ncyc, rtol, bnchm, xBC,zBC,sds,
                       top_cnv,bot_cnv) -> errs

Solve the momentum block at the current, fixed pressure with velocity
V-cycles until the fine velocity residual drops by `rtol` or `ncyc` cycles are
spent. Returns the fine velocity residual before the first and after each cycle.
"""
function velocity_mg_solve!(fluid::FluidState{T}, phase::PhaseState{T},
                            hier::MGHierarchy{T}, grid::Grid{T}, par::Parameters{T};
                            dt::Real, a1::Real = 1, a2::Real = 1, a3::Real = 0,
                            b1::Real = 1, b2::Real = 0, b3::Real = 0,
                            gamma::Real = par.gamma, γfact::Real = par.γfact_PT,
                            ncyc::Int = 50, rtol::Real = 1e-6, bnchm::Bool = false,
                            xBC::Symbol = :periodic, zBC::Symbol = :closed,
                            sds::Integer = -1, top_cnv::Integer = 1,
                            bot_cnv::Integer = 1) where {T}
    mfs_mean = _velocity_mg_setup!(fluid, phase, hier, grid, par; dt, a1, a2, a3, b1, b2, b3,
                                   gamma, γfact, bnchm, xBC, zBC, sds, top_cnv, bot_cnv)
    lvl1 = hier.levels[1]
    velocity_residual!(lvl1, par; a1, a2, a3, gamma, dt, fscale = one(T), xBC)
    errs = T[_velocity_norm(lvl1)]
    for _ in 1:ncyc
        velocity_vcycle!(hier, 1, par; a1, a2, a3, gamma, dt,
                         top_cnv, bot_cnv, mfs_mean0 = mfs_mean, xBC, sds)
        velocity_residual!(lvl1, par; a1, a2, a3, gamma, dt, fscale = one(T), xBC)
        push!(errs, _velocity_norm(lvl1))
        isfinite(errs[end]) || error("velocity_mg_solve!: non-finite residual after $(length(errs) - 1) cycles")
        errs[end] / errs[1] < rtol && break
    end
    return errs
end

# Cahouet–Chabard inertial Schur complement. With the momentum block reduced to
# its inertia, v_f = −dt/(c_f·ρ_f)·∂P (c_z = a1, c_x = a1 + γ on periodic x), and
# the face density cancels in ∇·(ρ_f v_f): the inertial Schur complement is
# dt·A with A = −∇·(c⁻¹∇·), a constant-coefficient 5-point operator with Neumann
# rows at closed walls (fixed normal velocity) and periodic x-links. A⁻¹ is
# approximated matrix-free by one V-cycle on the hierarchy's grids (weighted
# Jacobi smoothing, the p-transfers of the velocity hierarchy); only the step
# of the multiplier update depends on it, not the fixed point R_P = 0.

@kernel function _cc_residual_kernel!(r, @Const(p), @Const(f), wz, wx, invh2, periodic::Bool)
    iz, ix = @index(Global, NTuple)
    Nz, Nx = size(p)
    begin
        pc  = p[iz, ix]
        acc = zero(pc)
        iz > 1  && (acc += wz * (pc - p[iz - 1, ix]))
        iz < Nz && (acc += wz * (pc - p[iz + 1, ix]))
        if ix > 1
            acc += wx * (pc - p[iz, ix - 1])
        elseif periodic
            acc += wx * (pc - p[iz, Nx])
        end
        if ix < Nx
            acc += wx * (pc - p[iz, ix + 1])
        elseif periodic
            acc += wx * (pc - p[iz, 1])
        end
        r[iz, ix] = f[iz, ix] - acc * invh2
    end
end

@kernel function _cc_jacobi_kernel!(p, @Const(r), wz, wx, invh2, periodic::Bool, ω)
    iz, ix = @index(Global, NTuple)
    Nz, Nx = size(p)
    begin
        nz = (iz > 1) + (iz < Nz)
        nx = periodic ? 2 : (ix > 1) + (ix < Nx)
        p[iz, ix] += ω * r[iz, ix] / ((nz * wz + nx * wx) * invh2)
    end
end

# Per-level scratch (p, f, r) for the Poisson V-cycle, on the fluid's backend.
function _cc_workspace(hier::MGHierarchy{T}, backend) where {T}
    return [(p = xcore_zeros(backend, T, lv.grid.Nz, lv.grid.Nx),
             f = xcore_zeros(backend, T, lv.grid.Nz, lv.grid.Nx),
             r = xcore_zeros(backend, T, lv.grid.Nz, lv.grid.Nx),
             h = T(lv.grid.h)) for lv in hier.levels]
end

function _cc_sweeps!(w, n, wz, wx, periodic)
    backend = KernelAbstractions.get_backend(w.p)
    invh2 = inv(w.h^2)
    for _ in 1:n
        _cc_residual_kernel!(backend)(w.r, w.p, w.f, wz, wx, invh2, periodic; ndrange = size(w.p))
        _cc_jacobi_kernel!(backend)(w.p, w.r, wz, wx, invh2, periodic, oftype(w.h, 0.8); ndrange = size(w.p))
    end
    return nothing
end

function _cc_vcycle!(ws, l, wz, wx, periodic, xBC; ν = 2)
    w = ws[l]
    if l == length(ws)
        _cc_sweeps!(w, 20 * max(size(w.p)...), wz, wx, periodic)   # coarsest: Jacobi to near-convergence
        return nothing
    end
    _cc_sweeps!(w, ν, wz, wx, periodic)
    backend = KernelAbstractions.get_backend(w.p)
    _cc_residual_kernel!(backend)(w.r, w.p, w.f, wz, wx, inv(w.h^2), periodic; ndrange = size(w.p))
    restrict_p!(ws[l + 1].f, w.r)
    fill!(ws[l + 1].p, 0)
    _cc_vcycle!(ws, l + 1, wz, wx, periodic, xBC; ν)
    prolong_p!(w.r, ws[l + 1].p; xBC)
    w.p .+= w.r
    _cc_sweeps!(w, ν, wz, wx, periodic)
    return nothing
end

# P += (ω/dt)·A⁻¹(ρ·R_P), A⁻¹ ≈ one Poisson V-cycle (R_P is the 1/ρ-scaled
# continuity residual, hence the ρ). A has a constant null mode on
# Neumann/periodic domains, so the right-hand side and the result are mean-free.
function _cc_update!(ws, fluid::FluidState{T}, c, ω, a1, gamma, dt, xBC::Symbol) where {T}
    periodic = xBC === :periodic
    wz = T(inv(a1));  wx = T(inv(a1 + (periodic ? gamma : zero(gamma))))
    f = ws[1].f
    @. f = fluid.rho * c.R_P
    f .-= sum(f) / length(f)
    fill!(ws[1].p, 0)
    _cc_vcycle!(ws, 1, wz, wx, periodic, xBC)
    p = ws[1].p
    p .-= sum(p) / length(p)
    @views @. fluid.P[2:end-1, 2:end-1] += T(ω / dt) * p
    return nothing
end

"""
    fluidmech_dyrel_vmg!(fluid, phase, hier, grid, par, scales; dt, P0=nothing,
                         a1,a2,a3,b1,b2,b3, gamma, γfact=5, nv=1, rtol_PH=0,
                         bnchm=false, xBC,zBC,sds,top_cnv,bot_cnv, verbose=false)
        -> (; converged, relaxed, PH_iters, errs)

DYREL's Powell-Hestenes iteration with the inner velocity solve replaced by
`nv` velocity V-cycles (`velocity_vcycle!`) at fixed pressure, followed by the
multiplier update `P += ω·γ_eff·R_P`; with `cc = true` the update also adds the
inertial Cahouet–Chabard term `(ω/dt)·A⁻¹(ρ·R_P)`, with A⁻¹ one matrix-free
Poisson V-cycle (see `_cc_update!`), which
keeps the pressure iteration grid-independent when inertia dominates. It changes
only the step, not the fixed point `R_P = 0`. The velocity block carries the AL penalty
`γ_eff = γfact/s_P`; point-DR smoothing of that block degrades as `γfact` grows,
so the default is small. Stops as `converged` when the full residual meets
`par.atol_PH`, or as `relaxed` once it drops by `rtol_PH` below its value after
the first update. `errs[k]` is the full residual after PH step `k`.
"""
function fluidmech_dyrel_vmg!(fluid::FluidState{T}, phase::PhaseState{T},
                              hier::MGHierarchy{T}, grid::Grid{T}, par::Parameters{T},
                              scales::Scales{T};
                              dt::Real, P0 = nothing,
                              a1::Real = 1, a2::Real = 1, a3::Real = 0,
                              b1::Real = 1, b2::Real = 0, b3::Real = 0,
                              gamma::Real = par.gamma, γfact::Real = 5, nv::Int = 1,
                              cc::Bool = false,
                              rtol_PH::Real = 0, bnchm::Bool = false,
                              xBC::Symbol = :periodic, zBC::Symbol = :closed,
                              sds::Integer = -1, top_cnv::Integer = 1, bot_cnv::Integer = 1,
                              verbose::Bool = false) where {T}
    @timeit_debug TO "setup" begin
    P0arr = P0 === nothing ? copy(fluid.P) : P0
    mfs_mean = _velocity_mg_setup!(fluid, phase, hier, grid, par; dt, a1, a2, a3, b1, b2, b3,
                                   gamma, γfact, bnchm, xBC, zBC, sds, top_cnv, bot_cnv)
    lvl1 = hier.levels[1];  c = lvl1.cache
    ccws = cc && a1 > 0 ? _cc_workspace(hier, KernelAbstractions.get_backend(fluid.P)) : nothing
    end
    err0 = one(T);  converged = false;  relaxed = false;  PH_iters = 0
    errs = T[]
    for it in 1:par.maxit_PH
        PH_iters = it
        @timeit_debug TO "velocity V-cycle" for _ in 1:nv
            velocity_vcycle!(hier, 1, par; a1, a2, a3, gamma, dt,
                             top_cnv, bot_cnv, mfs_mean0 = mfs_mean, xBC, sds)
        end
        @timeit_debug TO "PH multiplier update" begin
        dyrel_residual_P!(c, fluid, grid, par, P0arr, Inf)
        @views @. fluid.P[2:end-1, 2:end-1] += par.pres_relax_PT * c.γ_eff * c.R_P
        end
        ccws === nothing || @timeit_debug TO "Cahouet–Chabard Poisson V-cycle" _cc_update!(ccws, fluid, c, par.pres_relax_PT, a1, gamma, dt, xBC)

        # full (penalty-free) residual: the AL term vanishes with R_P at the solution
        @timeit_debug TO "gauge + convergence residual" begin
        bnchm || dyrel_apply_pressure_gauge!(fluid, P0arr)
        dyrel_apply_pressure_bcs!(fluid; xBC)
        dyrel_residual_P!(c, fluid, grid, par, P0arr, Inf)
        dyrel_residual_V!(c, phase, fluid, grid, par; dr = false, a1, a2, a3, gamma, dt, xBC)
        nW, nU, nP = dyrel_residual_norms(c, fluid.rho)
        end
        err = max(nW, nU, nP)
        isfinite(err) || error("fluidmech_dyrel_vmg! produced a non-finite residual at PH step $it")
        push!(errs, err)
        it == 1 && (err0 = err + eps(T))
        verbose && @printf("  [vMG PH %02d] err=%.3e  (W=%.2e U=%.2e P=%.2e)\n", it, err, nW, nU, nP)
        if err < par.atol_PH
            converged = true;  break
        end
        if it > 1 && err < rtol_PH * err0
            relaxed = true;  break
        end
    end
    converged || relaxed || error("fluidmech_dyrel_vmg! failed to meet atol_PH=$(par.atol_PH) after $PH_iters PH steps")
    return (; converged, relaxed, PH_iters, errs)
end
