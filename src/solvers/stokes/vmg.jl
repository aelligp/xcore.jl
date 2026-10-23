# Powell–Hestenes iteration whose inner velocity solve is a velocity V-cycle at
# fixed pressure, A_γ·v = b − ∇P, A_γ carrying the augmented-Lagrangian term
# γ_eff·∇(∇·v) through P_num = γ_eff·R_P. The DR sweep at fixed P is the
# smoother. Coarse levels hold P = 0 and bext_P = 0, so their residual is
# A_γ·e − bext: the pressure never enters the hierarchy. No level carries the
# pseudo-transient term (the fine level references its own P, coarse levels have
# P = P0 = 0).

# Momentum residual b − A_γ·v into cache.Rv. The dr = false kernels omit the
# penalty; the dr = true form is (b − A_γ·v)/D, so it is multiplied back by D.
function velocity_residual!(lvl, op::StokesOp{D}; fscale) where {D}
    c = lvl.cache
    residual_continuity!(c, lvl.fluid, lvl.grid, lvl.fluid.P, Inf; fscale)
    residual_momentum!(c, lvl.fluid, lvl.grid, op; dr = true, fscale)
    for d in 1:D
        c.Rv[d] .*= c.Dv[d]
    end
    return nothing
end

velocity_norm(lvl) = maximum(velocity_norms(lvl.cache))

"""
    velocity_vcycle!(hier, l, par, op; mfs_mean0) -> nothing

One velocity V-cycle from level `l` at fixed pressure: `hier.npre`/`hier.npost`
smoothing steps of exactly `hier.inner` DR iterations; the coarsest level runs
adaptive DR until its velocity residual drops by `hier.coarse_rtol` (at most
`hier.ncoarse` sub-solves).
"""
function velocity_vcycle!(hier::MGHierarchy{T}, l::Int, par::Parameters, op::StokesOp{D};
                          mfs_mean0::Real) where {T, D}
    lvl = hier.levels[l];  fine = l == 1
    fscale   = fine ? one(T) : zero(T)
    mfs_mean = fine ? T(mfs_mean0) : zero(T)
    Ddom     = T(lvl.grid.li[1])
    P0       = fine ? lvl.fluid.P : lvl.cache.P0
    # the pseudo-transient term is off: P0 = P on the fine level, 0 = 0 on coarse ones
    sm(n, inner, adaptive) = smooth!(lvl, par, op; nsweeps = n, ninner = inner, fscale, P0,
                                     mfs_mean, Ddom, adaptive, update_P = false)
    res!() = velocity_residual!(lvl, op; fscale)
    if l == length(hier.levels)
        @timeit_debug TO "L$l coarse solve" begin
        res!();  r0 = velocity_norm(lvl) + eps(T)
        for _ in 1:hier.ncoarse
            sm(1, 20 * sum(cells(lvl.grid)), true)
            res!()
            velocity_norm(lvl) / r0 < hier.coarse_rtol && break
        end
        end
        return nothing
    end
    @timeit_debug TO "L$l pre-smooth" sm(hier.npre, hier.inner, false)
    @timeit_debug TO "L$l residual" res!()
    cl = hier.levels[l + 1]
    @timeit_debug TO "L$l restrict" begin
    foreach_axis(Val(D)) do comp
        d = unval(comp)
        restrict_face!(cl.cache.bext[d], lvl.cache.Rv[d], comp)
    end
    fill!(cl.cache.bext_P, 0)
    zero_solution!(cl)
    end
    velocity_vcycle!(hier, l + 1, par, op; mfs_mean0)
    @timeit_debug TO "L$l prolong + correct" begin
    v = velocity(lvl.fluid);  vc = velocity(cl.fluid)
    foreach_axis(Val(D)) do comp
        d = unval(comp);  e = lvl.cache.Rv[d]
        prolong_face!(e, face_interior(vc[d], d), comp, op)
        vd = face_interior(v[d], d)
        @. vd += e
    end
    apply_velocity_bcs!(lvl.fluid, lvl.grid, op; mfs_mean, Ddom)
    end
    @timeit_debug TO "L$l post-smooth" sm(hier.npost, hier.inner, false)
    return nothing
end

# Cahouet–Chabard inertial Schur complement. With the momentum block reduced to
# its inertia, v_d = −dt/(cin_d·ρ_d)·∂_d P and the face density cancels in
# ∇·(ρ_d v_d): the inertial Schur complement is dt·A with A = −∇·(cin⁻¹∇·), a
# constant-coefficient (2D+1)-point operator with Neumann rows at closed
# boundaries and periodic links elsewhere. A⁻¹ is approximated by one V-cycle on
# the hierarchy's grids (weighted Jacobi, cell transfers); it sets only the step
# of the multiplier update, not the fixed point R_P = 0.

@kernel function cc_residual_kernel!(r, @Const(p), @Const(f), w, periodic, invh2)
    I = @index(Global, Cartesian)
    D = length(I);  pc = p[I]
    acc = zero(pc)
    for k in 1:D
        n = size(p, k);  ek = unit_index(I, k)
        if I[k] > 1
            acc += w[k] * (pc - p[I - ek])
        elseif periodic[k]
            acc += w[k] * (pc - p[replace_index(I, k, n)])
        end
        if I[k] < n
            acc += w[k] * (pc - p[I + ek])
        elseif periodic[k]
            acc += w[k] * (pc - p[replace_index(I, k, 1)])
        end
    end
    r[I] = f[I] - acc * invh2
end

@kernel function cc_jacobi_kernel!(p, @Const(r), w, periodic, invh2, ω)
    I = @index(Global, Cartesian)
    D = length(I)
    diag = tuple_foldl(zero(eltype(p)), ntuple(identity, Val(D))) do s, k
        n = size(p, k)
        s + (periodic[k] ? 2 : (I[k] > 1) + (I[k] < n)) * w[k]
    end
    p[I] += ω * r[I] / (diag * invh2)
end

function cc_sweeps!(w, n, wk, periodic)
    backend = KernelAbstractions.get_backend(w.p)
    invh2 = inv(w.h^2)
    for _ in 1:n
        cc_residual_kernel!(backend)(w.r, w.p, w.f, wk, periodic, invh2; ndrange = size(w.p))
        cc_jacobi_kernel!(backend)(w.p, w.r, wk, periodic, invh2, oftype(w.h, 0.8); ndrange = size(w.p))
    end
    return nothing
end

function cc_vcycle!(ws, l, wk, periodic; ν = 3)
    w = ws[l]
    if l == length(ws)
        cc_sweeps!(w, 20 * maximum(size(w.p)), wk, periodic)   # coarsest: Jacobi to near-convergence
        return nothing
    end
    cc_sweeps!(w, ν, wk, periodic)
    backend = KernelAbstractions.get_backend(w.p)
    cc_residual_kernel!(backend)(w.r, w.p, w.f, wk, periodic, inv(w.h^2); ndrange = size(w.p))
    restrict_cell!(ws[l + 1].f, w.r)
    fill!(ws[l + 1].p, 0)
    cc_vcycle!(ws, l + 1, wk, periodic; ν)
    prolong_cell!(w.r, ws[l + 1].p, periodic)
    w.p .+= w.r
    cc_sweeps!(w, ν, wk, periodic)
    return nothing
end

# P += (ω/dt)·A⁻¹(ρ·R_P), A⁻¹ ≈ one Poisson V-cycle (R_P is 1/ρ-scaled, hence
# the ρ). A has a constant null mode, so the right-hand side and the result are
# made mean-free.
function cc_update!(ws, fluid, c::StokesCache{D, T}, ω, op::StokesOp{D}) where {D, T}
    wk = map(inv, op.cin)
    f = ws[1].f
    @. f = fluid.rho * c.R_P
    remove_mean!(f)
    fill!(ws[1].p, 0)
    cc_vcycle!(ws, 1, wk, op.periodic)
    p = ws[1].p
    remove_mean!(p)
    Pint = interior(fluid.P)
    @. Pint += T(ω / op.dt) * p
    return nothing
end

"""
    fluidmech_vmg!(fluid, phase, hier, grid, par, scales; dt, P0 = nothing,
                   a1, a2, a3, b1, b2, b3, gamma = par.gamma, γfact = 5, nv = 1,
                   cc = false, rtol_PH = 0, bnchm = false, xBC, yBC, zBC, sds,
                   top_cnv, bot_cnv, verbose = false)
        -> (; converged, relaxed, PH_iters, errs)

Powell–Hestenes iteration with the inner velocity solve replaced by `nv`
velocity V-cycles at fixed pressure, then the multiplier update
`P += ω·γ_eff·R_P`; `cc = true` adds the Cahouet–Chabard term
`(ω/dt)·A⁻¹(ρ·R_P)` (one matrix-free Poisson V-cycle), which keeps the pressure
iteration grid-independent when inertia dominates without changing the fixed
point. The velocity block carries `γ_eff = γfact/s_P` on every level; point-DR
smoothing of it degrades as `γfact` grows, so the default is small. Stops as
`converged` once the penalty-free full residual and the applied pressure change
(`pressure_change_norm`) are both below `par.atol_PH`, or as `relaxed` once the
residual alone drops by `rtol_PH` below its value after the first update;
throws otherwise after `par.maxit_PH` steps. `errs[k]` is the larger of the
residual and the pressure change after step `k`.
"""
function fluidmech_vmg!(fluid, phase, hier::MGHierarchy{T}, grid, par::Parameters,
                        scales::Scales;
                        dt::Real, P0 = nothing,
                        a1::Real = 1, a2::Real = 1, a3::Real = 0,
                        b1::Real = 1, b2::Real = 0, b3::Real = 0,
                        gamma::Real = par.gamma, γfact::Real = 5, nv::Int = 1,
                        cc::Bool = false, rtol_PH::Real = 0, bnchm::Bool = false,
                        xBC::Symbol = :periodic, yBC::Symbol = :periodic,
                        zBC::Symbol = :closed, sds::Integer = -1,
                        top_cnv::Integer = 1, bot_cnv::Integer = 1,
                        verbose::Bool = false) where {T}
    isfinite(dt) || throw(ArgumentError(
        "fluidmech_vmg!: dt must be finite (got $dt); the drunken-sailor term ∂ρ·g0·dt scales with dt"))
    op = StokesOp(T, dims_val(grid), par; a1, a2, a3, gamma, dt, xBC, yBC, zBC, sds, top_cnv, bot_cnv)
    c = hier.levels[1].cache
    @timeit_debug TO "setup" begin
    P0arr = P0 === nothing ? copyto!(c.P0, fluid.P) : P0
    bnchm || update_mass_source!(fluid, phase, par, dt; a1, a2, a3, b1, b2, b3)
    momentum_advection!(c, fluid, grid, par, op)
    mfs_mean = bnchm ? zero(T) : T(sum(fluid.MFS) / length(fluid.MFS))
    refresh_coefficients!(hier, par, op; γfact)
    apply_velocity_bcs!(fluid, grid, op; mfs_mean, Ddom = T(grid.li[1]))
    apply_pressure_bcs!(fluid, op)
    end
    err0 = one(T);  converged = false;  relaxed = false;  PH_iters = 0
    errs = T[]
    for it in 1:par.maxit_PH
        PH_iters = it
        @timeit_debug TO "velocity V-cycle" for _ in 1:nv
            velocity_vcycle!(hier, 1, par, op; mfs_mean0 = mfs_mean)
        end
        @timeit_debug TO "PH multiplier update" begin
        residual_continuity!(c, fluid, grid, P0arr, Inf)
        Pint = interior(fluid.P)
        c.dP .= Pint
        @. Pint += par.pres_relax_PT * c.γ_eff * c.R_P
        end
        cc && a1 > 0 &&
            @timeit_debug TO "Cahouet–Chabard Poisson V-cycle" cc_update!(hier.cc, fluid, c, par.pres_relax_PT, op)
        # full (penalty-free) residual: the penalty vanishes with R_P at the solution
        @timeit_debug TO "gauge + convergence residual" begin
        bnchm || pressure_gauge!(fluid, P0arr)
        apply_pressure_bcs!(fluid, op)
        residual_continuity!(c, fluid, grid, P0arr, Inf)
        residual_momentum!(c, fluid, grid, op; dr = false)
        nvn, nP = residual_norms(c, fluid.rho)
        c.dP .= Pint .- c.dP
        ndP = pressure_change_norm(c.dP, fluid, grid.h)
        end
        # The residuals bound the pressure error only up to the conditioning of
        # the Schur complement; the applied pressure change bounds it directly.
        res = max(nvn..., nP)
        err = max(res, ndP)
        isfinite(err) || error("fluidmech_vmg! produced a non-finite residual at PH step $it")
        push!(errs, err)
        it == 1 && (err0 = res + eps(T))
        verbose && @printf("  [vMG PH %02d] err=%.3e  (v=%s P=%.2e dP=%.2e)\n", it, err,
                           join((@sprintf("%.2e", x) for x in nvn), "/"), nP, ndP)
        if err < par.atol_PH
            converged = true;  break
        end
        if it > 1 && res < rtol_PH * err0
            relaxed = true;  break
        end
    end
    converged || relaxed ||
        error("fluidmech_vmg! failed to meet atol_PH=$(par.atol_PH) after $PH_iters PH steps")
    return (; converged, relaxed, PH_iters, errs)
end
