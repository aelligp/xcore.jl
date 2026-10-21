using Statistics
using LinearAlgebra: norm
using xcore: Parameters, Grid, compute_scales, FluidState, PhaseState,
             DyrelCache, fluidmech_dyrel!, fluidmech_gmg!, fluidmech_vmg!,
             build_hierarchy, fluidmech!
import xcore as XC
using xcore.Stokes: StokesOp, residual_continuity!, residual_momentum!, residual_norms,
                    apply_velocity_bcs!, penalty!, level_residual!, check_preconditioner
using KernelAbstractions: CPU

# momentum and continuity residual norms as one flat tuple
_flat_norms(cache, rho) = ((nv, nP) = residual_norms(cache, rho); (nv..., nP))

# Phase-1 DYREL validation: steady incompressible variable-viscosity Stokes.
# We don't have MMS source terms wired into the DR residual yet, so these
# tests check the two solver-agnostic invariants:
#   1. the PH/DR iteration converges (residual drops below atol_PH),
#   2. the converged velocity field is divergence-free (incompressible),
#   3. no NaNs / blow-ups,
# on a buoyancy-driven blob with both constant and contrasted viscosity.

function _dyrel_setup(T = Float64; Npts = 48, ηcontrast = 1.0, γfact = 20.0,
                      maxit_PH = 50)
    par = Parameters(T; D = 1.0, N = Npts, L = 1.0, etam0 = 1.0, g0 = 1.0,
                     γfact_PT = γfact, CFL_PT = 0.99, c_fact_PT = 0.5,
                     maxit_PH, maxit_PT = 50_000, n_tune_PT = 100,
                     atol_PH = 1.0e-6, rel_drop_PT = 1.0e-2)
    grid   = Grid(par)
    scales = compute_scales(par, grid)
    Nz, Nx = grid.Nz, grid.Nx
    fluid  = FluidState(T, CPU(), Nz, Nx)
    phase  = PhaseState(T, CPU(), Nz, Nx)
    dyrel  = DyrelCache(T, CPU(), grid)

    fill!(fluid.eta, one(T))
    fill!(fluid.etaco, one(T))
    fill!(fluid.MFS, zero(T))
    # constant unit density: the compressible continuity residual ∇·(ρv)
    # reduces to ∇·v, recovering the incompressible-Stokes check
    fill!(fluid.rho, one(T))
    fill!(fluid.rhow, one(T))
    fill!(fluid.rhou, one(T))

    # optional viscosity contrast: a stiff central block
    if ηcontrast != 1.0
        for j in 1:Nz, i in 1:Nx
            zc = (j - 0.5) * grid.h / grid.D
            xc = (i - 0.5) * grid.h / grid.L
            if abs(xc - 0.5) < 0.15 && abs(zc - 0.5) < 0.15
                fluid.eta[j, i] = T(ηcontrast)
            end
        end
        # corner viscosity = geometric-mean-ish; simple injection is fine for the test
        for j in 1:Nz+1, i in 1:Nx+1
            zc = (j - 1) * grid.h / grid.D
            xc = (i - 1) * grid.h / grid.L
            if abs(xc - 0.5) < 0.15 && abs(zc - 0.5) < 0.15
                fluid.etaco[j, i] = T(ηcontrast)
            end
        end
    end

    # buoyancy blob on z-faces (Nz+1, Nx)
    for j in 1:Nz+1, i in 1:Nx
        zc = (j - 1) * grid.h / grid.D
        xc = (i - 0.5) * grid.h / grid.L
        fluid.Drho[j, i] = exp(-(((xc - 0.5) / 0.1)^2 + ((zc - 0.5) / 0.1)^2))
    end

    return fluid, phase, dyrel, grid, par, scales
end

function _operator_setup(case::Symbol)
    T = Float64
    par = Parameters(T; D = 1.0, N = 32, L = 1.0, g0 = 1.0, gamma = 1.0e4,
                     γfact_PT = 50.0, maxit_PH = 100, maxit_PT = 50_000,
                     total_iterMax_PT = 50_000, n_tune_PT = 100,
                     atol_PH = 1.0e-8, alpha = 0.0)
    grid = Grid(par)
    fluid = FluidState(T, CPU(), grid.Nz, grid.Nx)
    phase = PhaseState(T, CPU(), grid.Nz, grid.Nx)
    fluid.eta .= case === :solcx ?
        [grid.Xc[i] < 0.5 ? 1.0 : 100.0 for j in 1:grid.Nz, i in 1:grid.Nx] :
        [exp(log(100.0) * (1 - grid.Zc[j])) for j in 1:grid.Nz, i in 1:grid.Nx]
    XC.compute_corner_eta!(fluid.etaco, fluid.eta; xBC = :closed, zBC = :closed)
    fluid.rho .= [1 + 0.1 * sin(2π * grid.Xc[i]) * cos(π * grid.Zc[j])
                  for j in 1:grid.Nz, i in 1:grid.Nx]
    XC.update_interpolations!(phase, fluid, grid, par; xBC = :closed)
    return fluid, phase, grid, par, compute_scales(par, grid)
end

_relative_pressure(a, b) = norm((a .- mean(a)) .- (b .- mean(b))) / (norm(b .- mean(b)) + eps(eltype(b)))

function _periodic_ns_trajectory(solver::Symbol; nsteps::Int = 3, N::Int = 32, forcing_PH::Real = 0, kw...)
    T = Float64
    # This is the periodic production configuration at a deliberately small
    # resolution.  It exercises transient inertia, BD2 after step one,
    # momentum advection, evolving density, MFS, and the outer Picard split.
    par = Parameters(T; runID = "test_periodic_ns", save_op = false,
                     D = 10.0, N, L = 15.0, t0end = 2.0,
                     d0 = 0.01, etam0 = 10.0, L0 = 0.1, l0 = 0.1,
                     Da = 0.01, Xi = 0.5, solver, atol = 1.0e-6, forcing_PH, kw...)
    grid = Grid(par)
    scales = compute_scales(par, grid)
    fluid = FluidState(T, CPU(), grid.Nz, grid.Nx)
    phase = PhaseState(T, CPU(), grid.Nz, grid.Nx)
    ns = XC.NoiseState(T, grid, scales)
    hst = XC.History(T)
    XC.initialize!(phase, fluid, ns, hst, grid, par, scales)
    time, dt = XC.run!(phase, fluid, ns, hst, grid, par, scales;
                       nsteps, verbose = false)
    return (; fluid, phase, time, dt, par)
end

@testset "DYREL constant-viscosity buoyancy Stokes" begin
    fluid, phase, dyrel, grid, par, scales = _dyrel_setup(Float64; Npts = 48)
    res = fluidmech_dyrel!(fluid, phase, dyrel, grid, par, scales;
                           dt = 1.0e30, top_cnv = -1, bot_cnv = -1, verbose = false)

    @test res.converged
    @test res.PH_iters ≤ par.maxit_PH

    # divergence-free check: recompute continuity residual (= -∇·v here)
    residual_continuity!(dyrel, fluid, grid, fluid.P, Inf)
    rms_div = sqrt(mean(abs2, dyrel.R_P))
    @test rms_div < 1.0e-4

    # sanity: finite, non-trivial flow
    @test all(isfinite, fluid.W)
    @test all(isfinite, fluid.U)
    @test all(isfinite, fluid.P)
    @test maximum(abs, fluid.W) > 0
end

@testset "DYREL viscosity-contrast buoyancy Stokes" begin
    fluid, phase, dyrel, grid, par, scales = _dyrel_setup(Float64; Npts = 48, ηcontrast = 1.0e3)
    res = fluidmech_dyrel!(fluid, phase, dyrel, grid, par, scales;
                           dt = 1.0e30, top_cnv = -1, bot_cnv = -1, verbose = false)

    @test res.converged
    residual_continuity!(dyrel, fluid, grid, fluid.P, Inf)
    @test sqrt(mean(abs2, dyrel.R_P)) < 1.0e-3
    @test all(isfinite, fluid.W) && all(isfinite, fluid.U)
end

@testset "DYREL convergence is invariant under a pressure offset" begin
    # Manufactured periodic flow with uniform pressure: the discrete streamfunction
    # makes (W, U) exactly divergence-free, and lagged momentum forcing built from
    # its residual makes (W, U, P = c) the exact solution for any constant c.
    T = Float64; dt = 10.0
    function solve_offset(c)
        par = Parameters(T; D = 1.0, N = 32, L = 1.0, etam0 = 1.0, g0 = 1.0,
                         alpha = 0.0, gamma = 0.0, atol_PH = 1.0e-7)
        grid = Grid(par); scales = compute_scales(par, grid)
        Nz, Nx, h = grid.Nz, grid.Nx, grid.h
        fluid = FluidState(T, CPU(), Nz, Nx); phase = PhaseState(T, CPU(), Nz, Nx)
        dyrel = DyrelCache(T, CPU(), grid)
        for A in (fluid.eta, fluid.etaco, fluid.rho, fluid.rhow, fluid.rhou)
            fill!(A, 1)
        end
        fill!(fluid.Drho, 0); fill!(fluid.MFS, 0); fill!(fluid.P, c)
        ψ = [1.0e-4 * sinpi((j - 1) * h) * sinpi(2 * (i - 1) * h) for j in 1:Nz+1, i in 1:Nx+1]
        fluid.W[:, 2:end-1] .= diff(ψ; dims = 2) ./ h
        fluid.U[2:end-1, :] .= .-diff(ψ; dims = 1) ./ h
        kw = (; a1 = 1, a2 = 1, a3 = 0, gamma = 0, dt)
        op = StokesOp(T, Val(2), par; kw..., top_cnv = -1, bot_cnv = -1)
        apply_velocity_bcs!(fluid, grid, op)
        residual_momentum!(dyrel, fluid, grid, op; dr = false)
        fluid.rhoWo .= .-dyrel.Rv[1] .* dt
        fluid.rhoUo .= .-dyrel.Rv[2] .* dt
        Wex, Uex = copy(fluid.W), copy(fluid.U)
        fill!(fluid.W, 0); fill!(fluid.U, 0)
        res = fluidmech_dyrel!(fluid, phase, dyrel, grid, par, scales; top_cnv = -1,
                               bot_cnv = -1, kw...)
        residual_momentum!(dyrel, fluid, grid, op; dr = false)
        residual_continuity!(dyrel, fluid, grid, fluid.P, Inf)
        return (; res, norms = _flat_norms(dyrel, fluid.rho), par,
                verr = max(maximum(abs, fluid.W .- Wex), maximum(abs, fluid.U .- Uex)),
                perr = maximum(abs, fluid.P[2:end-1, 2:end-1] .- c), Wamp = maximum(abs, Wex))
    end
    s0 = solve_offset(0.0)
    @test s0.res.converged
    @test all(<(s0.par.atol_PH), s0.norms)
    @test s0.verr < 1.0e-6 * s0.Wamp
    for c in (1.0, -1.0e3)
        sc = solve_offset(c)
        @test sc.res.converged
        @test sc.res.total_iter == s0.res.total_iter
        @test all(<(s0.par.atol_PH), sc.norms)
        @test sc.verr < 1.0e-6 * sc.Wamp
        @test isapprox(sc.perr, s0.perr; atol = 1.0e-9)
    end
end

@testset "multigrid pressure meets atol_PH independent of resolution" begin
    # Inertia-dominated solve: the residuals bound the pressure error only up to
    # the conditioning of the inertial Schur complement, which grows with N, so
    # the gate must also bound the applied pressure change.
    rel(a, b) = (a = a[2:end-1, 2:end-1] .- mean(a[2:end-1, 2:end-1]);
                 b = b[2:end-1, 2:end-1] .- mean(b[2:end-1, 2:end-1]); norm(a .- b) / norm(b))
    for N in (64, 128)
        P = map((:direct, :vmg, :gmg)) do solver
            fluid, phase, dyrel, grid, par, scales = _dyrel_setup(Float64; Npts = N, maxit_PH = 300)
            fluid.rhoo .= fluid.rho;  fluid.rhooo .= fluid.rho     # zero BD2 mass source
            kw = (; dt = 1.0e-2, a1 = 1, a2 = 1, a3 = 0, top_cnv = -1, bot_cnv = -1)
            if solver === :direct
                fluidmech!(fluid, grid, par; phase, sds = -1, xBC = :periodic, zBC = :closed, kw...)
            else
                hier = build_hierarchy(fluid, grid, par)
                r = solver === :gmg ? fluidmech_gmg!(fluid, phase, hier, grid, par, scales; kw...) :
                    fluidmech_vmg!(fluid, phase, hier, grid, par, scales; γfact = 5, cc = true, kw...)
                @test r.converged
            end
            (fluid.P, par.atol_PH)
        end
        (Pd, atol), (Pv, _), (Pg, _) = P
        @test rel(Pv, Pd) < 2 * atol
        @test rel(Pg, Pd) < 2 * atol
    end
end

@testset "solver entry points infer concrete return types" begin
    function inferred_solves(fluid, phase, dyrel, grid, par, scales)
        kw = (; dt = 1.0, a1 = 1.0, a2 = 0, a3 = 0, top_cnv = -1, bot_cnv = -1)
        hier = @inferred build_hierarchy(fluid, grid, par)
        @inferred fluidmech_dyrel!(fluid, phase, dyrel, grid, par, scales; kw...)
        @inferred fluidmech_gmg!(fluid, phase, hier, grid, par, scales; kw...)
        @inferred fluidmech_vmg!(fluid, phase, hier, grid, par, scales; γfact = 5, cc = true, kw...)
        return true
    end
    @test inferred_solves(_dyrel_setup(Float64; Npts = 32, maxit_PH = 200)...)
end

@testset "Gershgorin bound covers the penalized operator under density contrast" begin
    # λmax·D must be at least the absolute row sum of the penalized velocity
    # operator, assembled column by column from the dr = true residual. The
    # penalty couples faces through the 1/ρ-scaled mass flux, so a ±30 % density
    # field exposes any entry that drops the face/cell density ratio.
    T = Float64
    for (xBC, sds) in ((:periodic, -1), (:closed, 1))
        par = Parameters(T; D = 1.0, N = 8, L = 1.0, g0 = 1.0, gamma = 0.3, alpha = 0.0)
        grid = Grid(par); Nz, Nx, h = grid.Nz, grid.Nx, grid.h
        fluid = FluidState(T, CPU(), Nz, Nx); phase = PhaseState(T, CPU(), Nz, Nx)
        fluid.eta .= [1 + 0.5 * sinpi(2 * i / Nx) * cospi(j / Nz) for j in 1:Nz, i in 1:Nx]
        XC.compute_corner_eta!(fluid.etaco, fluid.eta; xBC, zBC = :closed)
        fluid.rho .= [1 + 0.3 * sinpi(2 * (i + 0.3) / Nx) * sinpi(3 * (j + 0.2) / Nz) for j in 1:Nz, i in 1:Nx]
        XC.update_interpolations!(phase, fluid, grid, par; xBC)
        dyrel = DyrelCache(T, CPU(), grid)
        op = StokesOp(T, Val(2), par; a1 = 1, a2 = 0, a3 = 0, gamma = par.gamma, dt = 1.0e3,
                      xBC, sds, top_cnv = -1, bot_cnv = -1)
        penalty!(dyrel, fluid, grid, op; γfact = 1.0e3)
        function applied()
            apply_velocity_bcs!(fluid, grid, op)
            residual_continuity!(dyrel, fluid, grid, fluid.P, Inf)
            residual_momentum!(dyrel, fluid, grid, op; dr = true)
            return vcat(vec(dyrel.Rv[1] .* dyrel.Dv[1]), vec(dyrel.Rv[2] .* dyrel.Dv[2]))
        end
        ucols = xBC === :periodic ? (1:Nx) : (2:Nx)
        cols = vcat([CartesianIndex(iz, ix + 1) => fluid.W for iz in 2:Nz for ix in 1:Nx],
                    [CartesianIndex(iz + 1, ix) => fluid.U for iz in 1:Nz for ix in ucols])
        fill!(fluid.W, 0); fill!(fluid.U, 0)
        r0 = applied()
        rowsum = zero(r0)
        for (I, A) in cols
            A[I] = 1;  rowsum .+= abs.(applied() .- r0);  A[I] = 0
        end
        bound = vcat(vec(dyrel.λmax[1] .* dyrel.Dv[1]), vec(dyrel.λmax[2] .* dyrel.Dv[2]))
        LW, LU = LinearIndices(dyrel.Rv[1]), LinearIndices(dyrel.Rv[2])
        rows = vcat([LW[iz, ix] for iz in 2:Nz for ix in 1:Nx],
                    [length(LW) + LU[iz, ix] for iz in 1:Nz for ix in ucols])
        @test maximum(rowsum[rows] ./ bound[rows]) ≤ 1 + 1.0e-12
    end
end

@testset "matrix-free solvers across a sharp density contrast" begin
    # A heavy circular inclusion with a sharp interface, buoyant through Drho.
    # The penalty couples faces through the 1/ρ-scaled mass flux, so the DR step
    # is stable only if λmax accounts for the density jump at the interface.
    T = Float64
    function inclusion(ratio)
        fluid, phase, dyrel, grid, par, scales = _dyrel_setup(T; Npts = 64, maxit_PH = 100)
        Nz, Nx = grid.Nz, grid.Nx
        fluid.rho .= [hypot((i - 0.5) / Nx - 0.5, (j - 0.5) / Nz - 0.5) < 0.2 ? ratio : 1.0
                      for j in 1:Nz, i in 1:Nx]
        XC.update_interpolations!(phase, fluid, grid, par; xBC = :periodic)
        # a2 = 1 with ρ_old = ρ keeps the BD2 mass source zero
        fluid.rhoo .= fluid.rho;  fluid.rhooo .= fluid.rho
        return fluid, phase, dyrel, grid, par, scales
    end
    kw(dt) = (; dt, a1 = 1, a2 = 1, a3 = 0, top_cnv = -1, bot_cnv = -1)
    rel(a, b) = norm(a .- b) / norm(b)
    function solve(solver, ratio, dt)
        fluid, phase, dyrel, grid, par, scales = inclusion(ratio)
        if solver === :direct
            fluidmech!(fluid, grid, par; phase, sds = -1, xBC = :periodic, zBC = :closed, kw(dt)...)
            return (; fluid, converged = true)
        elseif solver === :dyrel
            r = fluidmech_dyrel!(fluid, phase, dyrel, grid, par, scales; kw(dt)...)
            return (; fluid, r.converged)
        end
        hier = build_hierarchy(fluid, grid, par)
        r = solver === :gmg ?
            fluidmech_gmg!(fluid, phase, hier, grid, par, scales; kw(dt)...) :
            fluidmech_vmg!(fluid, phase, hier, grid, par, scales; γfact = 5, cc = true, kw(dt)...)
        return (; fluid, r.converged)
    end
    for (ratio, dt, solvers) in ((10.0, 1.0, (:dyrel, :gmg, :vmg)), (100.0, 1.0e-3, (:dyrel, :vmg)))
        ref = solve(:direct, ratio, dt).fluid
        for s in solvers
            out = solve(s, ratio, dt)
            @test out.converged
            @test rel(out.fluid.W[:, 2:end-1], ref.W[:, 2:end-1]) < 1.0e-5
            @test rel(out.fluid.U[2:end-1, :], ref.U[2:end-1, :]) < 1.0e-5
        end
    end
end

@testset "DYREL matches the direct nonperiodic operator" begin
    # The direct assembly is the reference contract.  `gamma` is deliberately
    # large, so using the periodic-only horizontal inertia at a side wall is
    # detectable; finite dt and nonuniform rho exercise the remaining terms.
    kw = (; dt = 100.0, a1 = 1.0, a2 = 0.0, a3 = 0.0,
            sds = -1, top_cnv = -1, bot_cnv = -1, xBC = :closed, zBC = :closed)
    for case in (:solcx, :solkz)
        direct, direct_phase, grid, par, _ = _operator_setup(case)
        fluidmech!(direct, grid, par; phase = direct_phase, kw...)

        dyrel, dyrel_phase, grid, par, scales = _operator_setup(case)
        info = fluidmech_dyrel!(dyrel, dyrel_phase, DyrelCache(Float64, CPU(), grid),
                                grid, par, scales;
                                require_convergence = true, kw...)
        @test info.converged
        @test norm(dyrel.W .- direct.W) / norm(direct.W) < 1.0e-5
        @test norm(dyrel.U .- direct.U) / norm(direct.U) < 1.0e-5
        @test _relative_pressure(dyrel.P[2:end-1, 2:end-1],
                                direct.P[2:end-1, 2:end-1]) < 1.0e-5
    end
end

@testset "DYREL periodic transient Navier-Stokes trajectory" begin
    # The production Picard workflow (fixed-P0 inexact fluid updates), not a
    # single frozen-operator solve.
    rel(a, b) = norm(a .- b) / (norm(b) + eps(eltype(b)))
    direct = _periodic_ns_trajectory(:direct)

    # Every fluid solve run to atol_PH: DYREL then follows the direct Picard
    # path sweep for sweep, so any operator mismatch shows up here.
    dyrel = _periodic_ns_trajectory(:dyrel; forcing_PH = 0)
    @test isapprox(dyrel.time, direct.time; rtol = 1.0e-8)
    @test isapprox(dyrel.dt, direct.dt; rtol = 1.0e-7)
    @test rel(dyrel.fluid.W, direct.fluid.W) < 1.0e-5
    @test rel(dyrel.fluid.U, direct.fluid.U) < 1.0e-5
    @test _relative_pressure(dyrel.fluid.P[2:end-1, 2:end-1],
                            direct.fluid.P[2:end-1, 2:end-1]) < 1.0e-5
    # both pin the pressure level in the same cell (fluidmech.m:298-308)
    @test rel(dyrel.fluid.P, direct.fluid.P) < 1.0e-5
    @test rel(dyrel.fluid.rho, direct.fluid.rho) < 1.0e-8
    @test rel(dyrel.phase.x, direct.phase.x) < 1.0e-8
    @test rel(dyrel.fluid.eta, direct.fluid.eta) < 1.0e-8

    # Anderson-accelerated multiplier updates reach the same solves; the
    # atol_PH gate leaves them a larger pressure error, which dt and the ramp
    # factors (both taken from the current iterate) carry into later sweeps.
    aa = _periodic_ns_trajectory(:dyrel; forcing_PH = 0, anderson_PH = 5)
    @test isapprox(aa.time, direct.time; rtol = 1.0e-8)
    @test isapprox(aa.dt, direct.dt; rtol = 1.0e-7)
    @test rel(aa.fluid.W, direct.fluid.W) < 1.0e-5
    @test rel(aa.fluid.U, direct.fluid.U) < 1.0e-5
    @test _relative_pressure(aa.fluid.P[2:end-1, 2:end-1],
                            direct.fluid.P[2:end-1, 2:end-1]) < 1.0e-4
    @test rel(aa.fluid.rho, direct.fluid.rho) < 1.0e-8

    # Inexact-solve forcing (the production default) takes a different Picard
    # path and so agrees with the direct result to the Picard tolerance.
    forced = _periodic_ns_trajectory(:dyrel; forcing_PH = 0.1)
    rtol = forced.par.rtol
    @test isapprox(forced.time, direct.time; rtol = 1.0e-8)
    @test isapprox(forced.dt, direct.dt; rtol)
    @test rel(forced.fluid.W, direct.fluid.W) < rtol
    @test rel(forced.fluid.U, direct.fluid.U) < rtol
    # pressure is the least converged field of a Picard sweep
    @test _relative_pressure(forced.fluid.P[2:end-1, 2:end-1],
                            direct.fluid.P[2:end-1, 2:end-1]) < 10rtol
    @test rel(forced.fluid.rho, direct.fluid.rho) < rtol
    @test rel(forced.phase.x, direct.phase.x) < rtol
    @test rel(forced.fluid.eta, direct.fluid.eta) < rtol
end

@testset "GMG periodic transient Navier-Stokes trajectory" begin
    # Production Picard workflow on :gmg with its default inexact-solve forcing.
    # N = 64 because the hierarchy needs both grid dimensions divisible by 32.
    rel(a, b) = norm(a .- b) / (norm(b) + eps(eltype(b)))
    direct = _periodic_ns_trajectory(:direct; N = 64)
    gmg = _periodic_ns_trajectory(:gmg; N = 64, forcing_PH = 0.1)
    @test isapprox(gmg.time, direct.time; rtol = 1.0e-8)
    @test isapprox(gmg.dt, direct.dt; rtol = 1.0e-7)
    @test rel(gmg.fluid.W, direct.fluid.W) < 1.0e-5
    @test rel(gmg.fluid.U, direct.fluid.U) < 1.0e-5
    @test _relative_pressure(gmg.fluid.P[2:end-1, 2:end-1],
                            direct.fluid.P[2:end-1, 2:end-1]) < gmg.par.rtol
    # the forced path reaches the fields that feed back into the next sweep
    # (dt, ramp factors) at the Picard tolerance, not the solve tolerance
    @test rel(gmg.fluid.rho, direct.fluid.rho) < 1.0e-8
    @test rel(gmg.phase.x, direct.phase.x) < 1.0e-7
    @test rel(gmg.fluid.eta, direct.fluid.eta) < 1.0e-6
end

@testset "VMG periodic transient Navier-Stokes trajectory" begin
    # Velocity V-cycles + Cahouet–Chabard inside the production Picard workflow.
    rel(a, b) = norm(a .- b) / (norm(b) + eps(eltype(b)))
    relp(a, b) = _relative_pressure(a.fluid.P[2:end-1, 2:end-1], b.fluid.P[2:end-1, 2:end-1])
    direct = _periodic_ns_trajectory(:direct; N = 64)

    # Every solve to atol_PH: VMG follows the direct Picard path, so velocity,
    # pressure and everything advected with them agree to the solve tolerance.
    exact = _periodic_ns_trajectory(:vmg; N = 64, forcing_PH = 0)
    atol = exact.par.atol_PH
    @test isapprox(exact.time, direct.time; rtol = 1.0e-8)
    @test isapprox(exact.dt, direct.dt; rtol = 1.0e-8)
    @test rel(exact.fluid.W, direct.fluid.W) < atol
    @test rel(exact.fluid.U, direct.fluid.U) < atol
    # the gate bounds the pressure change per update, not the pressure error
    @test relp(exact, direct) < 2atol
    @test rel(exact.fluid.rho, direct.fluid.rho) < 1.0e-8
    @test rel(exact.phase.x, direct.phase.x) < 1.0e-8
    @test rel(exact.fluid.eta, direct.fluid.eta) < 1.0e-8

    # Inexact-solve forcing (the production default) takes a different Picard
    # path and so agrees with the direct result to the Picard tolerance.
    forced = _periodic_ns_trajectory(:vmg; N = 64, forcing_PH = 0.1)
    rtol = forced.par.rtol
    @test isapprox(forced.time, direct.time; rtol)
    @test isapprox(forced.dt, direct.dt; rtol)
    @test rel(forced.fluid.W, direct.fluid.W) < rtol
    @test rel(forced.fluid.U, direct.fluid.U) < rtol
    @test relp(forced, direct) < rtol
    @test rel(forced.fluid.rho, direct.fluid.rho) < rtol
    @test rel(forced.phase.x, direct.phase.x) < rtol
    @test rel(forced.fluid.eta, direct.fluid.eta) < rtol
end

@testset "matrix-free solvers fail explicitly" begin
    fluid, phase, dyrel, grid, par, scales = _dyrel_setup(Float64; Npts = 32, maxit_PH = 1)
    @test_throws "fluidmech_dyrel! failed to meet atol_PH" fluidmech_dyrel!(
        fluid, phase, dyrel, grid, par, scales;
        dt = 1.0e6, top_cnv = -1, bot_cnv = -1,
        require_convergence = true)

    @test_throws DomainError check_preconditioner(DyrelCache(Float64, CPU(), grid))
end

@testset "direct cache rebuilds across side-boundary modes" begin
    fluid, phase, grid, par, _ = _operator_setup(:solkz)
    kw = (; dt = 100.0, a1 = 1.0, a2 = 0.0, a3 = 0.0,
            sds = -1, top_cnv = -1, bot_cnv = -1, zBC = :closed)
    fluidmech!(fluid, grid, par; phase, xBC = :closed, kw...)
    @test fluid.solver.sig_xBC === :closed
    fluidmech!(fluid, grid, par; phase, xBC = :periodic, kw...)
    @test fluid.solver.sig_xBC === :periodic
end

@testset "GMG gates on true mass balance and is stable at small dt" begin
    # Buoyant blob, periodic x, free slip. A converged solve must satisfy the
    # continuity equation without its artificial-compressibility term, and the
    # multiplier update must stay stable when ρ·dt < 1 (here ρ = 1).
    T = Float64
    function blob_gmg(dt; kw...)
        par = Parameters(T; D = 1.0, N = 32, L = 1.0, etam0 = 1.0, g0 = 1.0,
                         alpha = 0.0, gamma = 0.0, atol_PH = 1.0e-7, kw...)
        grid = Grid(par); scales = compute_scales(par, grid)
        fluid = FluidState(T, CPU(), grid.Nz, grid.Nx)
        phase = PhaseState(T, CPU(), grid.Nz, grid.Nx)
        for A in (fluid.eta, fluid.etaco, fluid.rho, fluid.rhow, fluid.rhou)
            fill!(A, 1)
        end
        fill!(fluid.MFS, 0)
        fluid.Drho .= [exp(-(((i - 0.5) * grid.h - 0.5)^2 + ((j - 1) * grid.h - 0.5)^2) / 0.01)
                       for j in 1:grid.Nz+1, i in 1:grid.Nx]
        hier = build_hierarchy(fluid, grid, par)
        kw = (; a1 = 1, a2 = 0, a3 = 0, gamma = 0, dt)
        res = fluidmech_gmg!(fluid, phase, hier, grid, par, scales;
                             top_cnv = -1, bot_cnv = -1, kw...)
        level_residual!(hier.levels[1], StokesOp(T, Val(2), par; kw..., top_cnv = -1, bot_cnv = -1);
                        P0 = fluid.P, fscale = 1)
        return (; res, nlev = length(hier.levels), par,
                norms = _flat_norms(hier.levels[1].cache, fluid.rho))
    end
    single = blob_gmg(1.0; mg_minlvl = 32, mg_ncoarse = 100)
    @test single.nlev == 1
    @test single.res.converged
    @test all(<(single.par.atol_PH), single.norms)
    for dt in (1.0, 0.1, 0.01)
        two = blob_gmg(dt)
        @test two.nlev == 2
        @test two.res.converged
        @test all(<(two.par.atol_PH), two.norms)
    end
end

@testset "GMG V-cycle contracts under viscosity contrast" begin
    # The coarse-grid correction has to survive a 10^6 span in η. This layout —
    # a stiff slab across the full width plus a soft column through it — puts
    # real energy in the long-wavelength error, which ONLY the coarse levels can
    # remove; a defective transfer still limps to the tolerance on a smooth
    # problem but shows up here as a contraction factor near 1. Npts = 64 meets
    # the GMG 32-multiple requirement, and no-slip (cnv = +1) exercises the
    # signed z-ghost reflection in the face prolongation.
    ηc = 1.0e3
    fluid, phase, _, grid, par, scales =
        _dyrel_setup(Float64; Npts = 64, γfact = 50.0)
    Nz, Nx = grid.Nz, grid.Nx
    for j in 1:Nz, i in 1:Nx
        zc = (j - 0.5) / Nz;  xc = (i - 0.5) / Nx
        fluid.eta[j, i] = abs(zc - 0.5) < 0.12 ? ηc :
                          (abs(xc - 0.5) < 0.08 ? inv(ηc) : 1.0)
    end
    XC.compute_corner_eta!(fluid.etaco, fluid.eta; xBC = :periodic, zBC = :closed)

    hier = build_hierarchy(fluid, grid, par)
    res = fluidmech_gmg!(fluid, phase, hier, grid, par, scales;
                         dt = 1.0e6, top_cnv = 1, bot_cnv = 1)

    @test res.converged
    @test all(isfinite, fluid.W) && all(isfinite, fluid.U) && all(isfinite, fluid.P)

    # Mean per-cycle contraction: ≈0.51 healthy, ≈0.77 with a mangled coarse
    # correction (measured by re-introducing the velocity mean-subtraction).
    ρ = (res.errs[end] / res.errs[1])^(1 / (length(res.errs) - 1))
    @info "GMG contraction under η contrast" ρ res.ncyc
    @test ρ < 0.65

    residual_continuity!(hier.levels[1].cache, fluid, grid, fluid.P, Inf)
    @test sqrt(mean(abs2, hier.levels[1].cache.R_P)) < 1.0e-3
end

@testset "matrix-free solvers match direct fluidmech! velocity field" begin
    # All three solvers attack the SAME linear system: fluidmech! does one exact
    # Newton step (LL⁻¹·RR), DYREL and the GMG V-cycle iterate to it. Started
    # from rest with frozen advection (=0 from rest), matched coefficients ⇒
    # same (W,U). N = 32 satisfies the GMG 32-multiple grid requirement.
    T = Float64;  Npts = 32
    par = Parameters(T; D = 1.0, N = Npts, L = 1.0, etam0 = 1.0, g0 = 1.0,
                     γfact_PT = 20.0, CFL_PT = 0.99, c_fact_PT = 0.5,
                     maxit_PH = 100, maxit_PT = 100_000, n_tune_PT = 100,
                     atol_PH = 1.0e-8, rel_drop_PT = 1.0e-3, gamma = 0.0)
    grid   = Grid(par);  scales = compute_scales(par, grid)
    Nz, Nx = grid.Nz, grid.Nx
    dt = 1.0e6   # large but finite (inertia ≈ 0); avoids 0·Inf in sailor

    function _setup!(fluid)
        fill!(fluid.eta, one(T));   fill!(fluid.etaco, one(T))
        fill!(fluid.rho, one(T));   fill!(fluid.rhow, one(T));  fill!(fluid.rhou, one(T))
        fill!(fluid.rhoo, one(T));  fill!(fluid.rhooo, one(T))
        fill!(fluid.MFS, zero(T))
        fill!(fluid.rhoWo, zero(T)); fill!(fluid.rhoWoo, zero(T))
        fill!(fluid.rhoUo, zero(T)); fill!(fluid.rhoUoo, zero(T))
        fill!(fluid.W, zero(T));    fill!(fluid.U, zero(T));    fill!(fluid.P, zero(T))
        fill!(fluid.drhodt, zero(T)); fill!(fluid.drhodto, zero(T)); fill!(fluid.drhodtoo, zero(T))
        for j in 1:Nz+1, i in 1:Nx
            zc = (j - 1) * grid.h;  xc = (i - 0.5) * grid.h
            fluid.Drho[j, i] = exp(-(((xc - 0.5) / 0.1)^2 + ((zc - 0.5) / 0.1)^2))
        end
        return fluid
    end

    # --- direct solver (one exact Newton step) ---
    fluid_d = FluidState(T, CPU(), Nz, Nx);  phase_d = PhaseState(T, CPU(), Nz, Nx)
    _setup!(fluid_d);  fill!(phase_d.advn_rho, zero(T))   # res_ρ = 0 ⇒ MFS unchanged
    fluidmech!(fluid_d, grid, par; phase = phase_d, dt = dt, a1 = 1, a2 = 1, a3 = 0,
               sds = -1, top_cnv = -1, bot_cnv = -1, xBC = :periodic, zBC = :closed)

    # --- DYREL (iterate to convergence) ---
    fluid_y = FluidState(T, CPU(), Nz, Nx);  phase_y = PhaseState(T, CPU(), Nz, Nx)
    dyrel   = DyrelCache(T, CPU(), grid)
    _setup!(fluid_y)
    fluidmech_dyrel!(fluid_y, phase_y, dyrel, grid, par, scales; dt = dt,
                     a1 = 1, a2 = 1, a3 = 0, top_cnv = -1, bot_cnv = -1, verbose = false)

    # --- GMG V-cycle (same iterate-to-convergence target) ---
    fluid_g = FluidState(T, CPU(), Nz, Nx);  phase_g = PhaseState(T, CPU(), Nz, Nx)
    _setup!(fluid_g)
    hier = build_hierarchy(fluid_g, grid, par)
    @test length(hier.levels) > 1                     # actually a hierarchy
    res_g = fluidmech_gmg!(fluid_g, phase_g, hier, grid, par, scales; dt = dt,
                           a1 = 1, a2 = 1, a3 = 0, top_cnv = -1, bot_cnv = -1)
    @test res_g.converged

    Wd = @view fluid_d.W[:, 2:end-1];  Ud = @view fluid_d.U[2:end-1, :]
    rel(a, b) = norm(a .- b) / (norm(b) + eps(T))
    relW = rel(@view(fluid_y.W[:, 2:end-1]), Wd)
    relU = rel(@view(fluid_y.U[2:end-1, :]), Ud)
    relW_g = rel(@view(fluid_g.W[:, 2:end-1]), Wd)
    relU_g = rel(@view(fluid_g.U[2:end-1, :]), Ud)
    @info "matrix-free vs direct" relW relU relW_g relU_g res_g.ncyc
    @test relW < 1.0e-3
    @test relU < 1.0e-3
    @test relW_g < 1.0e-3
    @test relU_g < 1.0e-3

    # The V-cycle exists to make the iteration count grid-independent: it must
    # reach the same tolerance in far fewer sweeps than the single-grid solver.
    @test res_g.ncyc < 20

    # Monotone contraction: each cycle must reduce the fine-level residual.
    @test all(<(1), res_g.errs[2:end] ./ res_g.errs[1:end-1])
end
