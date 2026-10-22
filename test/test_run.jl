using xcore: PhaseState, FluidState, NoiseState, Parameters, Grid, compute_scales,
             initialize!, run!, time_coefs, store_previous!, phsevo!
using KernelAbstractions: CPU

@testset "time_coefs scheme dispatch" begin
    for T in (Float64, Float32)
        # implicit: first 2 steps are BE, then BD2
        c1 = time_coefs(T, :bd2im, 1)
        c2 = time_coefs(T, :bd2im, 2)
        c3 = time_coefs(T, :bd2im, 3)
        @test c1 === (T(1), T(1), T(0), T(1), T(0), T(0))
        @test c2 === (T(1), T(1), T(0), T(1), T(0), T(0))
        @test c3 === (T(3/2), T(2), T(-1/2), T(1), T(0), T(0))

        # semi-implicit: BE then CN then BD2-SI
        s1 = time_coefs(T, :bd2si, 1)
        s2 = time_coefs(T, :bd2si, 2)
        s3 = time_coefs(T, :bd2si, 3)
        @test s1 === (T(1), T(1), T(0), T(1), T(0), T(0))
        @test s2 === (T(1), T(1), T(0), T(1/2), T(1/2), T(0))
        @test s3 === (T(3/2), T(2), T(-1/2), T(3/4), T(1/2), T(-1/4))

        @test_throws ArgumentError time_coefs(T, :bogus, 1)
    end
end

@testset "store_previous! shifts history buffers" begin
    T = Float64
    par = Parameters(T; D = 10.0, N = 8, L = 10.0, etam0 = 10.0, x0 = 0.1)
    grid = Grid(par)
    scales = compute_scales(par, grid)
    fluid = FluidState(T, CPU(), grid.ni[1], grid.ni[2])
    phase = PhaseState(T, CPU(), grid.ni[1], grid.ni[2])
    ns = NoiseState(T, grid, scales)
    hst = xcore.History(T)
    initialize!(phase, fluid, ns, hst, grid, par, scales)

    # mark a value, store, mark a new value, store again — verify the chain
    phase.X[1, 1] = 7.0
    store_previous!(fluid, phase)
    @test phase.Xo[1, 1] == 7.0
    phase.X[1, 1] = 9.0
    store_previous!(fluid, phase)
    @test phase.Xo[1, 1]  == 9.0
    @test phase.Xoo[1, 1] == 7.0
end

@testset "run! 5-step smoke test stays finite" begin
    # this is a sanity check, not a physics test — confirm the integrator
    # doesn't NaN or blow up over a handful of steps on smooth init data.
    T = Float64
    par = Parameters(T; D = 10.0, N = 16, L = 10.0, etam0 = 10.0,
                        x0 = 0.05, dxr = 0.1, Da = 0.0,
                        TINT = :bd2im, ADVN = :weno5, alpha = 0.5)
    grid   = Grid(par)
    scales = compute_scales(par, grid)
    fluid  = FluidState(T, CPU(), grid.ni[1], grid.ni[2])
    phase  = PhaseState(T, CPU(), grid.ni[1], grid.ni[2])
    ns     = NoiseState(T, grid, scales)
    hst    = xcore.History(T)
    initialize!(phase, fluid, ns, hst, grid, par, scales)

    time, dt = run!(phase, fluid, ns, hst, grid, par, scales;
                    nsteps = 5, dt = scales.dt0,
                    verbose = false)

    @test isfinite(time) && time > 0
    @test isfinite(dt) && dt > 0
    @test all(isfinite, phase.x)  && all(>(0), phase.x) && all(<(1), phase.x)
    @test all(isfinite, fluid.W)  && all(isfinite, fluid.U) && all(isfinite, fluid.P)
    @test all(isfinite, fluid.eta) && all(>(0), fluid.eta)
    @test all(isfinite, phase.etamix) && all(>(0), phase.etamix)
end

# Production configuration of usr/run_D1_dm2_e1.jl at a small resolution,
# integrated with the matrix-free solver (the one that runs in 3-D).
function run_case(mkgrid; nsteps = 3, kw...)
    T = Float64
    par = Parameters(T; save_op = false, D = 10.0, N = 8, L = 15.0, t0end = 2.0,
                     d0 = 0.01, etam0 = 10.0, L0 = 0.1, l0 = 0.1, Da = 0.01, Xi = 0.5,
                     solver = :dyrel, atol = 1.0e-6, kw...)
    grid = mkgrid(par)
    scales = compute_scales(par, grid)
    fluid = FluidState(T, CPU(), grid);  phase = PhaseState(T, CPU(), grid)
    ns = NoiseState(T, grid, scales);  hst = xcore.History(T)
    time, dt, step = initialize!(phase, fluid, ns, hst, grid, par, scales)
    run!(phase, fluid, ns, hst, grid, par, scales; nsteps, time, dt, step, verbose = false)
    return (; phase, fluid, hst, grid, par)
end

@testset "run! in 3-D stays finite and conserves mass" begin
    (; phase, fluid, hst, grid, par) = run_case(par -> Grid(par, Val(3)))
    @test cells(grid) == (8, 12, 12) && length(hst.EB) == 3
    @test all(isfinite, phase.x) && all(>(0), phase.x) && all(<(1), phase.x)
    @test all(A -> all(isfinite, A), (fluid.W, fluid.U, fluid.V, fluid.P, fluid.eta))
    @test maximum(abs, fluid.V) > 0                     # noise drives flow along y
    @test all(E -> maximum(abs, E) < 1e-10, (hst.EB, hst.EM, hst.EX))
    @test_throws "solver = :direct is 2-D only" xcore.make_solver(
        Parameters(Float64; N = 8, L = 15.0, D = 10.0), fluid, phase, grid)
end

@testset "run! in 3-D with one periodic y-cell tracks the 2-D run" begin
    # One periodic cell along y makes every 3-D field y-invariant: initialize!
    # draws the same random field and bump as in 2-D, and without noise (Xi = 0)
    # nothing forces a y-velocity. The 2-D and 3-D :dyrel solves take different
    # pseudo-transient paths, so the runs agree to the solver tolerance (here
    # 1e-8), except that the 3-D strain-rate invariant carries the out-of-plane
    # term ½·e_yy² with e_yy = −div/3, which the 2-D invariant omits.
    a = run_case(Grid; Xi = 0.0, atol = 1.0e-8)
    b = run_case(par -> (g = Grid(par); Grid((g.ni..., 1), (g.li..., g.h))); Xi = 0.0, atol = 1.0e-8)
    slice(A) = dropdims(A; dims = 3)
    rel(A3, A2) = maximum(abs, slice(A3) .- A2) / maximum(abs, A2)
    @test rel(b.phase.x, a.phase.x) < 1e-8
    @test rel(b.fluid.W[:, :, 2:2], a.fluid.W) < 1e-7 && rel(b.fluid.U[:, :, 2:2], a.fluid.U) < 1e-7
    @test rel(b.fluid.P[:, :, 2:2], a.fluid.P) < 1e-6
    @test all(iszero, b.fluid.V)
    eII2 = sqrt.(a.phase.eII .^ 2 .+ slice(b.phase.eyy) .^ 2 ./ 2)
    @test maximum(abs, slice(b.phase.eII) .- eII2) / maximum(a.phase.eII) < 1e-8
end

@testset "restart from a checkpoint continues the run" begin
    function restart_pair(mkgrid; kw...)
        outdir = mktempdir()
        return map((0, 1)) do restart    # restart = 1 resumes after step 3
            run_case(mkgrid; nsteps = 6, restart, save_op = true, plot_op = false, nop = 3,
                     outdir, runID = "restart", kw...)
        end
    end
    # :direct keeps no state between steps, so the restart is exact.
    a, b = restart_pair(Grid; solver = :direct)
    differing(s, t) = [f for f in fieldnames(typeof(s))
                       if getfield(s, f) isa AbstractArray && !isequal(getfield(s, f), getfield(t, f))]
    @test differing(a.phase, b.phase) == []
    @test differing(a.fluid, b.fluid) == []
    @test differing(a.hst, b.hst) == []
    # :dyrel warm-starts from its cache, which is not checkpointed, so the
    # restarted 3-D run agrees to the solver tolerance.
    a, b = restart_pair(par -> Grid(par, Val(3)); atol = 1.0e-10)
    rel(A, B) = maximum(abs, A .- B) / maximum(abs, A)
    @test all(f -> rel(getfield(a.fluid, f), getfield(b.fluid, f)) < 1e-10, (:W, :U, :V, :P))
    @test rel(a.phase.x, b.phase.x) < 1e-10
    @test rel(a.hst.x_mean, b.hst.x_mean) < 1e-10
end
