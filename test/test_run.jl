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
    fluid = FluidState(T, CPU(), grid.Nz, grid.Nx)
    phase = PhaseState(T, CPU(), grid.Nz, grid.Nx)
    ns = NoiseState(T, grid, scales)
    initialize!(phase, fluid, ns, grid, par, scales)

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
    fluid  = FluidState(T, CPU(), grid.Nz, grid.Nx)
    phase  = PhaseState(T, CPU(), grid.Nz, grid.Nx)
    ns     = NoiseState(T, grid, scales)
    initialize!(phase, fluid, ns, grid, par, scales)

    # silent callback for tests
    silent(args...) = nothing
    time, dt = run!(phase, fluid, ns, grid, par, scales;
                    nsteps = 5, dt = scales.dt0,
                    verbose = false, callback = silent)

    @test isfinite(time) && time > 0
    @test isfinite(dt) && dt > 0
    @test all(isfinite, phase.x)  && all(>(0), phase.x) && all(<(1), phase.x)
    @test all(isfinite, fluid.W)  && all(isfinite, fluid.U) && all(isfinite, fluid.P)
    @test all(isfinite, fluid.eta) && all(>(0), fluid.eta)
    @test all(isfinite, phase.etamix) && all(>(0), phase.etamix)
end
