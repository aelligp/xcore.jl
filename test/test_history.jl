using Statistics
using xcore: History, record_history!, PhaseState, FluidState, NoiseState,
             Parameters, Grid, compute_scales, initialize!
using KernelAbstractions: CPU

function _make_small(T = Float64)
    par    = Parameters(T; D = 10.0, N = 8, L = 10.0, etam0 = 10.0, x0 = 0.1, Da = 0.0)
    grid   = Grid(par)
    scales = compute_scales(par, grid)
    fluid  = FluidState(T, CPU(), grid.Nz, grid.Nx)
    phase  = PhaseState(T, CPU(), grid.Nz, grid.Nx)
    ns     = NoiseState(T, grid, scales)
    initialize!(phase, fluid, ns, grid, par, scales)
    return phase, fluid, ns, grid, scales
end

@testset "History constructor is empty" begin
    for T in (Float64, Float32)
        hst = History(T)
        @test isempty(hst.time)
        @test isempty(hst.x_min)
        @test isempty(hst.V_rms)
        @test isempty(hst.Ra_gm)
        @test isempty(hst.eta_min)
        @test isempty(hst.EB)
    end
end

@testset "record_history! appends one record" begin
    phase, fluid, ns, grid, _ = _make_small()
    hst = History(Float64)

    record_history!(hst, 0.0, 1.0, phase, fluid, ns, grid)

    @test length(hst.time)  == 1
    @test length(hst.x_min) == 1
    @test length(hst.V_rms) == 1
    @test length(hst.Ra_gm) == 1
    @test length(hst.EB)    == 1
    @test length(hst.EM)    == 1
    @test length(hst.EX)    == 1

    # conservation errors on the first record must be exactly zero
    @test hst.EB[1] == 0.0
    @test hst.EM[1] == 0.0
    @test hst.EX[1] == 0.0
end

@testset "record_history! ordering and finiteness" begin
    phase, fluid, ns, grid, scales = _make_small()
    hst = History(Float64)

    for k in 1:3
        record_history!(hst, Float64(k) * scales.dt0,
                        scales.dt0, phase, fluid, ns, grid)
    end

    @test length(hst.time) == 3

    # min ≤ mean ≤ max everywhere
    for k in 1:3
        @test hst.x_min[k] ≤ hst.x_mean[k] ≤ hst.x_max[k]
        @test hst.V_min[k] ≤ hst.V_rms[k]  ≤ hst.V_max[k] + sqrt(eps())
        @test hst.vx_min[k] ≤ hst.vx_rms[k] ≤ hst.vx_max[k] + sqrt(eps())
    end

    # all recorded values are finite
    for fn in fieldnames(History{Float64})
        v = getfield(hst, fn)
        @test all(isfinite, v)
    end
end

@testset "record_history! accumulates monotonically increasing time" begin
    phase, fluid, ns, grid, scales = _make_small()
    hst = History(Float64)
    dt  = scales.dt0

    t = 0.0
    for _ in 1:5
        t += dt
        record_history!(hst, t, dt, phase, fluid, ns, grid)
    end
    @test issorted(hst.time)
    @test length(hst.time) == 5
end
