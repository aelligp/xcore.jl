using xcore: StepResidual, snapshot!, compute_resnorm, report_iter,
             PhaseState, FluidState, NoiseState, Parameters, Grid, compute_scales,
             initialize!
using KernelAbstractions: CPU

function _make_diag_setup(T = Float64)
    par    = Parameters(T; D = 10.0, N = 8, L = 10.0, etam0 = 10.0, x0 = 0.1, Da = 0.0)
    grid   = Grid(par)
    scales = compute_scales(par, grid)
    fluid  = FluidState(T, CPU(), grid.Nz, grid.Nx)
    phase  = PhaseState(T, CPU(), grid.Nz, grid.Nx)
    ns     = NoiseState(T, grid, scales)
    initialize!(phase, fluid, ns, grid, par, scales)
    return phase, fluid, ns, grid, scales
end

@testset "StepResidual allocates with correct sizes" begin
    for T in (Float64, Float32)
        par  = Parameters(T; D = 10.0, N = 10, L = 10.0)
        grid = Grid(par)
        res  = StepResidual(grid)

        Nz, Nx = grid.Nz, grid.Nx
        @test size(res.Wo)   == (Nz + 1, Nx + 2)
        @test size(res.Uo)   == (Nz + 2, Nx + 1)
        @test size(res.Po)   == (Nz + 2, Nx + 2)
        @test size(res.Xo)   == (Nz,     Nx    )
        @test size(res.MFSo) == (Nz,     Nx    )
        @test res.resnorm0   == 1.0
    end
end

@testset "snapshot! copies primary fields into buffers" begin
    phase, fluid, _, grid, _ = _make_diag_setup()
    res = StepResidual(grid)

    fluid.W[2, 2] = 42.0
    fluid.U[3, 2] = -7.0
    phase.X[1, 1] = 0.3
    snapshot!(res, phase, fluid)

    @test res.Wo[2, 2]  == 42.0
    @test res.Uo[3, 2]  == -7.0
    @test res.Xo[1, 1]  == 0.3

    # mutating original must NOT affect the snapshot
    fluid.W[2, 2] = 0.0
    @test res.Wo[2, 2]  == 42.0
end

@testset "compute_resnorm is zero when nothing changed" begin
    phase, fluid, _, grid, scales = _make_diag_setup()
    res = StepResidual(grid)
    snapshot!(res, phase, fluid)

    rn, rm, rp = compute_resnorm(res, phase, fluid, scales.dt0)

    # no change between snapshot and current → residual should be exactly 0
    @test rn == 0.0
    @test rm == 0.0
    @test rp == 0.0
end

@testset "compute_resnorm is positive after field change" begin
    phase, fluid, _, grid, scales = _make_diag_setup()
    res = StepResidual(grid)
    snapshot!(res, phase, fluid)

    # perturb a primary unknown
    fluid.W .+= 1.0

    rn, rm, rp = compute_resnorm(res, phase, fluid, scales.dt0)

    @test rn > 0
    @test rp > 0   # momentum component must be positive
    @test isfinite(rn)
end

@testset "compute_resnorm updates resnorm0 when residual rises" begin
    phase, fluid, _, grid, scales = _make_diag_setup()
    res = StepResidual(grid)
    res.resnorm0 = 1e-10   # very small reference

    snapshot!(res, phase, fluid)
    fluid.W .+= 1.0
    rn, _, _ = compute_resnorm(res, phase, fluid, scales.dt0)

    # resnorm0 should have been bumped to ≈ rn (because rn > 1e-10)
    @test res.resnorm0 ≈ rn + 1e-32
end

@testset "report_iter does not throw on valid inputs" begin
    @test (report_iter(1,  1.5e-3, 2.0e-3, 1.0e-3, 5.0e-4); true)
    @test (report_iter(10, 8.0e-5, 2.0e-3, 4.0e-5, 4.0e-5); true)
    @test (report_iter(100, 1.0e-6, 2.0e-3, 5.0e-7, 5.0e-7); true)
end

@testset "report_iter throws on NaN residual" begin
    @test_throws ErrorException report_iter(1, NaN, 1.0, 0.5, 0.5)
end
