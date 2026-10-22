using xcore: StepResidual, snapshot!, compute_resnorm, report_iter,
             PhaseState, FluidState, NoiseState, Parameters, Grid, compute_scales,
             initialize!, History, phsevo!, store_previous!, time_coefs
using LinearAlgebra: norm
using KernelAbstractions: CPU

function _make_diag_setup(T = Float64)
    par    = Parameters(T; D = 10.0, N = 8, L = 10.0, etam0 = 10.0, x0 = 0.1, Da = 0.0)
    grid   = Grid(par)
    scales = compute_scales(par, grid)
    fluid  = FluidState(T, CPU(), grid.ni[1], grid.ni[2])
    phase  = PhaseState(T, CPU(), grid.ni[1], grid.ni[2])
    ns     = NoiseState(T, grid, scales)
    hst    = History(T)
    initialize!(phase, fluid, ns, hst, grid, par, scales)
    return phase, fluid, ns, grid, scales
end

@testset "StepResidual allocates with correct sizes" begin
    for T in (Float64, Float32)
        par  = Parameters(T; D = 10.0, N = 10, L = 10.0)
        grid = Grid(par)
        res  = StepResidual(grid, FluidState(T, CPU(), grid.ni[1], grid.ni[2]))

        Nz, Nx = grid.ni[1], grid.ni[2]
        @test size(res.vo[1])   == (Nz + 1, Nx + 2)
        @test size(res.vo[2])   == (Nz + 2, Nx + 1)
        @test size(res.Po)   == (Nz + 2, Nx + 2)
        @test size(res.MFSo) == (Nz,     Nx    )
    end
end

@testset "snapshot! copies primary fields into buffers" begin
    phase, fluid, _, grid, _ = _make_diag_setup()
    res = StepResidual(grid, fluid)

    fluid.W[2, 2] = 42.0
    fluid.U[3, 2] = -7.0
    snapshot!(res, fluid)

    @test res.vo[1][2, 2]  == 42.0
    @test res.vo[2][3, 2]  == -7.0

    # mutating original must NOT affect the snapshot
    fluid.W[2, 2] = 0.0
    @test res.vo[1][2, 2]  == 42.0
end

@testset "compute_resnorm is zero when nothing changed" begin
    phase, fluid, _, grid, scales = _make_diag_setup()
    res = StepResidual(grid, fluid)
    snapshot!(res, fluid)

    rn, rm, rp = compute_resnorm(res, fluid, scales.dt0, 0.0)

    # no change between snapshot and current → residual should be exactly 0
    @test rn == 0.0
    @test rm == 0.0
    @test rp == 0.0
end

@testset "compute_resnorm is positive after field change" begin
    phase, fluid, _, grid, scales = _make_diag_setup()
    res = StepResidual(grid, fluid)
    snapshot!(res, fluid)

    # perturb a primary unknown
    fluid.W .+= 1.0

    rn, rm, rp = compute_resnorm(res, fluid, scales.dt0, 0.0)

    @test rn > 0
    @test rp > 0   # momentum component must be positive
    @test isfinite(rn)
end

@testset "compute_resnorm mass term uses the unclamped X update" begin
    phase, fluid, _, grid, scales = _make_diag_setup()
    res = StepResidual(grid, fluid)
    snapshot!(res, fluid)
    rn, rm, rp = compute_resnorm(res, fluid, scales.dt0, 2.0)
    @test rm ≈ 2.0 / norm(fluid.rho)
    @test rp == 0.0
end

@testset "phsevo! returns the norm of the update before the clamp" begin
    phase, fluid, _, grid, scales = _make_diag_setup()
    par = Parameters(Float64; D = 10.0, N = 8, L = 10.0, etam0 = 10.0, x0 = 0.1, Da = 0.0)
    store_previous!(fluid, phase)
    a1, a2, a3, b1, b2, b3 = time_coefs(Float64, par.TINT, 1)
    step!() = (X0 = copy(phase.X);
               n = phsevo!(phase, fluid, grid, par, scales; a1, a2, a3, b1, b2, b3, dt = scales.dt0);
               (n, norm(phase.X .- X0)))

    # an old density 0.1ρ higher pulls X well inside (ρε, ρ(1-ε)): no clamp, so
    # the returned norm is the change in X
    phase.Xo .= phase.X .+ 0.1 .* fluid.rho
    nupd, dX = step!()
    @test nupd ≈ dX

    # a negative old density drives X below ρε: the clamp shrinks the change
    phase.Xo .= .-fluid.rho
    nupd, dX = step!()
    @test nupd > 2dX
    @test all(phase.X .>= fluid.rho .* eps())
end

@testset "report_iter does not throw on valid inputs" begin
    @test (report_iter(1,  1.5e-3, 2.0e-3, 1.0e-3, 5.0e-4); true)
    @test (report_iter(10, 8.0e-5, 2.0e-3, 4.0e-5, 4.0e-5); true)
    @test (report_iter(100, 1.0e-6, 2.0e-3, 5.0e-7, 5.0e-7); true)
end

@testset "report_iter throws on NaN residual" begin
    @test_throws ErrorException report_iter(1, NaN, 1.0, 0.5, 0.5)
end
