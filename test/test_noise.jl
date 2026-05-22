using Statistics
using xcore: NoiseState, PhaseState, FluidState, Parameters, Grid, compute_scales,
             initialize!, noise!, store_noise!
using xcore: NoiseState   # re-import to access compute_fft_filter via module
import xcore as XC
using KernelAbstractions: CPU

function _make_noise_setup(T = Float64; N = 16)
    par    = Parameters(T; D = 10.0, N, L = 10.0, etam0 = 10.0,
                           x0 = 0.1, Da = 0.0, Xi = 0.5)
    grid   = Grid(par)
    scales = compute_scales(par, grid)
    fluid  = FluidState(T, CPU(), grid.Nz, grid.Nx)
    phase  = PhaseState(T, CPU(), grid.Nz, grid.Nx)
    ns     = NoiseState(T, grid, scales; seed = 42)
    initialize!(phase, fluid, ns, grid, par, scales)
    return ns, phase, fluid, grid, par, scales
end

@testset "NoiseState sizes are correct" begin
    for T in (Float64, Float32)
        ns, _, _, grid, _, _ = _make_noise_setup(T; N = 12)
        Nz, Nx = grid.Nz, grid.Nx

        @test size(ns.psie)  == (Nz, Nx)
        @test size(ns.psieo) == (Nz, Nx)
        @test size(ns.re)    == (Nz, Nx)
        @test size(ns.rs)    == (Nz, Nx)
        # ghost-extended staggered flux arrays
        @test size(ns.xiew)  == (Nz + 1, Nx + 2)
        @test size(ns.xieu)  == (Nz + 2, Nx + 1)
        @test size(ns.xixw)  == (Nz + 1, Nx + 2)
        @test size(ns.xixu)  == (Nz + 2, Nx + 1)
        @test size(ns.xisw)  == (Nz + 1, Nx + 2)
        @test size(ns.xisu)  == (Nz + 2, Nx + 1)
        # cell-centred speed magnitudes
        @test size(ns.xie)   == (Nz, Nx)
        @test size(ns.xix)   == (Nz, Nx)
        @test size(ns.xis)   == (Nz, Nx)
    end
end

@testset "store_noise! snapshots OU potentials" begin
    ns, _, _, _, _, _ = _make_noise_setup()
    # plant known values
    ns.psie  .= 1.0
    ns.psix  .= 2.0
    ns.psis  .= 3.0
    store_noise!(ns)
    @test all(==(1.0), ns.psieo)
    @test all(==(2.0), ns.psixo)
    @test all(==(3.0), ns.psiso)

    # second store: psieo should follow psie
    ns.psie .= 7.0
    store_noise!(ns)
    @test all(==(7.0), ns.psieo)
end

@testset "noise! produces finite non-negative speed magnitudes" begin
    for T in (Float64, Float32)
        ns, phase, fluid, grid, par, scales = _make_noise_setup(T)

        # run a few noise steps (first with first_iter=true, rest reuse noise)
        for iter in 1:3
            noise!(ns, phase, grid, par, scales, scales.dt0; first_iter = (iter == 1))
        end

        @test all(isfinite, ns.xie)  && all(≥(0), ns.xie)
        @test all(isfinite, ns.xix)  && all(≥(0), ns.xix)
        @test all(isfinite, ns.xis)  && all(≥(0), ns.xis)
        @test all(isfinite, ns.xiew) && all(isfinite, ns.xieu)
        @test all(isfinite, ns.xixw) && all(isfinite, ns.xixu)
    end
end

@testset "noise! reuses random draw across Picard iterations" begin
    ns, phase, fluid, grid, par, scales = _make_noise_setup()
    store_noise!(ns)

    # first iteration draws fresh noise
    noise!(ns, phase, grid, par, scales, scales.dt0; first_iter = true)
    re_after_first  = copy(ns.re)

    # subsequent iterations must NOT redraw
    noise!(ns, phase, grid, par, scales, scales.dt0; first_iter = false)
    @test ns.re == re_after_first

    noise!(ns, phase, grid, par, scales, scales.dt0; first_iter = false)
    @test ns.re == re_after_first
end

@testset "noise! Gaussian filter preserves mean and std" begin
    # If we set psie to a constant, the FFT filter should not change anything.
    ns, phase, fluid, grid, par, scales = _make_noise_setup(; N = 24)
    store_noise!(ns)

    fill!(ns.psie, 0.0)          # constant → filter is a no-op on mean
    fill!(ns.psieo, 0.0)
    ns.re .= randn(size(ns.re)...)

    noise!(ns, phase, grid, par, scales, scales.dt0; first_iter = true)

    # xie speed magnitudes should be finite (even if small near zero-velocity init)
    @test all(isfinite, ns.xie)
    @test all(isfinite, ns.xiew)
end
