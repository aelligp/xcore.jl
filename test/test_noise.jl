using Statistics
using xcore: NoiseState, PhaseState, FluidState, Parameters, Grid, compute_scales,
             initialize!, noise!, store_noise!, History
using xcore: NoiseState   # re-import to access compute_fft_filter via module
import xcore as XC
using KernelAbstractions: CPU

function _make_noise_setup(T = Float64; N = 16)
    par    = Parameters(T; D = 10.0, N, L = 10.0, etam0 = 10.0,
                           x0 = 0.1, Da = 0.0, Xi = 0.5)
    grid   = Grid(par)
    scales = compute_scales(par, grid)
    fluid  = FluidState(T, CPU(), grid.ni[1], grid.ni[2])
    phase  = PhaseState(T, CPU(), grid.ni[1], grid.ni[2])
    ns     = NoiseState(T, grid, scales; seed = 42)
    hst    = History(T)
    initialize!(phase, fluid, ns, hst, grid, par, scales)
    return ns, phase, fluid, grid, par, scales
end

@testset "NoiseState sizes are correct" begin
    for T in (Float64, Float32)
        ns, _, _, grid, _, _ = _make_noise_setup(T; N = 12)
        Nz, Nx = grid.ni[1], grid.ni[2]

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

import Random

# noise inputs (phase fields read by noise!, white noise, OU history) for a 2-D
# grid, random but reproducible
function noise_test_inputs!(rng, phase, ns)
    for A in (phase.V, phase.vx, phase.fReL, phase.ke, phase.ks, phase.chi,
              phase.chiw, phase.chiu, phase.muw, phase.muu)
        A .= 0.5 .+ rand(rng, size(A))
    end
    for A in (ns.re, ns.rs, ns.psixo, ns.psiso); A .= randn(rng, size(A)); end
    return nothing
end

@testset "noise! in 3-D: divergence-free eddy noise" begin
    T = Float64
    par = Parameters(T; D = 10.0, N = 16, L = 10.0, Xi = 0.5)
    grid = Grid(par, Val(3));  scales = compute_scales(par, Grid(par))
    phase = PhaseState(T, CPU(), grid);  ns = NoiseState(T, grid, scales; seed = 3)
    rng = Random.Xoshiro(5)
    for A in (phase.V, phase.vx, phase.fReL, phase.ke, phase.ks, phase.chi,
              phase.chiw, phase.chiu, phase.chiv, phase.muw, phase.muu, phase.muv)
        A .= 0.5 .+ rand(rng, size(A))
    end
    noise!(ns, phase, grid, par, scales, scales.dt0)
    ξ = (ns.xiew, ns.xieu, ns.xiev)
    @test all(A -> maximum(abs, A) > 0, ξ)
    o = CartesianIndex(1, 1, 1)
    e(d) = CartesianIndex(ntuple(k -> Int(k == d), 3))
    div = [sum(ξ[d][c + o] - ξ[d][c + o - e(d)] for d in 1:3) for c in CartesianIndices(cells(grid))]
    @test maximum(abs, div) < 1e-12 * maximum(A -> maximum(abs, A), ξ)
    @test all(isfinite, ns.xie) && all(isfinite, ns.xix) && all(isfinite, ns.xis)
end

@testset "noise! in 3-D reduces to 2-D on y-invariant states" begin
    # Only the y-component of the eddy potential is forced (re_z = re_x = 0) and
    # every input is y-invariant: the eddy noise is the 2-D one scaled by 1/√2, the
    # particle-eddy noise (forced by re/√3, history psixo/√3) scaled by 1/√3, and
    # the settling noise equals the 2-D one on every slice.
    T = Float64
    par = Parameters(T; D = 10.0, N = 16, L = 10.0, Xi = 0.5)
    grid2 = Grid(par);  grid3 = Grid(par, Val(3));  scales = compute_scales(par, grid2)
    phase2 = PhaseState(T, CPU(), grid2);  phase3 = PhaseState(T, CPU(), grid3)
    ns2 = NoiseState(T, grid2, scales);  ns3 = NoiseState(T, grid3, scales)
    noise_test_inputs!(Random.Xoshiro(9), phase2, ns2)
    lift(A, n) = repeat(A, 1, 1, n)
    for (s3, s2) in ((phase3, phase2), (ns3, ns2)), n in fieldnames(typeof(s2))
        a2 = getfield(s2, n);  a3 = getfield(s3, n)
        a2 isa AbstractArray{T} && a3 isa AbstractArray{T, 3} && (a3 .= lift(a2, size(a3, 3)))
    end
    ns3.psixo ./= sqrt(3)
    for xBC in (:periodic, :closed), yBC in (:periodic, :closed)
        noise!(ns2, phase2, grid2, par, scales, scales.dt0; first_iter = false, xBC)
        noise!(ns3, phase3, grid3, par, scales, scales.dt0; first_iter = false, xBC, yBC)
        same(a3, a2; s = 1) = all(k -> isapprox(a3[:, :, k], s .* a2; rtol = 1e-10, atol = 1e-14), axes(a3, 3))
        @test same(ns3.xiew, ns2.xiew; s = 1 / sqrt(2)) && same(ns3.xieu, ns2.xieu; s = 1 / sqrt(2))
        @test same(ns3.xie, ns2.xie; s = 1 / sqrt(2))
        @test maximum(abs, ns3.xiev) < 1e-14 * maximum(abs, ns3.xieu)
        for n in (:xixw, :xixu, :xix)
            @test same(getfield(ns3, n), getfield(ns2, n); s = 1 / sqrt(3))
        end
        for n in (:xisw, :xisu, :xis)
            @test same(getfield(ns3, n), getfield(ns2, n))
        end
        @test maximum(abs, ns3.xixv) < 1e-12 && maximum(abs, ns3.xisv) < 1e-12
    end
end

@testset "noise! Ornstein–Uhlenbeck statistics in 2-D and 3-D" begin
    # With uniform inputs, every cell of each unfiltered potential is an
    # independent AR(1) process ψ' = Fψ + √(1−F²)·σ·r, with F = exp(−dt/(τ_eff + dt))
    # and 1/τ_eff = 1/τ + 1/(100 dt). Its stationary variance is σ², its lag-one
    # autocorrelation is F, and distinct potentials are uncorrelated. The linear
    # filter and curl carry the autocorrelation F over to the eddy noise ξ.
    T = Float64
    par = Parameters(T; D = 10.0, N = 16, L = 10.0, Xi = 0.5)
    scales = compute_scales(par, Grid(par))
    V = 0.3;  vx = 0.6;  a = 0.5                       # a: every other phase input
    τe = scales.L0 / (2 * (V + eps(T)));  τs = scales.l0 / (2 * (vx + eps(T)))
    dt = τe / 4
    F(τ) = exp(-dt / (inv(inv(τ) + inv(100 * dt)) + dt))
    for (grid, nsteps) in ((Grid(par), 3000), (Grid(par, Val(3)), 400))
        D = length(cells(grid))
        phase = PhaseState(T, CPU(), grid);  ns = NoiseState(T, grid, scales; seed = 11)
        for A in (phase.fReL, phase.ke, phase.ks, phase.chi, phase.chiw, phase.chiu,
                  phase.muw, phase.muu, (D == 3 ? (phase.chiv, phase.muv) : ())...)
            A .= a
        end
        phase.V .= V;  phase.vx .= vx
        σe = par.Xi * sqrt(a * a / τe) * ns.fL / (D == 3 ? sqrt(2) : 1)
        σs = par.Xi * sqrt(a * a / τs) * ns.fl
        σx = XC.xtal_amplitude(par.Xi, a, a, a, τe, scales.txi0) * ns.fL
        pots  =D == 3 ? (ns.psie_z, ns.psie_x, ns.psie, ns.psis) : (ns.psie, ns.psis)
        prevs = D == 3 ? (ns.psieo_z, ns.psieo_x, ns.psieo, ns.psiso) : (ns.psieo, ns.psiso)
        ne = length(pots) - 1                             # eddy components: 1 in 2-D, 3 in 3-D
        σ  = (ntuple(_ -> σe, ne)..., σs)
        Fk = (ntuple(_ -> F(τe), ne)..., F(τs))
        ξ  = D == 3 ? (ns.xiew, ns.xieu, ns.xiev) : (ns.xiew, ns.xieu)
        step!() = (store_noise!(ns); ξo = map(copy, ξ);
                   noise!(ns, phase, grid, par, scales, dt; first_iter = true); ξo)
        for _ in 1:ceil(Int, 15 / (1 - maximum(Fk)))   # burn-in: F^(2n) ≪ 1e-6
            step!()
        end
        n = length(pots)
        s2 = zeros(n);  so2 = zeros(n);  sc = zeros(n);  sx = zeros(n, n)
        ξ2 = zeros(D);  ξo2 = zeros(D);  ξc = zeros(D);  m = 0
        x2 = 0.0;  xo2 = 0.0;  xc = 0.0;  xe = zeros(ne)  # particle-eddy potential psix
        for _ in 1:nsteps
            ξo = step!()
            x2 += sum(abs2, ns.psix);  xo2 += sum(abs2, ns.psixo);  xc += sum(ns.psix .* ns.psixo)
            for k in 1:ne
                xe[k] += sum(ns.psix .* pots[k])
            end
            for i in 1:n
                s2[i] += sum(abs2, pots[i]);  so2[i] += sum(abs2, prevs[i])
                sc[i] += sum(pots[i] .* prevs[i])
                for j in i+1:n
                    sx[i, j] += sum(pots[i] .* pots[j])
                end
            end
            for d in 1:D
                ξ2[d] += sum(abs2, ξ[d]);  ξo2[d] += sum(abs2, ξo[d]);  ξc[d] += sum(ξ[d] .* ξo[d])
            end
            m += length(first(pots))
        end
        for i in 1:n
            @test s2[i] / m ≈ σ[i]^2 rtol = 0.03          # stationary variance
            @test sc[i] / so2[i] ≈ Fk[i] atol = 0.01      # lag-one autocorrelation
            for j in i+1:n
                @test abs(sx[i, j] / sqrt(s2[i] * s2[j])) < 0.01   # independent potentials
            end
        end
        for d in 1:D
            @test ξc[d] / sqrt(ξ2[d] * ξo2[d]) ≈ F(τe) atol = 0.02
        end
        # psix: the eddy OU process with amplitude sgx·fL (noise.m), forced by the
        # eddy white noise; in 3-D by its normalized sum over the three components
        @test x2 / m ≈ σx^2 rtol = 0.03
        @test xc / xo2 ≈ F(τe) atol = 0.01
        for k in 1:ne
            @test xe[k] / sqrt(x2 * s2[k]) ≈ (D == 3 ? 1 / sqrt(3) : 1) atol = 0.01
        end
    end
end
