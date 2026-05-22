using LinearAlgebra
using xcore: PhaseState, FluidState, Parameters, Grid, compute_scales,
             update_rheology!, update_kinematics!, update!
using KernelAbstractions: CPU

# Pure-Julia reference implementation of the permission-weight rheology.
# Used to cross-check the KA kernel without relying on the kernel itself.
function compute_rheology_reference(chi::T, mu::T, par::Parameters{T}) where {T}
    f  = (chi, mu)
    AA = par.AA;  BB = par.BB;  CC = par.CC

    Sf = T[(f[j] / BB[i, j])^(one(T) / CC[i, j]) for i in 1:2, j in 1:2]
    Sf = Sf ./ sum(Sf, dims = 2)                              # row-normalise

    Xf = similar(Sf)
    for i in 1:2
        a = AA[i, 1] * Sf[i, 1] + AA[i, 2] * Sf[i, 2]
        Xf[i, 1] = a * f[1] + (one(T) - a) * Sf[i, 1]
        Xf[i, 2] = a * f[2] + (one(T) - a) * Sf[i, 2]
    end

    kv   = (par.etax0, par.etam0)
    thtv = ntuple(2) do i
        (kv[1] / kv[i])^Xf[i, 1] * (kv[2] / kv[i])^Xf[i, 2]
    end
    etaf = (kv[1] * thtv[1], kv[2] * thtv[2])
    return chi * etaf[1] + mu * etaf[2]
end

@testset "update_rheology! permission weights match naive reference" begin
    for T in (Float64, Float32)
        par = Parameters{T}()
        # sweep (chi, mu) pairs spanning the rheology's interesting regimes
        for (chi, mu) in ((T(0.30), T(0.70)),
                          (T(0.50), T(0.50)),
                          (T(0.10), T(0.90)),
                          (T(0.90), T(0.10)))
            phase = PhaseState(T, CPU(), 1, 1)
            phase.chi[1, 1] = chi
            phase.mu[1, 1]  = mu
            update_rheology!(phase, par)
            expected = compute_rheology_reference(chi, mu, par)
            rt = T === Float64 ? 1e-12 : 1e-4
            @test phase.etamix[1, 1] ≈ expected rtol = rt
        end
    end
end

@testset "update_kinematics! 2nd-order strain rates on sinusoid" begin
    # incompressible velocity: W = sin(kz) cos(kx), U = -cos(kz) sin(kx)
    # ⇒ Div V = 0, exx = -k cos(kz_c) cos(kx_c), ezz = +k cos(kz_c) cos(kx_c)
    for T in (Float64, Float32)
        L = T(2π)
        errs_exx = T[];  errs_ezz = T[];  errs_div = T[]
        for N in (16, 32, 64)
            h = L / N
            k = T(2π) / L                                       # one full period in L
            par   = Parameters(T; D = L, N = N, L = L)
            grid  = Grid(par)
            fluid = FluidState(T, CPU(), N, N)
            phase = PhaseState(T, CPU(), N, N)

            # fill staggered velocity arrays (including ghost rows/cols) from the
            # analytic field; periodic BCs are automatic when k = 2π/L.
            for j in 1:size(fluid.W, 1), i in 1:size(fluid.W, 2)
                z = T(j - 1) * h
                x = T(i - T(1.5)) * h                           # ghost x-cells
                fluid.W[j, i] = sin(k * z) * cos(k * x)
            end
            for j in 1:size(fluid.U, 1), i in 1:size(fluid.U, 2)
                z = T(j - T(1.5)) * h                           # ghost z-cells
                x = T(i - 1) * h
                fluid.U[j, i] = -cos(k * z) * sin(k * x)
            end

            update_kinematics!(phase, fluid, grid)

            exact_exx = T[-k * cos(k * (T(jc) - T(0.5)) * h) *
                              cos(k * (T(ic) - T(0.5)) * h) for jc in 1:N, ic in 1:N]
            exact_ezz = -exact_exx                                # mirror sign

            push!(errs_exx, norm(phase.exx .- exact_exx) / norm(exact_exx))
            push!(errs_ezz, norm(phase.ezz .- exact_ezz) / norm(exact_ezz))
            push!(errs_div, norm(phase.Div_V))                    # exact = 0
        end
        @test log2(errs_exx[1] / errs_exx[2]) > 1.8
        @test log2(errs_exx[2] / errs_exx[3]) > 1.8
        @test log2(errs_ezz[1] / errs_ezz[2]) > 1.8
        @test log2(errs_ezz[2] / errs_ezz[3]) > 1.8
        # Div V should already be ≈ 0 to machine precision (cancellation of two
        # equal-magnitude exact terms): just check it stays small.
        @test errs_div[end] < (T === Float64 ? 1e-10 : 1e-3)
    end
end

@testset "update! end-to-end smoke test" begin
    # exercise the full pipeline on a smooth initial state; verify nothing
    # goes NaN, viscosities stay positive, and shapes line up.
    T = Float64
    par = Parameters(T; D = 10, N = 32, L = 10, etam0 = 10.0, x0 = 0.1)
    grid   = Grid(par)
    scales = compute_scales(par, grid)
    fluid  = FluidState(T, CPU(), grid.Nz, grid.Nx)
    phase  = PhaseState(T, CPU(), grid.Nz, grid.Nx)

    # gentle phase-fraction perturbation around x0
    for j in 1:grid.Nz, i in 1:grid.Nx
        zg = T(j - T(0.5)) * grid.h / grid.D
        xg = T(i - T(0.5)) * grid.h / grid.L
        phase.x[j, i] = T(0.1) + T(0.05) * sin(T(2π) * zg) * cos(T(2π) * xg)
        phase.m[j, i] = T(1) - phase.x[j, i]
    end

    # seed viscosity (Picard blend reads previous eta); seed velocity weakly
    fill!(fluid.eta,   par.etam0)
    fill!(fluid.etaco, par.etam0)
    for j in 1:size(fluid.W, 1), i in 1:size(fluid.W, 2)
        fluid.W[j, i] = T(1e-3) * sin(T(j - 1) * T(2π) / size(fluid.W, 1))
    end

    update!(phase, fluid, grid, par, scales)

    @test all(isfinite, fluid.rho)        && all(>(0), fluid.rho)
    @test all(isfinite, phase.etamix)     && all(>(0), phase.etamix)
    @test all(isfinite, fluid.eta)        && all(>(0), fluid.eta)
    @test all(isfinite, phase.eII)        && all(≥(0), phase.eII)
    @test all(isfinite, phase.tII)        && all(≥(0), phase.tII)
    @test size(phase.chi)   == (grid.Nz, grid.Nx)
    @test size(phase.exz)   == (grid.Nz + 1, grid.Nx + 1)
    @test size(phase.etamix) == (grid.Nz, grid.Nx)
end
