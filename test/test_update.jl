using LinearAlgebra, Random
import xcore
using xcore: PhaseState, FluidState, Parameters, Grid, cells, compute_scales,
             update_rheology!, update_kinematics!, update!, update_dt, NoiseState,
             store_previous!, update_phase_velocities!, phsevo!
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
    fluid  = FluidState(T, CPU(), grid.ni[1], grid.ni[2])
    phase  = PhaseState(T, CPU(), grid.ni[1], grid.ni[2])

    # gentle phase-fraction perturbation around x0
    for j in 1:grid.ni[1], i in 1:grid.ni[2]
        zg = T(j - T(0.5)) * grid.h / grid.li[1]
        xg = T(i - T(0.5)) * grid.h / grid.li[2]
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
    @test size(phase.chi)   == (grid.ni[1], grid.ni[2])
    @test size(phase.exz)   == (grid.ni[1] + 1, grid.ni[2] + 1)
    @test size(phase.etamix) == (grid.ni[1], grid.ni[2])
end

# 2-D state with a discretely divergence-free velocity (stream function on cell
# corners) and random phase fields; returns the state before `update!`.
function update_test_state(rng, par)
    T = Float64
    grid = Grid(par);  Nz, Nx = cells(grid)
    fluid = FluidState(T, CPU(), grid);  phase = PhaseState(T, CPU(), grid)
    ψ = 1e-3 .* randn(rng, Nz + 2, Nx + 2)
    for j in axes(fluid.U, 1)[2:end-1], i in axes(fluid.U, 2)
        fluid.U[j, i] = ψ[j + 1, i + 1] - ψ[j, i + 1]
    end
    for j in axes(fluid.W, 1), i in axes(fluid.W, 2)[2:end-1]
        fluid.W[j, i] = -(ψ[j + 1, i + 1] - ψ[j + 1, i])
    end
    # ghost layers outside the divergence stencil
    fluid.U[[1, end], :] .= 1e-3 .* randn(rng, 2, Nx + 1)
    fluid.W[:, [1, end]] .= 1e-3 .* randn(rng, Nz + 1, 2)
    phase.x .= 0.05 .+ 0.1 .* rand(rng, Nz, Nx);  phase.m .= 1 .- phase.x
    fluid.P .= 1e-3 .* randn(rng, size(fluid.P))
    for A in (phase.Wx, phase.Wm, phase.Ux, phase.Um, phase.wx, phase.wm)
        A .= 1e-3 .* randn(rng, size(A))
    end
    fluid.eta .= par.etam0 .* (1 .+ rand(rng, Nz, Nx))
    phase.etas .= par.etam0 .* (1 .+ rand(rng, Nz, Nx))
    phase.fReL .= rand(rng, Nz, Nx);  phase.fRel .= rand(rng, Nz, Nx)
    return phase, fluid, grid
end

@testset "update! in 3-D reduces to 2-D on y-invariant states" begin
    # A y-invariant 3-D state with zero y-velocities and a divergence-free (z, x)
    # flow is the 2-D state on every y-slice; y-gradients and the y-shear vanish.
    T = Float64
    par = Parameters(T; D = 10, N = 8, L = 10, etam0 = 10.0, x0 = 0.1)
    lift(A, n) = repeat(A, 1, 1, n)
    for xBC in (:periodic, :closed), yBC in (:periodic, :closed)
        phase2, fluid2, grid2 = update_test_state(Random.Xoshiro(7), par)
        grid3 = Grid(par, Val(3));  Ny = grid3.ni[3]
        fluid3 = FluidState(T, CPU(), grid3);  phase3 = PhaseState(T, CPU(), grid3)
        for (s3, s2) in ((fluid3, fluid2), (phase3, phase2)), n in fieldnames(typeof(s2))
            a2 = getfield(s2, n);  a3 = getfield(s3, n)
            a2 isa AbstractArray && (a3 .= lift(a2, size(a3, 3)))
        end
        scales = compute_scales(par, grid2)
        update!(phase2, fluid2, grid2, par, scales; xBC)
        update!(phase3, fluid3, grid3, par, scales; xBC, yBC)
        for (s3, s2) in ((fluid3, fluid2), (phase3, phase2)), n in fieldnames(typeof(s2))
            a2 = getfield(s2, n);  a3 = getfield(s3, n)
            a2 isa AbstractArray{T} || continue
            @test all(k -> isapprox(a3[:, :, k], a2; rtol = 1e-12, atol = 1e-15), axes(a3, 3))
        end
        @test fluid3.rhov ≈ lift(fluid2.rho, Ny + 1) rtol = 1e-14
        @test maximum(abs, phase3.eyy) < 1e-14 && all(iszero, phase3.eyz) && all(iszero, phase3.exy)
        @test update_dt(phase3, fluid3, grid3, par, 1e6) == update_dt(phase2, fluid2, grid2, par, 1e6)
        phase3.Vx .= 1e3
        @test update_dt(phase3, fluid3, grid3, par, 1e6) < update_dt(phase2, fluid2, grid2, par, 1e6)
    end
end

@testset "update_kinematics! in 3-D is exact for a linear velocity" begin
    # v = G·r with every face (ghosts included) set from its position: the
    # staggered differences and edge averages are exact, so the strain rates,
    # their invariant and the cell-center speed match the continuum values.
    T = Float64
    par = Parameters(T; D = 1, N = 6, L = 1)
    grid = Grid(par, Val(3));  h = grid.h
    fluid = FluidState(T, CPU(), grid);  phase = PhaseState(T, CPU(), grid)
    G = [0.3 -1.2 0.5; 0.7 0.1 -0.4; -0.9 0.6 -0.2]
    v = xcore.velocity(fluid)
    for d in 1:3, J in CartesianIndices(v[d])
        r = ntuple(k -> k == d ? (J[k] - 1) * h : (J[k] - 1.5) * h, 3)
        v[d][J] = sum(G[d, k] * r[k] for k in 1:3)
    end
    update_kinematics!(phase, fluid, grid)
    ε = (G .+ G') ./ 2 .- tr(G) / 3 .* I(3)
    @test all(≈(tr(G)), phase.Div_V)
    @test all(≈(ε[1, 1]), phase.ezz) && all(≈(ε[2, 2]), phase.exx) && all(≈(ε[3, 3]), phase.eyy)
    @test all(≈(ε[1, 2]), phase.exz) && all(≈(ε[1, 3]), phase.eyz) && all(≈(ε[2, 3]), phase.exy)
    @test all(≈(sqrt(sum(abs2, ε) / 2) + eps(T)), phase.eII)
    for c in CartesianIndices(phase.V)
        r = ntuple(k -> (c[k] - 0.5) * h, 3)
        @test phase.V[c] ≈ sqrt(sum(abs2, G * collect(r)))
    end
end

@testset "store_previous!, phase velocities and phsevo! in 3-D reduce to 2-D" begin
    # Same y-invariant construction as for update!, with random noise fluxes and
    # zero y-velocities: every field of the 3-D step is the 2-D one on each slice.
    T = Float64
    par = Parameters(T; D = 10, N = 8, L = 10, etam0 = 10.0, x0 = 0.1)
    lift(A, n) = repeat(A, 1, 1, n)
    coefs = (; a1 = 1.5, a2 = 2.0, a3 = -0.5, b1 = 1.0, b2 = 0.0, b3 = 0.0, dt = 1e-2)
    for xBC in (:periodic, :closed), yBC in (:periodic, :closed)
        rng = Random.Xoshiro(11)
        phase2, fluid2, grid2 = update_test_state(rng, par)
        scales = compute_scales(par, grid2)
        grid3 = Grid(par, Val(3));  Ny = grid3.ni[3]
        fluid3 = FluidState(T, CPU(), grid3);  phase3 = PhaseState(T, CPU(), grid3)
        ns2 = NoiseState(T, grid2, scales);  ns3 = NoiseState(T, grid3, scales)
        phase2.bndtaperw .= rand(rng, size(phase2.bndtaperw))
        for A in (ns2.xiew, ns2.xieu, ns2.xixw, ns2.xixu, ns2.xisw, ns2.xisu)
            A .= 1e-4 .* randn(rng, size(A))
        end
        for (s3, s2) in ((fluid3, fluid2), (phase3, phase2), (ns3, ns2)), n in fieldnames(typeof(s2))
            a2 = getfield(s2, n);  a3 = getfield(s3, n)
            a2 isa AbstractArray{T} && a3 isa AbstractArray{T, 3} && (a3 .= lift(a2, size(a3, 3)))
        end
        update!(phase2, fluid2, grid2, par, scales; xBC)
        update!(phase3, fluid3, grid3, par, scales; xBC, yBC)
        for (phase, fluid) in ((phase2, fluid2), (phase3, fluid3))
            phase.X .= phase.x .* fluid.rho;  phase.M .= phase.m .* fluid.rho
        end
        for (phase, fluid, ns, grid, kw) in ((phase2, fluid2, ns2, grid2, (; xBC)),
                                             (phase3, fluid3, ns3, grid3, (; xBC, yBC)))
            store_previous!(fluid, phase)
            update_phase_velocities!(phase, fluid, ns, grid, par; kw...)
        end
        n2 = phsevo!(phase2, fluid2, grid2, par, scales; xBC, coefs...)
        n3 = phsevo!(phase3, fluid3, grid3, par, scales; xBC, yBC, coefs...)
        @test n3 ≈ sqrt(Ny) * n2 rtol = 1e-12
        for (s3, s2) in ((fluid3, fluid2), (phase3, phase2)), n in fieldnames(typeof(s2))
            a2 = getfield(s2, n);  a3 = getfield(s3, n)
            a2 isa AbstractArray{T} || continue
            @test all(k -> isapprox(a3[:, :, k], a2; rtol = 1e-12, atol = 1e-15), axes(a3, 3))
        end
        @test all(iszero, phase3.Vx) && all(iszero, phase3.qy_advn_X) && all(iszero, phase3.qy_dffn_X)
    end
end
