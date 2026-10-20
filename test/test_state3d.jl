# Phase-1 smoke test for the 3-D GMG scaffold: the real `FluidState3D` /
# `DyrelCache3D` structs must be field-compatible with the validated
# `src/dyrel3D/schur.jl` kernel wrappers (which the standalone test
# `test_schur3d.jl` checks for correctness on NamedTuples). Here we confirm the
# struct field names/shapes line up and the wrappers run end-to-end producing
# finite, correctly-shaped output.
using Test
using xcore
using KernelAbstractions: CPU
using Random
Random.seed!(7)

const GMG3D = xcore.GMG3D
be = CPU()
T  = Float64
Nz, Nx, Ny = 8, 6, 4

f = FluidState3D(T, be, Nz, Nx, Ny)
c = DyrelCache3D(T, be, Nz, Nx, Ny)

# random, well-conditioned fill (viscosities/densities/diagonals strictly > 0)
randfill!(A)  = (A .= randn(size(A)...); A)
posfill!(A)   = (A .= rand(size(A)...) .+ 1.0; A)
for fld in (f.W, f.U, f.V, f.P, f.Drho, f.MFS,
            f.rhoWo, f.rhoWoo, f.rhoUo, f.rhoUoo, f.rhoVo, f.rhoVoo)
    randfill!(fld)
end
for fld in (f.eta, f.etaxz, f.etayz, f.etaxy, f.rho, f.rhow, f.rhou, f.rhov)
    posfill!(fld)
end
for fld in (c.D_W, c.D_U, c.D_V, c.γ_eff, c.η_b)
    posfill!(fld)
end
for fld in (c.bext_W, c.bext_U, c.bext_V, c.bext_P, c.advn_mz, c.advn_mx, c.advn_my)
    randfill!(fld)
end
P0 = randn(Nz + 2, Nx + 2, Ny + 2)

@testset "FluidState3D / DyrelCache3D shapes" begin
    # must match the field table in the header of src/dyrel3D/schur.jl
    @test size(f.W)   == (Nz + 1, Nx + 2, Ny + 2)
    @test size(f.U)   == (Nz + 2, Nx + 1, Ny + 2)
    @test size(f.V)   == (Nz + 2, Nx + 2, Ny + 1)
    @test size(c.R_W) == (Nz + 1, Nx, Ny)
    @test size(c.R_U) == (Nz, Nx + 1, Ny)
    @test size(c.R_V) == (Nz, Nx, Ny + 1)
    @test size(c.s_P) == (Nz, Nx, Ny)
end

h, g0, dt = 0.37, 9.81, 0.7

@testset "dyrel3d kernels on real structs (dr = $dr)" for dr in (false, true)
    GMG3D.dyrel3d_residual_P!(c, f, h, P0, dt)
    GMG3D.dyrel3d_residual_V!(c, f, h, g0; dr, a1 = 1.5, a2 = -0.6, a3 = 0.1,
                              gamma = 1e-3, dt)
    @test all(isfinite, c.R_W)
    @test all(isfinite, c.R_U)
    @test all(isfinite, c.R_V)
    @test all(isfinite, c.R_P)
    @test all(isfinite, c.P_num)
end

@testset "compute_schur_diag3d!" begin
    GMG3D.compute_schur_diag3d!(c, f, h)
    @test all(isfinite, c.s_P)
    @test all(>(0), c.s_P)
end

@testset "dyrel3d_update_pressure! touches the interior only" begin
    Pbefore = copy(f.P)
    GMG3D.dyrel3d_update_pressure!(c, f; ω = 0.8)
    expect = Pbefore[2:end-1, 2:end-1, 2:end-1] .+ 0.8 .* c.γ_eff .* c.R_P
    @test f.P[2:end-1, 2:end-1, 2:end-1] ≈ expect atol = 1e-13
    # ghost ring untouched
    @test f.P[1, :, :] == Pbefore[1, :, :]
    @test f.P[:, 1, :] == Pbefore[:, 1, :]
    @test f.P[:, :, 1] == Pbefore[:, :, 1]
end

@testset "Grid3D + coarsen_grid3d" begin
    g  = Grid3D(xcore.Parameters(T; N = 32, solver = :gmg))
    gc = coarsen_grid3d(g)
    @test (gc.Nz, gc.Nx, gc.Ny) == (g.Nz ÷ 2, g.Nx ÷ 2, g.Ny ÷ 2)
    @test gc.h ≈ 2g.h
end
