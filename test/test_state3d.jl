# 3-D state, grid and solver-cache layout: the shapes the dimension-generic
# Stokes kernels index, the per-axis accessors, and grid coarsening.
using Test
using xcore
import xcore as XC
using KernelAbstractions: CPU

let T = Float64, Nz = 8, Nx = 6, Ny = 4
    f = FluidState(T, CPU(), Nz, Nx, Ny)
    c = StokesCache(T, CPU(), (Nz, Nx, Ny))
    S = XC.Stokes

    @testset "3-D FluidState / StokesCache shapes" begin
        @test size(f.W) == (Nz + 1, Nx + 2, Ny + 2)
        @test size(f.U) == (Nz + 2, Nx + 1, Ny + 2)
        @test size(f.V) == (Nz + 2, Nx + 2, Ny + 1)
        @test size(f.P) == (Nz + 2, Nx + 2, Ny + 2)
        # compact face shapes N .+ e_d, edge viscosities N .+ e_a .+ e_b
        @test size.(c.Rv) == ((Nz + 1, Nx, Ny), (Nz, Nx + 1, Ny), (Nz, Nx, Ny + 1))
        @test size.(S.face_density(f)) == size.(c.Rv)
        @test size.(S.edge_viscosity(f)) == ((Nz + 1, Nx + 1, Ny), (Nz + 1, Nx, Ny + 1),
                                             (Nz, Nx + 1, Ny + 1))
        @test S.velocity(f) === (f.W, f.U, f.V)
        @test size(c.R_P) == size(c.γ_eff) == (Nz, Nx, Ny)
    end

    @testset "3-D Grid, coarsening and the multigrid hierarchy" begin
        par = Parameters(T; N = 64, L = 1.0, D = 1.0)
        g  = Grid(par, Val(3))
        gc = S.coarsen(g)
        @test (gc.ni[1], gc.ni[2], gc.ni[3]) == (g.ni[1] ÷ 2, g.ni[2] ÷ 2, g.ni[3] ÷ 2)
        @test gc.h ≈ 2g.h
        fl = FluidState(T, CPU(), g)
        hier = build_hierarchy(fl, g, par; minlvl = 16)
        @test [S.cells(l.grid) for l in hier.levels] == [(64, 64, 64), (32, 32, 32), (16, 16, 16)]
        @test hier.levels[1].fluid === fl
        @test_throws ArgumentError build_hierarchy(FluidState(T, CPU(), 48, 48, 48),
                                                   Grid((48, 48, 48), (1.0, 1.0, 1.0)), par)
    end
end
