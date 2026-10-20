using Test, xcore, LinearAlgebra
using xcore: fluidmech!, DirectMGSolver, DirectMGStats
isdefined(@__MODULE__, :sol_setup) || include("sol_benchmarks.jl")

# :gmg_direct solves the same assembled system as :direct, so both must agree
# to the FGMRES tolerance, viscous (a1 = 0) and with inertia (Cahouet–Chabard
# inertial term active).
@testset "gmg_direct matches direct" begin
    for (a1, dt) in ((0.0, 1.0e32), (1.0, 1.0e-2))
        par, grid, fluid, phase, _ = sol_setup(Float64, 32, :solcx; Δη = 1.0e6)
        kw = (; dt, a1, a2 = a1, a3 = 0.0, sds = -1, top_cnv = -1, bot_cnv = -1,
                xBC = :closed, zBC = :closed)
        fd = deepcopy(fluid);  fm = deepcopy(fluid)
        fluidmech!(fd, grid, par; phase = deepcopy(phase), kw...)
        s = DirectMGSolver(1.0e-10, 300, DirectMGStats(false, 0))
        fluidmech!(fm, grid, par; phase = deepcopy(phase), mg = s, kw...)
        @test s.stats.converged
        @test norm(fm.W - fd.W) / norm(fd.W) < 1.0e-7
        @test norm(fm.U - fd.U) / norm(fd.U) < 1.0e-7
    end
end
