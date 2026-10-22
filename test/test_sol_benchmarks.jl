using Test
include("sol_benchmarks.jl")

# Analytic Stokes benchmarks on the unit box with free slip on all four
# boundaries. See sol_benchmarks.jl for the setup.

_orders(errs) = [log2(errs[i][k] / errs[i + 1][k]) for i in 1:2, k in (:U, :W, :P)]

@testset "SolKz: analytic convergence at 1e6 viscosity range" begin
    # η = exp(log(Δη)·y) is smooth, so the staggered discretization is fully
    # second order even at the extreme viscosity range.
    errs = [sol_errors_for(N, :solkz) for N in (32, 64, 128)]
    orders = _orders(errs)
    @info "SolKz L2 errors" errs orders
    @test all(>(1.8), orders)
end

@testset "SolCx: analytic convergence at a 1e6 viscosity step" begin
    # A discontinuous η limits the staggered scheme to first order in L2 for the
    # velocities; the assertion guards that rate, not the second-order one.
    errs = [sol_errors_for(N, :solcx) for N in (32, 64, 128)]
    orders = _orders(errs)
    @info "SolCx L2 errors" errs orders
    @test all(>(0.9), orders)
end

@testset "SolCx: smooth limit is second order" begin
    # Δη = 1 removes the jump, leaving a smooth analytic Stokes solution that
    # pins the discretization order independently of the interface treatment.
    errs = [sol_errors_for(N, :solcx; Δη = 1.0) for N in (32, 64, 128)]
    orders = _orders(errs)
    @info "SolCx (Δη=1) L2 errors" errs orders
    @test all(>(1.9), orders)
end

@testset "free-slip walls: DYREL matches the direct solve" begin
    # The free-slip side boundaries are assembled in fluidmech.jl and applied
    # kernel-side in src/solvers/stokes/bcs.jl; agreement to well below the discretization
    # error is what ties the two implementations together.
    for case in (:solkz, :solcx)
        ref = sol_benchmark(Float64, 64, case, :direct)
        dyr = sol_benchmark(Float64, 64, case, :dyrel)
        @test dyr.info.converged
        @test norm(dyr.fluid.W .- ref.fluid.W) / norm(ref.fluid.W) < 1.0e-6
        @test norm(dyr.fluid.U .- ref.fluid.U) / norm(ref.fluid.U) < 1.0e-6
        @test isapprox(dyr.errs.U, ref.errs.U; rtol = 1.0e-3)
        @test isapprox(dyr.errs.W, ref.errs.W; rtol = 1.0e-3)
    end

end
