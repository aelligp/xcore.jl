@testset "Parameters" begin
    p64 = Parameters{Float64}()
    @test p64.D == 10.0
    @test p64.N == 100
    @test p64.ADVN === :weno5

    # Float32 narrowing
    p32 = Parameters{Float32}()
    @test p32.D isa Float32
    @test p32.D ≈ 10.0f0
    @test p32.AA isa Matrix{Float32}

    # convenience constructor with keyword overrides matches MATLAB driver pattern:
    # runID, D, N, etc. override par_default fields. Integer fields stay Int.
    p = Parameters(Float32; runID = "D1_dm2_e1_N100", D = 10, N = 100, L = 15.0,
                            d0 = 1e-2, etam0 = 1e1, t0end = 2.0,
                            CFL = 0.5, rtol = 1e-4, atol = 1e-9, maxit = 15,
                            alpha = 0.9, gamma = 1e-3)
    @test p.runID == "D1_dm2_e1_N100"
    @test p.D === 10.0f0
    @test p.N === 100
    @test p.L === 15.0f0
    @test p.maxit === 15
    @test p.etam0 === 10.0f0
end
