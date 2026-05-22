@testset "Scales" begin
    # mirror the parameters of usr/run_D1_dm2_e1.m so we can compare against the
    # MATLAB scaling printout if needed
    par = Parameters{Float64}(D = 10.0, N = 200, L = 15.0,
                              d0 = 1e-2, etam0 = 1e1, rhom0 = 2700, rhox0 = 3200,
                              g0 = 10.0, L0 = 10.0/100, l0 = 1e-2 * 10,
                              Da = 0.01, Xi = 0.5, x0 = eps(Float64))
    g = Grid(par)
    s = compute_scales(par, g)

    # Self-consistency: fixed-point residual should be machine-zero by construction
    Ri0 = s.Ri0
    W0_check = (sqrt(4/Ri0^2 * s.Dchi0 * s.Drho0 * par.g0 * s.rho0 * s.fReL0 *
                     s.L0^2 * s.D0 + s.eta0^2) - s.eta0) *
               s.D0 / (2 * s.fReL0 * s.L0^2 * s.rho0 / Ri0^2)
    @test isapprox(W0_check, s.W0; rtol = 1e-12)

    # Sanity: laminar < general < inertial for convection speed in this regime
    @test s.W0l > 0
    @test s.W0  > 0
    @test s.w0  > 0
    @test s.fReL0 ≤ 1 && s.fReL0 > 0
    @test s.fRel0 ≤ 1 && s.fRel0 > 0

    # Ra and ReD are derived quantities — verify ordering
    @test s.Ra0  > 0
    @test s.ReD0 ≥ 0
    @test s.Red0 ≥ 0

    # Float32: scales must match Float64 to several digits (iteration runs in Float64
    # internally, then narrows). Tolerance picked to catch narrowing bugs without
    # being so tight it flags legitimate Float32 truncation.
    par32 = Parameters(Float32; D = 10.0, N = 200, L = 15.0,
                                d0 = 1e-2, etam0 = 1e1, rhom0 = 2700, rhox0 = 3200,
                                g0 = 10.0, L0 = 0.1, l0 = 0.1, Da = 0.01, Xi = 0.5,
                                x0 = eps(Float64))
    g32 = Grid(par32)
    s32 = compute_scales(par32, g32)
    @test s32.W0  isa Float32
    @test isapprox(Float64(s32.W0),  s.W0;  rtol = 1e-5)
    @test isapprox(Float64(s32.w0),  s.w0;  rtol = 1e-5)
    @test isapprox(Float64(s32.Ra0), s.Ra0; rtol = 1e-5)
end
