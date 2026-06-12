using Statistics
using LinearAlgebra: norm
using xcore: Parameters, Grid, compute_scales, FluidState, PhaseState,
             DyrelCache, fluidmech_dyrel!, fluidmech!
import xcore as XC
using KernelAbstractions: CPU

# Phase-1 DYREL validation: steady incompressible variable-viscosity Stokes.
# We don't have MMS source terms wired into the DR residual yet, so these
# tests check the two solver-agnostic invariants:
#   1. the PH/DR iteration converges (residual drops below atol_PH),
#   2. the converged velocity field is divergence-free (incompressible),
#   3. no NaNs / blow-ups,
# on a buoyancy-driven blob with both constant and contrasted viscosity.

function _dyrel_setup(T = Float64; Npts = 48, ηcontrast = 1.0)
    par = Parameters(T; D = 1.0, N = Npts, L = 1.0, etam0 = 1.0, g0 = 1.0,
                     γfact_PT = 20.0, CFL_PT = 0.99, c_fact_PT = 0.5,
                     maxit_PH = 50, maxit_PT = 50_000, n_tune_PT = 100,
                     atol_PH = 1.0e-6, rel_drop_PT = 1.0e-2)
    grid   = Grid(par)
    scales = compute_scales(par, grid)
    Nz, Nx = grid.Nz, grid.Nx
    fluid  = FluidState(T, CPU(), Nz, Nx)
    phase  = PhaseState(T, CPU(), Nz, Nx)
    dyrel  = DyrelCache(T, CPU(), grid)

    fill!(fluid.eta, one(T))
    fill!(fluid.etaco, one(T))
    fill!(fluid.MFS, zero(T))
    # constant unit density: the compressible continuity residual ∇·(ρv)
    # reduces to ∇·v, recovering the incompressible-Stokes check
    fill!(fluid.rho, one(T))
    fill!(fluid.rhow, one(T))
    fill!(fluid.rhou, one(T))

    # optional viscosity contrast: a stiff central block
    if ηcontrast != 1.0
        for j in 1:Nz, i in 1:Nx
            zc = (j - 0.5) * grid.h / grid.D
            xc = (i - 0.5) * grid.h / grid.L
            if abs(xc - 0.5) < 0.15 && abs(zc - 0.5) < 0.15
                fluid.eta[j, i] = T(ηcontrast)
            end
        end
        # corner viscosity = geometric-mean-ish; simple injection is fine for the test
        for j in 1:Nz+1, i in 1:Nx+1
            zc = (j - 1) * grid.h / grid.D
            xc = (i - 1) * grid.h / grid.L
            if abs(xc - 0.5) < 0.15 && abs(zc - 0.5) < 0.15
                fluid.etaco[j, i] = T(ηcontrast)
            end
        end
    end

    # buoyancy blob on z-faces (Nz+1, Nx)
    for j in 1:Nz+1, i in 1:Nx
        zc = (j - 1) * grid.h / grid.D
        xc = (i - 0.5) * grid.h / grid.L
        fluid.Drho[j, i] = exp(-(((xc - 0.5) / 0.1)^2 + ((zc - 0.5) / 0.1)^2))
    end

    return fluid, phase, dyrel, grid, par, scales
end

@testset "DYREL constant-viscosity buoyancy Stokes" begin
    fluid, phase, dyrel, grid, par, scales = _dyrel_setup(Float64; Npts = 48)
    res = fluidmech_dyrel!(fluid, phase, dyrel, grid, par, scales;
                           dt = 1.0e30, top_cnv = -1, bot_cnv = -1, verbose = false)

    @test res.converged
    @test res.PH_iters ≤ par.maxit_PH

    # divergence-free check: recompute continuity residual (= -∇·v here)
    XC.dyrel_residual_P!(dyrel, fluid, grid, par, fluid.P, Inf)
    rms_div = sqrt(mean(abs2, dyrel.R_P))
    @test rms_div < 1.0e-4

    # sanity: finite, non-trivial flow
    @test all(isfinite, fluid.W)
    @test all(isfinite, fluid.U)
    @test all(isfinite, fluid.P)
    @test maximum(abs, fluid.W) > 0
end

@testset "DYREL viscosity-contrast buoyancy Stokes" begin
    fluid, phase, dyrel, grid, par, scales = _dyrel_setup(Float64; Npts = 48, ηcontrast = 1.0e3)
    res = fluidmech_dyrel!(fluid, phase, dyrel, grid, par, scales;
                           dt = 1.0e30, top_cnv = -1, bot_cnv = -1, verbose = false)

    @test res.converged
    XC.dyrel_residual_P!(dyrel, fluid, grid, par, fluid.P, Inf)
    @test sqrt(mean(abs2, dyrel.R_P)) < 1.0e-3
    @test all(isfinite, fluid.W) && all(isfinite, fluid.U)
end

@testset "DYREL matches direct fluidmech! velocity field" begin
    # Both solvers attack the SAME linear system: fluidmech! does one exact
    # Newton step (LL⁻¹·RR), DYREL iterates to it. Started from rest with
    # frozen advection (=0 from rest), matched coefficients ⇒ same (W,U).
    T = Float64;  Npts = 32
    par = Parameters(T; D = 1.0, N = Npts, L = 1.0, etam0 = 1.0, g0 = 1.0,
                     γfact_PT = 20.0, CFL_PT = 0.99, c_fact_PT = 0.5,
                     maxit_PH = 100, maxit_PT = 100_000, n_tune_PT = 100,
                     atol_PH = 1.0e-8, rel_drop_PT = 1.0e-3, gamma = 0.0)
    grid   = Grid(par);  scales = compute_scales(par, grid)
    Nz, Nx = grid.Nz, grid.Nx
    dt = 1.0e6   # large but finite (inertia ≈ 0); avoids 0·Inf in sailor

    function _setup!(fluid)
        fill!(fluid.eta, one(T));   fill!(fluid.etaco, one(T))
        fill!(fluid.rho, one(T));   fill!(fluid.rhow, one(T));  fill!(fluid.rhou, one(T))
        fill!(fluid.rhoo, one(T));  fill!(fluid.rhooo, one(T))
        fill!(fluid.MFS, zero(T))
        fill!(fluid.rhoWo, zero(T)); fill!(fluid.rhoWoo, zero(T))
        fill!(fluid.rhoUo, zero(T)); fill!(fluid.rhoUoo, zero(T))
        fill!(fluid.W, zero(T));    fill!(fluid.U, zero(T));    fill!(fluid.P, zero(T))
        fill!(fluid.drhodt, zero(T)); fill!(fluid.drhodto, zero(T)); fill!(fluid.drhodtoo, zero(T))
        for j in 1:Nz+1, i in 1:Nx
            zc = (j - 1) * grid.h;  xc = (i - 0.5) * grid.h
            fluid.Drho[j, i] = exp(-(((xc - 0.5) / 0.1)^2 + ((zc - 0.5) / 0.1)^2))
        end
        return fluid
    end

    # --- direct solver (one exact Newton step) ---
    fluid_d = FluidState(T, CPU(), Nz, Nx);  phase_d = PhaseState(T, CPU(), Nz, Nx)
    _setup!(fluid_d);  fill!(phase_d.advn_rho, zero(T))   # res_ρ = 0 ⇒ MFS unchanged
    fluidmech!(fluid_d, grid, par; phase = phase_d, dt = dt, a1 = 1, a2 = 1, a3 = 0,
               sds = -1, top_cnv = -1, bot_cnv = -1, xBC = :periodic, zBC = :closed)

    # --- DYREL (iterate to convergence) ---
    fluid_y = FluidState(T, CPU(), Nz, Nx);  phase_y = PhaseState(T, CPU(), Nz, Nx)
    dyrel   = DyrelCache(T, CPU(), grid)
    _setup!(fluid_y)
    fluidmech_dyrel!(fluid_y, phase_y, dyrel, grid, par, scales; dt = dt,
                     a1 = 1, a2 = 1, a3 = 0, top_cnv = -1, bot_cnv = -1, verbose = false)

    Wd = @view fluid_d.W[:, 2:end-1];  Wy = @view fluid_y.W[:, 2:end-1]
    Ud = @view fluid_d.U[2:end-1, :];  Uy = @view fluid_y.U[2:end-1, :]
    relW = norm(Wy .- Wd) / (norm(Wd) + eps(T))
    relU = norm(Uy .- Ud) / (norm(Ud) + eps(T))
    @info "DYREL vs direct" relW relU
    @test relW < 1.0e-3
    @test relU < 1.0e-3
end
