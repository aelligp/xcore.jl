using xcore
using KernelAbstractions: CPU
using Printf

# Mass-conservation stress test for the MULTIGRID pressure-projection solver
# (par.solver = :mg) on a REAL setup: the D1_dm2_e1 physics (compressible
# two-phase Navier-Stokes, Re>1), reduced to N=64 + a few steps for a quick
# turnaround. Reports the fractional conservation error EB per step (should stay
# at solver-floor levels, like the direct/dyrel solvers).
#
# Run:  julia --project=. usr/run_mg_conservation_test.jl

solver = get(ENV, "XCORE_SOLVER", "gmg") |> Symbol     # :gmg | :direct | :dyrel
N      = parse(Int, get(ENV, "XCORE_N", "64"))
nsteps = parse(Int, get(ENV, "XCORE_NSTEPS", "5"))

par = Parameters(Float64;
    runID   = "mg_cnsv_test",
    outdir  = joinpath(@__DIR__, "..", "out"),
    save_op = false,
    nrh     = 1,                # record history (EB) every step
    restart = 0,

    # domain (real D1_dm2_e1 setup)
    D  = 1e1,
    N  = N,
    L  = 1e1 * 1.5,             # 1.5 × D

    t0end = 1e6,                # don't stop on time — nsteps is the limiter

    # physics (real)
    d0    = 1e-2,
    etam0 = 1e1,
    L0    = 1e1 / 100,
    l0    = 1e-2 * 10,
    Da    = 0.01,
    Xi    = 0.5,

    solver = solver,

    # numerics
    CFL   = 0.9,
    γfact_PT = 20,              # JustRelax-standard (Schur-diagonal makes 1e3 unnecessary)
    pres_relax_PT = parse(Float64, get(ENV, "XCORE_OMEGA", "0.3")),  # SIMPLE under-relaxation of the MG pressure projection
    rtol  = 1e-4,
    atol  = 1e-9,
    maxit = 150,
    alpha = 0.9,
    gamma = 1e-3,
    TINT  = :bd2im,
    ADVN  = :weno5,
)

grid   = Grid(par)
scales = compute_scales(par, grid)
fluid  = FluidState(Float64, CPU(), grid.Nz, grid.Nx)
phase  = PhaseState(Float64, CPU(), grid.Nz, grid.Nx)
ns     = NoiseState(Float64, grid, scales)
hst    = History(Float64)

@printf("\n=== MG conservation test: solver=%s  grid=%d×%d  nsteps=%d ===\n",
        solver, grid.Nz, grid.Nx, nsteps)

time0, dt0, step0 = initialize!(phase, fluid, ns, hst, grid, par, scales)

final_time, final_dt = run!(phase, fluid, ns, hst, grid, par, scales;
                            time = time0, dt = dt0, step = step0,
                            nsteps = nsteps, verbose = true)

@printf("\n--- conservation summary (solver=%s) ---\n", solver)
@printf("%4s  %14s  %12s  %12s  %12s\n", "step", "sumB", "EB", "EM", "EX")
for k in eachindex(hst.EB)
    @printf("%4d  %14.6e  %12.3e  %12.3e  %12.3e\n",
            k, hst.sumB[k], hst.EB[k], hst.EM[k], hst.EX[k])
end
@printf("\nmax |EB| = %.3e   final W finite=%s  P finite=%s\n",
        maximum(abs, hst.EB), all(isfinite, fluid.W), all(isfinite, fluid.P))
