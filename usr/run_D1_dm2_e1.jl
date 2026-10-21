using xcore, KernelAbstractions
using CairoMakie   # save_op figures
# Backend: CPU(). For GPU, pass the GPU package's backend instead:
#   using CUDA;  be = CUDABackend()    # Float64 OK
#   using Metal; be = MetalBackend()   # Float32 only: also replace every Float64 below
be = CPU()

# Port of usr/run_D1_dm2_e1.m
# D = 1e1 m  |  d0 = 1e-2 m  |  etam0 = 1e1 Pa·s
# Runs ~2 dimensionless time units. N=128 ⇒ 128×192 grid (must be a multiple
# of 32 for the :gmg solver — N a multiple of 64 keeps Nx=1.5N also ÷32).

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

par = Parameters(Float64;
    runID   = "D1_dm2_e1_N1024_vmg",
    outdir  = joinpath(@__DIR__, "..", "out"),
    save_op = true,
    nop     = 10,          # save figures every nop steps
    nrh     = 1,            # record history every step
    restart = 0,            # set to frame number to restart from a specific checkpoint; -1 for most recent

    # domain
    D  = 1e1,
    N  = 1024,
    L  = 1e1 * 1.5,         # 1.5 × D

    # timing
    t0end = 2.0,            # stop at 2 dimensionless time units

    # physics
    d0    = 1e-2,
    etam0 = 1e1,
    L0    = 1e1 / 100,      # D/100
    l0    = 1e-2 * 10,      # d0×10
    Da    = 0.01,
    Xi    = 0.5,

    solver= :vmg,           # :direct (sparse-LU) | :dyrel (pseudo-transient) | :gmg (geometric MG) | :vmg (velocity MG + Cahouet–Chabard)
    atol = 1e-6,                # Picard convergence tolerance (nonlinear residual)
    # maxit_PH       = 100,                # outer Powell-Hestenes iterations
    # maxit_PT       = 50000,            # inner Dynamic-Relaxation iterations per PH step
    # total_iterMax_PT = 50000,          # global DR-iteration cap per fluidmech_dyrel! call (JustRelax `total_iterMax`); accept & let the outer Picard loop continue

    # --- geometric-multigrid (:gmg) controls — the dyrel sweep is the per-level
    # smoother; only the coarsest level is solved to tolerance. ---
    # mg_minlvl      = 16,                 # coarsen until min(Nz,Nx) would drop below this ⇒ coarsest stays ~16–31 cells/dim. 16² is the floor on purpose: the coarse grid must still RESOLVE the nonlinear rheology (power-law / T-dependent η) so the coarse correction sees the right operator (future development). Level count scales with N. GMG efficiency needs grids that coarsen cleanly to ~16 ⇒ resolution must be a multiple of 32 (asserted in build_hierarchy); awkward N (200=8·25) can't and is rejected.
    # mg_npre        = 2,                  # pre-smooth PH steps per level (V-cycle down)
    # mg_npost       = 2,                  # post-smooth PH steps per level (V-cycle up)
    # mg_ncoarse     = 20,                 # max coarsest-level PH steps (solve-to-tolerance budget)
    # mg_coarse_rtol   = 1e-4,            # coarsest-level relative residual drop (solve-to-tolerance)
    # mg_inner       = 32,                 # inner DR sweeps per smoothing PH step. WORK-OPTIMAL: there is a stability FLOOR (~32 on the buoyancy gate; below it the coarse correction amplifies ⇒ V-cycle DIVERGES) and above it more inner ⇒ fewer cycles, so total work (cycles·inner) is minimised just above the floor. Coarse-correction damping (γ_mg<1) does NOT lower the floor (it hurts). Much cheaper than a full sub-solve (≈2000).
    # mg_gamma         = 1,               # coarse-grid-correction damping γ_mg
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

grid   = Grid(par)
scales = compute_scales(par, grid)
print_scales(scales, par)

fluid  = FluidState(Float64, be, grid.Nz, grid.Nx)
phase  = PhaseState(Float64, be, grid.Nz, grid.Nx)
ns     = NoiseState(Float64, be, grid, scales)
hst    = History(Float64)

time0, dt0, step0 = initialize!(phase, fluid, ns, hst, grid, par, scales)

final_time, final_dt = run!(phase, fluid, ns, hst, grid, par, scales;
                            time = time0, dt = dt0, step = step0,
                            verbose = true)
println("\nDone. final_time = $final_time s  ($(round(final_time/scales.t0, sigdigits=4)) t₀)")
