using xcore
using KernelAbstractions: CPU

# Port of usr/run_D1_dm2_e1.m
# D = 1e1 m  |  d0 = 1e-2 m  |  etam0 = 1e1 Pa·s
# Runs ~2 dimensionless time units with 200×300 grid.

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

par = Parameters(Float64;
    runID   = "D1_dm2_e1_N150",
    outdir  = joinpath(@__DIR__, "..", "out"),
    save_op = true,
    nop     = 10,          # save figures every nop steps
    nrh     = 1,            # record history every step
    restart = 0,            # set to frame number to restart from a specific checkpoint; -1 for most recent

    # domain
    D  = 1e1,
    N  = 150,
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

    # numerics
    CFL   = 0.5,
    rtol  = 1e-4,
    atol  = 1e-9,
    maxit = 15,
    alpha = 0.9,
    gamma = 1e-3,
    TINT  = :bd2im,
    ADVN  = :weno5,
)

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

grid   = Grid(par)
scales = compute_scales(par, grid)
print_scales(scales, par)

fluid  = FluidState(Float64, CPU(), grid.Nz, grid.Nx)
phase  = PhaseState(Float64, CPU(), grid.Nz, grid.Nx)
ns     = NoiseState(Float64, grid, scales)
hst    = History(Float64)

time0, dt0, step0 = initialize!(phase, fluid, ns, hst, grid, par, scales)

final_time, final_dt = run!(phase, fluid, ns, hst, grid, par, scales;
                            time = time0, dt = dt0, step = step0,
                            verbose = true)
println("\nDone. final_time = $final_time s  ($(round(final_time/scales.t0, sigdigits=4)) t₀)")
