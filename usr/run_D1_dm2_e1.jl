using xcore
using KernelAbstractions: CPU

# Port of usr/run_D1_dm2_e1.m
# D = 1e1 m  |  d0 = 1e-2 m  |  etam0 = 1e1 Pa·s
# Runs ~2 dimensionless time units with 200×300 grid.

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

par = Parameters(Float64;
    runID   = "D1_dm2_e1_N50",
    outdir  = joinpath(@__DIR__, "..", "out"),
    save_op = true,
    nop     = 10,          # save figures every nop steps
    nrh     = 1,            # record history every step
    restart = 0,            # set to frame number to restart from a specific checkpoint; -1 for most recent

    # domain
    D  = 1e1,
    N  = 50,
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
    rtol  = 1e-6,
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

initialize!(phase, fluid, ns, grid, par, scales)

# ---------------------------------------------------------------------------
# Output callback: record history + save figures
# ---------------------------------------------------------------------------

function output_callback(step, time, dt, phase, fluid, ns)
    if step % par.nrh == 0
        record_history!(hst, time, dt, phase, fluid, ns, grid)
    end
    if par.save_op && step % par.nop == 0
        frame = step ÷ par.nop
        save_output(phase, fluid, ns, hst, grid, par, scales, time;
                    outdir = par.outdir, runID = par.runID,
                    frame, dt, step)   # dt + step go into the JLD2 checkpoint
    end
end

# To restart from the most recent checkpoint, uncomment:
# load_checkpoint!(restart_path(par.outdir, par.runID; frame = -1),
#                  phase, fluid, ns, hst)
# Or from a specific frame:  ...; frame = 12), phase, fluid, ns, hst)

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

tend   = par.t0end * scales.t0   # dimensional stop time
nsteps = ceil(Int, tend / scales.dt0) + 10   # upper bound; driver exits early

println("Running $(par.runID) for ≤ $nsteps steps (tend = $(round(tend, sigdigits=3)) s)")

final_time, final_dt = run!(phase, fluid, ns, grid, par, scales;
    nsteps,
    verbose  = true,
    callback = (step, t, dt, ph, fl, ns_) -> begin
        output_callback(step, t, dt, ph, fl, ns_)
        t >= tend && error("stop")   # early exit when target time is reached
    end)

println("\nDone. final_time = $final_time s  ($(round(final_time/scales.t0, sigdigits=4)) t₀)")
