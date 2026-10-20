# Repeated-solve profiler for every fluid-mechanics solver strategy.
#
# The matrix-free solvers take no MMS source terms, so the probe problem is the
# production step-1 state after `initialize!` (as in usr/run_gmg_harness.jl).
# Each rep restores fluid/phase arrays IN PLACE from that frozen state (the GMG
# hierarchy wraps the finest-level `fluid`/`phase`, so they cannot be replaced)
# and calls `solve_fluidmech!` once — one Picard sweep at production settings.
#
# Every solve is timed as a whole and broken down by the solvers' internal
# `@timeit_debug` regions: assemble/merge/scale/factor/solve for the assembled
# solvers; setup, PH/DR residuals, retunes and multiplier updates for :dyrel;
# per-level smoothing, residual, transfer and coarse solve for :gmg/:vmg.
# The regions do not synchronize the device, so on a GPU backend they measure
# kernel launches, not execution; profile the breakdown on the CPU.
#
# :gmg/:vmg need N to be a multiple of 64. Run from the repo root:
#   julia --project=. -t 8 usr/run_prof_solvers.jl
# Optional overrides:
#   XCORE_NS=64,128 XCORE_NREP=10 XCORE_SOLVERS=direct,vmg XCORE_FORCING=0 julia --project=. usr/run_prof_solvers.jl

using xcore
using KernelAbstractions: CPU
using TimerOutputs
using Printf

const T = Float64

NS      = parse.(Int, split(get(ENV, "XCORE_NS", "64,128,256"), ","))
NREP    = parse(Int, get(ENV, "XCORE_NREP", "10"))
SOLVERS = Symbol.(split(get(ENV, "XCORE_SOLVERS", "direct,dyrel,gmg,vmg"), ","))
# Production inexact-solve forcing (see `Parameters.forcing_PH`); 0 solves every
# sweep to `atol_PH`, the like-for-like comparison against :direct.
FORCING = parse(T, get(ENV, "XCORE_FORCING", "0.1"))

base_par(N, solver) = Parameters(T;
    runID = "prof_solvers", outdir = mktempdir(), save_op = false,
    nop = 10^6, nrh = 10^6,
    D = 1e1, N = N, L = 1.5e1, t0end = 2.0,
    d0 = 1e-2, etam0 = 1e1, L0 = 1e-1, l0 = 1e-1, Da = 0.01, Xi = 0.5,
    atol = 1e-6, forcing_PH = FORCING, solver)

"Copy every array field of `src` into the matching field of `dst`."
function restore!(dst, src)
    for f in fieldnames(typeof(src))
        a = getfield(src, f)
        a isa AbstractArray && copyto!(getfield(dst, f), a)
    end
    return dst
end

const BCS = (xBC = :periodic, zBC = :closed, sds = -1,
             top_cnv = 1, bot_cnv = 1, open_cnv = false)

enable_solver_timings()

for N in NS
    par0   = base_par(N, :direct)
    grid   = Grid(par0)
    scales = compute_scales(par0, grid)
    fluid0 = FluidState(T, CPU(), grid.Nz, grid.Nx)
    phase0 = PhaseState(T, CPU(), grid.Nz, grid.Nx)
    ns     = NoiseState(T, grid, scales)
    _, dt, _ = initialize!(phase0, fluid0, ns, History(T), grid, par0, scales)
    coefs  = NamedTuple{(:a1, :a2, :a3, :b1, :b2, :b3)}(time_coefs(T, par0.TINT, 1))

    walls = Dict{Symbol,Float64}()
    for s in SOLVERS
        par    = base_par(N, s)
        fluid  = deepcopy(fluid0);  phase = deepcopy(phase0)
        solver = xcore.make_solver(par, fluid, phase, grid; BCS.xBC)
        sweep!() = (restore!(fluid, fluid0); restore!(phase, phase0);
                    xcore.solve_fluidmech!(solver, fluid, phase, grid, par, scales;
                                     dt, coefs..., BCS...))

        # warm-up: JIT + one-time sparsity/symbolic-LU/hierarchy costs
        sweep!(); sweep!()
        reset_timer!(TO)
        nconv = 0
        for _ in 1:NREP
            nconv += @timeit TO "solve_fluidmech! total" sweep!()
        end
        walls[s] = TimerOutputs.time(TO["solve_fluidmech! total"]) / 1e9 / NREP

        @printf("\n========== %-10s N = %d  (%d sweeps, %d reached atol_PH) ==========\n",
                s, N, NREP, nconv)
        show(TO; allocations = true, sortby = :firstexec)
        println()
    end

    @printf("\n---------- summary N = %d: mean wall per sweep ----------\n", N)
    for s in SOLVERS
        @printf("  %-10s %9.4f s   (%.2fx direct)\n", s, walls[s],
                haskey(walls, :direct) ? walls[s] / walls[:direct] : NaN)
    end
end

disable_solver_timings()
