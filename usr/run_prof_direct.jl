# Repeated-solve profiler for the DIRECT (sparse-LU) fluid-mechanics solver.
#
# The VP benchmark calls `fluidmech!` once per N, so it only ever measures the
# FIRST call — which pays the one-time `sparse()` pattern build + UMFPACK
# symbolic factorization. Production runs do MANY Picard sweeps at the same
# (Nz, Nx), where those one-time costs are amortized and the steady-state cost
# is: assemble (scatter values) → merge `[KV GG; DM KP]` → scale `SCL*LL*SCL`
# → numeric refactor (`lu!`) → solve (`\`).
#
# This script drives that steady state and prints a TimerOutputs breakdown so we
# can see whether assembly/merge/scale (phase A) or the LU factor+solve
# (phase B) dominates — which gates the optimization plan.
#
# Run from the repo root:
#   julia --project=. usr/run_prof_direct.jl
# Optionally override the grid sweep and repetitions:
#   XCORE_NS=100,200 XCORE_NREP=30 julia --project=. usr/run_prof_direct.jl

using xcore
using KernelAbstractions: CPU
using LinearAlgebra
using TimerOutputs
using Printf

include(joinpath(@__DIR__, "..", "test", "mms_sources.jl"))

const T = Float64
const L = T(10)

NS   = parse.(Int, split(get(ENV, "XCORE_NS", "100,200,400"), ","))
NREP = parse(Int, get(ENV, "XCORE_NREP", "50"))

"""Build an MMS-loaded FluidState ready for repeated `fluidmech!` calls."""
function setup(N)
    mms = mms_sources(T, N, N, L)
    par = Parameters(T; D = L, N = N, L = L,
                     etam0 = 1e3, g0 = 10.0,
                     rhom0 = 2700, rhox0 = 3200, d0 = 1e-2,
                     Da = 0.0, Xi = 0.0, gamma = 0.0)
    grid  = Grid(par)
    fluid = FluidState(T, CPU(), grid.ni[1], grid.ni[2])
    fluid.eta   .= mms.eta;   fluid.etaco .= mms.etaco
    fluid.rho   .= mms.rho;   fluid.rhow  .= mms.rhow;  fluid.rhou .= mms.rhou
    fluid.Drho  .= mms.Drho;  fluid.MFS   .= mms.MFS
    bnchm = BnchmData{T, typeof(mms.src_W)}(
        mms.src_W, mms.src_U, mms.src_P,
        mms.W_exact, mms.U_exact, mms.P_exact)
    return par, grid, fluid, bnchm
end

solve!(fluid, grid, par, bnchm) = fluidmech!(fluid, grid, par;
    bnchm_data = bnchm, sds = -1, top_cnv = -1, bot_cnv = -1,
    open_cnv = false, dt = T(1e32), a1 = T(1), a2 = T(1), a3 = T(0))

enable_solver_timings()

for N in NS
    par, grid, fluid, bnchm = setup(N)

    # warm up: first call builds the sparsity + symbolic LU (one-time cost we
    # explicitly do NOT want in the steady-state numbers); a couple more let the
    # JIT settle. Reset the timer afterwards so only steady-state sweeps count.
    solve!(fluid, grid, par, bnchm)
    solve!(fluid, grid, par, bnchm)
    reset_timer!(TO)

    for _ in 1:NREP
        solve!(fluid, grid, par, bnchm)
    end

    ndof = (N + 2)^2 + 2 * (N + 1) * (N + 2)
    @printf("\n========== direct solver  N = %d  (%d dofs, %d steady sweeps) ==========\n",
            N, ndof, NREP)
    show(TO; allocations = true, sortby = :firstexec)
    println()
end

disable_solver_timings()
