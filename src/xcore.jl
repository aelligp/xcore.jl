module xcore

using LinearAlgebra
using Printf
using Random
using Statistics
using TOML, Crayons
using TimerOutputs

# Module-level timer for solver profiling. All timed regions use
# `@timeit_debug TO "..."`, which compiles to a no-op unless debug timings are
# enabled — so the default (production) path carries ZERO overhead. To profile:
#
#   xcore.enable_solver_timings(); reset_timer!(xcore.TO)
#   ... run solves ...
#   show(xcore.TO); println()
#
const TO = TimerOutput()

"""Enable `@timeit_debug` regions in xcore (e.g. the direct-solver phases)."""
enable_solver_timings()  = (TimerOutputs.enable_debug_timings(@__MODULE__); nothing)
"""Disable `@timeit_debug` regions, restoring zero-overhead production behaviour."""
disable_solver_timings() = (TimerOutputs.disable_debug_timings(@__MODULE__); nothing)

include("parameters.jl")
include("grid.jl")
include("scales.jl")

include("backend.jl")
include("kernels/stencils.jl")
include("kernels/ghosts.jl")
include("kernels/diffus.jl")
include("kernels/advect.jl")
include("state.jl")
include("fluidmech.jl")
# DYREL pseudo-transient Stokes solver (standalone — its cache is NOT a field
# of FluidState; construct a `DyrelCache` and pass it to `fluidmech_dyrel!`,
# mirroring JustRelax's `DYREL` + `solve_DYREL!` separation).
include("dyrel/cache.jl")
include("dyrel/init.jl")
include("dyrel/gershgorin.jl")
include("dyrel/stress.jl")
include("dyrel/advection.jl")
include("dyrel/residuals.jl")
include("dyrel/velocity.jl")
include("dyrel/bcs.jl")
include("dyrel/reductions.jl")
include("dyrel/solve.jl")
include("update.jl")
include("timing.jl")
include("noise.jl")
include("store.jl")
include("phsevo.jl")
include("history.jl")
include("diagnose.jl")
include("output.jl")
include("restart.jl")
include("run.jl")
# Geometric-multigrid Stokes solver (self-contained submodule; parallel to the
# single-grid dyrel/ path). Included last so all shared types + advect_centered!
# are defined before `module GMG` imports them.
include("dyrel_GMG/GMG.jl")
# Solver-strategy dispatch (needs both the single-grid solvers above and the GMG
# submodule); `run!` uses `make_solver`/`solve_fluidmech!` from here.
include("solver.jl")
include("plotting.jl")

export Parameters, Grid, Scales, VisScales
export compute_scales, compute_vis_scales, print_scales
export xcore_backend, xcore_zeros, interior
export ddx!, ddz!
export fill_ghosts!, embed_interior!
export diffus!, advect!
export FluidState, PhaseState, NoiseState, BnchmData, fluidmech!
export noise!, store_noise!
export update!, update_volume_fractions!, update_interpolations!,
       update_pressure!, update_rheology!, update_kinematics!,
       update_viscosity!, update_stresses!, update_dt
export time_coefs, store_previous!, update_phase_velocities!
export advect_centered!, diffus_centered!, phsevo!
export initialize!, run!
export History, record_history!
export StepResidual, snapshot!, compute_resnorm, report_iter, print_step!
export plot_fluid, plot_phase, plot_diffuse, plot_dimensionless, plot_profiles,
       plot_history, save_output
export plot_state, plot_mms_comparison
export save_checkpoint, load_checkpoint!, restart_path, resolve_restart
export DyrelCache, fluidmech_dyrel!
export TO, enable_solver_timings, disable_solver_timings

function _print_banner(io::IO)
    x = string(Crayon(foreground = (26, 12, 100)))
    c = string(Crayon(foreground = (44, 81, 146)))
    o = string(Crayon(foreground = (90, 139, 163)))
    r = string(Crayon(foreground = (179, 172, 149)))
    e = string(Crayon(foreground = (254, 242, 242)))
    res = string(Crayon(reset = true))

    str = """
     $(x)██╗  ██╗ $(c)██████╗ $(o)██████╗ $(r)██████╗ $(e)███████╗$(res)
     $(x)╚██╗██╔╝$(c)██╔════╝$(o)██╔═══██╗$(r)██╔══██╗$(e)██╔════╝$(res)
     $(x) ╚███╔╝ $(c)██║     $(o)██║   ██║$(r)██████╔╝$(e)█████╗  $(res)
     $(x) ██╔██╗ $(c)██║     $(o)██║   ██║$(r)██╔══██╗$(e)██╔══╝  $(res)
     $(x)██╔╝ ██╗$(c)╚██████╗$(o)╚██████╔╝$(r)██║  ██║$(e)███████╗$(res)
     $(x)╚═╝  ╚═╝ $(c)╚═════╝ $(o)╚═════╝ $(r)╚═╝  ╚═╝$(e)╚══════╝$(res)
     """
    printstyled(io, "\n\n", str, "\n",
"""
Version: $(TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))["version"])
Latest commit: $(try strip(read(`git log -1 --pretty=%B`, String)) catch _ "N/A" end)
Commit date: $(try strip(read(`git log -1 --pretty=%cd`, String)) catch _ "N/A" end)
""", bold=true, color=:default)
    return nothing
end

function __init__(io::IO = stdout)
    # Threaded BLAS — UMFPACK's dense kernels pick this up automatically;
    # 1.3-2× speedup on the Stokes factor at medium grids.
    LinearAlgebra.BLAS.set_num_threads(max(1, Threads.nthreads()))
    isa(stdout, Base.TTY) || return
    _print_banner(io)
    return nothing
end

end # module xcore
