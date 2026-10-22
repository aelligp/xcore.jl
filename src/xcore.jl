module xcore

using LinearAlgebra
using Printf
using Random
using Statistics
using TOML, Crayons
using TimerOutputs
using WriteVTK: vtk_grid, VTKCellData, VTKPointData, paraview_collection, collection_add_timestep

# Solver profiling timer. Timed regions use `@timeit_debug TO "..."`, a no-op
# unless debug timings are enabled. To profile:
#
#   xcore.enable_solver_timings(); reset_timer!(xcore.TO)
#   ... run solves ...
#   show(xcore.TO); println()
#
const TO = TimerOutput()

"""Enable the `@timeit_debug` regions of xcore and its Stokes submodule."""
enable_solver_timings()  = (foreach(TimerOutputs.enable_debug_timings, (@__MODULE__, Stokes)); nothing)
"""Disable the `@timeit_debug` regions."""
disable_solver_timings() = (foreach(TimerOutputs.disable_debug_timings, (@__MODULE__, Stokes)); nothing)

include("setup/parameters.jl")
include("setup/grid.jl")
include("setup/scales.jl")

include("setup/backend.jl")
include("kernels/stencils.jl")
include("kernels/diffus.jl")
include("kernels/advect.jl")
include("kernels/averages.jl")
include("setup/state.jl")
include("solvers/direct.jl")
include("physics/update.jl")
include("driver/timing.jl")
include("physics/noise.jl")
include("physics/store.jl")
include("physics/phsevo.jl")
include("driver/history.jl")
include("driver/diagnose.jl")
include("io/output.jl")
include("io/vtk.jl")
include("io/restart.jl")
include("driver/run.jl")
# Matrix-free solvers (:dyrel, :gmg, :vmg) for D = 2 and 3; they import kernels
# from update.jl and phsevo.jl, so they are included after those.
include("solvers/stokes/Stokes.jl")
using .Stokes: StokesCache, DyrelCache, fluidmech_dyrel!, fluidmech_gmg!, fluidmech_vmg!,
               build_hierarchy
# Solver dispatch: `make_solver` and `solve_fluidmech!`, used by `run!`.
include("solvers/dispatch.jl")
include("io/plotting.jl")

export Parameters, Grid, cells, Scales, VisScales
export compute_scales, compute_vis_scales, print_scales
export xcore_zeros, interior
export ddx!, ddz!
export diffus!, diffus_with_flux!, advect!, advect_with_flux!
export FluidState, PhaseState, NoiseState, BnchmData, fluidmech!
export noise!, store_noise!
export update!, update_volume_fractions!, update_interpolations!,
       update_pressure!, update_rheology!, update_kinematics!,
       update_viscosity!, update_stresses!, update_dt
export time_coefs, store_previous!, update_phase_velocities!
export phsevo!
export initialize!, run!
export History, record_history!, time_average
export StepResidual, snapshot!, compute_resnorm, report_iter, print_step!
export plot_fluid, plot_phase, plot_diffuse, plot_dimensionless, plot_profiles,
       plot_history, save_output
export plot_state, plot_mms_comparison
export save_checkpoint, load_checkpoint!, restart_path, resolve_restart
export save_vtk
export StokesCache, DyrelCache, fluidmech_dyrel!, fluidmech_gmg!, fluidmech_vmg!, build_hierarchy
export TO, enable_solver_timings, disable_solver_timings

function print_banner(io::IO)
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
    # UMFPACK's dense kernels use the BLAS threads.
    LinearAlgebra.BLAS.set_num_threads(max(1, Threads.nthreads()))
    isa(stdout, Base.TTY) || return
    print_banner(io)
    return nothing
end

end # module xcore
