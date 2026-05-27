module xcore

using LinearAlgebra
using Printf
using Random
using Statistics
using TOML, Crayons

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
include("plotting.jl")

export Parameters, Grid, Scales, VisScales
export compute_scales, compute_vis_scales, print_scales
export xcore_zeros, interior
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
    isa(stdout, Base.TTY) || return
    _print_banner(io)
    return nothing
end

end # module xcore
