# Dimension-generic matrix-free Stokes/Navier–Stokes solvers: one implementation
# for D = 2 (`FluidState`, `Grid`) and D = 3 (`FluidState3D`, `Grid3D`). Kernels
# run over `CartesianIndex{D}` with compile-time unit offsets; the velocity is the
# tuple `velocity(fluid)` with one component per axis. See `layout.jl` for the
# axis conventions and array shapes.
#
#   fluidmech_dyrel!  dynamic relaxation inside a Powell–Hestenes iteration
#   fluidmech_gmg!    coupled geometric-multigrid V-cycles, DR sweep as smoother
#   fluidmech_vmg!    Powell–Hestenes with velocity V-cycles and a Cahouet–Chabard
#                     pressure step (one matrix-free Poisson V-cycle)

module Stokes

using KernelAbstractions
using LinearAlgebra: pinv
using Printf
using TimerOutputs: @timeit_debug

using ..xcore: TO, Grid, Grid3D, Parameters, Scales, FluidState, FluidState3D,
               xcore_zeros, coarsen_grid3d,
               update_kinematics!, update_viscosity!,
               scheme_halo, _weno3_poly, _weno5_poly, _tvd_flux

include("layout.jl")
include("cache.jl")
include("residuals.jl")
include("gershgorin.jl")
include("bcs.jl")
include("reductions.jl")
include("advection.jl")
include("transfer.jl")
include("dyrel.jl")
include("hierarchy.jl")
include("gmg.jl")
include("vmg.jl")

end # module Stokes
