# Dimension-generic matrix-free Stokes/Navier–Stokes solvers: one implementation
# for D = 2 and D = 3 (`FluidState{T,D}`, `Grid{T,D}`). Kernels
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
using Printf
using TimerOutputs: @timeit_debug
using LinearAlgebra: pinv

using ..xcore: TO, Grid, Parameters, Scales, FluidState,
               xcore_zeros, interior, face_interior,
               velocity, edge_viscosity, face_density, momentum_lag1, momentum_lag2,
               face_mean!, edge_geomean!, foreach_axis,
               update_kinematics!, update_viscosity!,
               ADVN_SCHEMES, face_fluxes, ghost_index, ghost_face_index,
               unval, unit_index, replace_index, hz_axes, hz_vaxes, tuple_foldl

import ..xcore: cells

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
