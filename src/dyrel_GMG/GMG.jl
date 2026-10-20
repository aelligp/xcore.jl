# Geometric-multigrid (GMG) Stokes solver for xcore: a V-cycle wrapped around the
# single-grid DYREL solver in `src/dyrel/`. The DR kernels there ARE the per-level
# smoother (`smoother.jl::smooth!`) — they are imported, not copied, so the two
# solvers cannot drift apart. This submodule only adds what multigrid needs:
# transfer operators, the level hierarchy, the smoother driver, and the V-cycle.
#
# The smoother solves `A·v = b` for an arbitrary `b`, encoded by two knobs on the
# shared residual kernels: `fscale` (1 = physical forcing, 0 = homogeneous error
# equation) and `cache.bext_*` (injected restricted residual, zero on the fine
# level). At `fscale = 1, bext = 0` the smoother reduces exactly to the
# single-grid iteration. See `DESIGN_multigrid.md`.

module GMG

using KernelAbstractions
using LinearAlgebra
using Statistics
using Printf
using TimerOutputs: @timeit_debug

# State types + the single-grid DYREL solver kernels from the parent module.
# Nothing below is duplicated: both solvers run the same operator.
using ..xcore: TO, Grid, Parameters, Scales, FluidState, PhaseState,
               xcore_zeros, adapt_backend, advect_centered!, compute_corner_eta!,
               DyrelCache, compute_AL_penalty!, gershgorin!,
               update_dτ_α_β!, dyrel_momentum_advection!,
               dyrel_residual_V!, dyrel_residual_P!, dyrel_update_velocity!,
               dyrel_apply_velocity_bcs!, dyrel_apply_pressure_bcs!,
               dyrel_apply_pressure_gauge!,
               dyrel_residual_norms, dyrel_velocity_residual_norm,
               dyrel_rayleigh_λmin, _check_dyrel_preconditioner,
               ρ_want, γ_win_width, γ_max_retunes, γ_max_growth,
               compute_dyrel_residual_W_kernel!, compute_dyrel_residual_U_kernel!,
               compute_dyrel_residual_P_kernel!

include("transfer.jl")     # restrict/prolong kernels (p/w/u)
include("hierarchy.jl")    # MGLevel / MGHierarchy + coefficient coarsening
include("smoother.jl")     # smooth!(level; nsweeps) — truncated DR sub-solve + AL pressure update
include("vcycle.jl")       # V-cycle recursion + outer driver `fluidmech_gmg!`
include("velocity_vcycle.jl")  # velocity-block V-cycle at fixed pressure

end # module GMG
