# Geometric-multigrid (GMG) Stokes solver for xcore — a SELF-CONTAINED parallel
# copy of the single-grid DYREL solver (`src/dyrel/`) wrapped in a submodule so
# it can be developed without touching / breaking the working `src/dyrel/` path.
#
# The copied DR kernels (cache/gershgorin/stress/advection/residuals/velocity/
# bcs/reductions) provide the per-level SMOOTHER (`smoother.jl::smooth!`); the new
# files (transfer/hierarchy/vcycle) add the multigrid V-cycle on top. See
# `DESIGN_multigrid.md` for the full design.
#
# Names here (e.g. `DyrelCache`, the kernels) intentionally shadow the parent's —
# living in `module GMG` keeps them distinct (`GMG.DyrelCache` vs
# `xcore.DyrelCache`), so both solvers coexist.

module GMG

using KernelAbstractions
using LinearAlgebra
using Statistics
using Printf

# Shared state types + helpers from the parent xcore module (these are NOT
# duplicated — the GMG solver operates on the same FluidState/PhaseState).
using ..xcore: Grid, Parameters, Scales, FluidState, PhaseState,
               xcore_zeros, adapt_backend, advect_centered!, compute_corner_eta!

# --- copied single-grid DR kernels = the per-level smoother -------------------
include("cache.jl")
include("init.jl")
include("gershgorin.jl")
include("stress.jl")
include("advection.jl")
include("residuals.jl")
include("velocity.jl")
include("bcs.jl")
include("reductions.jl")

# --- geometric-multigrid layer (the WORKING GPU-friendly solver) --------------
include("transfer.jl")     # restrict/prolong kernels (p/w/u)
include("hierarchy.jl")    # MGLevel / MGHierarchy + coefficient coarsening
include("smoother.jl")     # smooth!(level; nsweeps) — truncated DR sub-solve + AL pressure update
include("vcycle.jl")       # V-cycle recursion + outer driver `fluidmech_gmg!`

end # module GMG
