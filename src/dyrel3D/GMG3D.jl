# Geometric-multigrid Stokes/Navier-Stokes solver in 3-D — the dimension lift of
# `src/dyrel_GMG/`. SELF-CONTAINED submodule (parallel to the 2-D `module GMG`),
# so the 3-D path can be developed without touching the working 2-D solvers.
#
# Phase 1 (this scaffold): the validated operator core only — the residual /
# Schur-diagonal / Powell-Hestenes kernels from `schur.jl`, now callable on the
# real `FluidState3D` / `DyrelCache3D` structs (field-name compatible).
#
# Still to come (see the memory plan `project_3d_implementation`):
#   Phase 2 — gershgorin3d, init3d, stress3d, velocity3d, bcs3d, reductions3d,
#             advection3d  (the per-level DR SMOOTHER)
#   Phase 3 — transfer3d, hierarchy3d, vcycle3d  (the V-cycle)
#   Phase 4 — 3-D update!/phsevo!, run3d! driver, Parameters3D
#   Phase 5 — save_vtk (WriteVTK), JLD2 checkpointing, ImplicitGlobalGrid MPI

module GMG3D

using KernelAbstractions
using LinearAlgebra
using Statistics
using Printf

# Shared 3-D state types from the parent module (NOT duplicated). The schur.jl
# kernel wrappers take dot-accessible structs, so importing the names documents
# intent even though the wrappers stay duck-typed.
using ..xcore: Grid3D, FluidState3D, PhaseState3D, DyrelCache3D

# --- validated operator core (W/U/V/P residuals, Schur diagonal, PH update) ---
include("schur.jl")

end # module GMG3D
