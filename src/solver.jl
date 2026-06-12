# Fluid-mechanics solver strategy.
#
# A thin dispatch layer over the three Stokes/Navier-Stokes solvers so the Picard
# loop in `run!` calls ONE polymorphic `solve_fluidmech!` instead of branching on
# `par.solver`. Each strategy owns whatever persistent state its solver needs
# (caches, MG hierarchy), allocated once by `make_solver` and reused across all
# Picard sweeps and time steps.
#
#   :direct → sparse-LU            (DirectSolver,  no state)
#   :dyrel  → pseudo-transient AL  (DyrelSolver,   DyrelCache)
#   :gmg    → geometric multigrid  (GMGSolver,     MGHierarchy; dyrel sweep = smoother)

abstract type FluidSolver end

struct DirectSolver <: FluidSolver end
struct DyrelSolver{C} <: FluidSolver;          cache::C;        end
struct GMGSolver{H} <: FluidSolver;            hier::H;         end

"""
    make_solver(par, fluid, phase, grid) -> FluidSolver

Construct the solver strategy selected by `par.solver`, allocating its persistent
state once (reused across the whole run). The GMG hierarchy wraps `fluid`/`phase`
at the finest level, so it must be rebuilt only if those are reallocated.
"""
function make_solver(par::Parameters{T}, fluid::FluidState{T}, phase::PhaseState{T},
                     grid::Grid{T}) where {T}
    be = KernelAbstractions.get_backend(fluid.W)
    if par.solver === :direct
        return DirectSolver()
    elseif par.solver === :dyrel
        return DyrelSolver(DyrelCache(T, be, grid))
    elseif par.solver === :gmg
        return GMGSolver(GMG.build_hierarchy(fluid, phase, grid, par))
    else
        error("unknown par.solver = $(par.solver) (use :direct, :dyrel, or :gmg)")
    end
end

"""
    solve_fluidmech!(solver, fluid, phase, grid, par, scales; dt, a1..b3,
                     sds, top_cnv, bot_cnv, open_cnv, xBC, zBC) -> nothing

One fluid-mechanics solve (one Picard sweep) with the chosen strategy. All four
methods take the same arguments; each forwards the subset its solver uses.
"""
function solve_fluidmech!(::DirectSolver, fluid, phase, grid, par, scales;
                          dt, a1, a2, a3, b1, b2, b3,
                          sds, top_cnv, bot_cnv, open_cnv, xBC, zBC, verbose = false)
    fluidmech!(fluid, grid, par; phase, sds, top_cnv, bot_cnv, open_cnv, xBC, zBC,
               dt, a1, a2, a3, b1, b2, b3, verbose = verbose)
    return nothing
end

function solve_fluidmech!(s::DyrelSolver, fluid, phase, grid, par, scales;
                          dt, a1, a2, a3, b1, b2, b3,
                          sds, top_cnv, bot_cnv, open_cnv, xBC, zBC, verbose = false)
    st = fluidmech_dyrel!(fluid, phase, s.cache, grid, par, scales;
                          dt, a1, a2, a3, b1, b2, b3, gamma = par.gamma,
                          xBC, zBC, top_cnv, bot_cnv, open_cnv, verbose = verbose)
    # surface budget-capped solves: an unconverged fluidmech leaves O(err) in
    # the momentum balance that the Picard loop then sees as residual wobble
    st.converged || println("      !  dyrel solve hit budget (itPH=$(st.PH_iters), itPT total=$(st.total_iter))")
    return nothing
end

function solve_fluidmech!(s::GMGSolver, fluid, phase, grid, par, scales;
                          dt, a1, a2, a3, b1, b2, b3,
                          sds, top_cnv, bot_cnv, open_cnv, xBC, zBC, verbose = false)
    st = GMG.fluidmech_gmg!(fluid, phase, s.hier, grid, par, scales;
                            dt, a1, a2, a3, b1, b2, b3, gamma = par.gamma,
                            xBC, zBC, top_cnv, bot_cnv, verbose = verbose)
    st.converged || println("      !  gmg solve hit V-cycle cap (ncyc=$(st.ncyc))")
    return nothing
end
