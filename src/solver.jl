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
#   :gmg_direct → :direct's assembled system, AMG-preconditioned FGMRES (DirectMGSolver)
#   :vmg    → DYREL Powell–Hestenes + velocity V-cycles + Cahouet–Chabard (VMGSolver, MGHierarchy)

abstract type FluidSolver end

struct DirectSolver <: FluidSolver end
struct DyrelSolver{C} <: FluidSolver;          cache::C;        end
struct GMGSolver{H} <: FluidSolver;            hier::H;         end
struct VMGSolver{H} <: FluidSolver;            hier::H;         end
mutable struct DirectMGSolver{T} <: FluidSolver
    const rtol::T
    const itmax::Int
    stats::DirectMGStats
end

"""
    make_solver(par, fluid, phase, grid) -> FluidSolver

Construct the solver strategy selected by `par.solver`, allocating its persistent
state once (reused across the whole run). The GMG hierarchy wraps `fluid`/`phase`
at the finest level, so it must be rebuilt only if those are reallocated.
"""
function make_solver(par::Parameters{T}, fluid::FluidState{T}, phase::PhaseState{T},
                     grid::Grid{T}; xBC::Symbol = :periodic) where {T}
    be = KernelAbstractions.get_backend(fluid.W)
    if par.solver === :direct
        return DirectSolver()
    elseif par.solver === :dyrel
        _check_iterative_tolerance("solver = :dyrel", par)
        return DyrelSolver(DyrelCache(T, be, grid))
    elseif par.solver === :gmg
        _check_iterative_tolerance("solver = :gmg", par)
        return GMGSolver(GMG.build_hierarchy(fluid, phase, grid, par; xBC))
    elseif par.solver === :vmg
        _check_iterative_tolerance("solver = :vmg", par)
        # Velocity V-cycle design point of `fluidmech_dyrel_vmg!`: 16 DR
        # iterations per smoothing step, one pre- and one post-smooth.
        return VMGSolver(GMG.build_hierarchy(fluid, phase, grid, par; xBC,
                                             inner = 16, npre = 1, npost = 1))
    elseif par.solver === :gmg_direct
        return DirectMGSolver(par.mg_direct_rtol, par.mg_direct_itmax, DirectMGStats(false, 0))
    else
        error("unknown par.solver = $(par.solver) (use :direct, :dyrel, :gmg, :vmg, or :gmg_direct)")
    end
end

"""
    solve_fluidmech!(solver, fluid, phase, grid, par, scales; dt, a1..b3,
                     sds, top_cnv, bot_cnv, open_cnv, xBC, zBC) -> converged

One fluid-mechanics solve (one Picard sweep) with the chosen strategy. All four
methods take the same arguments; each forwards the subset its solver uses.

Returns whether the solve met its own tolerance. An iterative solver that
returns `false` has left O(err) in the momentum balance, and the Picard
residual — a norm of the CHANGE between sweeps — cannot see the difference
between that and a converged sweep, so `run!` must not read the returned fields
as a fixed point.
"""
function solve_fluidmech!(::DirectSolver, fluid, phase, grid, par, scales;
                          dt, a1, a2, a3, b1, b2, b3,
                          sds, top_cnv, bot_cnv, open_cnv, xBC, zBC, verbose = false)
    fluidmech!(fluid, grid, par; phase, sds, top_cnv, bot_cnv, open_cnv, xBC, zBC,
               dt, a1, a2, a3, b1, b2, b3, verbose = verbose)
    return true      # sparse LU: solved exactly or it threw
end

function solve_fluidmech!(s::DirectMGSolver, fluid, phase, grid, par, scales;
                          dt, a1, a2, a3, b1, b2, b3,
                          sds, top_cnv, bot_cnv, open_cnv, xBC, zBC, verbose = false)
    fluidmech!(fluid, grid, par; phase, sds, top_cnv, bot_cnv, open_cnv, xBC, zBC,
               dt, a1, a2, a3, b1, b2, b3, mg = s, verbose = verbose)
    s.stats.converged || println("      !  gmg_direct FGMRES hit itmax ($(s.stats.niter))")
    return s.stats.converged
end

# The Picard residual is a norm of the change between sweeps, so a linear solve
# that stops above the nonlinear target makes that change small without the
# equations being solved — the loop then reports convergence on fields that are
# wrong by the linear solver's own tolerance.
function _check_iterative_tolerance(what::AbstractString, par)
    par.atol_PH <= par.atol || throw(ArgumentError(
        "$what: par.atol_PH = $(par.atol_PH) is looser than the Picard target " *
        "par.atol = $(par.atol); the nonlinear loop cannot converge below the " *
        "tolerance its linear solves stop at"))
    return nothing
end

# The matrix-free kernels implement periodic or solid-wall sides (slip set by
# `sds`) and a closed z. `fluidmech!` assembles combinations beyond that, so
# accepting them here would silently solve a different problem — check instead
# of assuming.
function _check_matrixfree_bcs(what::AbstractString, sds, xBC, zBC)
    sds in (-1, 1) || throw(ArgumentError(
        "$what: sds must be -1 (free slip) or +1 (no slip), got sds = $sds"))
    xBC === :periodic && sds != -1 && throw(ArgumentError(
        "$what: xBC = :periodic requires sds = -1 (the periodic wrap), got sds = $sds"))
    zBC === :closed || throw(ArgumentError(
        "$what: only zBC = :closed is implemented, got $zBC; use solver = :direct"))
    return nothing
end

function solve_fluidmech!(s::DyrelSolver, fluid, phase, grid, par, scales;
                          dt, a1, a2, a3, b1, b2, b3,
                          sds, top_cnv, bot_cnv, open_cnv, xBC, zBC, verbose = false)
    _check_matrixfree_bcs("solver = :dyrel", sds, xBC, zBC)
    st = fluidmech_dyrel!(fluid, phase, s.cache, grid, par, scales;
                          dt, a1, a2, a3, b1, b2, b3, gamma = par.gamma,
                          rtol_PH = par.forcing_PH,
                          xBC, zBC, sds, top_cnv, bot_cnv, open_cnv, verbose = verbose)
    st.converged || st.relaxed || println("      !  dyrel solve hit budget (itPH=$(st.PH_iters), itPT total=$(st.total_iter))")
    return st.converged
end

function solve_fluidmech!(s::GMGSolver, fluid, phase, grid, par, scales;
                          dt, a1, a2, a3, b1, b2, b3,
                          sds, top_cnv, bot_cnv, open_cnv, xBC, zBC, verbose = false)
    _check_matrixfree_bcs("solver = :gmg", sds, xBC, zBC)
    # The V-cycle's coarse levels solve the homogeneous error equation with a
    # closed bottom (no drain, so no net mass source is re-injected per level).
    # An open bottom would need its own homogeneous coarse-level form.
    open_cnv && throw(ArgumentError(
        "solver = :gmg does not implement open_cnv = true; use solver = :dyrel or :direct"))
    st = GMG.fluidmech_gmg!(fluid, phase, s.hier, grid, par, scales;
                            dt, a1, a2, a3, b1, b2, b3, gamma = par.gamma,
                            rtol_PH = par.forcing_PH,
                            xBC, zBC, sds, top_cnv, bot_cnv, verbose = verbose)
    st.converged || st.relaxed || println("      !  gmg solve hit V-cycle cap (ncyc=$(st.ncyc))")
    return st.converged
end

function solve_fluidmech!(s::VMGSolver, fluid, phase, grid, par, scales;
                          dt, a1, a2, a3, b1, b2, b3,
                          sds, top_cnv, bot_cnv, open_cnv, xBC, zBC, verbose = false)
    _check_matrixfree_bcs("solver = :vmg", sds, xBC, zBC)
    # The velocity V-cycle's coarse levels assume a closed bottom, as in :gmg.
    open_cnv && throw(ArgumentError(
        "solver = :vmg does not implement open_cnv = true; use solver = :dyrel or :direct"))
    st = GMG.fluidmech_dyrel_vmg!(fluid, phase, s.hier, grid, par, scales;
                                  dt, a1, a2, a3, b1, b2, b3, gamma = par.gamma,
                                  γfact = 5, cc = true, rtol_PH = par.forcing_PH,
                                  xBC, zBC, sds, top_cnv, bot_cnv, verbose = verbose)
    return st.converged
end
