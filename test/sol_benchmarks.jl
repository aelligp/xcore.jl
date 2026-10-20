using xcore
using xcore: Parameters, Grid, FluidState, PhaseState, DyrelCache,
             compute_scales, fluidmech!, fluidmech_dyrel!, compute_corner_eta!
using xcore.GMG: build_hierarchy, fluidmech_gmg!
using KernelAbstractions: CPU
using LinearAlgebra
using Statistics

include("solcx_solution.jl")
include("solkz_solution.jl")

# Analytic variable-viscosity Stokes benchmarks (SolCx, SolKz).
#
# Both reference solutions are posed on the unit box with FREE SLIP on all four
# sides, which xcore expresses directly: `xBC = :closed` with `sds = -1` for the
# side walls and `top_cnv = bot_cnv = -1` for the top and bottom.
#
# Orientation: xcore's z is depth (downward-positive W), the reference solutions
# use an upward y, so y = D - z and the vertical component reverses while the
# horizontal one does not:
#   W(x, z) = -v_y(x, D-z),  U(x, z) = +v_x(x, D-z),  P(x, z) = +P_ref(x, D-z).
#
# Viscosity convention: xcore's deviatoric stress is tau = eta(e - div(v)/3),
# without the conventional factor 2 (see the stress expressions in
# dyrel/residuals.jl and the 2/3 and 1/2 stencil weights in fluidmech.jl). The
# reference solutions are posed for tau = 2*eta_ref(e - div(v)/3), so the
# benchmark viscosity must be loaded as eta = 2*eta_ref or the computed
# velocities come out twice too large.
#
# The physics is reduced to steady incompressible Stokes by three choices:
# `rho` uniform (kills the drunken-sailor diagonal drho*g0*dt and makes the
# 1/rho-scaled continuity equation plain div(v) = 0), the buoyancy carried
# entirely by `Drho`, and a1 = a2 = a3 = 0 with a large dt (kills BD2 inertia
# and the lagged momentum flux). `par.alpha = 0` freezes the mass-flux source
# MFS at zero, which also leaves the bottom drain closed. Momentum advection is
# frozen from the entry velocity in every solver, so a zero initial state keeps
# it identically zero throughout.

# One explicit diffusion pass over the interior, with Neumann edges. Repeated
# application spreads a viscosity jump over a few cells; the 1/4.1 coefficient
# is the stable explicit step for the 5-point Laplacian.
function _smooth_eta!(eta::AbstractMatrix{T}, npass::Integer) where {T}
    npass > 0 || return eta
    buf = similar(eta)
    for _ in 1:npass
        i, j = axes(eta, 1)[2:(end - 1)], axes(eta, 2)[2:(end - 1)]
        @views buf[i, j] .= eta[i, j] .+ T(1 / 4.1) .*
            (eta[i, j .+ 1] .- 2 .* eta[i, j] .+ eta[i, j .- 1] .+
             eta[i .+ 1, j] .- 2 .* eta[i, j] .+ eta[i .- 1, j])
        @views buf[begin, :] .= buf[begin + 1, :]
        @views buf[end, :]   .= buf[end - 1, :]
        @views buf[:, begin] .= buf[:, begin + 1]
        @views buf[:, end]   .= buf[:, end - 1]
        eta .= buf
    end
    return eta
end

"""
    sol_setup(T, N, case; Δη = 1e6, L_b = 1, smooth_eta = 0)
        -> (par, grid, fluid, phase, scales)

Build the benchmark state for `case ∈ (:solcx, :solkz)` on an `N × N` grid over
the unit box of side `L_b`.

`atol_PH` is the residual the iterative solvers must reach. It only has to sit
well below the discretization error being measured; the default is far tighter
than that, and on a high-contrast SolCx the dynamic-relaxation residual plateaus
above it, so a resolution sweep should relax it (1e-7 is ample against a
discretization error of order 1e-4).

`smooth_eta` applies that many explicit diffusion passes to the viscosity field.
A pseudo-transient solve stalls on an unsmoothed SolCx jump beyond Δη ≈ 15, so
the matrix-free solvers need a few passes to converge at all. Smoothing changes
the operator, so every solver compared against the others must use the same
number of passes, and the smoothing width is a fixed number of cells rather
than a fixed length — it shrinks with h, so a resolution sweep taken with
`smooth_eta > 0` does not measure the discretization order of the sharp problem.
"""
function sol_setup(::Type{T}, N::Integer, case::Symbol; Δη::Real = 1.0e6,
                   L_b::Real = 1, smooth_eta::Integer = 0,
                   atol_PH::Real = 1.0e-10) where {T<:AbstractFloat}
    case in (:solcx, :solkz) || throw(ArgumentError("sol_setup: unknown case $case"))
    par  = Parameters(T; D = L_b, N = N, L = L_b, g0 = 1.0, gamma = 0.0,
                      Da = 0.0, Xi = 0.0, alpha = 0.0, atol_PH = atol_PH)
    grid = Grid(par)
    Nz, Nx = grid.Nz, grid.Nx

    fluid = FluidState(T, CPU(), Nz, Nx)
    phase = PhaseState(T, CPU(), Nz, Nx)

    # material fields in reference coordinates (y = D − z)
    ηf = case === :solcx ?
        (x, y) -> x <= T(L_b) / 2 ? one(T) : T(Δη) :
        (x, y) -> exp(T(log(Δη)) * y)
    ρf = case === :solcx ?
        (x, y) -> -sin(T(π) * y) * cos(T(π) * x) :
        (x, y) -> -sin(T(2) * y) * cos(T(3π) * x)

    D = T(L_b)
    fluid.eta .= [2 * ηf(grid.Xc[i], D - grid.Zc[j]) for j in 1:Nz, i in 1:Nx]
    _smooth_eta!(fluid.eta, smooth_eta)
    compute_corner_eta!(fluid.etaco, fluid.eta; xBC = :closed, zBC = :closed)
    # uniform inertial density; buoyancy lives in Drho on the z-faces
    fluid.rho .= 1;  fluid.rhow .= 1;  fluid.rhou .= 1
    fluid.Drho .= [ρf(grid.Xc[i], D - grid.Zf[j]) for j in 1:(Nz + 1), i in 1:Nx]

    return par, grid, fluid, phase, compute_scales(par, grid)
end

"""
    sol_solve!(fluid, phase, grid, par, scales, solver; dt = 1e32) -> info

Run one steady Stokes solve with free slip on all four boundaries. `solver` is
`:direct`, `:dyrel`, or `:gmg`; `info` is `nothing` for `:direct`.
"""
function sol_solve!(fluid::FluidState{T}, phase, grid, par, scales,
                    solver::Symbol; dt::Real = 1.0e32) where {T<:AbstractFloat}
    kw = (; dt = T(dt), a1 = T(0), a2 = T(0), a3 = T(0),
            sds = -1, top_cnv = -1, bot_cnv = -1, xBC = :closed, zBC = :closed)
    if solver === :direct
        fluidmech!(fluid, grid, par; phase, kw...)
        return nothing
    elseif solver === :dyrel
        cache = DyrelCache(T, CPU(), grid)
        return fluidmech_dyrel!(fluid, phase, cache, grid, par, scales; kw...)
    elseif solver === :gmg
        hier = build_hierarchy(fluid, phase, grid, par; xBC = :closed)
        return fluidmech_gmg!(fluid, phase, hier, grid, par, scales; kw...)
    end
    throw(ArgumentError("sol_solve!: unknown solver $solver"))
end

"""
    sol_errors(fluid, grid, case; Δη = 1e6, L_b = 1) -> (; U, W, P)

Relative L2 errors of the computed velocity and pressure against the analytic
solution. `P` is compared after removing its mean from both fields (the
pressure is determined only up to a constant).

`sol_errors_L1` reports the same discrepancies in the discrete L1 norm used for
order-of-accuracy measurement.
"""
function sol_errors(fluid::FluidState{T}, grid::Grid{T}, case::Symbol;
                    Δη::Real = 1.0e6, L_b::Real = 1) where {T<:AbstractFloat}
    exact = case === :solcx ?
        (x, y) -> _solCx_solution(x, y, 1.0, Δη) :
        (x, y) -> _solkz_solution(x, y; Δη = Δη)
    Nz, Nx, D = grid.Nz, grid.Nx, T(L_b)

    # U at (Zc, Xf): interior z-rows, all x-faces
    Uh = fluid.U[2:(end - 1), :]
    Ue = [exact(grid.Xf[i], D - grid.Zc[j])[1] for j in 1:Nz, i in 1:(Nx + 1)]

    # W at (Zf, Xc): all z-faces, interior x-columns
    Wh = fluid.W[:, 2:(end - 1)]
    We = [-exact(grid.Xc[i], D - grid.Zf[j])[2] for j in 1:(Nz + 1), i in 1:Nx]

    # P at (Zc, Xc): interior cells, gauge removed from both fields
    Ph = fluid.P[2:(end - 1), 2:(end - 1)]
    Pe = [exact(grid.Xc[i], D - grid.Zc[j])[3] for j in 1:Nz, i in 1:Nx]
    Ph = Ph .- mean(Ph);  Pe = Pe .- mean(Pe)

    return (; U = norm(Uh .- Ue) / norm(Ue),
              W = norm(Wh .- We) / norm(We),
              P = norm(Ph .- Pe) / norm(Pe))
end

# Discrete L1 norm of Duretz et al. (2011), doi:10.1029/2011GC003567, eq. (31):
# a one-point quadrature of ∫|e| dV over each node's control volume. Nodes on a
# domain-normal boundary own half a cell and are weighted accordingly (their
# eqs. (33) and (34)); cell-centred quantities all carry a full cell (eq. (32)).
function _L1(e::AbstractMatrix, h::Real; halfdim::Union{Int,Nothing} = nothing)
    w = fill(one(eltype(e)), axes(e))
    if halfdim !== nothing
        s = ntuple(d -> d == halfdim ? firstindex(w, d) : Colon(), ndims(w))
        t = ntuple(d -> d == halfdim ? lastindex(w, d)  : Colon(), ndims(w))
        w[s...] .= 1 // 2
        w[t...] .= 1 // 2
    end
    return sum(w .* abs.(e)) * h^2
end

"""
    sol_errors_L1(fluid, grid, case; Δη = 1e6, L_b = 1) -> (; V, P)

Discretization error in the discrete L1 norm of Duretz et al. (2011),
doi:10.1029/2011GC003567. Following their eq. (30) the two velocity components
enter as a single quantity, `V = ‖e_u‖₁ + ‖e_w‖₁`; `P` is their eq. (29), taken
after removing the mean from both fields since the pressure here is determined
only up to a constant.

These are absolute norms with the dimensions of the field times area, so their
magnitudes are only meaningful relative to each other across a resolution
sequence — which is what an order of accuracy is fitted from.
"""
function sol_errors_L1(fluid::FluidState{T}, grid::Grid{T}, case::Symbol;
                       Δη::Real = 1.0e6, L_b::Real = 1) where {T<:AbstractFloat}
    exact = case === :solcx ?
        (x, y) -> _solCx_solution(x, y, 1.0, Δη) :
        (x, y) -> _solkz_solution(x, y; Δη = Δη)
    Nz, Nx, D = grid.Nz, grid.Nx, T(L_b)

    Uh = fluid.U[2:(end - 1), :]
    Ue = [exact(grid.Xf[i], D - grid.Zc[j])[1] for j in 1:Nz, i in 1:(Nx + 1)]
    Wh = fluid.W[:, 2:(end - 1)]
    We = [-exact(grid.Xc[i], D - grid.Zf[j])[2] for j in 1:(Nz + 1), i in 1:Nx]
    Ph = fluid.P[2:(end - 1), 2:(end - 1)]
    Pe = [exact(grid.Xc[i], D - grid.Zc[j])[3] for j in 1:Nz, i in 1:Nx]
    Ph = Ph .- mean(Ph);  Pe = Pe .- mean(Pe)

    # U spans the x-faces (dimension 2 here), W the z-faces (dimension 1)
    return (; V = _L1(Uh .- Ue, grid.h; halfdim = 2) +
               _L1(Wh .- We, grid.h; halfdim = 1),
              P = _L1(Ph .- Pe, grid.h))
end

"""
    sol_benchmark(T, N, case, solver; Δη = 1e6, smooth_eta = 0, atol_PH = 1e-10)
        -> (; errs, errs_L1, info, fluid, grid)

Set up, solve, and measure one benchmark configuration. `errs` are relative L2
discrepancies per field; `errs_L1` is the Duretz L1 pair `(; V, P)`.
"""
function sol_benchmark(::Type{T}, N::Integer, case::Symbol, solver::Symbol;
                       Δη::Real = 1.0e6, smooth_eta::Integer = 0,
                       atol_PH::Real = 1.0e-10) where {T<:AbstractFloat}
    par, grid, fluid, phase, scales = sol_setup(T, N, case; Δη, smooth_eta, atol_PH)
    info = sol_solve!(fluid, phase, grid, par, scales, solver)
    return (; errs = sol_errors(fluid, grid, case; Δη),
              errs_L1 = sol_errors_L1(fluid, grid, case; Δη), info, fluid, grid)
end

"""
    sol_errors_for(N, case; Δη = 1e6) -> (; U, W, P)

Analytic errors of the direct solve at resolution `N` — the discretization
error of the scheme, independent of any iterative solver.
"""
sol_errors_for(N::Integer, case::Symbol; Δη::Real = 1.0e6) =
    sol_benchmark(Float64, N, case, :direct; Δη).errs
