# Analytic Stokes benchmark figures (SolCx, SolKz).
#
# The convergence study follows Duretz et al. (2011), doi:10.1029/2011GC003567:
# the discretization error is measured in their discrete L1 norm and the order
# of accuracy is fitted by least squares over a resolution sequence.
#
# Usage:  julia --project=. usr/run_sol_benchmarks.jl
#
# Produces, in `outdir` (default `usr/figs/`):
#   sol_convergence.png   discretization error vs h, in the L1 norm and log-log
#                         axes of Duretz et al. (2011), one panel per case.
#   solcx_fields.png      numerical / analytic / error panels for U, W, P.
#   solkz_fields.png      likewise.
#
# The benchmark setup itself lives in test/sol_benchmarks.jl (free slip on all
# four boundaries, `xBC = :closed` with `sds = -1`); this script only drives it
# and draws. Resolutions are a keyword so a quick pass can use fewer.

using CairoMakie
using Printf
using Statistics: mean
using LinearAlgebra

include(joinpath(@__DIR__, "..", "test", "sol_benchmarks.jl"))

# The three viscosity structures of Duretz et al. (2011): isoviscous, a
# discontinuous jump, and a large smooth variation.
#
# `smooth_eta` Laplacian passes are applied to the viscosity before solving. A
# pseudo-transient solve does not converge on an unsmoothed 10⁶ jump, so the
# jump case gets the 5 passes JustRelax's SolCx benchmarks use — applied for
# EVERY solver, so all three discretize the identical problem and the panel
# compares solvers rather than solvers-plus-problem. The smoothing spans a
# fixed number of cells, so it narrows as the grid refines: that panel measures
# the smoothed problem, not the sharp-jump order of accuracy the other two show.
const CASES = (
    (case = :solcx, Δη = 1.0,   smooth_eta = 0, title = "SolCx  (isoviscous)"),
    (case = :solcx, Δη = 1.0e6, smooth_eta = 5,
     title = "SolCx  (η* = 10⁶ jump, 5× smoothed)"),
    (case = :solkz, Δη = 1.0e6, smooth_eta = 0,
     title = "SolKz  (Δη* = 10⁶ smooth)"),
)

# `:gmg` builds its hierarchy only on 32-multiple grids, so a sequence all three
# solvers can share must be built from those.
const SOLVERS = (
    (solver = :direct, marker = :rect,    label = "direct"),
    (solver = :dyrel,  marker = :circle,  label = "dyrel"),
    # (solver = :gmg,    marker = :diamond, label = "gmg"),
    (solver = :vmg,    marker = :diamond, label = "vmg"),
)

"""
    convergence_data(case, Δη, Ns; solver, smooth_eta, atol_PH) -> (h, V, P)

Resolution sweep returning the discrete L1 velocity and pressure errors of
`sol_errors_L1` at each `N`. Resolutions whose iterative solve did not converge
are dropped — their fields carry the solver's own residual, not a
discretization error — so the returned vectors may be shorter than `Ns`.
"""
function convergence_data(case::Symbol, Δη::Real, Ns; solver::Symbol = :direct,
                          smooth_eta::Integer = 0, atol_PH::Real = 1.0e-7)
    h, V, P = Float64[], Float64[], Float64[]
    for N in Ns
        r = sol_benchmark(Float64, N, case, solver; Δη, smooth_eta, atol_PH)
        if r.info !== nothing && !r.info.converged
            @warn "$solver did not converge; point dropped" case Δη N
            continue
        end
        push!(h, 1 / N);  push!(V, r.errs_L1.V);  push!(P, r.errs_L1.P)
    end
    return (h, V, P)
end

# Order of accuracy r in ‖e‖₁ ≤ C hʳ, from a least-squares fit of log10‖e‖₁ on
# log10 h over the whole sequence (Duretz et al. 2011, eq. (35)).
function _rate(h, e)
    length(h) < 2 && return NaN
    x, y = log10.(h), log10.(e)
    x̄, ȳ = mean(x), mean(y)
    return sum((x .- x̄) .* (y .- ȳ)) / sum(abs2, x .- x̄)
end

"""
    convergence_figure(; Ns, solvers, atol_PH, outdir) -> Figure

Discretization error against grid spacing for each viscosity structure, in the
L1 norm and log10-log10 axes of Duretz et al. (2011). Velocity is solid and
pressure dashed; each solver has its own color and marker. The matrix-free
solves are converged well below the discretization error, so their curves lie
on top of the direct solve wherever they converge at all.
"""
function convergence_figure(; Ns = (32, 64, 96, 128, 192, 256),
                            solvers = SOLVERS, atol_PH::Real = 1.0e-7,
                            outdir = joinpath(@__DIR__, "figs"))
    fig = Figure(; size = (1600, 520), fontsize = 20)
    colors = Makie.wong_colors()
    for (k, c) in enumerate(CASES)
        ax = Axis(fig[1, k]; title = c.title, xlabel = L"\log_{10}(h)",
                  ylabel = k == 1 ? L"\log_{10}(\Vert e \Vert_1)" : "",
                  titlesize = 19, xlabelsize = 18, ylabelsize = 18)
        ref = nothing
        for (i, sv) in enumerate(solvers)
            h, V, P = convergence_data(c.case, c.Δη, Ns; sv.solver,
                                       c.smooth_eta, atol_PH)
            isempty(h) && continue
            ref === nothing && (ref = (h, V))
            for (e, ls) in ((V, :solid), (P, :dash))
                scatterlines!(ax, log10.(h), log10.(e); color = colors[i],
                              linewidth = 3, linestyle = ls, marker = sv.marker,
                              markersize = 12)
            end
            @printf("%-6s Δη=%-8g %-7s r_u = %7.4f  r_p = %7.4f  (%d/%d)\n",
                    c.case, c.Δη, sv.label, _rate(h, V), _rate(h, P),
                    length(h), length(Ns))
        end
        # first- and second-order reference slopes, anchored on the coarsest point
        if ref !== nothing
            hr, Vr = ref
            for r in (1, 2)
                lines!(ax, log10.(hr),
                       log10(first(Vr)) .+ r .* log10.(hr ./ first(hr));
                       color = (:gray, 0.8), linewidth = 1.5,
                       linestyle = r == 1 ? :dot : :dashdot)
            end
        end
    end

    solver_elems = [MarkerElement(; color = colors[i], marker = sv.marker,
                                  markersize = 12)
                    for (i, sv) in enumerate(solvers)]
    field_elems = [LineElement(; color = :black, linewidth = 3, linestyle = ls)
                   for ls in (:solid, :dash)]
    slope_elems = [LineElement(; color = (:gray, 0.8), linewidth = 1.5,
                               linestyle = ls) for ls in (:dot, :dashdot)]
    Legend(fig[1, length(CASES) + 1],
           [solver_elems, field_elems, slope_elems],
           [[sv.label for sv in solvers], ["velocity", "pressure"],
            [L"h^1", L"h^2"]],
           ["Solver", "Field", "Slope"])

    mkpath(outdir)
    path = joinpath(outdir, "sol_convergence.png")
    save(path, fig)
    @info "wrote $path"
    return fig
end

# Analytic fields on the same staggered points `sol_errors` compares, so the
# panels below show exactly the quantity the error norms measure.
function _analytic_fields(grid, case::Symbol; Δη::Real, L_b::Real = 1)
    exact = case === :solcx ?
        (x, y) -> _solCx_solution(x, y, 1.0, Δη) :
        (x, y) -> _solkz_solution(x, y; Δη = Δη)
    Nz, Nx, D = grid.Nz, grid.Nx, float(L_b)
    U = [exact(grid.Xf[i], D - grid.Zc[j])[1] for j in 1:Nz, i in 1:(Nx + 1)]
    W = [-exact(grid.Xc[i], D - grid.Zf[j])[2] for j in 1:(Nz + 1), i in 1:Nx]
    P = [exact(grid.Xc[i], D - grid.Zc[j])[3] for j in 1:Nz, i in 1:Nx]
    return (; U, W, P)
end

_panel!(fig, r, c, title, x, z, data, cmap) = begin
    ax = Axis(fig[r, c]; title = title, xlabel = "x", ylabel = "z",
              yreversed = true, aspect = DataAspect())
    m = max(maximum(abs, data), eps())
    hm = heatmap!(ax, x, z, permutedims(Float64.(data)); colormap = cmap,
                  colorrange = (-m, m))
    Colorbar(fig[r, c + 1], hm)
    ax
end

"""
    fields_figure(case; N, Δη, solver, outdir) -> Figure

Numerical / analytic / difference panels for U, W and P at one resolution —
the qualitative companion to the convergence plot.
"""
function fields_figure(case::Symbol; N::Integer = 64, Δη::Real = 1.0e6,
                       solver::Symbol = :direct,
                       outdir = joinpath(@__DIR__, "figs"))
    r = sol_benchmark(Float64, N, case, solver; Δη)
    grid, fluid = r.grid, r.fluid
    ana = _analytic_fields(grid, case; Δη)
    num = (; U = fluid.U[2:(end - 1), :],
             W = fluid.W[:, 2:(end - 1)],
             P = fluid.P[2:(end - 1), 2:(end - 1)])
    # the pressure is fixed only up to a constant; compare gauge-removed
    Pn = num.P .- sum(num.P) / length(num.P)
    Pa = ana.P .- sum(ana.P) / length(ana.P)

    fig = Figure(; size = (1450, 1200), fontsize = 17)
    # each staggered field carries its own coordinate vectors
    rows = ((:U, grid.Xf, grid.Zc, num.U, ana.U),
            (:W, grid.Xc, grid.Zf, num.W, ana.W),
            (:P, grid.Xc, grid.Zc, Pn,    Pa))
    for (i, (sym, x, z, n, a)) in enumerate(rows)
        _panel!(fig, i, 1, "$sym  numerical",  x, z, n, :balance)
        _panel!(fig, i, 3, "$sym  analytic",   x, z, a, :balance)
        _panel!(fig, i, 5, "$sym  difference", x, z, n .- a, :balance)
    end
    Label(fig[0, 1:6],
          @sprintf("%s   N = %d   Δη = %g   solver = %s", case, N, Δη, solver);
          fontsize = 22)
    mkpath(outdir)
    path = joinpath(outdir, "$(case)_fields.png")
    save(path, fig)
    @info "wrote $path"
    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    convergence_figure()
    fields_figure(:solcx)
    fields_figure(:solkz)
end
