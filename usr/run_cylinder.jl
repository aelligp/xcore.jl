# Flow past a circular cylinder — the Kármán vortex street of the `circle`
# example in WaterLily.jl — run through each of xcore's three
# Stokes/Navier-Stokes solvers (`:direct`, `:dyrel`, `:gmg`).
#
# Usage:  julia --project=. usr/run_cylinder.jl
#
# Produces, in `outdir` (default `usr/figs/`):
#   cylinder_vorticity.png  vorticity snapshots through the run, one row per
#                           solver, one column per time
#   cylinder_wake.png       cross-stream velocity at a wake probe against time,
#                           with the Strouhal number read off its period
#
# Setup. The stream runs along x, which is periodic; the channel walls at z = 0
# and z = D are free slip. Lengths are scaled on the cylinder diameter d,
# velocities on the free-stream speed U, and the density is 1, so Re = U d / nu
# is set by the viscosity alone. xcore's deviatoric stress is tau =
# eta(e - div(v)/3), without the conventional factor 2, so the field to load is
# eta = 2/Re.
#
# The cylinder is a direct-forcing immersed boundary: the velocity on faces
# inside it is reset to zero after every solve, and the following step's solve
# restores continuity around the masked field. The viscosity stays uniform. All
# of this sits outside the solvers, so the three of them see one identical
# problem.
#
# Drag drains streamwise momentum from a periodic box, so the bulk velocity is
# restored to U after each step — the constant-mass-flux forcing of periodic
# channel DNS. It adds a constant to U, which is divergence-free and so leaves
# continuity alone.
#
# The cylinder sits `zoff` off the channel centreline. The wake of a centred
# cylinder stays symmetric to round-off for a very long time; the offset seeds
# the shedding instability deterministically, at the cost of a slightly
# wall-biased wake.

using xcore
using xcore: make_solver, solve_fluidmech!, update_edge_viscosity!
using CairoMakie
using KernelAbstractions: CPU
using Printf
using Statistics: mean

const SOLVERS = (:direct, :dyrel, :gmg)

"""
    cylinder_setup(T = Float64; solver, Re, N, Ddom, Ldom, d, xc, zoff,
                   CFL, atol_PH, gfact) -> NamedTuple

Build the flow-past-a-cylinder state: parameters, grid, a uniform stream, the
viscosity, and the face masks marking the cylinder.

`N` sets the cross-stream resolution, so the cell size is `Ddom/N` and the
cylinder is resolved by `d·N/Ddom` cells. `:gmg` coarsens only on grids whose
both dimensions are multiples of 32, which constrains `N` and `Ldom/Ddom`
together.
"""
function cylinder_setup(::Type{T} = Float64; solver::Symbol = :direct,
                        Re::Real = 100, N::Integer = 96, Ddom::Real = 6,
                        Ldom::Real = 18, d::Real = 1, xc::Real = 6,
                        zoff::Real = 0.25, CFL::Real = 0.4,
                        atol_PH::Real = 1.0e-4,
                        gfact::Real = 50) where {T<:AbstractFloat}
    par = Parameters(T; solver, N, D = Ddom, L = Ldom, gamma = 0,
                     Da = 0, Xi = 0, alpha = 0, atol_PH = atol_PH,
                     γfact_PT = gfact)
    grid = Grid(par)
    Nz, Nx = grid.ni[1], grid.ni[2]
    fluid = FluidState(T, CPU(), Nz, Nx)
    phase = PhaseState(T, CPU(), Nz, Nx)

    zc = T(Ddom) / 2 + T(zoff) * T(d)
    r2 = (T(d) / 2)^2
    solid(x, z) = (x - T(xc))^2 + (z - zc)^2 <= r2

    maskU = falses(Nz + 2, Nx + 1)      # x-faces at (Zc, Xf), z-ghost rows
    maskW = falses(Nz + 1, Nx + 2)      # z-faces at (Zf, Xc), x-ghost columns
    for j in 1:Nz, i in 1:(Nx + 1)
        maskU[j + 1, i] = solid(grid.xvi[2][i], grid.xci[1][j])
    end
    for j in 1:(Nz + 1), i in 1:Nx
        maskW[j, i + 1] = solid(grid.xci[2][i], grid.xvi[1][j])
    end

    # Uniform viscosity: the cylinder is imposed on the velocity, not on the
    # rheology. eta = 2·nu carries the factor-2 stress convention.
    fluid.eta .= T(2) / T(Re)
    update_edge_viscosity!(fluid; xBC = :periodic, zBC = :closed)

    # Uniform density: continuity reduces to div(v) = 0 and the inertia is
    # plain rho·dv/dt. It also removes gravity, which needs no separate switch:
    # buoyancy is Drho·g0 and the drunken-sailor term is grad(rho)·g0·dt, so a
    # zero Drho and a uniform rho leave both identically zero. `par.g0` keeps
    # its default, since `compute_scales` divides by the free-fall speed.
    fluid.rho .= 1;  fluid.rhow .= 1;  fluid.rhou .= 1;  fluid.Drho .= 0

    Uinf = one(T)
    fluid.U .= Uinf
    fluid.U[maskU] .= 0

    return (; par, grid, fluid, phase, scales = compute_scales(par, grid),
              maskU, maskW, Uinf, Re = T(Re), d = T(d), xc = T(xc), zc,
              dt = T(CFL) * grid.h / Uinf)
end

"""
    enforce_body!(fluid, maskU, maskW, Uinf) -> nothing

Apply the immersed boundary and the bulk forcing to a freshly solved velocity
field: restore the mean streamwise velocity of the fluid cells to `Uinf`, then
zero the velocity on the faces inside the cylinder.
"""
function enforce_body!(fluid::FluidState, maskU::AbstractMatrix{Bool},
                       maskW::AbstractMatrix{Bool}, Uinf::Real)
    @views wet = .!maskU[2:(end - 1), :]
    @views bulk = sum(fluid.U[2:(end - 1), :][wet]) / count(wet)
    fluid.U .+= Uinf - bulk
    fluid.U[maskU] .= 0
    fluid.W[maskW] .= 0
    return nothing
end

"""
    vorticity(fluid, grid) -> Matrix

Out-of-plane vorticity `dU/dz - dW/dx` on the `(Zf, Xf)` corners, where both
differences are centred and need no interpolation.
"""
function vorticity(fluid::FluidState, grid::Grid)
    U, W = fluid.U, fluid.W
    return (@views (U[2:end, :] .- U[1:(end - 1), :]) .- (W[:, 2:end] .- W[:, 1:(end - 1)])) ./ grid.h
end

"""
    cylinder_run(solver; tend, nsnap, kw...) -> NamedTuple

March the cylinder flow to `tend` (in units of d/U) with `solver`, returning
`nsnap` evenly spaced vorticity snapshots and the full wake-probe time series.

Each step is one solve: the viscous and pressure terms are implicit, momentum
advection is explicit from the entry velocity, so no Picard loop is needed —
the viscosity field is imposed rather than computed from a rheology.
"""
function cylinder_run(solver::Symbol; tend::Real = 60, nsnap::Integer = 4,
                      verbose::Bool = true, kw...)
    st = cylinder_setup(; solver, kw...)
    (; par, grid, fluid, phase, scales, maskU, maskW, dt, Uinf) = st
    T = eltype(fluid.W)
    fsolver = make_solver(par, fluid, phase, grid)

    nsteps = ceil(Int, tend / dt)
    snapat = Set(round(Int, nsteps * k / nsnap) for k in 1:nsnap)
    # wake probe three diameters downstream, on the cylinder's centreline
    jp = argmin(abs.(grid.xvi[1] .- st.zc))
    ip = argmin(abs.(grid.xci[2] .- (st.xc + 3 * st.d))) + 1

    snaps = NamedTuple[]
    tp = T[];  wp = T[]
    elapsed = @elapsed for step in 1:nsteps
        (a1, a2, a3, b1, b2, b3) = time_coefs(T, par.TINT, step)
        store_previous!(fluid, phase)
        solve_fluidmech!(fsolver, fluid, phase, grid, par, scales;
                         dt, a1, a2, a3, b1, b2, b3, sds = -1,
                         top_cnv = -1, bot_cnv = -1, open_cnv = false,
                         xBC = :periodic, zBC = :closed)
        enforce_body!(fluid, maskU, maskW, Uinf)
        push!(tp, step * dt);  push!(wp, fluid.W[jp, ip])
        step in snapat && push!(snaps, (; t = step * dt, vort = vorticity(fluid, grid)))
    end
    verbose && @printf("%-7s %d steps to t = %.1f d/U in %.1f s\n",
                       solver, nsteps, nsteps * dt, elapsed)
    return (; solver, st, snaps, t = tp, w = wp)
end

# Strouhal number from the shedding period, measured by upward zero crossings of
# the wake probe over the second half of the record (the first half still holds
# the start-up transient).
function strouhal(t, w; d::Real = 1, U::Real = 1)
    i0 = max(1, length(t) ÷ 2)
    tt = @view t[i0:end]
    ww = @view w[i0:end]
    ww = ww .- mean(ww)
    cross = [k for k in 2:length(ww) if ww[k - 1] < 0 <= ww[k]]
    length(cross) < 2 && return NaN
    period = (tt[cross[end]] - tt[cross[1]]) / (length(cross) - 1)
    return d / (U * period)
end

"""
    vorticity_figure(runs; outdir) -> Figure

Vorticity snapshots for every solver on a shared colour range: one row per
solver, one column per snapshot time. The circle marks the immersed boundary.
"""
function vorticity_figure(runs; outdir = joinpath(@__DIR__, "figs"))
    nrow, ncol = length(runs), length(first(runs).snaps)
    grid = first(runs).st.grid
    # A fixed range, shared by every panel, so solvers and times are directly
    # comparable. It is set to resolve the wake; the boundary layer on the
    # cylinder is an order of magnitude stronger and saturates deliberately.
    lim = 5 * first(runs).st.Uinf / first(runs).st.d

    fig = Figure(; size = (360 * ncol + 120, 150 * nrow + 90), fontsize = 16)
    local hm
    for (i, r) in enumerate(runs), (k, s) in enumerate(r.snaps)
        ax = Axis(fig[i, k]; aspect = DataAspect(),
                  title = i == 1 ? @sprintf("t = %.0f d/U", s.t) : "",
                  ylabel = k == 1 ? String(r.solver) : "",
                  xticksvisible = false, xticklabelsvisible = false,
                  yticksvisible = false, yticklabelsvisible = false)
        hm = heatmap!(ax, grid.xvi[2], grid.xvi[1], permutedims(Float64.(s.vort));
                      colormap = :balance, colorrange = (-lim, lim))
        arc!(ax, Point2f(r.st.xc, r.st.zc), r.st.d / 2, -π, π;
             color = :black, linewidth = 1.5)
    end
    Colorbar(fig[1:nrow, ncol + 1], hm; label = "vorticity  (U/d)")
    Label(fig[0, 1:(ncol + 1)],
          @sprintf("Flow past a cylinder — vorticity   (Re = %g, %d×%d grid)",
                   first(runs).st.Re, grid.ni[1], grid.ni[2]);
          fontsize = 20)

    mkpath(outdir)
    path = joinpath(outdir, "cylinder_vorticity.png")
    save(path, fig)
    @info "wrote $path"
    return fig
end

"""
    wake_figure(runs; outdir) -> Figure

Cross-stream velocity at the wake probe against time. The shedding instability
shows as the growth of the oscillation out of the start-up transient; its
period gives the Strouhal number in the legend.
"""
function wake_figure(runs; outdir = joinpath(@__DIR__, "figs"))
    fig = Figure(; size = (1100, 480), fontsize = 18)
    ax = Axis(fig[1, 1]; xlabel = "t  (d/U)",
              ylabel = "cross-stream velocity  (U),  3d downstream",
              title = "Wake probe")
    colors = Makie.wong_colors()
    for (i, r) in enumerate(runs)
        St = strouhal(r.t, r.w; r.st.d, U = r.st.Uinf)
        lines!(ax, Float64.(r.t), Float64.(r.w); color = colors[i], linewidth = 2,
               label = @sprintf("%s   St = %.3f", r.solver, St))
        @printf("%-7s St = %.4f\n", r.solver, St)
    end
    axislegend(ax; position = :lt)

    mkpath(outdir)
    path = joinpath(outdir, "cylinder_wake.png")
    save(path, fig)
    @info "wrote $path"
    return fig
end

if abspath(PROGRAM_FILE) == @__FILE__
    runs = [cylinder_run(s) for s in SOLVERS]
    vorticity_figure(runs)
    wake_figure(runs)
end
