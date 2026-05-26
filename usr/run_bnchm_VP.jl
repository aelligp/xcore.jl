# Port of usr/run_bnchm_VP.m
#
# Manufactured-solution (MMS) benchmark for the Stokes / velocity-pressure
# solver. Builds an analytic (W, U, P) with known Stokes residuals, injects
# them as source terms via `bnchm_data`, runs `fluidmech!` once at each of
# three resolutions, and plots loglog error convergence vs grid step.
# Expects quadratic convergence in space.
#
# Run from the repo root:
#   julia --project=julia examples/run_bnchm_VP.jl

using xcore
using KernelAbstractions: CPU
using LinearAlgebra
using Statistics
using CairoMakie
using Printf

# pull the MMS source generator (test-only helper, not part of the package)
include(joinpath(@__DIR__, "..", "test", "mms_sources.jl"))

const T  = Float64
const L  = T(10)
const NS = (100, 200, 400)

results = NamedTuple[]

for N in NS
    println("\n=== bnchm_VP  N = $N ===")
    mms = mms_sources(T, N, N, L)

    par  = Parameters(T; D = L, N = N, L = L,
                       etam0 = 1e3, g0 = 10.0,
                       rhom0 = 2700, rhox0 = 3200, d0 = 1e-2,
                       Da = 0.0, Xi = 0.0, gamma = 0.0)
    grid = Grid(par)
    fluid = FluidState(T, CPU(), grid.Nz, grid.Nx)

    # inject analytic material fields (no constitutive blend in MMS mode)
    fluid.eta   .= mms.eta
    fluid.etaco .= mms.etaco
    fluid.rho   .= mms.rho
    fluid.rhow  .= mms.rhow
    fluid.rhou  .= mms.rhou
    fluid.Drho  .= mms.Drho
    fluid.MFS   .= mms.MFS

    bnchm_data = BnchmData{T, typeof(mms.src_W)}(
        mms.src_W, mms.src_U, mms.src_P,
        mms.W_exact, mms.U_exact, mms.P_exact,
    )

    t_solve = @elapsed fluidmech!(fluid, grid, par;
                                  bnchm_data = bnchm_data,
                                  sds = -1, top_cnv = -1, bot_cnv = -1,
                                  open_cnv = false,
                                  dt = T(1e32),
                                  a1 = T(1), a2 = T(1), a3 = T(0))

    EW = norm(fluid.W .- mms.W_exact) / norm(mms.W_exact)
    EU = norm(fluid.U .- mms.U_exact) / norm(mms.U_exact)
    EP = norm(fluid.P .- mms.P_exact) / norm(mms.P_exact)
    @info "errors" N EW EU EP t_solve

    push!(results, (N = N, h = Float64(L / N),
                    EW = EW, EU = EU, EP = EP, t = t_solve))
end

# ---------------------------------------------------------------------------
# Plot 1: error convergence in space
# ---------------------------------------------------------------------------

outdir = joinpath(@__DIR__, "..", "out", "bnchm_VP")
mkpath(outdir)

fig = Figure(size = (700, 550))
ax = Axis(fig[1, 1];
          xscale = log10, yscale = log10,
          xlabel = "grid step [m]",
          ylabel = "rel. numerical error [1]",
          title  = "Numerical convergence in space")

hs  = [r.h  for r in results]
EWs = [r.EW for r in results]
EUs = [r.EU for r in results]
EPs = [r.EP for r in results]

scatter!(ax, hs, EWs; marker = :rect,    markersize = 14, color = :steelblue, label = "error W")
scatter!(ax, hs, EUs; marker = :circle,  markersize = 14, color = :seagreen,  label = "error U")
scatter!(ax, hs, EPs; marker = :utriangle, markersize = 14, color = :tomato,  label = "error P")

# quadratic-convergence reference anchored at the coarsest point
gm0 = exp(mean(log, (EWs[1], EUs[1], EPs[1])))
lines!(ax, hs, gm0 .* (hs ./ hs[1]) .^ 2;
       color = :black, label = "quadratic")

axislegend(ax; position = :rb)
path = joinpath(outdir, "bnchm_VP_bnchm.png")
save(path, fig)
println("\nWrote: $path")

# ---------------------------------------------------------------------------
# Plot 2: time-to-solution vs DOFs (sparse LU scaling)
# ---------------------------------------------------------------------------

fig2 = Figure(size = (700, 550))
ax2 = Axis(fig2[1, 1];
           xscale = log10, yscale = log10,
           xlabel = "# dofs [1]",
           ylabel = "time to solution [s]",
           title  = "Scaling of direct solver")

dofs = [(r.N + 2)^2 + 2 * (r.N + 1) * (r.N + 2) for r in results]
ts   = [r.t for r in results]

scatter!(ax2, dofs, ts; marker = :cross, markersize = 14, color = :seagreen,
         label = "time to solution")

# reference lines anchored at the smallest problem
lines!(ax2, dofs, ts[1] .* (dofs ./ dofs[1]) .^ 1;
       color = :black, label = "linear")
lines!(ax2, dofs, ts[1] .* (dofs ./ dofs[1]) .^ 2;
       linestyle = :dash, color = :black, label = "quadratic")

axislegend(ax2; position = :rb)
path2 = joinpath(outdir, "bnchm_VP_sclng.png")
save(path2, fig2)
println("Wrote: $path2")
