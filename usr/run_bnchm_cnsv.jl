# Port of usr/run_bnchm_cnsv.m
#
# Runs the full simulation at three nonlinear tolerances and measures the
# residual rate of conservation drift (d|E|/dt) over the second half of the
# run. Plots loglog convergence of EB / EM / EX vs atol.
#
# Run from the repo root:
#   julia --project=julia examples/run_bnchm_cnsv.jl

using xcore
using KernelAbstractions: CPU
using LinearAlgebra
using Statistics
using CairoMakie
using Printf

const ATOL = (1e-3, 1e-6, 1e-9)

# storage for the loglog convergence plot
results = NamedTuple{(:atol, :EB, :EM, :EX), NTuple{4, Float64}}[]

for atol in ATOL
    println("\n=== bnchm_cnsv  atol = $(atol) ===")

    par = Parameters(Float64;
        runID   = "bnchm_cnsv",
        outdir  = joinpath(@__DIR__, "..", "out"),
        save_op = false,                          # PNGs not needed for the benchmark
        nop     = 10,
        nrh     = 1,

        # domain
        D  = 10.0,
        N  = 100,
        L  = 10.0,                               # square box

        # timing
        t0end = 10.0,

        # initial crystallinity
        x0  = 0.001,                              # xeq/10 with xeq=0.01
        dxr = 0.1,
        dxg = 0.0,

        # physics
        d0   = 1e-2,
        etam0 = 1e1,
        L0   = (10.0/100)/2,                      # h/2
        l0   = 1e-2 * 10,                         # d0*10
        Da   = 0.01,
        Xi   = 0.5,

        # numerics — the atol/rtol are the per-run knob
        CFL   = 0.5,
        maxit = 100,
        alpha = 0.9,
        atol  = atol,
        rtol  = atol / 1e6,
        TINT  = :bd2im,
        ADVN  = :weno5,
    )

    grid   = Grid(par)
    scales = compute_scales(par, grid)
    fluid  = FluidState(Float64, CPU(), grid.Nz, grid.Nx)
    phase  = PhaseState(Float64, CPU(), grid.Nz, grid.Nx)
    ns     = NoiseState(Float64, grid, scales)
    hst    = History(Float64)
    initialize!(phase, fluid, ns, grid, par, scales)

    run!(phase, fluid, ns, grid, par, scales;
         nsteps   = par.Nt,
         verbose  = true,
         callback = (step, t, dt, ph, fl, ns_, a1, a2, a3, b1, b2, b3) -> record_history!(hst, t, dt, ph, fl, ns_, grid, a1, a2, a3, b1, b2, b3))

    # rms of d|E|/dt over the second half of the run (matches bnchm_cnsv.m:58-60)
    rms(x) = sqrt(mean(abs2, x))
    half = length(hst.time) ÷ 2
    dt_h = diff(hst.time[half:end])
    EB_rate = rms(diff(hst.EB[half:end]) ./ dt_h)
    EM_rate = rms(diff(hst.EM[half:end]) ./ dt_h)
    EX_rate = rms(diff(hst.EX[half:end]) ./ dt_h)
    @info "convergence rates" atol EB = EB_rate EM = EM_rate EX = EX_rate

    push!(results, (atol = Float64(atol),
                    EB = EB_rate, EM = EM_rate, EX = EX_rate))
end

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

outdir = joinpath(@__DIR__, "..", "out", "bnchm_cnsv")
mkpath(outdir)

fig = Figure(size = (700, 550))
ax = Axis(fig[1, 1];
          xscale = log10, yscale = log10,
          xlabel = "Abs. residual tolerance [1]",
          ylabel = "Rel. conservation error rate [1/s]",
          title  = "Global conservation with nonlinear convergence")

atols = [r.atol for r in results]
EBs   = [max(r.EB, 1e-30) for r in results]
EMs   = [max(r.EM, 1e-30) for r in results]
EXs   = [max(r.EX, 1e-30) for r in results]

scatter!(ax, atols, EBs; marker = :rect,    markersize = 14, color = :steelblue, label = "error ρ̄")
scatter!(ax, atols, EMs; marker = :circle,  markersize = 14, color = :seagreen,  label = "error M")
scatter!(ax, atols, EXs; marker = :diamond, markersize = 14, color = :tomato,    label = "error X")

# machine-precision reference line (matches MATLAB's eps reference)
lines!(ax, atols, fill(eps(Float64), length(atols));
       linestyle = :dot, color = :black, label = "machine prec.")

axislegend(ax; position = :rb)
path = joinpath(outdir, "bnchm_cnsv_bd2im_weno5.png")
save(path, fig)
println("\nWrote: $path")
