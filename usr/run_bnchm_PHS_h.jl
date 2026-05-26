# Port of usr/run_bnchm_PHS_h.m
#
# Spatial-convergence benchmark for the phase-evolution solver. Same setup as
# the dt benchmark (constant U=1 advection, zero diffusion, no reaction, no
# noise) but holds dt fixed and sweeps grid resolution. Expects 5th-order
# convergence with WENO5.
#
# Run from the repo root:
#   julia --project=julia examples/run_bnchm_PHS_h.jl

using xcore
using KernelAbstractions: CPU
using LinearAlgebra
using CairoMakie
using Printf

function force_constant_advection!(phase, fluid)
    fill!(fluid.W, 0); fill!(fluid.U, 1); fill!(fluid.P, 0)
    fill!(phase.wx, 0); fill!(phase.wm, 0)
    fill!(phase.Wx, 0); fill!(phase.Wm, 0)
    fill!(phase.Ux, 1); fill!(phase.Um, 1)
    fill!(phase.kx, 0); fill!(phase.ke, 0); fill!(phase.ks, 0)
end

function run_bnchm_phs_h(N::Int; D = 10.0, L = 10.0,
                          dt = 0.0, nshft_base = 1, N_base = 30,
                          ADVN = :weno5, TINT = :bd2im,
                          atol = 1e-12, maxit = 100, alpha = 0.9)
    par = Parameters(Float64;
        runID = "bnchm_PHS_h",
        D = D, N = N, L = L,
        d0 = 1e-2, etam0 = 1e1,
        x0 = 0.01, dxr = 0.0, dxg = 1.0,           # Gaussian-blob initial
        Da = 0.0,                                    # no reaction
        Xi = 0.0,                                    # no noise
        CFL = 10.0, alpha = alpha, gamma = 0.0,
        atol = atol, rtol = atol / 1e6, maxit = maxit,
        dtmax = dt,
        TINT = TINT, ADVN = ADVN,
    )
    grid   = Grid(par)
    scales = compute_scales(par, grid)
    fluid  = FluidState(Float64, CPU(), grid.Nz, grid.Nx)
    phase  = PhaseState(Float64, CPU(), grid.Nz, grid.Nx)
    ns     = NoiseState(Float64, grid, scales)
    initialize!(phase, fluid, ns, grid, par, scales)
    force_constant_advection!(phase, fluid)

    # snapshot initial state and build the analytic shifted reference. The
    # shift in cells scales with resolution so the *physical* shift stays the
    # same as in the coarsest run.
    nshft = nshft_base * (N ÷ N_base)
    X0   = copy(phase.X);  M0   = copy(phase.M);  rho0 = copy(fluid.rho)
    Xref = circshift(X0,   (0, nshft))
    Mref = circshift(M0,   (0, nshft))
    rref = circshift(rho0, (0, nshft))

    Nsteps = round(Int, nshft_base * D / N_base / dt)

    res = StepResidual(grid)

    for step in 1:Nsteps
        a1, a2, a3, b1, b2, b3 = time_coefs(Float64, par.TINT, step)
        store_previous!(fluid, phase)
        resnorm0 = 1.0
        for iter in 1:par.maxit
            snapshot!(res, phase, fluid)
            phsevo!(phase, fluid, grid, par, scales;
                    ADVN = par.ADVN, xBC = :periodic, zBC = :closed,
                    a1, a2, a3, b1, b2, b3, dt)
            update_volume_fractions!(phase, fluid, par)
            force_constant_advection!(phase, fluid)
            resnorm, _, _ = compute_resnorm(res, phase, fluid, dt)
            iter == 1 && (resnorm0 = resnorm + eps(Float64))
            if iter ≥ 3 &&
               (resnorm ≤ par.atol || resnorm / resnorm0 ≤ par.rtol)
                break
            end
        end
    end

    EB = norm(fluid.rho .- rref) / norm(rref)
    EM = norm(phase.M   .- Mref) / norm(rref)
    EX = norm(phase.X   .- Xref) / norm(rref)
    return (; N, h = D / N, EB, EM, EX)
end

# ---------------------------------------------------------------------------
# Sweep three resolutions
# ---------------------------------------------------------------------------

const NS = (30, 60, 120)
const D_dom = 10.0
# fixed dt small enough for the finest resolution (matches MATLAB's `D/NN(3)/300`)
const dt_fixed = D_dom / NS[end] / 300

results = NamedTuple[]
for N in NS
    println("\n=== bnchm_PHS_h  N = $N ===")
    r = run_bnchm_phs_h(N; D = D_dom, dt = dt_fixed, nshft_base = 1, N_base = NS[1])
    @info "errors" N h = r.h EB = r.EB EM = r.EM EX = r.EX
    push!(results, r)
end

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

outdir = joinpath(@__DIR__, "..", "out", "bnchm_PHS_h")
mkpath(outdir)

fig = Figure(size = (700, 550))
ax = Axis(fig[1, 1];
          xscale = log10, yscale = log10,
          xlabel = "Grid step [m]",
          ylabel = "Rel. numerical error [1]",
          title  = "Numerical convergence in space")

hs  = [r.h  for r in results]
EBs = [r.EB for r in results]
EMs = [r.EM for r in results]
EXs = [r.EX for r in results]

scatter!(ax, hs, EBs; marker = :rect,    markersize = 14, color = :steelblue, label = "error ρ̄")
scatter!(ax, hs, EMs; marker = :circle,  markersize = 14, color = :seagreen,  label = "error M")
scatter!(ax, hs, EXs; marker = :diamond, markersize = 14, color = :tomato,    label = "error X")

# quintic-convergence reference anchored at the coarsest point
gm0 = exp(mean(log, (EBs[1], EMs[1], EXs[1])))
lines!(ax, hs, gm0 .* (hs ./ hs[1]) .^ 5;
       color = :black, label = "quintic")

axislegend(ax; position = :rb)
path = joinpath(outdir, "bnchm_PHS_h_weno5.png")
save(path, fig)
println("\nWrote: $path")
