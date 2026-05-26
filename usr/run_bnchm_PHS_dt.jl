# Port of usr/run_bnchm_PHS_dt.m
#
# Temporal-convergence benchmark for the phase-evolution solver. Sets velocity
# to constant (Ux = Um = 1, all others zero), zeros diffusion, removes the
# reaction (Da = 0), advects a Gaussian crystallinity blob for N timesteps,
# then compares to the analytic shifted solution. Repeats at three time-step
# sizes and plots loglog error vs dt (expects quadratic for BD2).
#
# Run from the repo root:
#   julia --project=julia examples/run_bnchm_PHS_dt.jl

using xcore
using KernelAbstractions: CPU
using LinearAlgebra
using CairoMakie
using Printf

# Constant rightward advection: U = 1, all other velocities zero, no settling,
# no noise, zero diffusion. Called after `initialize!` and after every Picard
# sweep so that `update_volume_fractions!` can't sneak diffusivity back in.
function force_constant_advection!(phase, fluid)
    fill!(fluid.W, 0); fill!(fluid.U, 1); fill!(fluid.P, 0)
    fill!(phase.wx, 0); fill!(phase.wm, 0)
    fill!(phase.Wx, 0); fill!(phase.Wm, 0)
    fill!(phase.Ux, 1); fill!(phase.Um, 1)
    fill!(phase.kx, 0); fill!(phase.ke, 0); fill!(phase.ks, 0)
end

function run_bnchm_phs_dt(dt::Float64; D = 10.0, L = 10.0, N = 100,
                          nshft = 8, ADVN = :weno5, TINT = :bd2im,
                          atol = 1e-12, maxit = 100, alpha = 0.9)
    par = Parameters(Float64;
        runID = "bnchm_PHS_dt",
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

    # snapshot the initial state for the analytic shifted reference
    X0   = copy(phase.X);  M0   = copy(phase.M);  rho0 = copy(fluid.rho)
    Xref = circshift(X0,   (0, nshft))
    Mref = circshift(M0,   (0, nshft))
    rref = circshift(rho0, (0, nshft))

    h = grid.h
    Nsteps = round(Int, nshft * h / dt)

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

    # relative numerical errors (mirrors bnchm_PHS_dt.m:136-138)
    EB = norm(fluid.rho .- rref) / norm(rref)
    EM = norm(phase.M   .- Mref) / norm(rref)
    EX = norm(phase.X   .- Xref) / norm(rref)
    return (; dt, EB, EM, EX, phase, fluid, Xref, Mref, rref)
end

# ---------------------------------------------------------------------------
# Sweep three time-step sizes
# ---------------------------------------------------------------------------

const D_dom = 10.0
const N_dom = 100
const h     = D_dom / N_dom
const DTS   = (h / 2, h / 4, h / 8)

results = NamedTuple[]
for dt in DTS
    println("\n=== bnchm_PHS_dt  dt = $(dt) ===")
    r = run_bnchm_phs_dt(dt; D = D_dom, N = N_dom)
    @info "errors" dt EB = r.EB EM = r.EM EX = r.EX
    push!(results, (dt = dt, EB = r.EB, EM = r.EM, EX = r.EX))
end

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

outdir = joinpath(@__DIR__, "..", "out", "bnchm_PHS_dt")
mkpath(outdir)

fig = Figure(size = (700, 550))
ax = Axis(fig[1, 1];
          xscale = log10, yscale = log10,
          xlabel = "Time step [s]",
          ylabel = "Rel. numerical error [1]",
          title  = "Numerical convergence in time")

dts = [r.dt for r in results]
EBs = [r.EB for r in results]
EMs = [r.EM for r in results]
EXs = [r.EX for r in results]

scatter!(ax, dts, EBs; marker = :rect,    markersize = 14, color = :steelblue, label = "error ρ̄")
scatter!(ax, dts, EMs; marker = :circle,  markersize = 14, color = :seagreen,  label = "error M")
scatter!(ax, dts, EXs; marker = :diamond, markersize = 14, color = :tomato,    label = "error X")

# quadratic-convergence reference anchored at the first point
gm0 = exp(mean(log, (EBs[1], EMs[1], EXs[1])))
lines!(ax, dts, gm0 .* (dts ./ dts[1]) .^ 2;
       color = :black, label = "quadratic")

axislegend(ax; position = :rb)
path = joinpath(outdir, "bnchm_PHS_dt_bd2im.png")
save(path, fig)
println("\nWrote: $path")
