# In-process GMG tuning harness: measures the per-V-cycle contraction factor ρ
# of `GMG.fluidmech_gmg!` on a FROZEN production state (step-1 IC after
# `initialize!`), across (mg_gfact, mg_inner, mg_minlvl) — the numeric
# cycle-factor measurement standard in MG practice. One Julia session, one
# `initialize!` per N, one `deepcopy` per config probe: no per-config JIT, no
# full Picard steps, no run!. A single-grid `fluidmech_dyrel!` reference per N
# anchors the wall-time comparison.
#
# Usage:  julia --project=. usr/run_gmg_harness.jl
# Tweak `Ns` / `configs(N)` below. Acceptance for a good smoother config:
# ρ ≤ ~0.5, flat in N and in hierarchy depth (mg_minlvl sweep).

using xcore
using KernelAbstractions: CPU
using Printf

# ---------------------------------------------------------------------------

base_par(N; kw...) = Parameters(Float64;
    runID = "gmg_harness", outdir = mktempdir(), save_op = false,
    nop = 10^6, nrh = 10^6,
    D = 1e1, N = N, L = 1.5e1, t0end = 2.0,
    d0 = 1e-2, etam0 = 1e1, L0 = 1e-1, l0 = 1e-1, Da = 0.01, Xi = 0.5,
    atol = 1e-4, kw...)

"""
Build the frozen probe state for resolution N. `evolve_steps > 0` advances the
state by that many full time steps with the (stable) single-grid `:dyrel`
solver at γfact_PT=500 first — the production GMG failure appears only on
EVOLVED states (developed η contrast), so tuning against the pristine IC state
is not sufficient (measured: IC-state ρ≈0.10 for the config that NaNs in
production at Picard iter 3).
"""
function build_state(N; evolve_steps::Int = 0)
    par    = base_par(N; solver = :dyrel, γfact_PT = 500.0)
    grid   = Grid(par)
    scales = compute_scales(par, grid)
    fluid  = FluidState(Float64, CPU(), grid.Nz, grid.Nx)
    phase  = PhaseState(Float64, CPU(), grid.Nz, grid.Nx)
    ns     = NoiseState(Float64, grid, scales)
    hst    = History(Float64)
    t0, dt0, s0 = initialize!(phase, fluid, ns, hst, grid, par, scales)
    if evolve_steps > 0
        _, dt0 = run!(phase, fluid, ns, hst, grid, par, scales;
                      nsteps = evolve_steps, time = t0, dt = dt0, step = s0,
                      verbose = false)
    end
    return (; grid, scales, fluid, phase, dt0)
end

"Run one GMG solve on a fresh copy of `base`; return (ncyc, ρ, errs, wall, conv)."
function probe(base, par)
    fc = deepcopy(base.fluid)
    pc = deepcopy(base.phase)
    hier = xcore.GMG.build_hierarchy(fc, pc, base.grid, par)
    logf = tempname()
    local st, wall
    open(logf, "w") do io
        redirect_stdout(io) do
            wall = @elapsed st = xcore.GMG.fluidmech_gmg!(
                fc, pc, hier, base.grid, par, base.scales;
                dt = base.dt0, a1 = 1, a2 = 1, a3 = 0, verbose = true)
        end
    end
    errs = Float64[]
    for ln in eachline(logf)
        m = match(r"err=([0-9.eE+\-]+|NaN|Inf)", ln)
        m === nothing || push!(errs, parse(Float64, m.captures[1]))
    end
    ρ = length(errs) ≥ 3 && errs[1] > 0 && isfinite(errs[end]) ?
        (errs[end] / errs[1])^(1 / (length(errs) - 1)) : NaN
    return (; ncyc = st.ncyc, ρ, errs, wall, conv = st.converged,
              nlev = length(hier.levels))
end

"Single-grid dyrel reference solve on a fresh copy of `base`."
function dyrel_ref(base, N)
    par = base_par(N; solver = :dyrel)
    fc = deepcopy(base.fluid);  pc = deepcopy(base.phase)
    cache = xcore.DyrelCache(Float64, CPU(), base.grid)
    wall = @elapsed st = xcore.fluidmech_dyrel!(fc, pc, cache, base.grid, par, base.scales;
                                                dt = base.dt0, a1 = 1, a2 = 1, a3 = 0)
    return (; st.converged, st.PH_iters, st.total_iter, wall)
end

# ---------------------------------------------------------------------------

# Ns      = [parse(Int, get(ENV, "XCORE_N", "256"))]
Ns      = [128,256,512]
EVOLVE  = parse(Int, get(ENV, "XCORE_EVOLVE", "0"))

# config axes: per-level smoothing penalty, exact DR count per sweep, depth.
# The strong-penalty configs (gf = γfact_PT) are the production candidates;
# the depth sweep is the stability-critical axis on evolved states.
configs(N) = vcat(
    [(gf = 50.0, ni = 32, ml = ml) for ml in (N == 128 ? [16, 64] : [16, 32, 64, 128])],
    [(gf = 50.0, ni = 8,  ml = 16)],
    [(gf = 2.0,  ni = 8,  ml = 16)],     # weak-penalty reference
)

for N in Ns
    println("\n================ N = $N  (evolve_steps = $EVOLVE) ================")
    base = build_state(N; evolve_steps = EVOLVE)
    r = dyrel_ref(base, N)
    @printf("dyrel ref:  conv=%-5s PH=%-3d itPT=%-6d wall=%.2fs\n",
            r.converged, r.PH_iters, r.total_iter, r.wall)
    @printf("%-7s %-6s %-7s %-5s | %-5s %-9s %-8s %s\n",
            "gfact", "inner", "minlvl", "lev", "ncyc", "rho", "wall", "conv")
    for c in configs(N)
        par = base_par(N; solver = :gmg, mg_gfact = c.gf, mg_inner = c.ni,
                       mg_minlvl = c.ml, maxit_PH = 16)   # bounded probe
        p = probe(base, par)
        @printf("%-7.1f %-6d %-7d %-5d | %-5d %-9.3g %-8.2f %s\n",
                c.gf, c.ni, c.ml, p.nlev, p.ncyc, p.ρ, p.wall, p.conv)
    end
end
