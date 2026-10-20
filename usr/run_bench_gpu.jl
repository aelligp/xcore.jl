# GPU-vs-CPU benchmark for the matrix-free :dyrel Stokes solver (buoyancy blob).
#
# Runs the SAME problem on CPU and a GPU backend across a grid sweep and reports,
# for each N: convergence flag, inner-iteration count, wall time, CPU/GPU speedup,
# and the velocity rel-diff vs CPU (correctness). The whole GPU solve runs under
# `allowscalar(false)`, so any scalar-indexing regression in the hot loop THROWS
# rather than silently falling back — this is the GPU-readiness gate.
#
# Why this script exists: the per-kernel `synchronize` removal + reduction rewrite
# (branch pa-dyrel_gpu_gmg) only pay off on GPU; on CPU they are perf-neutral.
# This measures the actual GPU behaviour. On Metal (Apple) the matrix-free solve
# is launch-latency-bound at small N (≈16 kernel dispatches/inner-iter); CUDA
# should fare better AND supports Float64, avoiding the F32 residual floor.
#
# Backend auto-detect (override with XCORE_GPU=cuda|metal):
#   CUDA  → Float64 (and Float32) — F64 reaches tight tolerances.
#   Metal → Float32 only          — Apple GPUs have no native F64.
#
# Run:
#   julia --project=. usr/run_bench_gpu.jl
#   XCORE_NS=128,256,512 XCORE_GPU=cuda XCORE_FLOAT=64 julia --project=. usr/run_bench_gpu.jl

using xcore
using KernelAbstractions: CPU
using LinearAlgebra: norm
using Printf

# ---------------------------------------------------------------------------
# Backend selection — load the requested/available GPU package AT TOP LEVEL (a
# `using` inside a function won't be visible in the same world age), then expose
# the backend + its allowscalar/synchronize + F64 support as a NamedTuple.
# ---------------------------------------------------------------------------
const _WANT = lowercase(get(ENV, "XCORE_GPU", "auto"))

const GPU = let
    sel = nothing
    if _WANT in ("cuda", "auto")
        try
            @eval using CUDA
            if CUDA.functional()
                sel = (be=CUDA.CUDABackend(), name="CUDA",
                       allowscalar=CUDA.allowscalar, sync=() -> CUDA.synchronize(), f64=true)
            end
        catch e
            _WANT == "cuda" && rethrow(e)
        end
    end
    if sel === nothing && _WANT in ("metal", "auto")
        try
            @eval using Metal
            sel = (be=Metal.MetalBackend(), name="Metal",
                   allowscalar=Metal.allowscalar, sync=() -> Metal.synchronize(), f64=false)
        catch e
            _WANT == "metal" && rethrow(e)
        end
    end
    sel === nothing &&
        error("no functional GPU backend (set XCORE_GPU=cuda|metal)")
    sel
end

# ---------------------------------------------------------------------------
# Buoyancy-blob steady Stokes problem (same as test/test_dyrel.jl), built on an
# arbitrary backend. Tolerances differ by precision: F64 reaches 1e-8; F32 has a
# higher residual floor, so we use a looser atol and report the honest conv flag.
# ---------------------------------------------------------------------------
function build(::Type{T}, backend, N) where {T}
    atolPH  = T <: Float32 ? 5.0f-4 : 1.0e-8
    reldrop = T <: Float32 ? 1.0f-2 : 1.0e-3
    par = Parameters(T; D=1.0, N=N, L=1.0, etam0=1.0, g0=1.0,
                     γfact_PT=50.0, CFL_PT=0.99, c_fact_PT=0.5,
                     maxit_PH=50, maxit_PT=50_000, n_tune_PT=50,
                     atol_PH=atolPH, rel_drop_PT=reldrop, gamma=0.0)
    grid = Grid(par);  scales = compute_scales(par, grid)
    Nz, Nx = grid.Nz, grid.Nx
    fluid = FluidState(T, backend, Nz, Nx)
    phase = PhaseState(T, backend, Nz, Nx)
    dy    = DyrelCache(T, backend, grid)
    for f in (fluid.eta, fluid.etaco, fluid.rho, fluid.rhow, fluid.rhou,
              fluid.rhoo, fluid.rhooo); fill!(f, one(T)); end
    for f in (fluid.MFS, fluid.rhoWo, fluid.rhoWoo, fluid.rhoUo, fluid.rhoUoo,
              fluid.W, fluid.U, fluid.P, fluid.drhodt, fluid.drhodto,
              fluid.drhodtoo); fill!(f, zero(T)); end
    fill!(phase.advn_rho, zero(T))
    Drho_h = zeros(T, Nz + 1, Nx)             # build on host, copy to device once
    for j in 1:Nz+1, i in 1:Nx
        zc = (j - 1) * grid.h;  xc = (i - T(0.5)) * grid.h
        Drho_h[j, i] = exp(-(((xc - T(0.5)) / T(0.1))^2 + ((zc - T(0.5)) / T(0.1))^2))
    end
    copyto!(fluid.Drho, Drho_h)
    return par, grid, scales, fluid, phase, dy
end

solve!(f, p, dy, g, par, sc, dt) =
    fluidmech_dyrel!(f, p, dy, g, par, sc; dt=dt, a1=1, a2=1, a3=0,
                     top_cnv=-1, bot_cnv=-1)

# ---------------------------------------------------------------------------
function run_bench(::Type{T}) where {T}
    be = GPU.be;  name = GPU.name;  allowscalar = GPU.allowscalar;  gsync = GPU.sync
    NS = parse.(Int, split(get(ENV, "XCORE_NS", "64,128,256"), ","))
    dt = T(1.0e6)
    @printf("\n==== :dyrel  CPU vs %s   (%s)  ====\n", name, T)
    @printf("%-5s | %-26s | %-26s | %-8s | %-9s\n",
            "N", "CPU", name, "speedup", "rel-diff")
    for N in NS
        par, g, sc, fc, pc, dyc = build(T, CPU(), N);  solve!(fc, pc, dyc, g, par, sc, dt)
        par, g, sc, fc, pc, dyc = build(T, CPU(), N)
        tc = @elapsed rc = solve!(fc, pc, dyc, g, par, sc, dt)
        Wc = Array(@view fc.W[:, 2:end-1])

        par, g, sc, fm, pm, dym = build(T, be, N)
        allowscalar(false)
        solve!(fm, pm, dym, g, par, sc, dt);  gsync()         # warm
        par, g, sc, fm, pm, dym = build(T, be, N)
        tm = @elapsed (rm = solve!(fm, pm, dym, g, par, sc, dt); gsync())
        allowscalar(true)
        Wm = Array(@view fm.W[:, 2:end-1])

        rel = norm(Wm .- Wc) / (norm(Wc) + eps(T))
        @printf("%-5d | conv=%-5s it=%-6d %7.3fs | conv=%-5s it=%-6d %7.3fs | %6.2fx | %.2e\n",
                N, rc.converged, rc.total_iter, tc,
                rm.converged, rm.total_iter, tm, tc/tm, rel)
    end
end

# pick precision: F64 default on CUDA, F32 on Metal (auto), or force XCORE_FLOAT
let floatenv = get(ENV, "XCORE_FLOAT", "auto")
    if floatenv == "32"
        run_bench(Float32)
    elseif floatenv == "64"
        run_bench(Float64)
    else  # auto: F64 on CUDA, F32 on Metal
        run_bench(GPU.f64 ? Float64 : Float32)
    end
end
