# :dyrel / :gmg GPU performance assessment — handoff

Branch: `pa-dyrel_gpu_gmg`. This file is the cross-machine handoff (Claude Code
chat history and the local agent-memory do **not** transfer between machines —
this doc + the committed code do). A fresh session on a CUDA box: read this,
then run `usr/run_bench_gpu.jl`.

## What was done (committed on this branch)

Matrix-free solver GPU-readiness, applied to **both** `src/dyrel/` and
`src/dyrel_GMG/` (GMG keeps duplicated kernel copies). Reference: JustRelax
`src/DYREL/solver.jl`.

- **Removed per-kernel `KernelAbstractions.synchronize(backend)`** from the hot
  wrappers (stress, residuals_V/P, velocity, gershgorin, schur_diag, dτ_α_β).
  Was ~6 host↔device syncs per inner iteration → fatal to GPU pipelining. KA
  keeps same-backend launches ordered; sync is only needed before host
  scalar-reads (the reductions provide it).
- **Fixed allocating + scalar-indexing reductions** in `reductions.jl`. The
  multi-arg `mapreduce(f,+,A,B,C)` allocated ~96 KiB/call AND forced scalar
  indexing on GPU. Replaced with `sum(_bc(f, A,B,C))` (lazy `Broadcasted`
  reduce) — ~96 B/call, GPU-friendly, mirrors JustRelax `norm(D .* R)`.

## Verified

- dyrel matches direct solver: relW ~1e-6, bit-stable, same iteration count (CPU F64).
- Reduction allocs 98.6 KB → 16 B/call; full-solve allocs N=128 74.5 → 26 MiB (−65%).
- No scalar indexing: dyrel runs under `allowscalar(false)` on JLArrays AND on
  real Metal hardware (F32).
- Mass conservation preserved: dyrel EB 2.7e-15, gmg 5.8e-11 (production setup).

## CPU profiling (production run_D1_dm2_e1, N=128, 128×192)

The fluid-mechanics solve dominates each step (phsevo ~1.6 ms/iter, update
~5 ms/iter are negligible). Per-Picard-sweep cost (warm): direct ~0.38 s,
dyrel ~1.0–1.5 s, gmg 1.74→0.64→0.40 s (falls as field settles; by step 3
gmg ≈ direct). dyrel inner-loop tuning is already optimal (swept
n_tune/c_fact/rel_drop/CFL — defaults win). On CPU, direct (UMFPACK) wins at all
N tested; the matrix-free payoff is GPU / 3D / very-high-N.

## Metal findings (Float32 — Apple GPUs have no native F64)

`usr/run_bench_gpu.jl`, buoyancy blob, CPU-F32 vs Metal-F32:

| N   | CPU-F32 | Metal-F32 | speedup |
|-----|--------:|----------:|--------:|
| 64  | 2.2 s   | 51 s      | 0.04×   |
| 128 | 17 s    | 96 s      | 0.18×   |
| 256 | 34 s    | 51 s      | 0.66×   |

Two conclusions:
1. **Latency-bound at these sizes.** ~16 kernel dispatches per inner iteration ×
   tens of thousands of iterations; at 64²–256² the GPU can't amortize Metal's
   per-dispatch latency. Speedup rises with N (0.04→0.18→0.66×) → break-even is
   beyond N≈300, *unless* we cut launches/iter (see "Next").
2. **F32 residual floor.** The F32 solve does not reach tight tolerances (CPU-F32
   and Metal-F32 agree and both stall ~5e-4). Not a bug — a precision floor.
   **CUDA F64 avoids this entirely.**

## What to run on CUDA

```
# F64 sweep (CUDA supports F64 — use it; reaches tight tolerances)
XCORE_NS=128,256,512,1024 XCORE_GPU=cuda XCORE_FLOAT=64 julia --project=. usr/run_bench_gpu.jl
# F32 sweep for comparison (throughput ceiling)
XCORE_NS=128,256,512,1024 XCORE_GPU=cuda XCORE_FLOAT=32 julia --project=. usr/run_bench_gpu.jl
```

Then the production end-to-end on GPU: in `usr/run_D1_dm2_e1.jl` build
`FluidState`/`PhaseState` on `CUDA.CUDABackend()` (NOTE: the shared phsevo /
advect / update / ghost kernels still call per-launch `synchronize`, which works
on CUDA but is a follow-up to strip; and noise FFT/RNG + IC-gen are still
CPU-bound — see memory `project_gpu_readiness`).

## Info needed to continue the assessment on CUDA

1. GPU model + VRAM (e.g. A100 40GB, RTX 4090) and CUDA toolkit/driver version.
2. Confirm `]add CUDA` to the project + `]instantiate` done, `CUDA.functional()` true.
3. Target N range — production is 128/256; for the "GPU wins" regime push to
   512/1024 (and eventually 3D).
4. Whether F64 is the production target (assumed yes; CUDA does F64 well).
5. Paste the `run_bench_gpu.jl` output back so the next steps can be decided.

## Next optimization (now justified by the Metal data)

Reduce **kernel launches per inner iteration** (the GPU bottleneck at these
sizes), which CPU profiling had made look marginal:
- Fuse the ~8 BC slice-broadcasts in `dyrel_apply_velocity_bcs!` into 1–2 kernels
  ("Phase C").
- Fuse `P_num = γ_eff·R_P` into the P-residual kernel; consider fusing
  stress+residual passes.
- Goal: ~16 dispatches/iter → ~5–6, directly cutting the latency-bound time.
This matters most where arrays are small relative to launch latency; on CUDA
(lower dispatch latency than Metal) the break-even N is lower, but the same lever
applies. Re-measure with `run_bench_gpu.jl` after each fusion.

## 2026-06-11 update — solver fixes + backend API + full-loop GPU cleanliness

> CORRECTION ×2 (later same day): the two V-cycle fixes below are real but were
> never the production blocker. After a full audit the actual cause is the AL
> artificial-compressibility `comp` term (∝ 1/γfact); the matrix-free operator
> is IDENTICAL to direct. Raising `γfact_PT` (~5000) makes both `:dyrel` and
> `:gmg` converge the N=256 production Picard loop. See the rewritten subsection
> below; the intermediate "operator mismatch" claim was a test artifact.

**V-cycle fixes (real, but insufficient — see correction):** in the inertia-
dominated production regime ((ρ0/dt)/(η/h²) ≈ 5) the GMG V-cycle with γ_mg=1
DIVERGED geometrically (×1.63/cycle) once ≥3 levels deep — undamped coarse-grid
correction overshoot compounding through the recursion. Plus a second,
independent stall: the unweighted pressure gauge conflicted with the global
mass solvability constraint ∫comp=0, leaving a bit-exact constant P-residual
floor. Fixes: `mg_gamma=0.5` (new default) and the γ_eff-weighted gauge
`Σ(P−P0)/γ_eff=0` (`dyrel_apply_pressure_gauge!(fluid, P0, γ_eff)`, both
solver trees). These make GMG converge ITS OWN residual gate in 9–32 V-cycles —
but to a solution that is wrong by the operator mismatch below.

**THE REAL PRODUCTION BLOCKER — the AL artificial-compressibility `comp` term
(NOT an operator bug).** Full audit: the matrix-free momentum operator is
IDENTICAL to direct. Evaluating the dyrel/gmg residual at the exact direct
solution, *with momentum advection frozen at the entry velocity (as both
`fluidmech!` and `fluidmech_dyrel!` actually freeze it)*, gives `|R_W|=9.4e-12`,
`|R_U|=9.4e-12`, `|R_P|=2.4e-14` — machine zero. (An earlier "6.7e-2 mismatch"
was a TEST ARTIFACT: it recomputed the frozen advection from the *solved*
velocity instead of the entry velocity, injecting a spurious residual.)

The real reason `:dyrel`/`:gmg` don't converge the production Picard loop is the
augmented-Lagrangian term `comp=(P−P0)/(η_b·dt)` (η_b=γ_eff=γfact/s_P) they add
to continuity. dyrel converges to `∇·ρv = MFS − comp`; direct to `∇·ρv = MFS`.
`comp` vanishes only at the OUTER Picard fixed point (P=P0), and the per-solve
difference from direct scales EXACTLY as 1/γfact: at N=64 a fully-converged
dyrel (true residual ~1e-12) differs from direct by relW=2.7e-3 at γfact=50 →
2.8e-5 at γfact=5000 (PH 519→21; relP_gaugefree 8.3e-3→8.4e-5; relP_raw≈8.6e-2
is a harmless gauge constant — direct pins one P cell, dyrel pins the weighted
mean). At default γfact=50 the comp perturbation exceeds Picard `rtol=1e-4`,
breaking the outer loop's contraction ⇒ mmnt wobbles to maxit. `:direct` has no
comp ⇒ converges the same loop in 9 iters.

**PROPER FIX (implemented in `:gmg`, no tuning): pseudo-transient P0-tracking.**
Advance the `comp` reference to the current pressure after each V-cycle
(`P0arr .= fluid.P` in `src/dyrel_GMG/vcycle.jl::fluidmech_gmg!`) — the standard
PT artificial-compressibility scheme (P0 = previous PT iterate, not the frozen
entry pressure). `comp` then measures the per-cycle increment and → 0 as the
V-cycle converges, so the fixed point is `∇·ρv = MFS` exactly at ANY γfact.
Validated at the DEFAULT γfact=50: `:gmg` converges the N=256 production Picard
loop in ~9 iters (rel ~5e-5), matching `:direct` iter-by-iter; the steady blob
gate (relW≈1e-4 vs direct) and machine-precision mass conservation (EB≈1e-11)
are preserved; MMS guarded by `bnchm`. The SAME fix on single-grid `:dyrel` was
HARMFUL (weakens the restoring comp force ⇒ itPH caps at 100, Picard wobble
worsens) and was reverted — `:dyrel`'s high-N stall is the separate O(N²)
smooth-mode PH wall, cured by `:gmg` or larger γfact.

**Alternative lever (both solvers): raise `γfact_PT`** (~5000). comp ∝
1/(γfact·dt), so larger N (smaller dt) wants larger γfact. Validated: γfact=5000
converges both `:dyrel` and `:gmg` at N=256 in ~9 iters. Use this if running
`:dyrel` standalone.

**EFFICIENCY — `:gmg` now BEATS `:direct` (`mg_gamma` 0.5 → 1.0).** The
P0-tracking fix also eliminated the divergence that had forced `mg_gamma=0.5`
(the frozen-P0 biased per-cycle system over-shot under the full correction). A
verbose V-cycle trace showed the bottleneck: U solved in ~3 cycles but the
coupled smooth (W,P) mode decayed at a CONSTANT 0.866/cycle for 64 cycles (the
damped coarse correction was only ~26% effective on it). With P0-tracking each
cycle is a well-posed per-increment solve, so the full correction `mg_gamma=1.0`
is stable and optimal: FLAT 6 V-cycles/solve (textbook, factor≈0.21) vs 64.
`mg_gamma` default restored to 1.0. **Timing (3 production steps, CPU): N=128
gmg 5.87s vs direct 6.22s (1.06×); N=256 gmg 19.7s vs direct 40.3s (2.04×)** —
the speedup GROWS with N (O(N²·6) vs ~O(N³)), so `:gmg` is the large-grid solver
of choice and the right GPU target. Conservation identical to direct. Note: GMG's
`atol_PH` is still a NO-OP (relative gate `err/err0 < rtol` fires first).

**Backend selection:** `xcore_backend()` (CPU), `xcore_backend(:metal)`
(Metal is a project dep; F32 only), `xcore_backend(:cuda)` (CUDA is a WEAK dep
via `ext/xcoreCUDAExt.jl` — on the cluster `pkg> activate; add CUDA` once in
the default env, then `using CUDA` in the run script). No more
KernelAbstractions/CUDA name juggling in scripts.

**GPU cleanliness:** the FULL `initialize!` + `run!` step (incl. noise!,
StepResidual snapshot, history) now runs under `allowscalar(false)` on
JLArrays for BOTH `:dyrel` and `:gmg`. Fixed: noise magnitude diagnostics loop
→ KA kernel; `StepResidual` → backend-generic (`StepResidual(grid, fluid)`).

## Direct solver on GPU — assessment (cuBLAS question)

Short answer: cuBLAS does NOT help `:direct`. LinearAlgebra-on-CuArray
dispatches to cuBLAS for DENSE arrays only; the Stokes matrix is sparse
(~3·N·1.5N unknowns ⇒ ~3×10⁵ rows at N=256 — dense LU would need ~700 GB).
The CPU path is SuiteSparse UMFPACK via `lu(::SparseMatrixCSC)`, and
factorization is 85–91% of solve time (see memory `project_direct_solver_perf`).

GPU options for a SPARSE direct solve, in order of promise:
1. **CUDSS.jl** (NVIDIA cuDSS): proper sparse LU/LDLᵀ on GPU with separate
   analysis/factorization/solve phases AND cheap refactorization for a fixed
   sparsity pattern — which is exactly the Picard loop (pattern fixed, values
   change per sweep). Worth a benchmark on the cluster: assemble on CPU as now,
   `CuSparseMatrixCSR` upload (~40 MB at N=256, negligible), `cudss("G")` LU.
   Caveats: needs the same penalty/pivot-robust formulation UMFPACK handles
   (saddle-point zero P-block), F64 fine on data-center GPUs.
2. CUSOLVER `csrlsvqr` (what you get "for free" via `\`): QR-based, generally
   slower than CPU UMFPACK at these sizes — not recommended.
3. Keep `:direct` on CPU. It is the conservation-grade reference; the matrix-free
   `:gmg`/`:dyrel` ARE the GPU path and (with γfact_PT raised, ~5000) now converge
   the full production model — no operator bug, just the comp/γfact tuning above.

Recommendation: the GPU path (`:gmg`/`:dyrel`) is unblocked — run the CUDA
benchmark with γfact_PT≳1e3. CUDSS.jl is a fallback only if a GPU *direct* solve
is wanted. The remaining matrix-free efficiency work is the GMG V-cycle cap
(smoother/coarse balance), not correctness.
