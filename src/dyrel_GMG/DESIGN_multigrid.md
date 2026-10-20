# Geometric multigrid with PT/DR smoothing — design for xcore-DYREL

**Goal:** kill the O(N) growth in pseudo-transient iteration count by wrapping the
existing Dynamic-Relaxation (DR) solver as a *smoother* inside a geometric
multigrid (GMG) V-cycle. PT/DR damps high-frequency error in a few sweeps but
crawls on smooth error (so #iters ~ 1/h ~ N). MG resolves smooth error on coarse
grids where it is high-frequency again ⇒ **grid-independent iteration counts,
O(N) work**, matrix-free, GPU-portable (restrict/prolong are stencils, smoother is
the existing KA kernels).

Correction-scheme V-cycle with staggered restrict/prolong for p/w/u. The MATLAB
reference implementation (`~/Documents/PhD/xcore/src/fluidmech.m`) is a direct
sparse solve with no multigrid, so only the operator, BCs and coefficient
averaging are inherited from it; the V-cycle below is not validated against it.

---

## 1. Scheme choice

- **Correction-scheme (linear) MG, not FAS.** Within one `fluidmech_dyrel!` call
  the viscosity `η`/`etaco`, densities `ρ`/`rhow`/`rhou`, advection, and BD2
  history are *frozen* ⇒ the momentum+continuity operator `A` is **linear** in
  `(W,U,P)`. CS-MG (restrict residuals, prolong corrections) is therefore exact.
  This matches the MATLAB code (it restricts `res_*` and prolongs `upd_*`).
- **MG is the inner Stokes solver; the existing Picard loop in `run!` stays.**
  Picard handles the nonlinearity (η, ρ, advection updates); each Picard sweep
  calls one (or a few) MG V-cycles instead of the bare PH/DR iteration. Do NOT
  fold Picard into FAS unless later desired.
- **MG replaces the inner DR loop**, not the Powell–Hestenes pressure structure:
  the smoother still does coupled velocity-DR + pressure-penalty updates; MG just
  accelerates the convergence of that coupled relaxation across scales.

---

## 2. Operator in residual form `A·v = b` (the key refactor)

Current `compute_dyrel_residual_{W,U,P}_kernel!` compute `R = A·v − b` with the
RHS `b` (buoyancy `body`, BD2 `lagged`, momentum advection `adv`, continuity
`MFS`) **baked in**. For MG the smoother must solve `A·v = b_ext` for an
*arbitrary* `b_ext` (on coarse levels `b_ext` = restricted fine residual).

**Refactor:** add optional additive external-RHS arrays to the residual kernels:

```
R_W[iz_f,ix] = (KV·W + GG·P)|_fine_terms  −  rr_W  −  bext_W[iz_f,ix]
R_U[iz,ix]   = … − rr_U − bext_U[iz,ix]
R_P[iz,ix]   = (MFS − ∇·ρv − comp)/ρ  −  bext_P[iz,ix]
```

- `bext_*` default to `nothing`/zero ⇒ **fine-grid behaviour unchanged** (the
  existing physical RHS `rr` stays in place; this is the level-0 problem).
- On coarse levels the physical `rr` terms (`body`, `lagged`, `adv`, `MFS`) are
  set to zero and `bext_*` carries the restricted residual — i.e. coarse levels
  solve the homogeneous *error* equation `A_c·e_c = r_c`.
- Cleanest implementation: a `level::Int` or a `homogeneous::Bool` flag selecting
  whether the physical-forcing terms are included, plus the `bext_*` arrays in
  the per-level cache. `bext_*` shapes = `R_W (Nz+1,Nx)`, `R_U (Nz,Nx+1)`,
  `R_P (Nz,Nx)`.

Expose the smoother as:

```
smooth!(v::LevelState, b::LevelRHS, lvl::MGLevel; nsweeps, homogeneous) -> nothing
```

solving `A_lvl · v = b` with `nsweeps` DR sweeps (+ the periodic Gershgorin /
Rayleigh retune already in the DR loop). `v=(W,U,P)`, `b=(bext_W,bext_U,bext_P)`.

---

## 3. Field shapes & staggering (xcore convention: index `(iz, ix)`, z=dim1, x=dim2)

| field | full (FluidState) | interior / cache | location |
|---|---|---|---|
| W (z-vel) | `(Nz+1, Nx+2)` | `R_W (Nz+1, Nx)` | z-faces |
| U (x-vel) | `(Nz+2, Nx+1)` | `R_U (Nz, Nx+1)` | x-faces |
| P / δP | `(Nz+2, Nx+2)` | `R_P (Nz, Nx)` | cell centres |
| η | `(Nz, Nx)` | — | cell |
| etaco | `(Nz+1, Nx+1)` | — | corner |
| ρ | `(Nz, Nx)` | — | cell |
| rhow | `(Nz+1, Nx)` | — | z-face |
| rhou | `(Nz, Nx+1)` | — | x-face |

Coarsening: `Nz_c = Nz÷2`, `Nx_c = Nx÷2`, `h_c = 2h`. Requires `Nz, Nx` divisible
by 2 down to `minlvl` (enforce in the hierarchy builder; pad/abort otherwise).
NB MATLAB stores `(comp, x, z)` (x=dim2, z=dim3); xcore is `(z, x)` — every
MATLAB `dim2↔x`, `dim3↔z` maps to xcore `dim2↔ix`, `dim1↔iz`. The operators
below are already written in xcore `(iz, ix)` order.

---

## 4. Restriction (fine → coarse), as KA kernels

Cell-centred (p, and any cell coefficient), 2×2 average — MATLAB `restrict_p`:
```
Pc[J,I] = ( P[2J-1,2I-1] + P[2J,2I-1] + P[2J-1,2I] + P[2J,2I] ) / 4      # (Nz/2, Nx/2)
```
z-face (w) — coarse z-face coincides with odd fine z-face, average the 2 fine
x-cells — MATLAB `restrict_w`:
```
Wc[J,I] = ( W[2J-1, 2I-1] + W[2J-1, 2I] ) / 2                            # (Nz/2+1, Nx/2)
```
x-face (u) — symmetric, average the 2 fine z-rows — MATLAB `restrict_u`:
```
Uc[J,I] = ( U[2J-1, 2I-1] + U[2J, 2I-1] ) / 2                            # (Nz/2, Nx/2+1)
```
(Boundary faces: coarse z-face `J=Nz_c+1` ← fine `iz_f=Nz+1`; coarse x-face
`I=Nx_c+1` ← fine `ix=Nx+1`, mirroring the MATLAB end-index handling.)

---

## 5. Prolongation (coarse → fine), as KA kernels

Bilinear interpolation with BC-aware ghosts (periodic-x, closed-z), porting
MATLAB `prolong_p/w/u`:
- **p (cell)**: tensor-product `{1/4,3/4}` ⊗ `{1/4,3/4}` ⇒ the `9/16,3/16,3/16,1/16`
  4-point stencil over the 4 fine cells in each coarse cell. Ghost ring filled
  periodic-x / closed-z (`closed`: linear extrapolation `2·a−b`) before the
  weighted gather.
- **w (z-face)**: fine z-faces that coincide with coarse faces use `{1/4,3/4}` in
  x only; intermediate fine z-faces average the two coarse faces in z then `{1/4,3/4}`
  in x (the `1/8,1/8,3/8,3/8` stencil). x-ghosts periodic.
- **u (x-face)**: mirror of w with z↔x.

Each prolong is a gather kernel over the fine array reading the (ghost-extended)
coarse array — GPU-clean.

---

## 6. Coefficient coarsening + per-level autotune

The coarse smoother needs coarse operators. Per level, restrict:
- `η` (cell, restrict_p), `ρ` (cell, restrict_p), `rhow` (z-face, restrict_w),
  `rhou` (x-face, restrict_u). `etaco` (corner) — restrict via corner average OR
  recompute from coarse `η` with the existing `compute_corner_eta!` (preferred:
  consistent with the fine-grid construction).
- Then recompute **per level**: `gershgorin!` → `D_W/D_U/λmax` (with `h_c`,
  inertia coeff using the same `a1,γ,dt`), `init_dyrel_penalty!` → `γ_eff`,
  `update_dτ_α_β!`. Exactly the MATLAB `closures` rebuild per level. The
  inertia/sailor `dt` is the *same* physical dt across levels (only `h` changes).

This coefficient hierarchy is the **bulk of the implementation** and the main
correctness risk (get the face-density restriction + `h_c` scaling right).

---

## 7. Smoother (`relax`) in correction form

Reuse the inner-DR body from `solve.jl` (snapshot residuals → stress → continuity
residual → Schur penalty `P_num=γ_eff·R_P` → preconditioned DR momentum residual
with `bext` injected → damped velocity update → BCs → periodic Gershgorin/Rayleigh
retune), but:
- driven for a fixed small `nsweeps` (pre/post-smoothing counts, e.g. 2–4) rather
  than to convergence;
- `homogeneous=true` on coarse levels (drop `body/lagged/adv/MFS`, use `bext`);
- BCs: fine level uses the physical BCs incl. the **bottom-drain** `W_bot =
  MFSmean·D/ρw`; coarse levels (error equation) use **homogeneous** BCs
  (`W_bot=0`, no drain), since the correction must not re-inject the net source.

The Powell–Hestenes pressure-penalty update `P += γ_eff·R_P` stays as part of the
smoother (couples the velocity smoothing to continuity).

---

## 8. V-cycle orchestrator (mirrors MATLAB `multigrid_solve`)

```
vcycle!(v, b, lvl):
    if lvl is coarsest (Nz_c ≤ minlvl):
        smooth!(v, b, lvl; nsweeps = ncoarse)          # "solve" coarsest
        return
    smooth!(v, b, lvl; nsweeps = npre)                 # pre-smooth
    r = residual(v, b, lvl)                            # r_W,r_U,r_P  (= b − A·v)
    bc = ( restrict_w(r_W), restrict_u(r_U), restrict_p(r_P) )   # → coarse RHS
    vc = zeros(coarse)                                 # CS: coarse init 0
    vcycle!(vc, bc, lvl-1)                              # recurse
    # prolong correction, mean-remove (null space), damped apply
    e = ( prolong_w(vc_W), prolong_u(vc_U), prolong_p(vc_P) )
    e_P .-= mean(e_P)                                  # pressure gauge
    (periodic) e_W .-= mean(e_W); e_U .-= mean(e_U)
    v .-= γ_mg .* e                                    # MATLAB: u -= gamma*upd
    smooth!(v, b, lvl; nsweeps = npost)                # post-smooth
```

The outer driver runs V-cycles until the (unscaled, ρ-weighted) residual norm
< `atol_PH`, reusing the existing convergence gate. `γ_mg` is the MATLAB `NUM.gamma`
correction-damping (start 1.0; the MATLAB also relaxes α,β per level — we instead
recompute α,β from per-level Gershgorin, which is cleaner).

---

## 9. Data structures

```
struct MGLevel{T,A}
    grid::Grid{T}                 # coarse grid (Nz_c,Nx_c,h_c)
    η::A; etaco::A; ρ::A; rhow::A; rhou::A      # restricted coefficients
    cache::DyrelCache{T,A}        # per-level D_W/D_U/γ_eff/dτ/α/β/residual scratch
    W::A; U::A; P::A              # level solution (W (Nz_c+1,Nx_c+2) etc.)
    bext_W::A; bext_U::A; bext_P::A             # injected RHS (restricted residual)
end
struct MGHierarchy{T,A}; levels::Vector{MGLevel{T,A}}; minlvl::Int; npre::Int; npost::Int; ncoarse::Int; γ_mg::T; end
```

Built once per `fluidmech_dyrel!` call (coefficients change each Picard sweep) —
OR cache the hierarchy and only re-restrict coefficients + re-tune each call
(cheaper; allocate once). Prefer allocate-once + refresh.

New `Parameters`: `mg::Bool=false` (opt-in), `mg_minlvl::Int`, `mg_npre::Int=2`,
`mg_npost::Int=2`, `mg_ncoarse::Int=8`, `mg_gamma::T=1`, plus reuse `n_tune_PT`,
`CFL_PT`, `c_fact_PT`, `γfact_PT` inside the smoother.

---

## 10. Null space & BCs

- **Pressure** is defined up to a constant (closed-z + periodic-x) ⇒ mean-remove
  the prolonged pressure correction (`e_P .-= mean(e_P)`), as MATLAB does.
- **Periodic-x velocity** also carries a constant mode ⇒ mean-remove `e_W`,`e_U`
  when x is periodic (MATLAB guards this with `any(strcmp(BC,'periodic'))`).
- Coarse-level BCs are **homogeneous** (error equation): `W=0` top/bot (no drain),
  periodic-x ghosts, U reflection — reuse `dyrel_apply_velocity_bcs!` with
  `mfs_mean=0`, and `dyrel_apply_pressure_bcs!`.

---

## 11. Validation plan

1. **2-level first** (one coarsening): verify a V-cycle reduces the residual and
   the converged `(W,U,P)` matches the direct solver to ~1e-8 on N=32. De-risks
   the operator/coarsening/correction-scheme before the full recursion.
2. **Grid-independence**: V-cycles-to-converge should stay ~flat across
   N = 32,64,128,256 (the whole point). Plot iters vs N — must NOT grow ~N.
3. **Conservation**: EB still ~1e-12 (the bottom-drain is on the fine level only;
   coarse corrections are homogeneous so they don't perturb the mass budget).
4. **Cost crossover**: wall-clock vs direct at N=128/200/256 — MG should win at
   large N where bare PT loses and direct's LU factorisation dominates.
5. **GPU**: restrict/prolong/coarsen kernels on JLArray (scalar-off) like the rest.

---

## 12. Open questions / risks

1. **Face-density restriction consistency.** `restrict_w(rhow)` vs recomputing
   coarse `rhow` from coarse `ρ`. Recompute-from-coarse-ρ is more consistent with
   the fine construction (`update_interpolations!`). Decide + test.
2. **etaco coarsening** — restrict corners vs recompute via `compute_corner_eta!`
   from coarse η (prefer recompute).
3. **Smoothing factor of the DR smoother** for this NS+penalty operator — the
   penalty `γ_eff` stiffens high frequencies; may need `γ`-aware smoothing or a
   few extra sweeps. Measure the 2-grid smoothing factor empirically.
4. **Anisotropy at L=1.5D** and non-square coarsening if `Nx` odd at some level
   (MATLAB has the `Nx>=2` guard for 1-D collapse — port it).
5. **Compressible continuity on coarse grids**: the `1/ρ` scaling in `R_P` and the
   Schur penalty interact with restriction; verify the coarse continuity operator
   is the Galerkin-consistent coarsening (or accept rediscretisation, which is
   what geometric MG does — usually fine for smooth coefficients).
6. **Cost of per-call hierarchy refresh** vs convergence gain — profile.

---

## 13. File layout (`src/dyrel/multigrid/`)

```
multigrid/
├── DESIGN_multigrid.md   ← this file
├── transfer.jl           ← restrict_{p,w,u}! + prolong_{p,w,u}! KA kernels
├── hierarchy.jl          ← MGLevel/MGHierarchy structs + build/refresh + coefficient coarsening
├── smoother.jl           ← smooth!(v,b,lvl;…) (refactored inner-DR, external-RHS, homogeneous flag)
├── vcycle.jl             ← vcycle!(...) recursion + outer driver to atol_PH
└── (residuals.jl gains optional bext_{W,U,P} args; solve.jl dispatches par.mg)
```

Touches: `residuals.jl` (external-RHS injection), `solve.jl` (dispatch on
`par.mg`), `parameters.jl` (mg knobs), `xcore.jl` (includes). Keeps the existing
single-grid DR path intact (`par.mg=false`).
