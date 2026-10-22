# Matrix-free Stokes solvers (`src/solvers/stokes/`)

One dimension-generic implementation (`FluidState{T,D}`/`Grid{T,D}`, D = 2, 3) serves `par.solver ∈ (:dyrel, :gmg, :vmg)`;
`src/solvers/dispatch.jl` dispatches. `:direct` (sparse LU, `src/solvers/direct.jl`) is the
reference: all solvers share one operator and agree to solver tolerance. Each
call solves one frozen-coefficient system; the outer Picard loop in `run!`
refreezes η, ρ, advection and the BD2 history.

## Equations

xcore is compressible two-phase Navier–Stokes at Re > 1, not creeping Stokes.
For velocity component d, with τ = η(ε̇ − ⅓(∇·v)I) (**no factor 2**):

    ∇·τ − ∇P − cin_d ρ_d v_d/dt − (∂_dρ) g0 dt v_d
        + [d=1] Drho g0 + (a2 m° + a3 m°°)/dt − advn_d = 0       (momentum)
    ∇·(ρv) = MFS                                                  (continuity)

- BD2 inertia is an implicit diagonal term plus a lagged momentum-flux history.
  Momentum advection is frozen at the entry velocity (`advection.jl`).
- The drunken-sailor term ∂_dρ·g0·dt·v_d stabilizes density gradients, so `dt`
  must be finite.
- The constraint is on the mass flux ∇·(ρv). `MFS` carries the ∂ₜρ
  compressibility correction and is updated once per solve from
  `phase.advn_rho`.
- The continuity block has no pressure term, matching the assembled direct
  system.
- The residual sign is R = b − A·v, so the DR update is a descent.

## Grid conventions (`layout.jl`)

- Axis 1 is z (gravity, always bounded); axes 2..D are x (then y). Index order
  is (z, x[, y]).
- Component d lives on faces normal to axis d. Ghosted shapes are v[d]:
  N + 2 − e_d and P: N + 2. Cache arrays are compact: R[d]: N + e_d, cell
  fields N.
- The edge viscosity of axis pair (a < b) has shape N + e_a + e_b, stored
  lexicographically (2-D `(etaco,)`; 3-D `(etaco, etayz, etaxy)`). Edge η is
  the geometric mean of the four surrounding cells; face ρ is the arithmetic
  mean of the two adjacent cells.
- Normal stress lives at cells, shear stress at edges.
- Boundary conditions (`bcs.jl`):
  - z: W = 0 at the top. The bottom is closed, draining (W_bot = mfs_mean·D/ρ_w,
    which passes the net volume source out), or open (W from the discrete
    continuity of the bottom cell row).
  - Tangential ghosts reflect: v_ghost = −cnv·v (cnv = +1 no slip, −1 free
    slip; `top_cnv`/`bot_cnv`).
  - Horizontal axes are periodic or solid walls with ghost = −sds·v (sds = −1
    free slip, +1 no slip; periodic requires sds = −1).
  - P is Neumann in z and at walls.
  - Only closed z is supported; `:gmg`/`:vmg` reject `open_cnv`.

## Powell–Hestenes / augmented Lagrangian (`dyrel.jl`)

R_P = (MFS − ∇·(ρv))/ρ is **1/ρ-scaled**. The penalty γ_eff·R_P ~ γ_eff·∇v must
match the Gershgorin bound, which carries no ρ; without the scaling the
explicit step sees a penalty ≈ ρ times too strong and diverges.

Three nested loops:

1. Picard (`run!`, nonlinear).
2. Powell–Hestenes (PH): multiplier update P += ω·γ_eff·R_P, ω =
   `pres_relax_PT`. It is not Schur-scaled; the Schur diagonal enters only
   through γ_eff.
3. Dynamic relaxation (DR): solves (K + penalty)v = f − ∇P at fixed P, with the
   penalty P_num = γ_eff·R_P injected as −∂_d P_num. Preconditioned residual:
   R_DR = (R_d − ∂_d P_num)/D_d.

At the fixed point R_P = 0, so the multiplier update and the penalty vanish and
(v, P) solves the unpenalized saddle point. γ_eff and ω change only the
convergence rate.

**Penalty (`gershgorin.jl`).** γ_eff = γfact/s_P per cell, where
s_P = (1/(h²ρ_c))·Σ_faces ρ_f/D_f is the Schur diagonal of the penalty-free
momentum diagonal. Dirichlet faces are excluded, since counting them deflates
γ_eff in boundary cells. 1/s_P is a viscosity that carries both the viscous and
the inertial scaling, so the dimensionless γfact needs no retuning with ρ/dt.

**γfact self-tuning (`dyrel.jl`).** The contraction is ρ = 1/(1 + γfact·λ),
with λ = λmin(diag(s_P)⁻¹S) measured from ‖R_P‖ over a window of 4 sweeps.
γfact is retuned in closed form to reach ρ_want = 0.5 (growth ≤ 50× per retune,
≤ 6 retunes). Anderson acceleration (`anderson_PH = m`) is optional and resets
on every retune.

## DR autotune (Duretz et al. 2025; `gershgorin.jl`, `reductions.jl`)

    dvdτ ← α·dvdτ + R_DR,   v += β·dτ·dvdτ
    dτ = 2·CFL/√λmax,   c = 2·√λmin·c_fact
    α = (2 − c·dτ)/(2 + c·dτ),   β = 2·dτ/(2 + c·dτ)

- D_d and λmax_d are Gershgorin row sums per face. The viscous part uses
  cn = 2/3, cc = 1/3, cs = 1/2, so D is the true Jacobi diagonal of the
  assembled operator.
- The row sums include the penalty (coupling γ_c·ρ_g/ρ_c from cell to face),
  the BD2 inertia δ = cin·ρ_d/dt and |σ| of the sailor term. D includes only
  max(σ, 0), so an inverted density gradient cannot make D non-positive.
- Wall faces and z-boundary W faces have D = λmax = 1.
- λmin is the Rayleigh quotient |ΣΔv·ΔR|/ΣΔv² from the last step, re-estimated
  every `n_tune_PT` iterations (the only non-local reduction).

## Multigrid (`gmg.jl`, `vmg.jl`, `hierarchy.jl`, `transfer.jl`)

DR needs O(N) iterations (dτ ~ h). The operator is linear within a solve, so
the V-cycle is a correction scheme (not FAS).

- Level 1 wraps the live fluid state and carries the physical forcing
  (`fscale = 1`) and the bottom drain. Coarse levels solve the homogeneous error
  equation A e = bext with `fscale = 0`, no drain, and the restricted residual
  in `bext`.
- Coarsening halves every axis (h_c = 2h); dimensions must be multiples of 32.
  η and ρ are restricted by arithmetic cell means (geometric and harmonic means
  contract more slowly under viscosity contrast). Face densities and edge
  viscosities are rederived from the coarse cell fields, not restricted.
- Transfers: cell restriction averages 2^D cells; face restriction averages the
  2^(D−1) coinciding fine faces (coarse face J = fine face 2J − 1).
  Prolongation is the tensor-product {3/4, 1/4}; faces copy or average along d.
  Ghost neighbors: periodic wrap, Neumann for cells at walls and in z,
  reflection by −sds/−cnv for faces.
- Only the prolonged pressure correction is made mean-free. No velocity
  component has a null mode (Dirichlet or reflected at z).
- The coarse-grid correction is v += γ_mg·e with γ_mg = `mg_gamma` = 1.
- The smoother is a fixed `mg_inner`-iteration DR sweep at frozen P with fixed
  damping c = mg_cfact·√(mean λmax) (no reductions), followed by one multiplier
  update. The coarsest level solves adaptively (early-stop poll, Rayleigh
  retune, and a penalty retune once it is inertia-dominated).

**`:gmg`** is coupled. Every level carries the pseudo-transient term
(P − P0)/(η_b·dt) in R_P, with η_b = γ_eff·max(1, 1/(ρ·dt)). On the fine level
**P0 is advanced to the latest iterate after every V-cycle**, so the term
vanishes at convergence and the fixed point is still ∇·(ρv) = MFS. Coarse
levels use P0 = 0. The P0 tracking is what makes γ_mg = 1 stable. The gate
evaluates the residual with P0 = P (term removed) and also requires the
per-cycle pressure change to be small.

**`:vmg`** is a PH loop whose velocity solve is `nv` velocity V-cycles at fixed
P. No level carries the pseudo-transient term, and the hierarchy never sees P.
It runs with γfact = 5, because point-DR smoothing of the penalized block
degrades as γfact grows. `cc = true` adds the Cahouet–Chabard step
P += (ω/dt)·A⁻¹(ρR_P), where A = −∇·(cin⁻¹∇·) is a constant-coefficient Poisson
operator: when the momentum block reduces to its inertia,
v_d = −dt/(cin ρ_d)∂_dP and ρ cancels. A⁻¹ is one V-cycle (weighted Jacobi
ω = 0.8, ν = 2, mean-free right-hand side and result). It sets only the step
size, not the fixed point.

## Convergence gate

Error = max(RMS velocity residuals relative to the first sweep, RMS ρ·R_P,
pressure-update norm) < `atol_PH`. The pressure-change term is required because
residuals bound the pressure error only up to the Schur conditioning.
`atol_PH ≤ atol` is enforced. `forcing_PH` allows an inexact solve
(`relaxed`); the Picard loop still ends only on a solve that reached `atol_PH`.
Failure to converge throws.

## Precision and GPU

Float32 stalls at a residual floor (~5e-4 on a buoyancy-blob benchmark, CPU and
Metal alike). It is a precision floor, not a bug; use Float64 for tight
tolerances. Reductions use lazy `Broadcasted` sums to avoid scalar indexing on
GPU arrays, and kernels launch without per-kernel `synchronize`.

## References

- Powell (1969); Hestenes (1969): augmented Lagrangian.
- Frankel (1950); Räss, Utkin, Duretz, Omlin, Podladchikov (2022): accelerated
  pseudo-transient iteration.
- Duretz, de Montserrat, Sevilla, Räss, Utkin, Spang (2025), egusphere-2025-5641:
  autotuned DYREL (Gershgorin λmax, Rayleigh λmin).
- JustRelax.jl `src/DYREL/` (solver and Gershgorin kernels).
- Cahouet & Chabard (1988): inertial Schur preconditioner.
- Brandt (1977); Trottenberg, Oosterlee, Schüller (2001): multigrid.
