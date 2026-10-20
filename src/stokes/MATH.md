# Mathematical basis of the xcore DYREL Stokes solver

This documents the equations the DYREL (Dynamic-Relaxation / pseudo-transient)
solver in `src/stokes/` actually solves, the augmented-Lagrangian / Powell–Hestenes
structure, the pseudo-transient velocity relaxation, and — in detail — the
**Schur-complement-scaled pressure update** (`P += ω·R_P/s_P`). Symbols match the
code (`R_W`,`R_U`,`R_P`, `D_W`,`D_U`, `γ_eff`, `s_P`, …).

---

## 1. Continuous problem

xcore is **compressible two-phase Navier–Stokes at Re > 1** (not creeping Stokes).
Per momentum component, with deviatoric stress `τ = 2η(ε̇ − ⅓(∇·v)I)`:

```
∇·τ − ∇P + ρg − a₁ ρ ∂ₜv − (advection) = 0           (momentum)
∇·(ρv) = MFS                                          (mass / continuity)
```

- **Inertia** enters via the BD2 time derivative `a₁ρv/dt` (current) minus the lagged momentum-flux history `(a₂ρv° + a₃ρv°°)/dt`.
- **Momentum advection** `∇·(v⊗ρv)` is frozen at the entry velocity each solve.
- **Drunken-sailor** stabilisation `g·∂ρ·dt·v`.
- **Compressible continuity**: the constraint is on the **mass flux** `∇·(ρv)`, not `∇·v`; the source `MFS` carries the `∂ₜρ` compressibility correction accumulated by the outer Picard loop.

Boundaries: periodic in x; `W=0` (closed) top, prescribed drain `W_bot = MFSmean·D/ρ_w` bottom (passes the net mass source out — see `bcs.jl`); free/
no-slip tangential via `top_cnv`/`bot_cnv`.

---

## 2. Discrete saddle-point system

On the staggered MAC grid, freezing the nonlinear coefficients (η, ρ, advection,
BD2 history) — valid **within one solve**, the outer Picard loop in `run!`
updates them — the linearised system is a saddle point:

```
[ K   G ] [ v ]   [ f ]
[ Gᵀ  0 ] [ P ] = [ b_c ]
```

- `v = (W, U)` face velocities; `P` cell pressures.
- `K` = momentum operator: viscous `−∇·2ηε̇` (SPD) **plus** the BD2 inertia
  diagonal `a₁ρ/dt` (and `(a₁+γ)ρ/dt` for U) and the drunken-sailor diagonal.
  At Re>1 / large ρ the inertia term is a *large positive diagonal* — `K` is
  strongly diagonally dominant (this matters in §5).
- `G = −∇` (pressure gradient in momentum); `Gᵀ = ∇·(ρ·)` (mass-flux divergence).
- `f` = buoyancy + lagged inertia + advection; `b_c = MFS`.

The xcore residuals (`residuals.jl`, sign `R = −(K v + G P − f)` so the velocity
update is a descent — see [[project-governing-equations]]):

```
R_W = ∂z τzz + ∂x τxz − ∂z P − inertia − sailor + (body + lagged − adv)        (= f_W − (Kv+GP)_W)
R_U = ∂x τxx + ∂z τxz − ∂x P − inertia − sailor + (lagged − adv)
R_P = ( MFS − ∇·(ρv) ) / ρ                                                    (continuity residual, 1/ρ-scaled)
```

The continuity residual carries **no pressure term**: the discrete constraint is
`∇·(ρv) = MFS` with an empty pressure block (`fluidmech.jl`), and the physical
compressibility reaches it through `MFS`, which the outer Picard loop accumulates
from `∂ₜρ`. An artificial-compressibility term `(P−P₀)/(η_b·dt)` added here has no
counterpart in that system: held at a fixed `P₀` it does not vanish at the DYREL
fixed point, which then solves `∇·(ρv) = MFS − comp`, biased off the direct
solution by `O(1/γfact)`. The multigrid smoother (§7) does carry such a term, as a
*pseudo-transient* regulariser whose `P₀` tracks the previous V-cycle iterate, so
there it vanishes at convergence.

The **1/ρ scaling** of `R_P` is essential: the Schur penalty injected into
momentum is `γ_eff·R_P ~ γ_eff·∇v`, matching the Gershgorin `λmax` (which carries
no ρ); without it the explicit pseudo-transient step sees a penalty ρ≈2700×
too strong and diverges.

---

## 3. Augmented-Lagrangian / Powell–Hestenes outer loop

We solve the saddle point by an augmented-Lagrangian (Powell–Hestenes) iteration.
Augment the momentum operator with the penalty `γ_eff Gᵀ`-term and update the
pressure multiplier:

```
(K + γ_eff·penalty) v = f − G P^k          (velocity solve, §4)
P^{k+1} = P^k + Δ(R_P)                      (multiplier update, §5)
```

- The penalty term is injected as `P_num = γ_eff·R_P`, added to the momentum
  residual as `−∂(P_num)` (`dyrel.jl`, `dr=true`). It stiffens `K` against the
  constraint, accelerating the multiplier iteration. `γ_eff = γfact·⟨η⟩`.
- At a fixed point `R_P → 0`, so the multiplier update vanishes and `(v,P)`
  solves the saddle point regardless of the update *coefficient* — the
  coefficient sets only the convergence **rate** (§5).

This is the **outer (PH) loop** `itPH`. It is NOT the Picard loop (which is the
outer *nonlinear* fixed-point in `run!`, re-freezing η/ρ/advection), nor the
pseudo-transient loop (§4).

---

## 4. Pseudo-transient (dynamic-relaxation) velocity solve

The velocity sub-solve `(K+penalty)v = f − GP` is driven to steady state by a
**damped second-order pseudo-transient** iteration (Frankel / dynamic
relaxation; Räss et al. 2022; Duretz et al. 2025):

```
dV/dτ  = α·dV/dτ + R_DR              (damped pseudo-rate;  R_DR = preconditioned residual)
V     += β·dτ·dV/dτ
```

with per-face Jacobi preconditioning `R_DR = R / D` (D = Gershgorin diagonal),
and autotuned coefficients:

```
dτ = 2·CFL/√λmax        (per face;  λmax from Gershgorin row sums)
c  = 2·√λmin·c_fact      (global damping;  λmin from a Rayleigh quotient)
α  = (2 − c·dτ)/(2 + c·dτ),   β = 2·dτ/(2 + c·dτ)
```

`D_W`, `D_U`, `λmax` (`gershgorin.jl`) include the viscous, penalty (`γ_eff`),
**and BD2 inertia** contributions to the operator diagonal — so `D` is the true
momentum diagonal in the Re>1 regime. This is the **inner (DR) loop** `itPT`.

---

## 5. The Schur-complement pressure update (the key point)

> **CURRENT DESIGN (2026-06): operator-derived per-cell augmented-Lagrangian
> penalty.** The penalty is `γ_eff[i] = γfact / s_P_phys[i]`, where `s_P_phys`
> is the Schur diagonal (§5.2) of the PENALTY-FREE momentum operator (viscous
> η-convention + BD2 inertia; `gershgorin!` with `γ_eff=0`). This same per-cell
> `γ_eff` is used CONSISTENTLY in (i) the momentum penalty `P_num = γ_eff·R_P`,
> (ii) the multiplier update `P += ω·γ_eff·R_P` (`ω = pres_relax_PT`, default 1),
> and (iii) `η_b = γ_eff`, the coefficient of the pseudo-transient term the
> multigrid smoother adds to the continuity residual (§2). Because `1/s_P` is a
> viscosity `[Pa·s]` carrying the full operator scaling (viscous + `a₁ρh²/dt`
> inertia), **`γfact` is a single DIMENSIONLESS, scale-invariant constant** —
> validated `γfact=50` converges in ~O(10) Picard iters across η∈[1e-1,1e4],
> D∈[10,100], ρ×0.1, N∈[40,80] with no per-run tuning. The earlier scalar
> `γfact·⟨η⟩` (creeping-only, needed ρ/dt-retuning at Re>1) and the
> Schur-scaled `P += ω·R_P/s_P` (decoupled from the penalty ⇒ Uzawa-slow, and
> NaN with `comp`) are both superseded. §5.2 derives `s_P`; §5.3–5.4 below are
> retained as background.

### 5.1 Why a constant penalty is wrong

The classical PH/Uzawa multiplier update is `P += ω·R_P` with a **constant**
`ω = γ_eff = γfact·⟨η⟩` (JustRelax uses `γfact≈20`). For the multiplier iteration
`P^{k+1} = P^k + ω·R_P(P^k)` to contract, `ω` must approximate the inverse of the
**Schur complement**

```
S = Gᵀ K⁻¹ G        (maps a pressure to the continuity defect it induces).
```

The optimal/Newton step is `P += S⁻¹ R_P`. A scalar `ω` only works when `S` is
near-uniform. In xcore it is **not**, because `K` carries the BD2 inertia diagonal
`a₁ρ/dt`: for incompressible creeping Stokes `K ~ η∇²` and `S ~ η⁻¹` (so a
viscosity-scaled constant works — JustRelax's `γfact≈20`); but at Re>1 with
`ρ≈2700`, `K ~ a₁ρ/dt + η∇²` is inertia-dominated, `K⁻¹` is much smaller, and
`S ~ Gᵀ(a₁ρ/dt)⁻¹G` is correspondingly different and spatially variable. A
constant `ω` then mis-scales the update, which is exactly why xcore needed
`γfact≈10³` (≈50× JustRelax) to limp along, and why the PH loop *caps* in early
Picard sweeps (`conv=false`, `itPH=maxit_PH`) — the multiplier simply converges
too slowly. The same mis-scaling makes the decoupled smoother non-contractive on
coarse multigrid levels (penalty-locking).

### 5.2 The Schur **diagonal** as a cheap, local, adaptive scaling

We approximate `S⁻¹` by its diagonal — a Jacobi preconditioner for the pressure
Schur complement. For cell `i`, the diagonal of `S = Gᵀ K⁻¹ G` is assembled from
the four face couplings, using the **Jacobi/Gershgorin diagonal `D_f`** as the
cheap stand-in for `K` on face `f`:

```
S_ii = Σ_{faces f of cell i}  (∂R_P_i/∂v_f) · (K⁻¹)_ff · (∂R_v_f/∂P_i).
```

The couplings on the staggered grid:

- continuity wrt a face velocity:  `∂R_P_i/∂v_f = ∓ ρ_f /(h·ρ_i)`  (from `∇·(ρv)/ρ`),
- pressure gradient in momentum:   `∂R_v_f/∂P_i = ± 1/h`,
- momentum response (Jacobi):      `(K⁻¹)_ff ≈ 1/D_f`.

Multiplying and summing the four faces gives the **Schur diagonal** computed in
`schur_diagonal!`:

```
s_P[i] = (1/(h²ρ_i)) · Σ_f  ρ_f / D_f
       = invh² /ρ · ( ρu_W/D_U_W + ρu_E/D_U_E + ρw_N/D_W_N + ρw_S/D_W_S ).
```

Because `D_f` contains the inertia term, `s_P` automatically encodes the Re>1
scaling and the local grid spacing — no hand-tuned `γfact`.

### 5.3 The update and the ρ-cancellation

The pressure (multiplier) update becomes the **diagonally-preconditioned
Newton/Uzawa step** (`dyrel.jl`):

```
P_i += ω · R_P_i / s_P[i],        ω = par.pres_relax_PT ~ O(1)   (Newton step at ω=1).
```

The `1/ρ` factor in `R_P` and in `s_P` **cancels**:

```
R_P_i / s_P[i] = [ (MFS−∇·ρv)_i /ρ_i ] / [ (Σ_f ρ_f/D_f)·invh²/ρ_i ]
              =   (MFS − ∇·ρv)_i / ( invh² Σ_f ρ_f/D_f ),
```

i.e. the update is the **physical mass-flux defect divided by the Schur diagonal**
— dimensionally a pressure, independent of the (large, constant) ρ that forced
the `γfact` blow-up. `ω≈1` is the full Newton–Uzawa step; `ω<1` under-relaxes for
robustness. The momentum penalty `P_num=γ_eff·R_P` is retained for DR
stabilisation with the JustRelax-scale `γfact≈20`.

### 5.4 Multigrid coarse correction — status, failed coupled-smoother attempts, and the recommended architecture

**The problem.** As a multigrid smoother on coarse levels, the augmented-Lagrangian
Uzawa smoother (velocity DR with penalty `P_num=γ_eff·R_P`, then `P += ω R_P/s_P`)
converges *as a standalone 1-level solver* but the **2-level V-cycle diverges**.
Instrumented diagnosis (N=32→16): the coarse *solve* converges, yet its correction
is almost pure pressure (`e_P≈17`, `e_W≈1.7e-3`) — because the inertia-dominated
`K` barely lets the coarse velocity respond — and the prolonged `e_P` produces a
fine pressure-gradient (`∂e_P/h≈55`) that swamps the fine momentum residual (~13).
The rediscretised coarse operator is not consistent *enough* with the fine one for
the correction to reduce the fine residual (classic Stokes/saddle-point MG issue).

**Failed coupled-smoother attempts** (all in xcore's flipped `R=b−Av`, 1/ρ-scaled
convention; the prototypes are not part of the repository):

| attempt | pressure step | velocity correction | result |
|---|---|---|---|
| Vanka (local saddle) | `δP=+ω(R_P−∇·(ρR_v/D)/ρ)/s_P` | `v−=∇δP/D` | V-cycle ~1e8×/cycle → NaN |
| SIMPLE projection | `φ=−ω R_P/s_P` | `v−=∇φ/D`, `P+=φ` | even 1-level diverges (2e119) |

The coupled smoothers destabilise even the 1-level solve — the velocity-correction
sign in the flipped staggered convention and the removal of the AL penalty (`P_num`)
both contribute. Conclusion: **hand-tuning the coupled-smoother signs by V-cycle
trial is the wrong method**; it needs an isolated per-step "does ‖R_P‖ decrease?"
sign-calibration, not full-cycle trial-and-error.

**Recommended architecture (not yet implemented — the right backbone for 3-D/GPU).**
At **Re>1 the momentum block `K ~ a₁ρ/dt·I + η∇²` is inertia-dominated and
well-conditioned** — Jacobi/DR converges on it in O(1) iterations, so `K` does **not
need multigrid**. The only operator with `O(N)` conditioning is the **pressure Schur
complement** `S = Gᵀ K⁻¹ G ~ (dt/ρ)∇²` — a scalar variable-coefficient Laplacian.
Therefore the principled structure is **not** a full velocity+pressure saddle-point
V-cycle, but:

```
outer SIMPLE/Uzawa:
    velocity solve   K v = f − G P        (cheap DR/Jacobi — K well-conditioned)
    pressure-Poisson  (Gᵀ D⁻¹ G) φ = −ρ R_P   ← solve this scalar elliptic by GEOMETRIC MULTIGRID
    correct          v −= D⁻¹ G φ ,  P += φ
```

i.e. apply MG to the **scalar pressure-Poisson** only (textbook cell-centred Poisson
MG: `restrict_p`/`prolong_p` + a Jacobi/red-black Poisson smoother + the
variable-coefficient Laplacian `GᵀD⁻¹G` whose diagonal is `ρ·s_P`). Scalar elliptic
MG is robust, has no saddle-point pathology, is GPU-friendly (pure stencils, optional
red-black), and extends to 3-D mechanically. This reuses the existing `transfer.jl`
cell operators and the hierarchy; it needs a Poisson operator-apply + Poisson
smoother instead of the coupled saddle smoother.

**Prototyped** (not part of the repository) as an operator `GᵀD⁻¹G` + red-black
Gauss-Seidel smoother and a `PoissonMG` hierarchy +
V-cycle + `poisson_solve!`), with coefficient rediscretisation down the levels
(face-coefficient restriction on `κU=ρ_u/D_U`, `κW=ρ_w/D_W`) and null-space
handling (RHS mean-removed for Neumann compatibility, φ gauge-fixed). Validated
on a manufactured discrete problem: solves to solver
tolerance (`sol_err≈1e-9`) and is **grid-independent — 7–8 V-cycles to `rel 1e-10`
for N = 32, 64, 128, 256** (constant + variable coefficient). This is the textbook
MG signature the full-saddle V-cycle never achieved, confirming the reframing.
The full-saddle V-cycle (`gmg.jl`) is superseded by this for the pressure;
remaining work is wiring `PoissonMG` into an outer SIMPLE/Uzawa loop (cheap DR
velocity + Poisson-MG pressure projection — the WaterLily.jl-style architecture).

### 5.5 Fixed point unchanged

Both the constant-`γ_eff` update and the `S⁻¹`-scaled update drive `R_P→0`; at

### 5.5 Fixed point unchanged

Both the constant-`γ_eff` update and the `S⁻¹`-scaled update drive `R_P→0`; at
the fixed point the increment is zero either way, so the **converged `(v,P)` is
identical** — only the convergence rate changes. Hence the Schur scaling is a
*preconditioner* of the multiplier iteration, not a change of the equations
(safe wrt the machine-precision mass conservation established with the
bottom-drain BC).

---

## 6. Loop structure (three nested iterations)

```
Picard           (run!, nonlinear)          : re-freeze η,ρ,advection;  resnorm = ‖Δ(v,P,X)‖
└─ Powell–Hestenes (itPH, multiplier)        : P += ω·R_P/s_P;   convergence gate atol_PH
   └─ Dynamic Relaxation (itPT, pseudo-trans): damped 2nd-order velocity update to steady state
```

The Picard loop is the genuine nonlinear coupling (velocity ↔ phase transport
within a BD2 step). Because the DR solve warm-starts from the current `(v,P)`,
the PH/DR iterations *interleave* with Picard across sweeps; demanding a fully
converged linear solve every sweep is unnecessary (a remnant of the `:direct`
pattern) — an inexact inner solve per sweep is admissible.

---

## 7. Multigrid (planned, `src/stokes/`)

PT/DR convergence degrades as `O(N)` (`dτ ~ h` ⇒ iters ~ `1/h`). Geometric
multigrid with the DR solver as a per-level **smoother** restores
grid-independent iteration counts. The same Schur-diagonal pressure update is
essential there: a constant penalty tuned for the fine grid penalty-locks on
coarse levels, whereas `s_P` (built from the per-level `D_f`) auto-rescales.
See `stokes/DESIGN_multigrid.md` and `MATH.md` §5.

---

## References

- Powell (1969); Hestenes (1969) — augmented Lagrangian / method of multipliers.
- Uzawa; Benzi, Golub & Liesen (2005) — saddle-point solvers, Schur complement.
- Frankel (1950); Räss, Utkin, Duretz, Omlin, Podladchikov (2022) — accelerated
  pseudo-transient / dynamic relaxation.
- Duretz, de Montserrat, Sevilla, Räss, Utkin, Spang (2025), egusphere-2025-5641
  — autotuned PT (Gershgorin λmax, Rayleigh λmin).
- Brandt (1977); Trottenberg, Oosterlee, Schüller (2001) — multigrid.
