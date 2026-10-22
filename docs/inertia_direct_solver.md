---
title: "Inertia in xcore: a code walkthrough of the direct solver vs creeping Stokes"
date: 2026-06-12
tags:
  - xcore
  - solver
  - inertia
  - navier-stokes
  - code-walkthrough
aliases:
  - xcore inertia walkthrough
  - direct solver inertia note
---

# Inertia in xcore: a code walkthrough of the direct solver vs creeping Stokes

This note answers one question concretely, at the level of code lines: **where
and how do the inertial terms enter xcore's momentum solve, and what exactly
changes relative to a slow creeping-Stokes problem?** The reference is the
**direct solver** `fluidmech!` (`src/solvers/direct.jl`) — the faithful port of the
MATLAB `fluidmech.m` and the discretisation behind the methods paper. The
matrix-free solvers (`:dyrel`, `:gmg`) replicate it term for term (§7); the
solver-level consequences of inertia (Gershgorin bound, Schur penalty) are in
the companion theory note (`docs/dyrel_GMG_theory.md`) and are only referenced
here.

> [!abstract] TL;DR
> A creeping-Stokes code solves $-\nabla\!\cdot\!\boldsymbol\tau + \nabla P =
> \Delta\rho\,g$: the velocity block is the viscous stencil and the RHS is
> buoyancy. xcore adds exactly **four** terms, in two places:
> **two diagonals into the matrix** — the implicit BD2 inertia
> $a_1\rho_f/\Delta t$ ([fluidmech.jl:262](../src/solvers/direct.jl#L262),
> [:337](../src/solvers/direct.jl#L337)) and the drunken-sailor stabiliser
> $g_0\,\partial\rho\,\Delta t$ ([:269](../src/solvers/direct.jl#L269),
> [:344](../src/solvers/direct.jl#L344)) — and **two forcings onto the RHS** — the
> lagged BD2 momentum-flux history $(a_2\rho\mathbf v^{\circ} +
> a_3\rho\mathbf v^{\circ\circ})/\Delta t$ ([:303](../src/solvers/direct.jl#L303),
> [:381](../src/solvers/direct.jl#L381)) and the momentum advection
> $\nabla\!\cdot\!(\mathbf v\otimes\rho\mathbf v)$ evaluated at the current
> Picard iterate ([:297–302](../src/solvers/direct.jl#L297-L302)). The continuity
> blocks are untouched by inertia. Crucially, BD2 is applied to the
> **momentum flux $\rho\mathbf v$, not to the velocity** — the stored history
> is the product field `rhoWo = ρ_w·W` ([store.jl:24-27](../src/physics/store.jl#L24-L27)).
> Setting $\Delta t \to \infty$ (the default `dt = 1e32`, used by the MMS
> benchmark) kills all four terms and the same code *is* the creeping-Stokes
> solver.

---

## 1. The creeping-Stokes baseline (what the code reduces to)

For slow creeping flow the momentum equation is force balance with no time
derivative:

$$
-\nabla\!\cdot\!\big[\,2\eta\,(\dot{\boldsymbol\varepsilon} - \tfrac13(\nabla\!\cdot\!\mathbf v)\,\mathbf I)\,\big] + \nabla P = \Delta\rho\, g_0\,\hat{\mathbf z},
\qquad \nabla\!\cdot\!(\rho \mathbf v) = \mathrm{MFS}.
$$

In the assembled system $\begin{pmatrix}K_V & G\\ D_M & K_P\end{pmatrix}
\begin{pmatrix}\mathbf v\\ P\end{pmatrix} = \begin{pmatrix}R_V\\ R_P\end{pmatrix}$
this means: $K_V$ contains only the viscous stress-divergence stencil, and
$R_V$ contains only buoyancy. In the code that baseline is visible verbatim —
the viscous bands of the W-equation rows
([fluidmech.jl:263-274](../src/solvers/direct.jl#L263-L274)):

```julia
append!(AAL, T(2/3).*(EtaP1.+EtaP2).*invh2 .+ T(1/2).*(EtaC1.+EtaC2).*invh2)    # 2 center    (ii,ii)
append!(AAL, -T(2/3) .* EtaP1 .* invh2)                                         # 3 above     (ii,jj1)
append!(AAL, -T(2/3) .* EtaP2 .* invh2)                                         # 4 below     (ii,jj2)
append!(AAL, -T(1/2) .* EtaC1 .* invh2)                                         # 5 left      (ii,jj3)
append!(AAL, -T(1/2) .* EtaC2 .* invh2)                                         # 6 right     (ii,jj4)
...
append!(AAL, -(T(1/2).*EtaC1 .- T(1/3).*EtaP1) .* invh2)                        # 8  Utl — W–U cross-coupling
```

The $2/3,\,1/2,\,1/3$ coefficients are the
$\tau = 2\eta(\dot\varepsilon - \tfrac13\nabla\!\cdot\!\mathbf v\,\mathbf I)$
convention on the staggered grid (cell viscosity `eta` for normal stress,
corner viscosity `etaco` for shear), and bands 8–11 are the U-velocity
cross-couplings through $\tau_{xz}$ and the $-\tfrac13\nabla\!\cdot\!\mathbf v$
term. **Everything else in the W/U rows is inertia-related and is the subject
of this note.**

The code contains this limit as a switch: `fluidmech!` defaults to
`dt = T(1e32)` ([fluidmech.jl:137](../src/solvers/direct.jl#L137)), so every
$1/\Delta t$ term is numerically zero and the drunken-sailor band is skipped in
benchmark mode (`!bnchm` guard, [:268](../src/solvers/direct.jl#L268)). The MMS
verification (`test/test_mms.jl`) therefore exercises exactly the
creeping-Stokes operator. Production runs pass the real, CFL-limited `dt` from
`run.jl` ([run.jl:311](../src/driver/run.jl#L311), `update_dt`).

---

## 2. The time discretisation: BD2 on the momentum flux, not the velocity

The momentum balance keeps $\partial_t(\rho \mathbf v)$ in **conservative
form**. With the BDF2 weights from
[timing.jl:18-36](../src/driver/timing.jl#L18-L36),

$$
\partial_t(\rho \mathbf v)\Big|^{\,n} \;\approx\;
\frac{a_1\,(\rho\mathbf v)^n - a_2\,(\rho \mathbf v)^{n-1} - a_3\,(\rho\mathbf v)^{n-2}}{\Delta t},
\qquad
(a_1,a_2,a_3) = \big(\tfrac32,\; 2,\; -\tfrac12\big),
$$

falling back to backward Euler $(1,1,0)$ for the first two steps of a run
(`:bd2im` is the default `TINT`). The decisive implementation detail is *what
gets stored as history*. At the end of each accepted time step,
`store_previous!` saves the **product** field, not the velocity
([store.jl:24-27](../src/physics/store.jl#L24-L27)):

```julia
fluid.rhoWoo .= fluid.rhoWo
@views fluid.rhoWo .= fluid.rhow .* fluid.W[:, 2:end-1]   # (ρ_w W)^{n-1} on z-faces
fluid.rhoUoo .= fluid.rhoUo
@views fluid.rhoUo .= fluid.rhou .* fluid.U[2:end-1, :]   # (ρ_u U)^{n-1} on x-faces
```

So `rhoWo` is $(\rho_w W)^{n-1}$ — the z-face momentum density of the previous
step, with the previous step's density baked in. The time derivative is of the
conserved quantity; when $\rho$ evolves between steps (it does — two-phase
compaction changes it every step), discretising $\rho\,\partial_t \mathbf v$
instead would silently drop the $\mathbf v\,\partial_t\rho$ momentum exchange.
This is a methods-paper-level point: **xcore's inertia is a discrete
$\partial_t(\rho\mathbf v)$, consistent with the mass equation
$\partial_t\rho = -\nabla\!\cdot\!(\rho\mathbf v)$ that the MFS machinery
enforces.**

The three BD2 terms then split between matrix and RHS:

| term | time level | treatment | where |
|---|---|---|---|
| $a_1 (\rho\mathbf v)^n/\Delta t$ | current (unknown) | **implicit diagonal** in $K_V$: density frozen at the current Picard iterate, velocity is the matrix unknown — $a_1\rho_f^{(k)}/\Delta t$ on the diagonal | [fluidmech.jl:262](../src/solvers/direct.jl#L262) (W), [:337](../src/solvers/direct.jl#L337) (U) |
| $a_2 (\rho\mathbf v)^{n-1}/\Delta t$ | known | RHS forcing | [:303](../src/solvers/direct.jl#L303), [:381](../src/solvers/direct.jl#L381) |
| $a_3 (\rho\mathbf v)^{n-2}/\Delta t$ | known | RHS forcing (same lines) | — |

The two assembly lines, verbatim:

```julia
# W rows — z-face inertial diagonal (band 1 of the stencil block)
append!(AAL, a1 .* vec(rhow[2:end-1, :]) ./ T(dt))                 # fluidmech.jl:262
# W rows — RHS: buoyancy + lagged momentum flux − advection
rr = Drho[2:end-1, :] .* g0 .+
     (a2 .* rhoWo[2:end-1, :] .+ a3 .* rhoWoo[2:end-1, :]) ./ T(dt) .- advn_mz   # :303
```

```julia
# U rows — x-face inertial diagonal; note the extra γ (§5)
append!(AAL, vec((a1 + gamma) .* rhou ./ T(dt)))                   # fluidmech.jl:337
rr = (a2 .* rhoUo .+ a3 .* rhoUoo) ./ T(dt) .- advn_mx             # :381
```

(U has no buoyancy term — gravity is z-only; the W RHS uses the density
*anomaly* `Drho`, the background hydrostatic part having been absorbed into
the pressure.)

---

## 3. Momentum advection: explicit per sweep, implicit at Picard convergence

$\nabla\!\cdot\!(\mathbf v\otimes\rho\mathbf v)$ is genuinely nonlinear, so the
direct solver handles it by Picard freezing: each `fluidmech!` call evaluates
it **at the current iterate** and puts it on the RHS
([fluidmech.jl:297-302](../src/solvers/direct.jl#L297-L302)):

```julia
f_mz = rhow[2:end-1, :] .* W[2:end-1, 2:end-1]        # advected quantity: z-momentum ρ_w W
u_mz = (U[2:end-2, :] .+ U[3:end-1, :]) ./ T(2)       # transport velocity, averaged to the face
w_mz = (W[1:end-1, 2:end-1] .+ W[2:end, 2:end-1]) ./ T(2)
advect_centered!(advn_mz, f_mz, u_mz, w_mz, h, par.ADVN; xBC, zBC)
```

Again the advected quantity is the **momentum density** $\rho_w W$, transported
by the face-averaged velocity — conservative form, scheme selected by
`par.ADVN` (the same `advect_centered!` machinery as phase transport;
`src/kernels/advect.jl`). The x-momentum analogue is at
[:371-380](../src/solvers/direct.jl#L371-L380), including the periodic
column-averaging of the two duplicated boundary columns
([:378-380](../src/solvers/direct.jl#L378-L380)).

Because `run.jl`'s Picard loop ([run.jl:259-289](../src/driver/run.jl#L259-L289))
re-calls `fluidmech!` each sweep — with at least 3 sweeps enforced — the
advection term is re-evaluated at each new velocity. At the Picard fixed point
it is therefore evaluated at $\mathbf v^n$ itself: **the converged step is
fully implicit** (nonlinearly so), even though within any single linear solve
the advection sits frozen on the RHS. The same holds for the inertial
diagonal's density $\rho^{(k)} \to \rho^n$.

---

## 4. The drunken-sailor diagonal

The fourth and least standard term, on both W and U rows
([fluidmech.jl:269](../src/solvers/direct.jl#L269),
[:344](../src/solvers/direct.jl#L344)):

```julia
append!(AAL, vec((rho[2:end, :] .- rho[1:end-1, :]) .* invh) .* g0 .* T(dt))   # W diag
append!(AAL, vec((rho[:, icx[2:end]] .- rho[:, icx[1:end-1]]) .* invh) .* g0 .* T(dt))  # U diag
```

i.e. a diagonal $g_0\,(\partial\rho)\,\Delta t$ multiplying the implicit
velocity. This damps the "drunken-sailor" instability of buoyancy-driven flow
with explicit-in-Picard density: a parcel overshooting in $z$ through a density
gradient feels a restoring force only one sweep later, and at large $\Delta t$
the lag oscillates. The term scales **with** $\Delta t$ (unlike inertia's
$1/\Delta t$), vanishes in the creeping/MMS limit only because `bnchm` skips
it, and is sign-indefinite ($\partial_z\rho$ can have either sign) — which is
why the matrix-free solvers exclude it from the Gershgorin diagonal bound
while keeping it in the residual (see `src/dyrel_GMG/gershgorin.jl`, header
note).

---

## 5. The γ augmentation of the U inertia

The U-diagonal is $(a_1 + \gamma)\,\rho_u/\Delta t$ with $\gamma =$
`par.gamma` $= 10^{-3}$ ([parameters.jl:80](../src/setup/parameters.jl#L80)). The
MATLAB original documents its purpose (`usr/par_default.m:73`):

```matlab
gamma = 1e-3;   % artificial horizontal inertia parameter (only applies if periodic)
```

i.e. a deliberate **artificial inertia** added to the x-momentum diagonal,
specific to the periodic-$x$ configuration — there the mean horizontal
velocity is a translational near-null mode that nothing else in the U
equation constrains at finite $\Delta t$, and the extra $\gamma\rho_u/\Delta t$
weakly damps it. It is *not* part of the physical BD2 weight (the lagged RHS
uses plain $a_2, a_3$), so the discrete momentum balance carries an
$O(\gamma/a_1) = O(10^{-3})$ bias on U; at $\Delta t\to\infty$ it disappears
entirely (MMS instead pins one U cell explicitly,
[fluidmech.jl:536-541](../src/solvers/direct.jl#L536-L541)).

---

## 6. Where inertia does **not** go — and why that matters

The inertial terms touch only the $K_V$ block and $R_V$. The pressure-gradient
block `GG` ([fluidmech.jl:402-427](../src/solvers/direct.jl#L402-L427)), the
mass-flux divergence block `DM` (the $\pm\rho_f/h$ stencil,
[:437-440](../src/solvers/direct.jl#L437-L440)), and the continuity RHS (`MFS`)
are **identical to the creeping-Stokes case**. Summarising the full delta:

| # | term | enters as | matrix/RHS | code |
|---|---|---|---|---|
| 1 | $a_1\rho_f/\Delta t$ (BD2, current) | diagonal of $K_V$ | matrix | [262](../src/solvers/direct.jl#L262), [337](../src/solvers/direct.jl#L337) |
| 2 | $(a_2\rho\mathbf v^\circ + a_3\rho\mathbf v^{\circ\circ})/\Delta t$ (BD2, lagged) | forcing | RHS | [303](../src/solvers/direct.jl#L303), [381](../src/solvers/direct.jl#L381) |
| 3 | $\nabla\!\cdot\!(\mathbf v\otimes\rho\mathbf v)$ at Picard iterate | forcing | RHS | [297–302](../src/solvers/direct.jl#L297-L302), [371–380](../src/solvers/direct.jl#L371-L380) |
| 4 | $g_0\,\partial\rho\,\Delta t$ (sailor) | diagonal of $K_V$ | matrix | [269](../src/solvers/direct.jl#L269), [344](../src/solvers/direct.jl#L344) |

The asymmetry — momentum block changed, constraint blocks unchanged — is the
single most important structural fact downstream. Comparing diagonal
magnitudes of $K_V$, inertia dominates the viscous entry when

$$
\frac{a_1\rho\,h^2}{\eta\,\Delta t} \;\gg\; \tfrac{4}{3} + 1
\quad\Longleftrightarrow\quad
\Delta t \;\ll\; \frac{a_1 \rho h^2}{\eta} \;\;(\text{the viscous diffusion time of one cell}),
$$

which holds throughout xcore's production regime ($\rho\sim 2700$,
CFL-limited $\Delta t$). Then:

- $K_V$ becomes **strongly diagonally dominant** — good news for any
  Jacobi-type iteration on the velocity (the DR inner loop converges fast),
  and the reason the direct solver's symmetric Jacobi scaling
  ([fluidmech.jl:584-595](../src/solvers/direct.jl#L584-L595)) keeps UMFPACK
  well-behaved.
- the **Schur complement** $S = D_M K_V^{-1} G$ becomes inertia-scaled and
  spatially variable, $S \sim (\Delta t/a_1\rho)\nabla^2$, instead of the
  creeping-Stokes $S\sim\eta^{-1}$. The direct solver never notices (LU
  factors the coupled system exactly); every *iterative* pressure strategy
  does. This is precisely the departure point from Duretz et al. (2025)
  developed in the companion theory note — the per-cell penalty
  $\gamma_{\rm eff} = \gamma_{\rm fact}/s_P$ exists *because* rows 1 and 4 of
  the table above sit in $K_V$.

---

## 7. One operator, three solvers

The direct solver applies the operator by assembling it; `:dyrel`/`:gmg`
apply it matrix-free. The W-residual kernel
([src/dyrel/residuals.jl:50-77](../src/dyrel/residuals.jl#L50-L77)) is the
line-by-line mirror of the assembly — same four inertial terms, same
Picard-frozen advection, flipped overall sign ($R = -(K_V v + G P - rr)$ so
the velocity update is a descent direction):

```julia
inertia  = a1 * rhow[iz_f, ix] * Wv * inv_dt                       # ↔ matrix band 1
sailor   = ddz_rho * g0 * dt * Wv                                  # ↔ matrix band 7
lagged   = (a2 * rhoWo[iz_f, ix] + a3 * rhoWoo[iz_f, ix]) * inv_dt # ↔ RHS lagged
adv      = advn_mz[iz_f - 1, ix]                                   # ↔ RHS advection
r = d_tzz_dz + d_txz_dx - dP_dz + body - inertia - sailor + lagged - adv
```

with `inv_dt = isfinite(dt) ? inv(dt) : 0`
([residuals.jl:190](../src/dyrel/residuals.jl#L190)) reproducing the same
$\Delta t \to \infty$ creeping-Stokes switch. The matrix-free operator was
verified against the assembled one to $10^{-12}$, so all three solvers share
one discretisation — `:direct` is its definition, and the methods paper can
describe the physics once, independent of the solution algorithm.

What inertia means for the *iterative* solvers (the $\delta = a_1\rho_f/\Delta t$
term in the Gershgorin diagonal and $\lambda_{\max}$, the Schur diagonal
$s_P$, the penalty scaling) is deliberately not duplicated here — see
`docs/dyrel_GMG_theory.md` §4–5.

---

## 8. Checklist for the methods paper

Facts established by this walkthrough that the paper's discretisation section
should state:

1. Momentum is integrated as $\partial_t(\rho\mathbf v)$ (conservative form),
   BDF2 in time ($3/2, 2, -1/2$; BE for the first two steps), with the
   momentum-flux history stored as the product $(\rho_f v)^{n-1,n-2}$.
2. Within a Picard sweep the system is linear: inertial diagonal implicit with
   Picard-frozen density; lagged flux, advection (conservative,
   $\rho_f v$ transported by face-averaged velocity, scheme `par.ADVN`) and
   buoyancy on the RHS. At Picard convergence the step is nonlinearly
   implicit.
3. A drunken-sailor diagonal $g_0\partial\rho\,\Delta t$ stabilises the
   buoyancy–density lag at large $\Delta t$.
4. Continuity (mass-flux divergence + MFS) is unmodified by inertia; the
   creeping-Stokes limit is recovered exactly as $\Delta t\to\infty$ and is
   what the MMS benchmark verifies.
5. The horizontal momentum diagonal carries a $+\gamma\rho_u/\Delta t$
   ($\gamma = 10^{-3}$) regularisation absent from the lagged terms.
