# DYREL solver design for xcore.jl

Pseudo-transient / dynamic relaxation solver for xcore's compressible
incompressible Stokes system with two-phase mass-flux source, modelled after
Duretz et al. (GMD 2025) and the JustRelax.jl `src/DYREL/` reference
implementation.

**Goal**: replace `src/fluidmech.jl`'s sparse-LU Stokes solve with a
matrix-free iterative scheme that (a) runs on GPU via KernelAbstractions
unchanged, (b) scales O(N²) instead of O(N³) per Picard sweep, (c) drops the
UMFPACK dependency entirely.

**Scope**: this replaces ONLY `fluidmech!`. Phase evolution (`phsevo!`),
constitutive update (`update!`), noise, segregation speed — all unchanged.

---

## 1. What xcore-DYREL has to solve

### 1.1 Continuous equations

Momentum (per face, with deviatoric stress):

```
∇·τ − ∇P − ρg − ρ ∂_t v = 0
τ   = 2η [D − (1/3)(∇·v)I]     # D = symmetric velocity gradient
```

Continuity (with mass-flux source, NOT pure ∇·v = 0):

```
∇·(ρ v) = MFS
```

The MFS source comes from the density-evolution residual:

```
res_ρ   = (a1·ρ − a2·ρ_o − a3·ρ_oo)/dt − (b1·∂_tρ + b2·∂_tρ_o + b3·∂_tρ_oo)
MFS     = MFS − α·res_ρ                         # accumulated correction
```

where `∂_tρ = −∇·(ρv)` from phsevo's `advn_rho`.

### 1.2 Time integration

BD2 with momentum-flux history (`rhoWo`, `rhoWoo` for z-faces; `rhoUo`,
`rhoUoo` for x-faces). Enters the momentum residual as a linearized inertial
term: `a1·(ρ_face · v)/dt` on the LHS, lagged `a2·ρv_old/dt + a3·ρv_oo/dt`
on the RHS.

### 1.3 Constitutive

`η` (cell-centred), `etaco` (corner-averaged) are produced by `update!`
*before* fluidmech runs. **xcore has no shear modulus G or plasticity** — η
enters the stress directly as `τ = 2η ε`. This is structurally simpler than
JustRelax's visco-elasto-plastic stress kernel.

### 1.4 Boundary conditions

| boundary | velocity | pressure |
|---|---|---|
| top (z=0)    | `W = 0` (Dirichlet) | repeat: `P[1,:] = P[2,:]` |
| bottom (z=D) | `W = 0` (closed) OR `∂_z W = −∂_x U·ρ_ratio` (open_cnv) | repeat |
| left/right (x=0,L) | periodic | periodic |

Plus tangential at top/bot: `U[1,:] + top_cnv·U[2,:] = 0` (free-slip if
top_cnv=−1, no-slip if top_cnv=+1). Same with bot_cnv.

### 1.5 Sizing (matches FluidState exactly)

| field | size | location |
|---|---|---|
| `W`     | `(Nz+1, Nx+2)` | z-faces, x-ghost cols |
| `U`     | `(Nz+2, Nx+1)` | x-faces, z-ghost rows |
| `P`     | `(Nz+2, Nx+2)` | cells + ghost rings |
| `η`     | `(Nz,   Nx  )` | cell-centred |
| `etaco` | `(Nz+1, Nx+1)` | corner-staggered |
| `ρ`     | `(Nz,   Nx  )` | cell-centred |
| `ρw`    | `(Nz+1, Nx  )` | z-face density |
| `ρu`    | `(Nz,   Nx+1)` | x-face density |
| `Drho`  | `(Nz+1, Nx  )` | density anomaly on z-faces |
| `MFS`   | `(Nz,   Nx  )` | cell-centred source |

---

## 2. xcore vs JustRelax DYREL — the deltas

| feature | JustRelax | xcore | impact on solver |
|---|---|---|---|
| Visco-elastic G·dt | yes (`η_ve = η·G·dt/(η+G·dt)`) | NO (pure viscous) | drop the G·dt terms in Gershgorin formulas |
| Plasticity (λ, ΔPψ, yield F) | yes | NO | drop entirely |
| Continuity source | thermal expansion `α·ΔT/dt` + Q/dt | `MFS` (drift correction) | replace term-for-term in `R_P` |
| BD2 inertia in momentum | absent | `a1·ρv/dt + lagged` | new term in `R_W`, `R_U` |
| Drunken-sailor term | absent | `g·∂_zρ·dt·W_old` | new diagonal contribution |
| Open-bottom BC | absent | `open_cnv = true` modifies bot-W BC | special BC kernel branch |
| Adaptive viscosity inside | yes (some configs) | NO (η is fixed for fluidmech's lifetime) | simpler: skip viscosity rebuild in DR loop |
| Marker-in-cell | yes | NO | irrelevant (already eulerian) |

**Bottom line**: xcore-DYREL is structurally *simpler* than JustRelax. The
visco-elasto-plastic complexity is gone; the new complexity is MFS and BD2
inertia, both of which are scalar additions to the residual.

---

## 3. Algorithmic structure (Powell-Hestenes outer / DR inner)

```
solve_dyrel!(state, grid, par, …):
    cache = state.dyrel                          # StokesCache, lazy-initialized

    # one-time setup per fluidmech! call: re-tune for current viscosity field
    compute_bulk_viscosity_and_penalty!(cache, η, par.γfact)
    gershgorin!(cache.D_W, cache.D_U, cache.λmax_W, cache.λmax_U, η, etaco, γ_eff, a1, dt, h)
    update_dτ_α_β!(cache, par.CFL_PT)

    # Powell-Hestenes outer loop
    for itPH in 1:par.maxit_PH:
        # 1. recompute strain rate, stress, residuals at current (W,U,P)
        strain_rate!(state, grid)
        stress!(state, grid)                       # τ_zz, τ_xx, τ_xz from η·ε
        residual_PH!(R_W, R_U, R_P, state, cache, …)

        # 2. convergence check on UN-preconditioned residuals
        errW = norm(R_W);  errU = norm(R_U);  errP = norm(R_P)
        max(errW/errW₀, errU/errU₀, errP/errP₀) < par.atol_PH && break

        # 3. inner DR velocity solve
        ϵ_vel = max(errW, errU) · par.rel_drop_PT
        for m in 1:par.maxit_PT:
            copyto!(R_W_old, R_W);  copyto!(R_U_old, R_U)
            strain_rate!(…);  stress!(…)
            residual_P!(R_P, state, cache.η_b)
            P_num = cache.γ_eff .* R_P             # Schur-complement penalty
            residual_DR!(R_W, R_U, state, P_num, cache.D_W, cache.D_U, …)
            damping_update!(dWdτ, R_W, α_W);  damping_update!(dUdτ, R_U, α_U)
            velocity_update!(W, dWdτ, β_W, dτ_W);  velocity_update!(U, dUdτ, β_U, dτ_U)
            apply_velocity_BCs!(W, U, BCs, …)

            # 4. every n_tune iterations: re-estimate λ_max, λ_min, rebuild dτ/α/β
            if m % par.n_tune_PT == 0:
                errW = norm(D_W .* R_W);  errU = norm(D_U .* R_U)
                err  = max(errW/errW₀₀, errU/errU₀₀)
                err < ϵ_vel && break

                # Rayleigh quotient (the ONE non-local reduction)
                λmin = abs(Σ(dW_step·(R_W − R_W_old)) + Σ(dU_step·(R_U − R_U_old)))
                     / (Σ(dW_step²) + Σ(dU_step²))
                cache.c = 2·√λmin · par.c_fact

                gershgorin!(cache, η, etaco, γ_eff, a1, dt, h)
                update_dτ_α_β!(cache, par.CFL_PT)

        # 5. PH pressure penalty update
        P += cache.γ_eff .* R_P
        # 6. MFS source update (xcore-specific, every PH iter)
        MFS -= par.alpha · res_ρ                   # uses drhodt from outside
```

---

## 4. Residual formulas (the actual math)

Notation: `_d_xa, _d_yi` = forward/centred staggered differences; same as
JustRelax's primitives. Subscripts: cell-centred unless noted.

### 4.1 Powell-Hestenes momentum (uncondtioned — for convergence check)

```
R_W[j,i] = ∂_z(τzz)[j,i] + ∂_x(τxz)[j,i]
         − ∂_z(P)[j,i]
         − ρw[j,i_int] · g0                          # body force; xcore uses Drho · g0
         + (a2·rhoWo[j,i] + a3·rhoWoo[j,i])/dt        # BD2 lagged momentum-flux
         − a1·rhoW[j,i]·W[j,i+1]/dt                   # BD2 implicit momentum-flux (linearized)
         − advn_mz[j,i]                               # advect(rhoW·W) — momentum advection
         + g0·∂_z(ρ)[j,i]·dt·W[j,i+1]                 # drunken-sailor stabilization
```

(Indexing here is schematic; actual kernels follow current fluidmech.jl's
staggered indexing exactly.)

`R_U[j,i]` similar with `(a1 + γ)·rhoU/dt`, `ρu` body force, ∂_x(P), etc.

### 4.2 DR momentum (preconditioned — for inner DR step)

```
R_W_DR[j,i] = (R_W[j,i] − ∂_z(P_num)[j,i]) / D_W[j,i]
```

`P_num = γ_eff · R_P` is the Schur-complement penalty injected from the
continuity residual. Division by `D_W` is the Jacobi preconditioner (the
Gershgorin diagonal).

### 4.3 Continuity residual

```
R_P[j,i] = MFS[j,i] − ∇·v[j,i]                       # mass-flux source
         − (P[j,i] − P_o[j,i]) / (η_b[j,i] · dt)     # compressibility penalty (= 0 if incompressible)
```

For xcore with K → ∞ (effective incompressibility), `η_b = γfact · ⟨η⟩` so
this term is the *numerical* penalty term that drives ∇·v → 0.

---

## 5. Auto-tuning — what to compute when

### 5.1 Gershgorin diagonal & λ_max (per-face, every n_tune DR iters)

For each Z-face (Vw position) `(j, i)`:

```
# viscosities at surrounding 4 nodes (no G·dt — pure viscous)
η_N = etaco[j+1, i+1];  η_S = etaco[j,   i+1]
η_W = η[j,   i];        η_E = η[j+1, i]
γ_W = γ_eff[j, i];      γ_E = γ_eff[j+1, i]

# Gershgorin row sums (translates directly from JustRelax Cxx, Cxy formulas)
C_zz = (η_N + η_S)·(1/h²) + (γ_E + 4/3·η_E)·(1/h²) + (γ_W + 4/3·η_W)·(1/h²)
       + (η_N + η_S)·(1/h²) + (γ_E + γ_W + 4/3·(η_E + η_W))·(1/h²)
C_zx = ((γ_E − 2/3·η_E + η_N) + (γ_E − 2/3·η_E + η_S))·(1/h²)
       + ((γ_W − 2/3·η_W + η_N) + (γ_W − 2/3·η_W + η_S))·(1/h²)

# diagonal entry + max eigenvalue estimate
D_W[j,i]    = (η_N + η_S)·(1/h²) + (γ_E + γ_W + 4/3·(η_E + η_W))·(1/h²)
λmax_W[j,i] = (C_zz + C_zx) / D_W[j,i]
```

Same shape for `D_U[j,i]`, `λmax_U[j,i]` on X-faces, swapping z/x.

### 5.2 λ_min via Rayleigh quotient (global scalar, every n_tune)

```
dW_step[j,i] = dWdτ[j,i] · β_W[j,i] · dτ_W[j,i]      # velocity increment from last step
dU_step[j,i] = analogous

# the single global reduction
λmin = |Σ_{j,i} dW_step·(R_W − R_W_old) + Σ_{j,i} dU_step·(R_U − R_U_old)|
       /  (Σ_{j,i} dW_step² + Σ_{j,i} dU_step²)

c = 2·√λmin · par.c_fact                              # damping coefficient (global scalar)
```

### 5.3 dτ, α, β update (per-face)

```
dτ_W[j,i] = 2·par.CFL_PT / √λmax_W[j,i]
β_W[j,i]  = 2·dτ_W[j,i] / (2 + c·dτ_W[j,i])
α_W[j,i]  = (2 − c·dτ_W[j,i]) / (2 + c·dτ_W[j,i])
```

(Same for U.)

### 5.4 γ_eff (Powell-Hestenes penalty, every fluidmech! call)

```
γ_num     = par.γfact · ⟨η⟩            # numerical
γ_phy     = K_b · dt                   # physical (= γ_num for incompressible)
γ_eff[j,i] = γ_num · γ_phy / (γ_num + γ_phy)   # harmonic mean
η_b[j,i]   = K_b · dt                  # or γ_num if K → ∞
```

For xcore (incompressible): `K_b → ∞`, so `γ_eff = γ_num = par.γfact · ⟨η⟩`.

---

## 6. Cache data structure

```julia
mutable struct StokesCache{T<:AbstractFloat, A<:AbstractMatrix{T}}
    # --- penalty + compressibility ---
    γ_eff::A          # (Nz,   Nx  )
    η_b::A            # (Nz,   Nx  )

    # --- per-face autotune state, W (z-faces, Nz+1 × Nx)
    D_W::A;     λmax_W::A
    dτ_W::A;    α_W::A;     β_W::A
    dWdτ::A;    dW_step::A;  R_W_old::A

    # --- per-face autotune state, U (x-faces, Nz × Nx+1)
    D_U::A;     λmax_U::A
    dτ_U::A;    α_U::A;     β_U::A
    dUdτ::A;    dU_step::A;  R_U_old::A

    # --- residual scratch (interior sizing, matches W/U interior)
    R_W::A;     R_U::A;     R_P::A

    # --- damping (global scalar, but cached as a Ref so we don't break type stability)
    c::Base.RefValue{T}

    # --- iteration counters (for telemetry)
    last_PH_iters::Base.RefValue{Int}
    last_DR_iters::Base.RefValue{Int}
end
```

Constructor allocates everything via `xcore_zeros(backend, T, …)` exactly
like `FluidState`. Lives at `state.dyrel::StokesCache{T}`, parametric on the
backend.

---

## 7. New `Parameters` knobs

```julia
# DR/PH solver controls
solver::Symbol        = :direct          # :direct | :dyrel
maxit_PH::Int         = 50               # outer Powell-Hestenes iters
maxit_PT::Int         = 100_000          # inner DR iters per PH outer
n_tune_PT::Int        = 100              # re-tune λ every n iters
CFL_PT::T             = T(0.99)          # < 1 for stability
c_fact_PT::T          = T(0.5)           # damping scaling factor (Eq. 19, [1/2, 1])
γfact_PT::T           = T(20.0)          # PH penalty numerical factor
rel_drop_PT::T        = T(1e-2)          # inner DR tolerance = outer err · rel_drop
atol_PH::T            = T(1e-6)          # outer PH absolute tol
```

---

## 8. Kernels — one-to-one mapping from JustRelax

| JustRelax kernel | xcore equivalent | new/reuse |
|---|---|---|
| `compute_∇V_strain_rate!` | `update_kinematics!` already does this | **reuse** existing |
| `compute_stress_DYREL!` | `update_stresses!` (without plasticity) | **reuse + simplify** |
| `compute_PH_residual_V!` | `_dyrel_residual_W_PH_kernel!`, `_dyrel_residual_U_PH_kernel!` | **NEW** |
| `compute_DR_residual_V!` | `_dyrel_residual_W_DR_kernel!`, `_dyrel_residual_U_DR_kernel!` | **NEW** |
| `compute_residual_P!` | `_dyrel_residual_P_kernel!` (with MFS) | **NEW** |
| `update_V_damping!` | `_dyrel_damping_W_kernel!`, `_dyrel_damping_U_kernel!` | **NEW** |
| `update_DR_V!` | `_dyrel_update_W_kernel!`, `_dyrel_update_U_kernel!` | **NEW** |
| `Gershgorin_Stokes2D_SchurComplement!` | `_dyrel_gershgorin_kernel!` | **NEW** (drop G·dt terms) |
| `update_dτV_α_β!` | `_dyrel_update_dτ_α_β_kernel!` | **NEW** |
| `compute_bulk_viscosity_and_penalty!` | `_dyrel_init_penalty!` | **NEW** |
| BC kernels (flow_bcs!) | adapt from existing xcore BC patterns in fluidmech.jl | **NEW** |

Plus reductions: `_dyrel_rayleigh_quotient` (2 dot products + 2 norms),
`_residual_norms` (3 norms).

---

## 9. File layout under `src/stokes/`

One implementation serves D = 2 (`FluidState`, `Grid`) and D = 3 (`FluidState3D`,
`Grid3D`); kernels run over `CartesianIndex{D}`.

```
src/stokes/
├── Stokes.jl         ← module Stokes, includes below
├── layout.jl         ← index helpers, state accessors, StokesOp (frozen coefficients + BCs)
├── cache.jl          ← StokesCache, AndersonPH
├── residuals.jl      ← momentum / continuity residuals, DR velocity update
├── gershgorin.jl     ← Gershgorin diagonals + λmax, Schur diagonal, penalty!, dτ/α/β
├── bcs.jl            ← velocity / pressure BCs, pressure gauges
├── reductions.jl     ← Rayleigh quotient, residual norms, pressure-update gate
├── advection.jl      ← D-dimensional flux-form advection, frozen momentum advection
├── transfer.jl       ← multigrid restriction / prolongation
├── dyrel.jl          ← fluidmech_dyrel! (Powell–Hestenes + dynamic relaxation)
├── hierarchy.jl      ← MGLevel / MGHierarchy, coefficient coarsening
├── gmg.jl            ← smoother, coarse solve, V-cycle, fluidmech_gmg!
└── vmg.jl            ← velocity V-cycle, Cahouet–Chabard Poisson V-cycle, fluidmech_vmg!
```

`src/solver.jl` dispatches `par.solver ∈ (:dyrel, :gmg, :vmg)` to these drivers.

---

## 10. Open questions / things to validate

1. **MFS update cadence**. JustRelax has no MFS; xcore currently updates MFS
   inside `fluidmech!` (line 71-82). In DYREL we could (a) update MFS once
   per PH outer iter, or (b) update every DR inner iter. The MATLAB code
   updates per-Picard-sweep; (a) is closest to current semantics.

2. **drhodt computation**. The current code does `drhodt = -advn_rho` where
   `advn_rho` comes from `phsevo!`. In DYREL, the DR inner loop never calls
   phsevo, so `drhodt` stays frozen at the value `phsevo!` produced before
   `fluidmech_dyrel!` was called. Verify this is correct.

3. **BD2 inertial term linearization**. Currently `a1·rhoW·W/dt` enters as a
   diagonal in `KV`. In DYREL it enters the momentum residual. Linearization
   is implicit: treat `W` in this term as the current iterate (so it acts
   like a relaxation toward the steady inertial balance). Verify this is
   stable.

4. **Drunken-sailor coupling**. The `g·∂_zρ·dt·W` term is also a "diagonal"
   contribution (it scales with W). Same linearization choice as #3.

5. **Warm-starting the inner DR**. Start from the previous Picard sweep's
   `W`, `U`, `P` (warm), with `dWdτ`, `dUdτ` zeroed. JustRelax warm-starts
   velocity and zeros the rate. Match.

6. **MMS test fidelity**. The existing `test/test_mms.jl` calls
   `fluidmech!` once with `dt → ∞` (kills inertia) and pins P/U. DYREL with
   `K → ∞` and large γ_eff should converge to the same solution to a similar
   tolerance — but tolerance has to be defined relative to DR iteration count
   not LU exactness. Need to choose comparable atol.

7. **GPU branch fidelity**. KernelAbstractions.jl + xcore's existing
   patterns should make GPU work "for free" once kernels are KA-native. But
   reductions (Rayleigh, norms) need backend-specific paths (the existing
   `maximum`/`sum` on KA arrays handles this on CUDA; verify on Metal).

---

## 11. Sanity benchmarks to pass before merging

1. **MMS Stokes-only** (`test/test_mms.jl`): match UMFPACK error
   convergence rate on N = 16, 32, 64.
2. **Conservation** (`usr/run_bnchm_cnsv.jl`): `EB/EM/EX` should stay at the
   same order of magnitude as the direct solver (currently ~1e-13 to 1e-14
   with cache+BLAS).
3. **Phase-shift benchmark** (`usr/run_bnchm_PHS_*.jl`): same convergence
   order in dt and h.
4. **Production run** (`usr/run_D1_dm2_e1.jl` at N = 50, 100, 150):
   trajectory of `mean(x)`, `mean(eta)`, etc. matches direct solver within
   inner-DR tolerance.
5. **Cost comparison** at N = 50, 100, 150: wall-clock / Picard sweep
   should match or beat the cached UMFPACK path by N = 100, beat it
   substantially by N = 150+.

---

## References

- Duretz, T., de Montserrat, A., Sevilla, R., Räss, L., Utkin, I., Spang, A.
  (2025). *Automatic tuning of iterative pseudo-transient solvers for
  modelling the deformation of heterogeneous media.* Preprint
  egusphere-2025-5641. https://doi.org/10.5194/egusphere-2025-5641
- JustRelax.jl, `src/DYREL/` (CPUBackend reference implementation).
- Räss, L., Utkin, I., Duretz, T., Omlin, S., Podladchikov, Y. (2022).
  *Assessing the robustness and scalability of the accelerated
  pseudo-transient method.*
- Papadrakakis, M. (1981); Oakley, D. R. and Knight, N. F. (1995) — original
  DR parameter derivations.
