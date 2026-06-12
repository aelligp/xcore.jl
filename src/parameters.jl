"""
    Parameters{T}

All runtime knobs for a single xcore simulation, parameterised on the floating-point
type `T` (Float32 or Float64). Mirrors `usr/par_default.m`; user driver scripts build
this with keyword overrides, the same way MATLAB user files override defaults.

Fields with physical units live alongside numerical-method and IO switches. Integer
counts (resolution, iteration caps, output cadence) stay `Int`; scheme selectors are
`Symbol` so kernels can dispatch on `Val{:weno5}` etc.
"""
Base.@kwdef struct Parameters{T<:AbstractFloat}
    # --- run identification & IO ---
    runID::String       = "default"
    srcdir::String      = "../src"
    outdir::String      = "../out"
    restart::Int        = 0
    nrh::Int            = 1
    nop::Int            = 100
    ndm_op::Bool        = false
    plot_op::Bool       = true
    save_op::Bool       = false
    plot_cv::Bool       = false
    colourmap::Symbol   = :lapaz

    # --- unit conversions ---
    hr::T               = T(3600)
    yr::T               = T(24 * 365.25 * 3600)
    cm::T               = T(0.01)
    km::T               = T(1000)

    # --- domain ---
    D::T                = T(10)
    N::Int              = 100
    L::T                = T(10)            # set equal to h for 1-D mode; defaults to D

    # --- timing ---
    Nt::Int             = Int(1e6)
    t0end::T            = T(2)
    xend::T             = T(1.00)
    tend::T             = T(10 * 24 * 365.25 * 3600)   # 10 years

    # --- initial phase fraction ---
    x0::T               = T(eps(Float64))   # MATLAB `eps` is Float64 machine eps
    dxr::T              = T(0.1)
    dxg::T              = T(0)
    seed::Int           = 15

    # --- buoyancy ---
    rhom0::T            = T(2700)
    rhox0::T            = T(3200)
    d0::T               = T(0.01)
    g0::T               = T(9.81)

    # --- rheology ---
    etam0::T            = T(1e1)
    etax0::T            = T(1e18)
    # permission-weight matrices [2x2]; mirror layout of par_default.m
    AA::Matrix{T}       = T[0.72 0.19; 0.81 0.20]
    BB::Matrix{T}       = T[0.63 0.37; 0.999 0.001]
    CC::Matrix{T}       = T[2.09 0.09; 0.37 1.45]

    # --- physical control / noise ---
    L0::T               = T(0.1)            # default D/100 with D=10
    l0::T               = T(0.1)            # default d0*10 with d0=0.01
    Da::T               = T(0.01)
    Xi::T               = T(0.5)
    Ptop::T             = T(1e5)
    open_cnv::Bool      = false
    open_sgr::Bool      = false

    # --- numerics ---
    TINT::Symbol        = :bd2im            # :be1im :bd2im :cn2si :bd2si
    ADVN::Symbol        = :weno5            # :centr :upwd1 :quick :fromm :weno3 :weno5 :tvdim
    CFL::T              = T(0.50)
    rtol::T             = T(1e-4)
    atol::T             = T(1e-9)
    maxit::Int          = 25
    alpha::T            = T(0.9)
    gamma::T            = T(1e-3)
    kmin::T             = T(1e-16)
    kmax::T             = T(1e16)
    dtmax::T            = T(1e32)
    etacntr::T          = T(1e8)

    # --- Stokes solver selection + DYREL (pseudo-transient) controls ---
    solver::Symbol      = :direct           # :direct (sparse-LU) | :dyrel (pseudo-transient) | :gmg (geometric multigrid, dyrel sweep as smoother) | :mg (WIP scalar-Poisson projection)
    maxit_PH::Int       = 100                # outer Powell-Hestenes iterations
    maxit_PT::Int       = 50_000            # inner Dynamic-Relaxation iterations per PH step
    total_iterMax_PT::Int = 50_000          # global DR-iteration cap per fluidmech_dyrel! call (JustRelax `total_iterMax`); accept & let the outer Picard loop continue
    n_tune_PT::Int      = 25                # re-estimate λ (Gershgorin + Rayleigh) every n iters
    CFL_PT::T           = T(0.99)           # DR pseudo-time CFL (< 1)
    c_fact_PT::T        = T(0.5)            # damping scaling factor (Eq. 19, [1/2, 1])
    γfact_PT::T         = T(50)             # DIMENSIONLESS augmented-Lagrangian over-penalisation factor. The penalty is operator-derived per-cell, γ_eff = γfact/s_P_phys (s_P = Schur diagonal, init.jl). It does double duty: (i) the multiplier/penalty step P_num=γ_eff·R_P, and (ii) the artificial-compressibility η_b=γ_eff in comp=(P−P0)/(η_b·dt). KEY (2026-06-11): comp ∝ 1/(γfact·dt) is a perturbation of each linear solve away from the EXACT (direct) solution — it vanishes only when P→P0 (matrix-free operator itself is identical to :direct, verified 1e-12). At FINITE dt in the inertial production regime a frozen entry-P0 leaves comp > the outer Picard rtol (1e-4) at the default γfact=50, so the Picard loop WOBBLES. TWO cures: (a) :gmg now uses PSEUDO-TRANSIENT P0-tracking (advances P0 each V-cycle, src/dyrel_GMG/vcycle.jl) ⇒ comp→0 at convergence ⇒ :gmg converges at the DEFAULT γfact=50 (no tuning) — PREFERRED for high-N production. (b) a LARGER γfact (≳1e3–5e3) shrinks comp directly and converges BOTH :dyrel and :gmg (use for :dyrel standalone, which can't P0-track without losing its restoring force). The default 50 is the steady/low-N sweet spot (dt→∞, comp≈0): 5→~40 iters, 20→~13, 50→~8, 200→~7.
    rel_drop_PT::T      = T(1e-2)           # inner-DR tol = outer PH err × rel_drop
    atol_PH::T          = atol              # per-fluidmech (linear) solve tolerance — AUTO-TRACKS the nonlinear `atol` by default. The outer Picard loop only needs each fluidmech solve as accurate as `atol`; the old fixed 1e-6 made the solver OVER-SOLVE when atol was looser (e.g. atol=1e-3 ⇒ ~1000× wasted work) — catastrophic for :gmg/:mg (each extra V-cycle is expensive) and the cause of dyrel's Picard wobble (linear looser than nonlinear target ⇒ can't reach atol). At matched atol_PH=atol, :gmg is ~21× faster than :dyrel on bnchm_cnsv (128², atol=1e-3). Override explicitly to decouple.
    pres_relax_PT::T    = T(1.0)            # augmented-Lagrangian multiplier under-relaxation ω in P += ω·γ_eff·R_P (ω=1 = Newton-Uzawa step; <1 under-relaxes for robustness).
    verbose_PH::Bool    = false             # print outer Powell-Hestenes itPH convergence (JustRelax-style)
    verbose_DR::Bool    = false             # print inner Dynamic-Relaxation itDR convergence
    linear_viscosity::Bool = false          # freeze η during the dyrel solve (constant-viscosity tests / MMS). false = recompute the strain-rate rheology η(eII) INSIDE the PT loop (the nonlinear, JustRelax-style path)
    viscosity_relaxation::T = T(0.5)        # blend weight for the in-loop η update (JustRelax `viscosity_relaxation`; 0.5 = the update.m Picard blend)

    # --- geometric-multigrid (:gmg) controls — the dyrel sweep is the per-level
    # smoother; only the coarsest level is solved to tolerance. ---
    mg_minlvl::Int      = 16                 # coarsen until min(Nz,Nx) would drop below this ⇒ coarsest stays ~16–31 cells/dim. 16² is the floor on purpose: the coarse grid must still RESOLVE the nonlinear rheology (power-law / T-dependent η) so the coarse correction sees the right operator (future development). Level count scales with N. GMG efficiency needs grids that coarsen cleanly to ~16 ⇒ resolution must be a multiple of 32 (asserted in build_hierarchy); awkward N (200=8·25) can't and is rejected.
    mg_npre::Int        = 2                  # pre-smooth PH steps per level (V-cycle down)
    mg_npost::Int       = 2                  # post-smooth PH steps per level (V-cycle up)
    mg_ncoarse::Int     = 20                 # max coarsest-level PH steps (solve-to-tolerance budget)
    mg_coarse_rtol::T   = T(1e-2)            # coarsest-level relative residual drop (solve-to-tolerance)
    mg_inner::Int       = 32                 # DR iterations per smoothing PH step — an EXACT count (no polling/reductions inside the smoother). The old ~32 stability floor came from the smoother running UNDAMPED (Rayleigh c never computed within a short sweep); with the fixed per-level damping `mg_cfact` much smaller counts are stable — tune down for cheaper V-cycles.
    mg_cfact::T         = T(1)               # per-level smoother damping scale: c_level = mg_cfact·√(mean λmax), the critical damping of the upper half-spectrum (λ ≳ λmax/4) the level must smooth (coarser levels handle the rest). 0 reproduces the old undamped smoother.
    mg_gfact::T         = γfact_PT           # SMOOTHING-level AL over-penalization (coarsest always keeps γfact_PT). MEASURED (harness, step-1 state, N=128/256): REDUCING it below γfact_PT makes the V-cycle WORSE — ρ_cycle 0.55→1.8→3.7 for 50→2→1 at N=128 full depth — because the per-sweep multiplier update γ_eff·R_P is the only high-frequency PRESSURE smoothing the level has; weakening it starves intermediate levels (they neither smooth nor solve P) and the cycle amplifies. The "small γ inside MG" theory (grad-div spectrum compression) is falsified for this scheme; keep the penalty strong on all levels. Knob retained for experiments.
    mg_gamma::T         = T(1.0)             # coarse-grid-correction damping γ_mg. γ_mg=1 = full (undamped) correction = textbook MG. HISTORY: with a FROZEN entry-P0 comp term, γ_mg=1 DIVERGED (ρ_cycle≈1.6 → 1e16) because the over-penalised, biased per-cycle system over-corrected; the workaround was γ_mg=0.5. The PSEUDO-TRANSIENT P0-tracking fix (vcycle.jl, advance P0 each V-cycle) makes each cycle solve a well-posed per-increment system, which STABILISES the full correction: measured at N=256 production, γ_mg=1.0 ⇒ a FLAT 6 V-cycles/solve (textbook), vs 0.5 ⇒ ~19–64 (the constant-0.87 factor mode). γ_mg=1.2 over-shoots/diverges, so 1.0 is the optimum. (Restoring γ_mg=1 is THE efficiency win that lets :gmg beat :direct at large N — O(N²·6) vs O(N³).)

    # --- modes ---
    bnchm::Bool         = false             # MMS benchmark mode
    postprc::Bool       = false             # post-processing only
end

"""
    Parameters(::Type{T}; kwargs...) where {T}

Convenience constructor matching MATLAB driver scripts: `Parameters(Float64; D=10, N=200, ...)`.
Any keyword argument with a numeric value is converted to `T` before construction so users
can pass `D=10` rather than `D=Float32(10)`.
"""
function Parameters(::Type{T}; kwargs...) where {T<:AbstractFloat}
    int_fields = (:N, :Nt, :seed, :restart, :nrh, :nop, :maxit)
    converted = Dict{Symbol,Any}()
    for (k, v) in kwargs
        if k in int_fields
            converted[k] = Int(v)                       # accept 1e6 etc., narrow to Int
        elseif v isa Real
            converted[k] = T(v)
        elseif v isa AbstractArray && eltype(v) <: Real
            converted[k] = T.(v)
        else
            converted[k] = v                            # strings, symbols, bools pass through
        end
    end
    return Parameters{T}(; converted...)
end
