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
    vtk_op::Bool        = false             # ParaView frames (save_vtk) every nop steps
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
    L::T                = D                # chamber width; set equal to h for 1-D mode

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
    # permission-weight matrices [2x2], same layout as par_default.m
    AA::Matrix{T}       = T[0.72 0.19; 0.81 0.20]
    BB::Matrix{T}       = T[0.63 0.37; 0.999 0.001]
    CC::Matrix{T}       = T[2.09 0.09; 0.37 1.45]

    # --- physical control / noise ---
    L0::T               = D / 100          # eddy correlation length
    l0::T               = d0 * 10          # phase-fluctuation correlation length
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
    solver::Symbol      = :direct  # Stokes solver: :direct (sparse LU), :dyrel (pseudo-transient), :gmg (geometric multigrid with the dyrel sweep as smoother), :vmg (DYREL Powell-Hestenes + velocity V-cycle + Cahouet-Chabard)
    maxit_PH::Int       = 100                # outer Powell-Hestenes iterations
    maxit_PT::Int       = 50_000            # inner Dynamic-Relaxation iterations per PH step
    total_iterMax_PT::Int = 50_000          # global DR-iteration cap per fluidmech_dyrel! call
    n_tune_PT::Int      = 25                # re-estimate λ (Gershgorin + Rayleigh) every n iters
    CFL_PT::T           = T(0.99)           # DR pseudo-time CFL (< 1)
    c_fact_PT::T        = T(0.5)            # damping scaling factor (Eq. 19, [1/2, 1])
    γfact_PT::T         = T(50)  # augmented-Lagrangian factor, γ_eff = γfact/s_P_phys; starting value for :dyrel (which retunes it from the measured Powell-Hestenes contraction), used as given by :gmg via mg_gfact
    rel_drop_PT::T      = T(1e-2)           # inner-DR target: reduce the velocity residual by this factor WITHIN each PH sweep
    atol_PH::T          = atol  # linear-solve tolerance per fluidmech call; looser than `atol` and the Picard loop cannot reach `atol`, much tighter and every sweep over-solves
    forcing_PH::T       = T(0.1)  # inexact-solve forcing for the matrix-free solvers: a solve may stop once its error has dropped by this factor (:dyrel from entry, :gmg from its first V-cycle); such a solve does not count as converged. Changes the Picard path, so results agree with :direct to the Picard tolerance. 0 solves to `atol_PH`
    pres_relax_PT::T    = T(1.0)  # multiplier under-relaxation ω in P += ω·γ_eff·R_P (1 = Newton-Uzawa step)
    anderson_PH::Int    = 0  # Anderson depth m for the :dyrel Powell-Hestenes pressure update (least-squares mix of the last m iterates); 0 = plain step. History restarts every solve and whenever γ_eff is retuned
    rheology_in_PT::Bool = false  # refresh η(eII) inside the :dyrel iteration instead of in the Picard loop; makes :dyrel solve a different problem from `fluidmech!` and overwrites a prescribed η, so keep off unless the strain-rate dependence is strong
    visc_relax_PT::T    = T(0.25)  # weight w of the in-solve η refresh, η ← w·η(eII) + (1−w)·η, applied every n_tune_PT iterations

    # --- geometric multigrid (:gmg): the dyrel sweep smooths each level, only the
    # coarsest level is solved to tolerance ---
    mg_minlvl::Int      = 16  # coarsen until min(Nz,Nx) would drop below this; the coarsest grid must still resolve the nonlinear rheology. Resolution must be a multiple of 32 (checked in build_hierarchy)
    mg_npre::Int        = 2                  # pre-smooth PH steps per level (V-cycle down)
    mg_npost::Int       = 2                  # post-smooth PH steps per level (V-cycle up)
    mg_ncoarse::Int     = 20                 # max coarsest-level PH steps (solve-to-tolerance budget)
    mg_coarse_rho::T    = T(0.5)  # target Powell-Hestenes contraction the coarsest level tunes γfact to (`coarse_solve!`); ≤ 0 keeps `γfact_PT`. A retune that worsens the measured contraction is undone
    mg_coarse_rtol::T   = T(1e-4)  # coarsest-level relative residual drop (`coarse_solve!`)
    mg_inner::Int       = 32  # DR iterations per smoothing PH step (exact count, no reductions inside the smoother)
    mg_cfact::T         = T(1)  # per-level smoother damping scale: c_level = mg_cfact·√(mean λmax); 0 leaves the smoother undamped
    mg_gfact::T         = γfact_PT  # augmented-Lagrangian penalty on the smoothing levels (the coarsest keeps γfact_PT); keep at γfact_PT, a weaker penalty makes the V-cycle diverge
    mg_gamma::T         = T(1.0)  # coarse-grid-correction damping γ_mg; 1 = full correction, ~6 V-cycles per solve independent of N

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
            converted[k] = Int(v)                       # accept 1e6
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
