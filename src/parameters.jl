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
    g0::T               = T(10)

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
    maxit::Int          = 15
    alpha::T            = T(0.9)
    gamma::T            = T(1e-3)
    kmin::T             = T(1e-16)
    kmax::T             = T(1e16)
    dtmax::T            = T(1e32)
    etacntr::T          = T(1e8)

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
