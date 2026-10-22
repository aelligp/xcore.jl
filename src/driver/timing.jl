"""
    time_coefs(::Type{T}, TINT::Symbol, step::Integer) -> NTuple{6, T}

Return the six BDF-style coefficients `(a1, a2, a3, b1, b2, b3)` used by
`fluidmech!` and `phsevo!` for the LHS time-derivative weights `(a*)` and the
RHS rate-of-change weights `(b*)`. Direct port of `src/timing.m`.

Schemes:
* `:be1im`             — 1st-order backward Euler, fully implicit.
* `:bd2im`             — 2nd-order BDF, fully implicit (default).
* `:cn2si`             — 2nd-order Crank–Nicolson, semi-implicit.
* `:bd2si`             — 2nd-order BDF, semi-implicit.

The first two time steps always run with the BE coefficients (for `*im`
schemes) or BE then CN (for `*si` schemes); BD2 kicks in once enough history
is available.
"""
function time_coefs(::Type{T}, TINT::Symbol, step::Integer) where {T<:AbstractFloat}
    name = String(TINT)
    if endswith(name, "im")
        if TINT === :be1im || step <= 2
            return (T(1), T(1), T(0), T(1), T(0), T(0))
        elseif TINT === :bd2im
            return (T(3//2), T(4//2), T(-1//2), T(1), T(0), T(0))
        end
    elseif endswith(name, "si")
        if step == 1
            return (T(1), T(1), T(0), T(1), T(0), T(0))
        elseif TINT === :cn2si || step == 2
            return (T(1), T(1), T(0), T(1//2), T(1//2), T(0))
        elseif TINT === :bd2si
            return (T(3//2), T(4//2), T(-1//2), T(3//4), T(2//4), T(-1//4))
        end
    end
    throw(ArgumentError("time_coefs: unsupported TINT scheme $TINT (use :be1im, :bd2im, :cn2si, or :bd2si)"))
end
