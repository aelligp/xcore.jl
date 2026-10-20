# Global reductions: Rayleigh-quotient λmin, residual norms and the
# Powell-Hestenes pressure-update gate. Each reduces over a lazy `Broadcasted`
# (`sum(_bc(f, A, B))`), a single on-device pass with no temporary array and no
# host-side zip of the operands (which would scalar-index a GPU array).

@inline _bc(f, args...) = Broadcast.instantiate(Broadcast.broadcasted(f, args...))

_dot_diff(a, b, c) = sum(_bc((ai, bi, ci) -> ai * (bi - ci), a, b, c))

"""
    rayleigh_λmin(cache) -> λmin

Rayleigh-quotient estimate of the smallest eigenvalue of the preconditioned
velocity operator from the last increment `dv_step` and the residual change
`Rv − Rv_old`: `|Σ Δv·ΔR| / Σ Δv²` (0 when there is no increment).
"""
function rayleigh_λmin(cache::StokesCache{D, T}) where {D, T}
    num = abs(foldl(+, ntuple(d -> _dot_diff(cache.dv_step[d], cache.Rv[d], cache.Rv_old[d]), Val(D))))
    den = foldl(+, ntuple(d -> sum(abs2, cache.dv_step[d]), Val(D)))
    return den > zero(T) ? T(num / den) : zero(T)
end

"""
    velocity_norms(cache) -> NTuple{D}

RMS of each momentum residual `cache.Rv[d]`.
"""
velocity_norms(cache::StokesCache{D, T}) where {D, T} =
    ntuple(d -> T(sqrt(sum(abs2, cache.Rv[d])) / sqrt(length(cache.Rv[d]))), Val(D))

"""
    residual_norms(cache, rho) -> (norms_v::NTuple{D}, normP)

RMS of the momentum residuals and of the mass imbalance `ρ·R_P = MFS − ∇·(ρv)`
(the 1/ρ scaling of `R_P` would otherwise understate it by ρ).
"""
function residual_norms(cache::StokesCache{D, T}, rho) where {D, T}
    normP = T(sqrt(sum(_bc((r, ρ) -> abs2(r * ρ), cache.R_P, rho))) / sqrt(length(cache.R_P)))
    return velocity_norms(cache), normP
end

"""
    velocity_residual_norm(cache) -> err

Inner-DR convergence measure `max_d RMS(D_d·R_d)` of the preconditioned residual
(diagonal-weighted, so it is comparable across viscosity contrasts).
"""
velocity_residual_norm(cache::StokesCache{D, T}) where {D, T} =
    maximum(d -> T(sqrt(sum(_bc((dg, r) -> abs2(dg * r), cache.Dv[d], cache.Rv[d]))) /
                   sqrt(length(cache.Rv[d]))), 1:D)

"""
    pressure_update_norm(cache, fluid, h) -> RMS(γ_eff·R_P) / max(RMS(P̃), RMS(τ_II))

Size of the pending multiplier update `γ_eff·R_P` against the pressure scale of
the flow: `P̃ = P − mean(P)` (relative part) and the deviatoric stress invariant
τ_II (absolute part). Both are invariant under a constant pressure offset, and
the stress keeps the scale nonzero under a uniform pressure; with neither, the
update is returned unscaled. Gating on the update rather than on `R_P` alone
makes the accepted pressure independent of the penalty strength.
"""
function pressure_update_norm(cache::StokesCache{D, T}, fluid, h::Real) where {D, T}
    dP = sqrt(sum(_bc((g, r) -> abs2(g * r), cache.γ_eff, cache.R_P)) / length(cache.R_P))
    return _relative_to_pressure_scale(T(dP), fluid, h)
end

"""
    pressure_change_norm(ΔP, fluid, h) -> RMS(ΔP − mean ΔP) / max(RMS(P̃), RMS(τ_II))

The same mixed criterion for a pressure change `ΔP` (interior cells) that an
update has already applied. Its constant part is the gauge mode and is excluded.
With a Schur-complement-preconditioned update (Cahouet–Chabard, or a multigrid
cycle) `ΔP` approximates the remaining pressure error, which the residual norms
bound only up to the conditioning of the Schur complement — a factor that grows
with grid resolution.
"""
pressure_change_norm(ΔP, fluid, h::Real) =
    _relative_to_pressure_scale(_deviation_rms(ΔP), fluid, h)

function _relative_to_pressure_scale(dP::T, fluid, h::Real) where {T}
    scale = max(_pressure_deviation_rms(fluid.P), _deviatoric_stress_rms(fluid, T(inv(h))))
    return scale > zero(T) ? dP / scale : dP
end

# RMS of an array about its mean (two passes: the mean can dominate)
function _deviation_rms(A)
    m = sum(A) / length(A)
    return sqrt(sum(_bc(a -> abs2(a - m), A)) / length(A))
end
_pressure_deviation_rms(P) = _deviation_rms(_interior(P))

# view of a ghosted velocity component: `r_along` along axis `along`, the full
# range along every axis in `full`, the interior (ghosts removed) elsewhere
function _slab(A::AbstractArray{<:Any, D}, along, r_along::UnitRange{Int}, full) where {D}
    return view(A, ntuple(k -> k == along ? r_along : (k in full ? (1:size(A, k)) : (2:size(A, k) - 1)), Val(D))...)
end

# RMS of τ_II = √(½Σ τ_dd² + Σ_{a<b} τ_ab²): normal stresses at cells, shear at
# edges (boundary ghosts carry the BCs), with the residual kernels' stencils.
function _deviatoric_stress_rms(fluid, invh::T) where {T}
    v = velocity(fluid);  ηe = edge_viscosity(fluid)
    D = length(v)
    third = T(1//3);  half = T(1//2)
    hi(d) = _slab(v[d], d, 2:size(v[d], d), ())
    lo(d) = _slab(v[d], d, 1:size(v[d], d) - 1, ())
    args = ntuple(i -> isodd(i) ? hi((i + 1) ÷ 2) : lo(i ÷ 2), Val(2D))
    normal = sum(_bc(fluid.eta, args...) do η, x...
        n = Val(length(x) ÷ 2)
        g = ntuple(d -> (x[2d - 1] - x[2d]) * invh, n)
        dv = _hsum(g) * third
        half * sum(ntuple(d -> abs2(η * (g[d] - dv)), n))
    end) / length(fluid.eta)
    shear = zero(T)
    for a in 1:D-1, b in a+1:D
        η = ηe[_pair(a, b, D)]
        # v_b differenced along a, v_a along b; full range along both pair axes
        vb, va = v[b], v[a]
        shear += sum(_bc(_slab(vb, a, 2:size(vb, a), (a, b)), _slab(vb, a, 1:size(vb, a) - 1, (a, b)),
                         _slab(va, b, 2:size(va, b), (a, b)), _slab(va, b, 1:size(va, b) - 1, (a, b)),
                         η) do b1, b0, a1, a0, ηv
            abs2(ηv * ((b1 - b0) + (a1 - a0)) * invh * half)
        end) / length(η)
    end
    return sqrt(normal + shear)
end
