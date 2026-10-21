# Gershgorin auto-tune of the dynamic-relaxation iteration, the Schur-complement
# diagonal, and the operator-derived augmented-Lagrangian penalty.
#
# Stress convention τ = η(ε̇ − ⅓∇·v I): normal coefficient cn = 2/3, deviatoric
# cross coefficient cc = 1/3, shear cs = 1/2, so `Dv` is the true Jacobi diagonal
# of the assembled operator. Per face of component d, with cell viscosity η and
# penalty γ on either side along d and edge viscosity η_k on either side along
# each other axis k:
#
#   D_d    = (Σ_k cs·(η_k⁺ + η_k⁻)/h + cn·(η⁺ + η⁻)/h)/h + γ_diag/h² + δ + max(σ, 0)
#   λmax_d = (C_same + C_cross + δ + |σ|)/D_d
#
# C_same/C_cross are the row sums over the same component and the other
# components; δ = cin·ρ_d/dt is the BD2 inertia, σ = ∂_dρ·g0·dt the drunken-sailor
# diagonal (in D only by its positive part, so an inverted density gradient
# cannot make the preconditioner non-positive). R_P is 1/ρ-scaled and
# differences the mass flux, so the penalty coupling from cell c to any face g
# of it is γ_c·ρ_g/ρ_c; every entry carries that factor (γ_diag = ρ_d·Σ γ/ρ_cell
# on the diagonal), so the row sum bounds the spectrum under density contrasts.
# Wall faces of a bounded horizontal axis are Dirichlet: D = λmax = 1. The
# z-boundary W faces repeat the boundary cell.

@kernel function _gershgorin_kernel!(Dd, λd, comp, @Const(η), @Const(ηe), @Const(γ_eff),
                                     @Const(ρf), @Const(rho), periodic,
                                     invh2, invh, cn, cc, cs, iner, g0dt)
    J = @index(Global, Cartesian)
    @inbounds begin
        d  = _val(comp)
        D  = length(J)
        Nd = size(η, d)
        ed = _unit(J, d)
        ρd = ρf[d]
        wall = d != 1 && !periodic[d] && (J[d] == 1 || J[d] == Nd + 1)
        lo = J[d] == 1      ? (d == 1 ? J : _setidx(J, d, Nd)) : J - ed
        hi = J[d] == Nd + 1 ? (d == 1 ? J - ed : _setidx(J, d, 1)) : J
        η_hi = η[hi];  η_lo = η[lo]
        gρ_hi = γ_eff[hi] / rho[hi];  gρ_lo = γ_eff[lo] / rho[lo]
        γ_diag = ρd[J] * (gρ_hi + gρ_lo)
        ks = _others(comp, Val(D))
        shear2 = _tfoldl(zero(η_hi), ks) do s, k
            @inbounds s + cs * (ηe[_pair(d, k, D)][J + _unit(J, k)] + ηe[_pair(d, k, D)][J]) * invh2
        end
        shear1 = _tfoldl(zero(η_hi), ks) do s, k
            @inbounds s + cs * (ηe[_pair(d, k, D)][J + _unit(J, k)] + ηe[_pair(d, k, D)][J]) * invh
        end
        # cross coupling to component k: the k-faces of cells hi and lo; the
        # upper face shares the upper edge viscosity, the lower face the lower
        cross = _tfoldl(zero(η_hi), ks) do s, k
            @inbounds begin
                ek = _unit(J, k);  ηp = ηe[_pair(d, k, D)]
                e1 = ηp[J + ek];  e2 = ηp[J];  ρk = ρf[k]
                s + (abs(gρ_hi * ρk[hi + ek] - cc * η_hi + cs * e1) +
                     abs(gρ_hi * ρk[hi]      - cc * η_hi + cs * e2) +
                     abs(gρ_lo * ρk[lo + ek] - cc * η_lo + cs * e1) +
                     abs(gρ_lo * ρk[lo]      - cc * η_lo + cs * e2)) * invh2
            end
        end
        diagv  = (shear1 + cn * (η_hi + η_lo) * invh) * invh
        # same-component neighbors: the d-faces across cells hi and lo
        C_same = shear2 + (gρ_hi * ρd[hi + ed] + cn * η_hi) * invh2 +
                 (gρ_lo * ρd[lo] + cn * η_lo) * invh2 + diagv + γ_diag * invh2
        δ = iner * ρd[J]
        σ = (rho[hi] - rho[lo]) * invh * g0dt
        Dv = diagv + γ_diag * invh2 + δ + max(σ, zero(σ))
        Dd[J] = wall ? one(Dv) : Dv
        λd[J] = wall ? one(Dv) : (C_same + cross + δ + abs(σ)) / Dv
    end
end

"""
    gershgorin!(cache, fluid, grid, op) -> nothing

Jacobi diagonals `cache.Dv` and Gershgorin bounds `cache.λmax` of every
component at the current coefficients and penalty `cache.γ_eff`. `op.dt` must be
finite: the drunken-sailor diagonal scales with it.
"""
function gershgorin!(cache::StokesCache{D, T}, fluid, grid, op::StokesOp{D}) where {D, T}
    backend = KernelAbstractions.get_backend(cache.R_P)
    invh = T(inv(grid.h));  invh2 = invh * invh
    ηe = edge_viscosity(fluid);  ρf = face_density(fluid)
    g0dt = op.g0 * op.dt
    _each_axis(Val(D)) do comp
        d = _val(comp)
        _gershgorin_kernel!(backend)(cache.Dv[d], cache.λmax[d], comp, fluid.eta, ηe,
                                     cache.γ_eff, ρf, fluid.rho, op.periodic,
                                     invh2, invh, T(2//3), T(1//3), T(1//2),
                                     op.cin[d] * op.inv_dt, g0dt; ndrange = size(cache.Dv[d]))
    end
    return nothing
end

# Schur-complement diagonal s_P[c] = (1/(h²ρ_c)) Σ_faces ρ_f/D_f. Dirichlet faces
# (z-boundary W, walls of a bounded axis) cannot respond to a pressure change
# and are left out; counting them would deflate the penalty γ_eff = γfact/s_P
# exactly in the boundary cells.
@kernel function _penalty_kernel!(γ_eff, @Const(Dv), @Const(ρf), @Const(rho), periodic, invh2, γ)
    c = @index(Global, Cartesian)
    @inbounds begin
        D = length(c);  Z = zero(eltype(γ_eff))
        sfac = _tfoldl(Z, _hz_axes(Val(D))) do s, k
            ek = _unit(c, k)
            fixed = k == 1 || !periodic[k]
            @inbounds begin
                lo = fixed && c[k] == 1            ? Z : ρf[k][c] / Dv[k][c]
                hi = fixed && c[k] == size(γ_eff, k) ? Z : ρf[k][c + ek] / Dv[k][c + ek]
            end
            s + lo + hi
        end
        s_P = sfac * invh2 / rho[c]
        γ_eff[c] = γ / s_P
    end
end

"""
    penalty!(cache, fluid, grid, op; γfact) -> nothing

Operator-derived augmented-Lagrangian penalty `γ_eff = γfact/s_P_phys`, with
`s_P_phys` the Schur diagonal of the penalty-free momentum diagonal, and
`η_b = γ_eff·max(1, 1/(ρ·dt))`. Leaves `cache.Dv`/`cache.λmax` holding the
penalized diagonal.

`γfact` is dimensionless: `1/s_P` is a viscosity carrying the viscous and
inertial scaling of the operator. The same γ_eff is the momentum penalty
(`P_num = γ_eff·R_P`), the multiplier step (`P += ω·γ_eff·R_P`) and, through
η_b, the pseudo-transient coefficient of the multigrid smoother; flooring η_b at
γ_eff/(ρ·dt) caps that term's diagonal ω·γ_eff/(η_b·ρ·dt) in the multiplier
update at ω, which keeps the update stable when ρ·dt < 1/2.
"""
function penalty!(cache::StokesCache{D, T}, fluid, grid, op::StokesOp{D};
                  γfact::Real) where {D, T}
    fill!(cache.γ_eff, zero(T))
    gershgorin!(cache, fluid, grid, op)
    _penalty_kernel!(KernelAbstractions.get_backend(cache.γ_eff))(
        cache.γ_eff, cache.Dv, face_density(fluid), fluid.rho, op.periodic,
        T(inv(grid.h)^2), T(γfact); ndrange = size(cache.γ_eff))
    dtT = op.dt
    @. cache.η_b = cache.γ_eff * max(one(T), inv(fluid.rho * dtT))
    gershgorin!(cache, fluid, grid, op)
    return nothing
end

"""
    check_preconditioner(cache) -> nothing

Throw a `DomainError` if the penalty or any Jacobi diagonal / Gershgorin bound
is non-finite or non-positive.
"""
function check_preconditioner(cache::StokesCache{D, T}) where {D, T}
    for A in (cache.γ_eff, cache.η_b, cache.Dv..., cache.λmax...)
        all(isfinite, A) && all(>(zero(T)), A) ||
            throw(DomainError(A, "the Stokes solver produced a non-finite or non-positive preconditioner entry"))
    end
    return nothing
end
