# Momentum and continuity residuals of the compressible two-phase Navier–Stokes
# system (Re > 1), the matrix-free form of the operator `fluidmech!` assembles.
#
# Momentum, component d (R = b − A·v, driven to zero by dynamic relaxation):
#   R_d = ∂_d τ_dd + Σ_{k≠d} ∂_k τ_dk − ∂_d P            [stress divergence, pressure]
#       − cin_d·ρ_d·v_d/dt − (∂_d ρ)·g0·dt·v_d            [BD2 inertia, drunken sailor]
#       + fscale·([d = 1]·Drho·g0 + (a2·m° + a3·m°°)/dt − advn_d) + bext_d
# with τ = η(ε̇ − ⅓(∇·v)I) (normal stress at cells, shear at edges). `fscale = 1`,
# `bext = 0` is the physical problem; multigrid coarse levels solve the
# homogeneous error equation with `fscale = 0` and the restricted residual in
# `bext`. With `dr = true` the residual is preconditioned for the DR update:
# (R_d − ∂_d P_num)/D_d, P_num = γ_eff·R_P the Schur (augmented-Lagrangian) penalty.
#
# Continuity, 1/ρ-scaled so the penalty γ_eff·R_P is ~γ_eff·∇·v, matching the
# Gershgorin bound that carries γ_eff with no density factor:
#   R_P = (fscale·MFS − ∇·(ρv) − (P − P0)/(η_b·dt))/ρ + bext_P.
# The last term is the multigrid smoother's pseudo-transient regularizer; the
# single-grid solver passes dt = Inf, which removes it.

# normal strain rates ∂_k v_k at cell c, all k
Base.@propagate_inbounds @inline function _strain_rates(v, c::CartesianIndex{D}, invh) where {D}
    o = _ones(c)
    return ntuple(k -> @inbounds((v[k][c + o] - v[k][c + o - _unit(c, k)]) * invh), Val(D))
end

# normal deviatoric stress τ_dd at cell c
Base.@propagate_inbounds @inline function _normal_stress(v, eta, c, d, invh, third)
    g = _strain_rates(v, c, invh)
    return eta[c] * (g[d] - _hsum(g) * third)
end

# shear stress τ_ab at the edge E of the axis pair (a, b), a < b
Base.@propagate_inbounds @inline function _shear_stress(v, ηe, E::CartesianIndex{D}, a, b,
                                                        invh, half) where {D}
    ea = _unit(E, a);  eb = _unit(E, b)
    o  = _ones(E) - ea - eb
    η  = ηe[_pair(a, b, D)][E]
    return η * ((v[b][E + o + ea] - v[b][E + o]) * invh +
                (v[a][E + o + eb] - v[a][E + o]) * invh) * half
end

# ∂_k τ_dk at the d-face J
Base.@propagate_inbounds @inline function _shear_divergence(v, ηe, J, d, k, invh, half)
    a, b = minmax(d, k)
    return (_shear_stress(v, ηe, J + _unit(J, k), a, b, invh, half) -
            _shear_stress(v, ηe, J, a, b, invh, half)) * invh
end

# Cells on either side of the d-face J; periodic wrap across the domain edge.
# Boundary faces of a bounded axis never reach these (they are Dirichlet), and
# z is always bounded, so its faces need no wrap.
@inline _cell_hi(J, d, Nd) = d != 1 && J[d] == Nd + 1 ? _setidx(J, d, 1) : J
@inline _cell_lo(J, d, Nd) = d != 1 && J[d] == 1 ? _setidx(J, d, Nd) : J - _unit(J, d)

# Dirichlet face: z-boundary faces of W, wall faces of a bounded horizontal axis
@inline _dirichlet_face(J, d, n, periodic) =
    (d == 1 || !periodic[d]) && (J[d] == 1 || J[d] == n)

@kernel function _momentum_residual_kernel!(R, comp, @Const(v), @Const(eta), @Const(ηe), @Const(P),
                                            @Const(P_num), @Const(Drho), @Const(rho),
                                            @Const(ρf), @Const(mo), @Const(moo),
                                            @Const(advn), @Const(Dd), @Const(bext),
                                            periodic, invh, g0, third, half, cin, a2, a3,
                                            inv_dt, dt, fscale, dr::Bool)
    J = @index(Global, Cartesian)
    @inbounds begin
        d  = _val(comp)
        D  = length(J)
        if _dirichlet_face(J, d, size(R, d), periodic)
            R[J] = zero(eltype(R))
        else
            o  = _ones(J);  ed = _unit(J, d)
            Nd = size(eta, d)
            hi = _cell_hi(J, d, Nd);  lo = _cell_lo(J, d, Nd)
            vd = v[d][J + o - ed]
            dτn = (_normal_stress(v, eta, hi, d, invh, third) -
                   _normal_stress(v, eta, lo, d, invh, third)) * invh
            stress  = _tfoldl((acc, k) -> @inbounds(acc + _shear_divergence(v, ηe, J, d, k, invh, half)),
                              dτn, _others(comp, Val(D)))
            dP      = (P[J + o] - P[J + o - ed]) * invh
            inertia = cin * ρf[J] * vd * inv_dt
            sailor  = (rho[hi] - rho[lo]) * invh * g0 * dt * vd
            lagged  = (a2 * mo[J] + a3 * moo[J]) * inv_dt
            force   = d == 1 ? Drho[J] * g0 + lagged - advn[J] : lagged - advn[J]
            r = stress - dP - inertia - sailor + fscale * force + bext[J]
            if dr
                dPnum = (P_num[hi] - P_num[lo]) * invh
                R[J]  = (r - dPnum) / Dd[J]
            else
                R[J] = r
            end
        end
    end
end

@kernel function _continuity_residual_kernel!(R_P, P_num, @Const(v), @Const(P), @Const(P0),
                                              @Const(MFS), @Const(rho), @Const(ρf), @Const(η_b),
                                              @Const(bext_P), @Const(γ_eff),
                                              invh, inv_dt, fscale)
    c = @index(Global, Cartesian)
    @inbounds begin
        D = length(c);  o = _ones(c)
        flux = ntuple(k -> @inbounds((ρf[k][c + _unit(c, k)] * v[k][c + o] -
                                      ρf[k][c] * v[k][c + o - _unit(c, k)]) * invh), Val(D))
        div_ρv = _hsum(flux)
        comp   = (P[c + o] - P0[c + o]) / η_b[c] * inv_dt
        rp     = (fscale * MFS[c] - div_ρv - comp) / rho[c] + bext_P[c]
        R_P[c]   = rp
        P_num[c] = γ_eff[c] * rp
    end
end

"""
    residual_momentum!(cache, fluid, grid, op; dr, fscale = 1) -> nothing

Momentum residuals of every component into `cache.Rv`: `b − A·v` (`dr = false`),
or the Jacobi-preconditioned DR form `(b − A·v − ∇P_num)/D` (`dr = true`, which
reads `cache.P_num`, so call `residual_continuity!` first). Stresses are
evaluated inline from the current velocity.
"""
function residual_momentum!(cache::StokesCache{D, T}, fluid, grid, op::StokesOp{D};
                            dr::Bool, fscale::Real = 1) where {D, T}
    backend = KernelAbstractions.get_backend(cache.R_P)
    v  = velocity(fluid);  ηe = edge_viscosity(fluid);  ρf = face_density(fluid)
    mo = momentum_lag1(fluid);  moo = momentum_lag2(fluid)
    invh = T(inv(grid.h));  fs = T(fscale);  third = T(1//3);  half = T(1//2)
    _each_axis(Val(D)) do comp
        d = _val(comp)
        _momentum_residual_kernel!(backend)(
            cache.Rv[d], comp, v, fluid.eta, ηe, fluid.P, cache.P_num, fluid.Drho,
            fluid.rho, ρf[d], mo[d], moo[d], cache.advn[d], cache.Dv[d], cache.bext[d],
            op.periodic, invh, op.g0, third, half, op.cin[d], op.a2, op.a3, op.inv_dt,
            op.dt, fs, dr; ndrange = size(cache.Rv[d]))
    end
    return nothing
end

"""
    residual_continuity!(cache, fluid, grid, P0, dt; fscale = 1) -> nothing

Continuity residual into `cache.R_P` and the fused Schur penalty
`cache.P_num = γ_eff·R_P`. `dt = Inf` removes the pseudo-transient term
`(P − P0)/(η_b·dt)`.
"""
function residual_continuity!(cache::StokesCache{D, T}, fluid, grid, P0, dt::Real;
                              fscale::Real = 1) where {D, T}
    backend = KernelAbstractions.get_backend(cache.R_P)
    inv_dt = isfinite(dt) ? T(inv(dt)) : zero(T)
    _continuity_residual_kernel!(backend)(
        cache.R_P, cache.P_num, velocity(fluid), fluid.P, P0, fluid.MFS, fluid.rho,
        face_density(fluid), cache.η_b, cache.bext_P, cache.γ_eff, T(inv(grid.h)),
        inv_dt, T(fscale); ndrange = size(cache.R_P))
    return nothing
end

@kernel function _velocity_update_kernel!(v, dvdτ, dv_step, @Const(R), @Const(α), @Const(β),
                                          @Const(dτ), comp)
    J = @index(Global, Cartesian)
    @inbounds begin
        d = _val(comp)
        rate = α[J] * dvdτ[J] + R[J]
        dvdτ[J] = rate
        step = β[J] * dτ[J] * rate
        dv_step[J] = step
        v[J + _ones(J) - _unit(J, d)] += step
    end
end

"""
    update_velocity!(cache, fluid) -> nothing

One damped DR pseudo-time step of every component from the preconditioned
residual: `dvdτ ← α·dvdτ + R`, `v += β·dτ·dvdτ` (increment kept in `dv_step` for
the Rayleigh-quotient estimate).
"""
function update_velocity!(cache::StokesCache{D}, fluid) where {D}
    backend = KernelAbstractions.get_backend(cache.R_P)
    v = velocity(fluid)
    _each_axis(Val(D)) do comp
        d = _val(comp)
        _velocity_update_kernel!(backend)(v[d], cache.dvdτ[d], cache.dv_step[d], cache.Rv[d],
                                          cache.α[d], cache.β[d], cache.dτ[d], comp;
                                          ndrange = size(cache.Rv[d]))
    end
    return nothing
end
