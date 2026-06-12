using KernelAbstractions

# Residual kernels for xcore-DYREL — full compressible two-phase Navier-Stokes
# (Re > 1), NOT steady Stokes. Transcribed term-for-term from the matrix
# assembly in `src/fluidmech.jl` so the DYREL converged solution matches the
# direct sparse-LU solver.
#
# Momentum residual (R = KV·v + GG·P − rr, driven to 0 by DR), per face:
#   R_W = ∂_z(τzz) + ∂_x(τxz)            [viscous stress divergence]
#       − ∂_z(P)                          [pressure gradient]
#       + a1·ρw·W/dt                       [BD2 inertia, current — diagonal]
#       + (∂_zρ)·g0·dt·W                   [drunken-sailor stabilization — diagonal]
#       − Drho·g0                          [buoyancy]
#       − (a2·ρWo + a3·ρWoo)/dt            [BD2 inertia, lagged momentum flux]
#       + advn_mz                          [momentum advection div(v·ρW)]
#   (U analogous; inertia coefficient is (a1+γ); sailor uses ∂_xρ; no buoyancy)
#
# Continuity residual (compressible — mass-flux divergence, NOT ∇·v):
#   R_P = MFS − ∇·(ρv) − (P − P0)/(η_b·dt)
#
# Stress (txx,tzz cell; txz corner) from `dyrel_update_stress!`; momentum
# advection (advn_mz,advn_mx) from `dyrel_momentum_advection!` (Phase 2c).
#
# Index mapping (cache ↔ FluidState):
#   R_W[iz_f, ix] ↔ state.W[iz_f,   ix+1]   (z-faces; rows 1,Nz+1 = Dirichlet 0)
#   R_U[iz,   ix] ↔ state.U[iz+1,   ix  ]   (x-faces, interior z-rows)
#   R_P[iz,   ix] ↔ cell (iz, ix) = state.P[iz+1, ix+1]
#   advn_mz[iz_f-1, ix]  (interior z-faces only, size (Nz-1, Nx))
#   advn_mx[iz,    ix]   (all x-faces, size (Nz, Nx+1))

# ---------------------------------------------------------------------------
# W momentum residual (z-velocity). `dr` toggles the Schur-penalty + Jacobi
# preconditioner (true ⇒ inner DR form; false ⇒ unconditioned PH form).
# ---------------------------------------------------------------------------

@kernel function compute_dyrel_residual_W_kernel!(R_W, @Const(W), @Const(U),
                                                  @Const(eta), @Const(etaco),
                                                  @Const(P), @Const(P_num), @Const(Drho),
                                                  @Const(rho), @Const(rhow),
                                                  @Const(rhoWo), @Const(rhoWoo),
                                                  @Const(advn_mz), @Const(D_W),
                                                  invh, g0, third, half, a1, a2, a3, inv_dt, dt,
                                                  dr::Bool)
    iz_f, ix = @index(Global, NTuple)
    @inbounds begin
        Nzf = size(R_W, 1)                    # = Nz + 1
        if iz_f == 1 || iz_f == Nzf
            R_W[iz_f, ix] = zero(eltype(R_W))  # Dirichlet boundary face
        else
            Wv       = W[iz_f, ix + 1]
            # τzz computed INLINE (cells iz_f and iz_f-1) instead of read from a
            # precomputed array — same formula as dyrel/stress.jl, so bit-exact,
            # but saves the stress kernel launch + the txx/tzz/txz round-trip.
            Ux_hi  = (U[iz_f + 1, ix + 1] - U[iz_f + 1, ix]) * invh
            Wz_hi  = (W[iz_f + 1, ix + 1] - W[iz_f,     ix + 1]) * invh
            tzz_hi = eta[iz_f, ix] * (Wz_hi - (Ux_hi + Wz_hi) * third)
            Ux_lo  = (U[iz_f, ix + 1] - U[iz_f, ix]) * invh
            Wz_lo  = (W[iz_f, ix + 1] - W[iz_f - 1, ix + 1]) * invh
            tzz_lo = eta[iz_f - 1, ix] * (Wz_lo - (Ux_lo + Wz_lo) * third)
            d_tzz_dz = (tzz_hi - tzz_lo) * invh
            # τxz INLINE at corners (iz_f, ix) and (iz_f, ix+1)
            txz_l = etaco[iz_f, ix]     * ((U[iz_f + 1, ix]     - U[iz_f, ix])     * invh +
                                           (W[iz_f, ix + 1]     - W[iz_f, ix])     * invh) * half
            txz_r = etaco[iz_f, ix + 1] * ((U[iz_f + 1, ix + 1] - U[iz_f, ix + 1]) * invh +
                                           (W[iz_f, ix + 2]     - W[iz_f, ix + 1]) * invh) * half
            d_txz_dx = (txz_r - txz_l) * invh
            dP_dz    = (P[iz_f + 1, ix + 1] - P[iz_f, ix + 1]) * invh
            body     = Drho[iz_f, ix] * g0
            inertia  = a1 * rhow[iz_f, ix] * Wv * inv_dt
            ddz_rho  = (rho[iz_f, ix] - rho[iz_f - 1, ix]) * invh
            sailor   = ddz_rho * g0 * dt * Wv
            lagged   = (a2 * rhoWo[iz_f, ix] + a3 * rhoWoo[iz_f, ix]) * inv_dt
            adv      = advn_mz[iz_f - 1, ix]
            # R = −(KV·W + GG·P − rr): viscous (+∇·τ) and pressure (−∂P) keep
            # their sign; forcing/diagonal terms (buoyancy, inertia, sailor,
            # lagged, advection) carry the opposite sign to match fluidmech!.
            r = d_tzz_dz + d_txz_dx - dP_dz + body - inertia - sailor + lagged - adv
            if dr
                dPnum_dz = (P_num[iz_f, ix] - P_num[iz_f - 1, ix]) * invh
                R_W[iz_f, ix] = (r - dPnum_dz) / D_W[iz_f, ix]
            else
                R_W[iz_f, ix] = r
            end
        end
    end
end

# ---------------------------------------------------------------------------
# U momentum residual (x-velocity). Periodic in x; inertia coeff (a1+γ); no
# buoyancy (gravity is z-only); drunken-sailor uses ∂_xρ.
# ---------------------------------------------------------------------------

@kernel function compute_dyrel_residual_U_kernel!(R_U, @Const(U), @Const(W),
                                                  @Const(eta), @Const(etaco),
                                                  @Const(P), @Const(P_num),
                                                  @Const(rho), @Const(rhou),
                                                  @Const(rhoUo), @Const(rhoUoo),
                                                  @Const(advn_mx), @Const(D_U),
                                                  invh, g0, third, half, a1, a2, a3, gamma, inv_dt, dt,
                                                  dr::Bool)
    iz, ix = @index(Global, NTuple)
    @inbounds begin
        Nx  = size(eta, 2)
        ixL = ix == 1      ? Nx : ix - 1       # periodic-x cell columns
        ixR = ix == Nx + 1 ? 1  : ix
        Uv       = U[iz + 1, ix]
        # τxx computed INLINE (cells ixR and ixL) — same formula as dyrel/stress.jl
        UxR   = (U[iz + 1, ixR + 1] - U[iz + 1, ixR]) * invh
        WzR   = (W[iz + 1, ixR + 1] - W[iz, ixR + 1]) * invh
        txx_R = eta[iz, ixR] * (UxR - (UxR + WzR) * third)
        UxL   = (U[iz + 1, ixL + 1] - U[iz + 1, ixL]) * invh
        WzL   = (W[iz + 1, ixL + 1] - W[iz, ixL + 1]) * invh
        txx_L = eta[iz, ixL] * (UxL - (UxL + WzL) * third)
        d_txx_dx = (txx_R - txx_L) * invh
        # τxz INLINE at corners (iz+1, ix) and (iz, ix)
        txz_t = etaco[iz + 1, ix] * ((U[iz + 2, ix] - U[iz + 1, ix]) * invh +
                                     (W[iz + 1, ix + 1] - W[iz + 1, ix]) * invh) * half
        txz_b = etaco[iz,     ix] * ((U[iz + 1, ix] - U[iz,     ix]) * invh +
                                     (W[iz, ix + 1]     - W[iz,     ix]) * invh) * half
        d_txz_dz = (txz_t - txz_b) * invh
        dP_dx    = (P[iz + 1, ix + 1] - P[iz + 1, ix]) * invh
        inertia  = (a1 + gamma) * rhou[iz, ix] * Uv * inv_dt
        ddx_rho  = (rho[iz, ixR] - rho[iz, ixL]) * invh
        sailor   = ddx_rho * g0 * dt * Uv
        lagged   = (a2 * rhoUo[iz, ix] + a3 * rhoUoo[iz, ix]) * inv_dt
        adv      = advn_mx[iz, ix]
        # forcing/diagonal terms carry opposite sign to the +∇·τ−∂P core
        # (R = −(KV·U + GG·P − rr); matches fluidmech!)
        r = d_txx_dx + d_txz_dz - dP_dx - inertia - sailor + lagged - adv
        if dr
            dPnum_dx = (P_num[iz, ixR] - P_num[iz, ixL]) * invh
            R_U[iz, ix] = (r - dPnum_dx) / D_U[iz, ix]
        else
            R_U[iz, ix] = r
        end
    end
end

# ---------------------------------------------------------------------------
# Continuity residual: compressible mass-flux divergence ∇·(ρv) with the MFS
# source. `comp` is the artificial-compressibility relaxation (→0 as dt→∞).
# ---------------------------------------------------------------------------

@kernel function compute_dyrel_residual_P_kernel!(R_P, P_num, @Const(W), @Const(U),
                                                  @Const(P), @Const(P0),
                                                  @Const(MFS), @Const(rho),
                                                  @Const(rhow), @Const(rhou),
                                                  @Const(η_b), @Const(γ_eff), invh, inv_dt)
    iz, ix = @index(Global, NTuple)
    @inbounds begin
        # ∇·(ρv): mass-flux through the four faces of cell (iz,ix)
        fU_R = rhou[iz, ix + 1] * U[iz + 1, ix + 1]
        fU_L = rhou[iz, ix]     * U[iz + 1, ix]
        fW_B = rhow[iz + 1, ix] * W[iz + 1, ix + 1]
        fW_T = rhow[iz, ix]     * W[iz, ix + 1]
        div_ρv = (fU_R - fU_L) * invh + (fW_B - fW_T) * invh
        comp   = (P[iz + 1, ix + 1] - P0[iz + 1, ix + 1]) / η_b[iz, ix] * inv_dt
        # Scale the whole constraint by 1/ρ so the Schur penalty γ_eff·R_P is
        # ~γ_eff·∇v (matching the Gershgorin λmax, which carries γ_eff with no
        # ρ factor). Without this the penalty feedback is ρ× too strong and the
        # explicit DR step (dτ from λmax) diverges. Constraint ∇·(ρv)=MFS is
        # unchanged (1/ρ > 0 just rescales the pressure multiplier).
        rp = (MFS[iz, ix] - div_ρv - comp) / rho[iz, ix]
        R_P[iz, ix]   = rp
        # Fused Schur penalty P_num = γ_eff·R_P (was a separate broadcast each DR
        # iter) — saves one kernel launch per inner iteration on GPU. P_num is
        # only read by the dr=true momentum residual, which always runs after.
        P_num[iz, ix] = γ_eff[iz, ix] * rp
    end
end

# ---------------------------------------------------------------------------
# Host-side wrappers
# ---------------------------------------------------------------------------

"""
    dyrel_residual_V!(cache, phase, fluid, grid, par; dr, coeffs) -> nothing

Compute the W/U momentum residuals into `cache.R_W`, `cache.R_U`. `dr=false`
gives the un-preconditioned Powell-Hestenes form (convergence check); `dr=true`
gives the Jacobi-preconditioned DR form with the Schur penalty `cache.P_num`.
`coeffs = (a1, a2, a3, gamma, dt)` are the BD2 time-integration coefficients.
"""
function dyrel_residual_V!(cache::DyrelCache{T}, phase::PhaseState{T},
                           fluid::FluidState{T}, grid::Grid{T}, par::Parameters{T};
                           dr::Bool, a1::Real, a2::Real, a3::Real,
                           gamma::Real, dt::Real) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(cache.R_W)
    invh   = T(inv(grid.h));  g0 = T(par.g0)
    inv_dt = isfinite(dt) ? T(inv(dt)) : zero(T)
    dtT    = T(dt);  third = T(1//3);  half = T(1//2)
    # Stress is recomputed inline in the residual kernels (no phase.txx/tzz/txz
    # round-trip, no separate stress kernel) — they take W, U, eta, etaco.
    compute_dyrel_residual_W_kernel!(backend)(
        cache.R_W, fluid.W, fluid.U, fluid.eta, fluid.etaco, fluid.P, cache.P_num, fluid.Drho,
        fluid.rho, fluid.rhow, fluid.rhoWo, fluid.rhoWoo, cache.advn_mz, cache.D_W,
        invh, g0, third, half, T(a1), T(a2), T(a3), inv_dt, dtT, dr; ndrange = size(cache.R_W))
    compute_dyrel_residual_U_kernel!(backend)(
        cache.R_U, fluid.U, fluid.W, fluid.eta, fluid.etaco, fluid.P, cache.P_num,
        fluid.rho, fluid.rhou, fluid.rhoUo, fluid.rhoUoo, cache.advn_mx, cache.D_U,
        invh, g0, third, half, T(a1), T(a2), T(a3), T(gamma), inv_dt, dtT, dr; ndrange = size(cache.R_U))
    # No synchronize: KA keeps same-backend launches ordered; sync happens at
    # the next host read (the reductions / convergence checks in solve.jl).
    return nothing
end

"""
    dyrel_residual_P!(cache, fluid, grid, par, P0, dt) -> nothing

Compute the compressible continuity residual `cache.R_P = MFS − ∇·(ρv) −
(P−P0)/(η_b·dt)`. Pass `dt = Inf` to drop the artificial-compressibility term
(steady-state limit). `P0` is the previous-step pressure.
"""
function dyrel_residual_P!(cache::DyrelCache{T}, fluid::FluidState{T},
                           grid::Grid{T}, par::Parameters{T},
                           P0::AbstractMatrix{T}, dt::Real) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(cache.R_P)
    invh   = T(inv(grid.h))
    inv_dt = isfinite(dt) ? T(inv(dt)) : zero(T)
    compute_dyrel_residual_P_kernel!(backend)(
        cache.R_P, cache.P_num, fluid.W, fluid.U, fluid.P, P0, fluid.MFS, fluid.rho,
        fluid.rhow, fluid.rhou, cache.η_b, cache.γ_eff, invh, inv_dt; ndrange = size(cache.R_P))
    # No synchronize: KA keeps same-backend launches ordered; sync happens at
    # the next host read (the reductions / convergence checks in solve.jl).
    return nothing
end
