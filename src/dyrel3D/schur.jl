using KernelAbstractions

# 3-D Schur-complement kernel set for xcore-DYREL — the dimension lift of the
# 2-D kernels in `src/dyrel/residuals.jl` / `src/dyrel_GMG/residuals.jl` plus
# the Schur diagonal (`gershgorin.jl::compute_schur_diag_kernel!`) and the
# Powell-Hestenes pressure (multiplier) update.
#
# Adapted from T. Duretz's `Stokes3D_SchurComplement_kernels_v1.jl`
# (@parallel_indices / ParallelStencil), with these deltas:
#   * KernelAbstractions `@kernel` + ndrange over the residual arrays (the only
#     manual guard is the W Dirichlet boundary rows), matching xcore.
#   * xcore stress convention τ = η(ε − ⅓(∇·v)I) — NOT the source's 2η·ε̇.
#   * The Schur penalty is NOT folded into the normal stress (the source's
#     `τxx = 2ηε̇xx + γ∇V`): xcore's constraint is the full compressible
#     continuity residual R_P = (MFS − ∇·(ρv) − comp)/ρ (1/ρ-scaled), so the
#     penalty enters as a separate gradient −∂(P_num), P_num = γ_eff·R_P,
#     exactly as in 2-D. At MFS = comp = 0, ρ = 1 the two forms coincide.
#   * Full Navier-Stokes terms (BD2 inertia, drunken-sailor, lagged momentum
#     flux, frozen advection) + the GMG hooks `fscale`/`bext_*`, mirroring
#     `src/dyrel_GMG/residuals.jl` term-for-term so a future 3-D GMG smoother
#     can reuse these kernels unchanged.
#
# NOT wired into the build yet (no include from xcore.jl): this is the 3-D
# operator core only. Still missing for a runnable 3-D solver: 3-D Gershgorin
# (D_W/D_U/D_V are *inputs* here), BC kernels, momentum advection, transfer
# operators, and 3-D state structs (the host wrappers below take NamedTuples
# with the same field names a future FluidState3D/DyrelCache3D would carry).
#
# ---------------------------------------------------------------------------
# Conventions: axes (iz, ix, iy) = (z, x, y); z = dim1 (gravity), x = dim2 and
# y = dim3 (both periodic). Velocities: W = z, U = x, V = y. Grid spacing h
# isotropic (as in 2-D). Field shapes (ghosted state / compact cache):
#
#   field   | state (ghosted)        | cache (compact)     | location
#   --------|------------------------|---------------------|--------------------
#   W       | (Nz+1, Nx+2, Ny+2)     | R_W (Nz+1, Nx, Ny)  | z-faces
#   U       | (Nz+2, Nx+1, Ny+2)     | R_U (Nz, Nx+1, Ny)  | x-faces
#   V       | (Nz+2, Nx+2, Ny+1)     | R_V (Nz, Nx, Ny+1)  | y-faces
#   P       | (Nz+2, Nx+2, Ny+2)     | R_P (Nz, Nx, Ny)    | cells
#   eta,rho | (Nz, Nx, Ny)           |                     | cells
#   etaxz   | (Nz+1, Nx+1, Ny)       |                     | z-x edges (τxz)
#   etayz   | (Nz+1, Nx,   Ny+1)     |                     | z-y edges (τyz)
#   etaxy   | (Nz,   Nx+1, Ny+1)     |                     | x-y edges (τxy)
#   rhow    | (Nz+1, Nx, Ny)         |                     | z-faces
#   rhou    | (Nz, Nx+1, Ny)         |                     | x-faces
#   rhov    | (Nz, Nx, Ny+1)         |                     | y-faces
#   Drho    | (Nz+1, Nx, Ny)         |                     | z-faces
#   advn_mz | (Nz-1, Nx, Ny)         | interior z-faces    |
#   advn_mx | (Nz, Nx+1, Ny)         | all x-faces         |
#   advn_my | (Nz, Nx, Ny+1)         | all y-faces         |
#
# Index mapping (cache ↔ state): R_W[iz_f,ix,iy] ↔ W[iz_f,ix+1,iy+1];
# R_U[iz,ix,iy] ↔ U[iz+1,ix,iy+1]; R_V[iz,ix,iy] ↔ V[iz+1,ix+1,iy];
# R_P[iz,ix,iy] ↔ P[iz+1,ix+1,iy+1]. In 2-D, `etaco` is the z-x edge
# viscosity: etaxz is its direct generalisation (etayz/etaxy are new).

# ---------------------------------------------------------------------------
# W momentum residual (z-velocity). Dirichlet rows iz_f = 1, Nz+1 zeroed.
# ---------------------------------------------------------------------------

@kernel function compute_dyrel3d_residual_W_kernel!(R_W, @Const(W), @Const(U), @Const(V),
                                                    @Const(eta), @Const(etaxz), @Const(etayz),
                                                    @Const(P), @Const(P_num), @Const(Drho),
                                                    @Const(rho), @Const(rhow),
                                                    @Const(rhoWo), @Const(rhoWoo),
                                                    @Const(advn_mz), @Const(D_W), @Const(bext_W),
                                                    invh, g0, third, half, a1, a2, a3,
                                                    inv_dt, dt, fscale, dr::Bool)
    iz_f, ix, iy = @index(Global, NTuple)
    @inbounds begin
        Nzf = size(R_W, 1)                       # = Nz + 1
        if iz_f == 1 || iz_f == Nzf
            R_W[iz_f, ix, iy] = zero(eltype(R_W))  # Dirichlet boundary face
        else
            Wv = W[iz_f, ix + 1, iy + 1]
            # τzz INLINE at cells (iz_f, ix, iy) and (iz_f-1, ix, iy); the 3-D
            # divergence carries the extra Vy term.
            Ux_hi  = (U[iz_f + 1, ix + 1, iy + 1] - U[iz_f + 1, ix, iy + 1]) * invh
            Vy_hi  = (V[iz_f + 1, ix + 1, iy + 1] - V[iz_f + 1, ix + 1, iy]) * invh
            Wz_hi  = (W[iz_f + 1, ix + 1, iy + 1] - W[iz_f,     ix + 1, iy + 1]) * invh
            tzz_hi = eta[iz_f, ix, iy] * (Wz_hi - (Ux_hi + Vy_hi + Wz_hi) * third)
            Ux_lo  = (U[iz_f, ix + 1, iy + 1] - U[iz_f, ix, iy + 1]) * invh
            Vy_lo  = (V[iz_f, ix + 1, iy + 1] - V[iz_f, ix + 1, iy]) * invh
            Wz_lo  = (W[iz_f, ix + 1, iy + 1] - W[iz_f - 1, ix + 1, iy + 1]) * invh
            tzz_lo = eta[iz_f - 1, ix, iy] * (Wz_lo - (Ux_lo + Vy_lo + Wz_lo) * third)
            d_tzz_dz = (tzz_hi - tzz_lo) * invh
            # τxz at the two z-x edges flanking this face in x
            txz_l = etaxz[iz_f, ix, iy]     * ((U[iz_f + 1, ix, iy + 1]     - U[iz_f, ix, iy + 1])     * invh +
                                               (W[iz_f, ix + 1, iy + 1]     - W[iz_f, ix, iy + 1])     * invh) * half
            txz_r = etaxz[iz_f, ix + 1, iy] * ((U[iz_f + 1, ix + 1, iy + 1] - U[iz_f, ix + 1, iy + 1]) * invh +
                                               (W[iz_f, ix + 2, iy + 1]     - W[iz_f, ix + 1, iy + 1]) * invh) * half
            d_txz_dx = (txz_r - txz_l) * invh
            # τyz at the two z-y edges flanking this face in y
            tyz_b = etayz[iz_f, ix, iy]     * ((V[iz_f + 1, ix + 1, iy]     - V[iz_f, ix + 1, iy])     * invh +
                                               (W[iz_f, ix + 1, iy + 1]     - W[iz_f, ix + 1, iy])     * invh) * half
            tyz_f = etayz[iz_f, ix, iy + 1] * ((V[iz_f + 1, ix + 1, iy + 1] - V[iz_f, ix + 1, iy + 1]) * invh +
                                               (W[iz_f, ix + 1, iy + 2]     - W[iz_f, ix + 1, iy + 1]) * invh) * half
            d_tyz_dy = (tyz_f - tyz_b) * invh
            dP_dz    = (P[iz_f + 1, ix + 1, iy + 1] - P[iz_f, ix + 1, iy + 1]) * invh
            body     = Drho[iz_f, ix, iy] * g0
            inertia  = a1 * rhow[iz_f, ix, iy] * Wv * inv_dt
            ddz_rho  = (rho[iz_f, ix, iy] - rho[iz_f - 1, ix, iy]) * invh
            sailor   = ddz_rho * g0 * dt * Wv
            lagged   = (a2 * rhoWo[iz_f, ix, iy] + a3 * rhoWoo[iz_f, ix, iy]) * inv_dt
            adv      = advn_mz[iz_f - 1, ix, iy]
            # R = b − A·v: operator-diagonal inertia/sailor always present; pure
            # forcing (body + lagged − adv) scaled by fscale (0 on GMG coarse
            # error levels); bext_W injects the restricted residual (0 on fine).
            r = d_tzz_dz + d_txz_dx + d_tyz_dy - dP_dz - inertia - sailor +
                fscale * (body + lagged - adv) + bext_W[iz_f, ix, iy]
            if dr
                dPnum_dz = (P_num[iz_f, ix, iy] - P_num[iz_f - 1, ix, iy]) * invh
                R_W[iz_f, ix, iy] = (r - dPnum_dz) / D_W[iz_f, ix, iy]
            else
                R_W[iz_f, ix, iy] = r
            end
        end
    end
end

# ---------------------------------------------------------------------------
# U momentum residual (x-velocity). Periodic in x; inertia coeff (a1+γ); no
# buoyancy (gravity is z-only); drunken-sailor uses ∂_xρ.
# ---------------------------------------------------------------------------

@kernel function compute_dyrel3d_residual_U_kernel!(R_U, @Const(U), @Const(W), @Const(V),
                                                    @Const(eta), @Const(etaxz), @Const(etaxy),
                                                    @Const(P), @Const(P_num),
                                                    @Const(rho), @Const(rhou),
                                                    @Const(rhoUo), @Const(rhoUoo),
                                                    @Const(advn_mx), @Const(D_U), @Const(bext_U),
                                                    invh, g0, third, half, a1, a2, a3, gamma,
                                                    inv_dt, dt, fscale, dr::Bool)
    iz, ix, iy = @index(Global, NTuple)
    @inbounds begin
        Nx  = size(eta, 2)
        ixL = ix == 1      ? Nx : ix - 1       # periodic-x cell columns
        ixR = ix == Nx + 1 ? 1  : ix
        Uv  = U[iz + 1, ix, iy + 1]
        # τxx INLINE at cells (iz, ixR, iy) and (iz, ixL, iy)
        UxR   = (U[iz + 1, ixR + 1, iy + 1] - U[iz + 1, ixR, iy + 1]) * invh
        WzR   = (W[iz + 1, ixR + 1, iy + 1] - W[iz, ixR + 1, iy + 1]) * invh
        VyR   = (V[iz + 1, ixR + 1, iy + 1] - V[iz + 1, ixR + 1, iy]) * invh
        txx_R = eta[iz, ixR, iy] * (UxR - (UxR + WzR + VyR) * third)
        UxL   = (U[iz + 1, ixL + 1, iy + 1] - U[iz + 1, ixL, iy + 1]) * invh
        WzL   = (W[iz + 1, ixL + 1, iy + 1] - W[iz, ixL + 1, iy + 1]) * invh
        VyL   = (V[iz + 1, ixL + 1, iy + 1] - V[iz + 1, ixL + 1, iy]) * invh
        txx_L = eta[iz, ixL, iy] * (UxL - (UxL + WzL + VyL) * third)
        d_txx_dx = (txx_R - txx_L) * invh
        # τxz at the z-x edges above/below this face in z
        txz_t = etaxz[iz + 1, ix, iy] * ((U[iz + 2, ix, iy + 1] - U[iz + 1, ix, iy + 1]) * invh +
                                         (W[iz + 1, ix + 1, iy + 1] - W[iz + 1, ix, iy + 1]) * invh) * half
        txz_b = etaxz[iz,     ix, iy] * ((U[iz + 1, ix, iy + 1] - U[iz,     ix, iy + 1]) * invh +
                                         (W[iz, ix + 1, iy + 1]     - W[iz,     ix, iy + 1]) * invh) * half
        d_txz_dz = (txz_t - txz_b) * invh
        # τxy at the x-y edges fore/back of this face in y
        txy_f = etaxy[iz, ix, iy + 1] * ((U[iz + 1, ix, iy + 2] - U[iz + 1, ix, iy + 1]) * invh +
                                         (V[iz + 1, ix + 1, iy + 1] - V[iz + 1, ix, iy + 1]) * invh) * half
        txy_b = etaxy[iz, ix, iy]     * ((U[iz + 1, ix, iy + 1] - U[iz + 1, ix, iy]) * invh +
                                         (V[iz + 1, ix + 1, iy]     - V[iz + 1, ix, iy]) * invh) * half
        d_txy_dy = (txy_f - txy_b) * invh
        dP_dx    = (P[iz + 1, ix + 1, iy + 1] - P[iz + 1, ix, iy + 1]) * invh
        inertia  = (a1 + gamma) * rhou[iz, ix, iy] * Uv * inv_dt
        ddx_rho  = (rho[iz, ixR, iy] - rho[iz, ixL, iy]) * invh
        sailor   = ddx_rho * g0 * dt * Uv
        lagged   = (a2 * rhoUo[iz, ix, iy] + a3 * rhoUoo[iz, ix, iy]) * inv_dt
        adv      = advn_mx[iz, ix, iy]
        r = d_txx_dx + d_txz_dz + d_txy_dy - dP_dx - inertia - sailor +
            fscale * (lagged - adv) + bext_U[iz, ix, iy]
        if dr
            dPnum_dx = (P_num[iz, ixR, iy] - P_num[iz, ixL, iy]) * invh
            R_U[iz, ix, iy] = (r - dPnum_dx) / D_U[iz, ix, iy]
        else
            R_U[iz, ix, iy] = r
        end
    end
end

# ---------------------------------------------------------------------------
# V momentum residual (y-velocity) — exact mirror of the U kernel with x↔y
# (verified by the permutation-symmetry test). Periodic in y.
# ---------------------------------------------------------------------------

@kernel function compute_dyrel3d_residual_V_kernel!(R_V, @Const(V), @Const(W), @Const(U),
                                                    @Const(eta), @Const(etayz), @Const(etaxy),
                                                    @Const(P), @Const(P_num),
                                                    @Const(rho), @Const(rhov),
                                                    @Const(rhoVo), @Const(rhoVoo),
                                                    @Const(advn_my), @Const(D_V), @Const(bext_V),
                                                    invh, g0, third, half, a1, a2, a3, gamma,
                                                    inv_dt, dt, fscale, dr::Bool)
    iz, ix, iy = @index(Global, NTuple)
    @inbounds begin
        Ny  = size(eta, 3)
        iyL = iy == 1      ? Ny : iy - 1       # periodic-y cell layers
        iyR = iy == Ny + 1 ? 1  : iy
        Vv  = V[iz + 1, ix + 1, iy]
        # τyy INLINE at cells (iz, ix, iyR) and (iz, ix, iyL)
        VyR   = (V[iz + 1, ix + 1, iyR + 1] - V[iz + 1, ix + 1, iyR]) * invh
        WzR   = (W[iz + 1, ix + 1, iyR + 1] - W[iz, ix + 1, iyR + 1]) * invh
        UxR   = (U[iz + 1, ix + 1, iyR + 1] - U[iz + 1, ix, iyR + 1]) * invh
        tyy_R = eta[iz, ix, iyR] * (VyR - (VyR + WzR + UxR) * third)
        VyL   = (V[iz + 1, ix + 1, iyL + 1] - V[iz + 1, ix + 1, iyL]) * invh
        WzL   = (W[iz + 1, ix + 1, iyL + 1] - W[iz, ix + 1, iyL + 1]) * invh
        UxL   = (U[iz + 1, ix + 1, iyL + 1] - U[iz + 1, ix, iyL + 1]) * invh
        tyy_L = eta[iz, ix, iyL] * (VyL - (VyL + WzL + UxL) * third)
        d_tyy_dy = (tyy_R - tyy_L) * invh
        # τyz at the z-y edges above/below this face in z
        tyz_t = etayz[iz + 1, ix, iy] * ((V[iz + 2, ix + 1, iy] - V[iz + 1, ix + 1, iy]) * invh +
                                         (W[iz + 1, ix + 1, iy + 1] - W[iz + 1, ix + 1, iy]) * invh) * half
        tyz_b = etayz[iz,     ix, iy] * ((V[iz + 1, ix + 1, iy] - V[iz,     ix + 1, iy]) * invh +
                                         (W[iz, ix + 1, iy + 1]     - W[iz,     ix + 1, iy]) * invh) * half
        d_tyz_dz = (tyz_t - tyz_b) * invh
        # τxy at the x-y edges left/right of this face in x
        txy_r = etaxy[iz, ix + 1, iy] * ((V[iz + 1, ix + 2, iy] - V[iz + 1, ix + 1, iy]) * invh +
                                         (U[iz + 1, ix + 1, iy + 1] - U[iz + 1, ix + 1, iy]) * invh) * half
        txy_l = etaxy[iz, ix, iy]     * ((V[iz + 1, ix + 1, iy] - V[iz + 1, ix, iy]) * invh +
                                         (U[iz + 1, ix, iy + 1]     - U[iz + 1, ix, iy]) * invh) * half
        d_txy_dx = (txy_r - txy_l) * invh
        dP_dy    = (P[iz + 1, ix + 1, iy + 1] - P[iz + 1, ix + 1, iy]) * invh
        inertia  = (a1 + gamma) * rhov[iz, ix, iy] * Vv * inv_dt
        ddy_rho  = (rho[iz, ix, iyR] - rho[iz, ix, iyL]) * invh
        sailor   = ddy_rho * g0 * dt * Vv
        lagged   = (a2 * rhoVo[iz, ix, iy] + a3 * rhoVoo[iz, ix, iy]) * inv_dt
        adv      = advn_my[iz, ix, iy]
        r = d_tyy_dy + d_tyz_dz + d_txy_dx - dP_dy - inertia - sailor +
            fscale * (lagged - adv) + bext_V[iz, ix, iy]
        if dr
            dPnum_dy = (P_num[iz, ix, iyR] - P_num[iz, ix, iyL]) * invh
            R_V[iz, ix, iy] = (r - dPnum_dy) / D_V[iz, ix, iy]
        else
            R_V[iz, ix, iy] = r
        end
    end
end

# ---------------------------------------------------------------------------
# Continuity residual: ∇·(ρv) over the six cell faces, MFS source, artificial
# compressibility; 1/ρ-scaled; fused Schur penalty P_num = γ_eff·R_P (one
# kernel launch, as in 2-D).
# ---------------------------------------------------------------------------

@kernel function compute_dyrel3d_residual_P_kernel!(R_P, P_num, @Const(W), @Const(U), @Const(V),
                                                    @Const(P), @Const(P0),
                                                    @Const(MFS), @Const(rho),
                                                    @Const(rhow), @Const(rhou), @Const(rhov),
                                                    @Const(η_b), @Const(bext_P), @Const(γ_eff),
                                                    invh, inv_dt, fscale)
    iz, ix, iy = @index(Global, NTuple)
    @inbounds begin
        fU_R = rhou[iz, ix + 1, iy] * U[iz + 1, ix + 1, iy + 1]
        fU_L = rhou[iz, ix, iy]     * U[iz + 1, ix, iy + 1]
        fV_F = rhov[iz, ix, iy + 1] * V[iz + 1, ix + 1, iy + 1]
        fV_B = rhov[iz, ix, iy]     * V[iz + 1, ix + 1, iy]
        fW_B = rhow[iz + 1, ix, iy] * W[iz + 1, ix + 1, iy + 1]
        fW_T = rhow[iz, ix, iy]     * W[iz, ix + 1, iy + 1]
        div_ρv = (fU_R - fU_L) * invh + (fV_F - fV_B) * invh + (fW_B - fW_T) * invh
        comp   = (P[iz + 1, ix + 1, iy + 1] - P0[iz + 1, ix + 1, iy + 1]) / η_b[iz, ix, iy] * inv_dt
        rp = (fscale * MFS[iz, ix, iy] - div_ρv - comp) / rho[iz, ix, iy] + bext_P[iz, ix, iy]
        R_P[iz, ix, iy]   = rp
        P_num[iz, ix, iy] = γ_eff[iz, ix, iy] * rp
    end
end

# ---------------------------------------------------------------------------
# Schur-complement diagonal (six-face version of compute_schur_diag_kernel!):
#   s_P[i] = (1/(h²ρ_i)) Σ_{6 faces} ρ_f / D_f
# Feeds the operator-derived AL penalty γ_eff = γfact/s_P_phys exactly as 2-D.
# ---------------------------------------------------------------------------

@kernel function compute_schur_diag3d_kernel!(s_P, @Const(D_W), @Const(D_U), @Const(D_V),
                                              @Const(rhow), @Const(rhou), @Const(rhov),
                                              @Const(rho), invh2)
    iz, ix, iy = @index(Global, NTuple)
    @inbounds begin
        sfac = rhou[iz, ix, iy]     / D_U[iz, ix, iy]     + rhou[iz, ix + 1, iy] / D_U[iz, ix + 1, iy] +
               rhov[iz, ix, iy]     / D_V[iz, ix, iy]     + rhov[iz, ix, iy + 1] / D_V[iz, ix, iy + 1] +
               rhow[iz, ix, iy]     / D_W[iz, ix, iy]     + rhow[iz + 1, ix, iy] / D_W[iz + 1, ix, iy]
        s_P[iz, ix, iy] = sfac * invh2 / rho[iz, ix, iy]
    end
end

# ---------------------------------------------------------------------------
# Powell-Hestenes multiplier update (UpdatePressure_SchurComplement! analog):
#   P += ω·γ_eff·R_P  on interior cells (ghost ring untouched; refresh BCs
# after). In 2-D this is a host broadcast; kernel form saves the view-broadcast
# launch and is the 3-D-ready shape.
# ---------------------------------------------------------------------------

@kernel function update_pressure_schur3d_kernel!(P, @Const(R_P), @Const(γ_eff), ω)
    iz, ix, iy = @index(Global, NTuple)
    @inbounds P[iz + 1, ix + 1, iy + 1] += ω * γ_eff[iz, ix, iy] * R_P[iz, ix, iy]
end

# ---------------------------------------------------------------------------
# Host-side wrappers. `f` (fields) and `c` (cache) are NamedTuples (or any
# dot-accessible structs) carrying the arrays named in the header table — the
# same names a future FluidState3D/DyrelCache3D would have, so these wrappers
# survive the move to real structs unchanged.
# ---------------------------------------------------------------------------

"""
    dyrel3d_residual_V!(c, f, h, g0; dr, a1, a2, a3, gamma, dt, fscale = 1)

Compute the W/U/V momentum residuals into `c.R_W`, `c.R_U`, `c.R_V`.
`dr=false` gives the un-preconditioned Powell-Hestenes form; `dr=true` the
Jacobi-preconditioned DR form with the Schur penalty `c.P_num` (must be
filled by `dyrel3d_residual_P!` first). 3-D lift of `dyrel_residual_V!`.
"""
function dyrel3d_residual_V!(c, f, h::Real, g0::Real; dr::Bool,
                             a1::Real, a2::Real, a3::Real, gamma::Real, dt::Real,
                             fscale::Real = 1)
    T = eltype(c.R_W)
    backend = KernelAbstractions.get_backend(c.R_W)
    invh   = T(inv(h))
    inv_dt = isfinite(dt) ? T(inv(dt)) : zero(T)
    dtT    = T(dt);  fs = T(fscale);  third = T(1//3);  half = T(1//2)
    compute_dyrel3d_residual_W_kernel!(backend)(
        c.R_W, f.W, f.U, f.V, f.eta, f.etaxz, f.etayz, f.P, c.P_num, f.Drho,
        f.rho, f.rhow, f.rhoWo, f.rhoWoo, c.advn_mz, c.D_W, c.bext_W,
        invh, T(g0), third, half, T(a1), T(a2), T(a3), inv_dt, dtT, fs, dr;
        ndrange = size(c.R_W))
    compute_dyrel3d_residual_U_kernel!(backend)(
        c.R_U, f.U, f.W, f.V, f.eta, f.etaxz, f.etaxy, f.P, c.P_num,
        f.rho, f.rhou, f.rhoUo, f.rhoUoo, c.advn_mx, c.D_U, c.bext_U,
        invh, T(g0), third, half, T(a1), T(a2), T(a3), T(gamma), inv_dt, dtT, fs, dr;
        ndrange = size(c.R_U))
    compute_dyrel3d_residual_V_kernel!(backend)(
        c.R_V, f.V, f.W, f.U, f.eta, f.etayz, f.etaxy, f.P, c.P_num,
        f.rho, f.rhov, f.rhoVo, f.rhoVoo, c.advn_my, c.D_V, c.bext_V,
        invh, T(g0), third, half, T(a1), T(a2), T(a3), T(gamma), inv_dt, dtT, fs, dr;
        ndrange = size(c.R_V))
    # No synchronize: KA keeps same-backend launches ordered; sync happens at
    # the next host read (reductions / convergence checks).
    return nothing
end

"""
    dyrel3d_residual_P!(c, f, h, P0, dt; fscale = 1)

Compute the compressible continuity residual `c.R_P = (fscale·MFS − ∇·(ρv) −
comp)/ρ + c.bext_P` and the fused Schur penalty `c.P_num = γ_eff·R_P`.
3-D lift of `dyrel_residual_P!`.
"""
function dyrel3d_residual_P!(c, f, h::Real, P0, dt::Real; fscale::Real = 1)
    T = eltype(c.R_P)
    backend = KernelAbstractions.get_backend(c.R_P)
    invh   = T(inv(h))
    inv_dt = isfinite(dt) ? T(inv(dt)) : zero(T)
    compute_dyrel3d_residual_P_kernel!(backend)(
        c.R_P, c.P_num, f.W, f.U, f.V, f.P, P0, f.MFS, f.rho,
        f.rhow, f.rhou, f.rhov, c.η_b, c.bext_P, c.γ_eff, invh, inv_dt, T(fscale);
        ndrange = size(c.R_P))
    return nothing
end

"""
    compute_schur_diag3d!(c, f, h)

Fill `c.s_P` from the current momentum diagonals `c.D_W/D_U/D_V` and the face
densities. 3-D lift of `compute_schur_diag!`.
"""
function compute_schur_diag3d!(c, f, h::Real)
    T = eltype(c.s_P)
    backend = KernelAbstractions.get_backend(c.s_P)
    compute_schur_diag3d_kernel!(backend)(c.s_P, c.D_W, c.D_U, c.D_V,
                                          f.rhow, f.rhou, f.rhov, f.rho, T(inv(h)^2);
                                          ndrange = size(c.s_P))
    return nothing
end

"""
    dyrel3d_update_pressure!(c, f; ω = 1)

Powell-Hestenes multiplier update `P += ω·γ_eff·R_P` on interior cells.
Refresh the pressure ghost ring (BCs) and re-pin the gauge afterwards.
"""
function dyrel3d_update_pressure!(c, f; ω::Real = 1)
    T = eltype(c.R_P)
    backend = KernelAbstractions.get_backend(c.R_P)
    update_pressure_schur3d_kernel!(backend)(f.P, c.R_P, c.γ_eff, T(ω);
                                             ndrange = size(c.R_P))
    return nothing
end
