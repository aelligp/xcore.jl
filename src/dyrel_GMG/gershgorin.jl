using KernelAbstractions

# Gershgorin auto-tune for xcore-DYREL. Direct port of JustRelax's
# `_Gershgorin_Stokes2D_SchurComplement!` from src/DYREL/Gershgorin.jl,
# adapted for:
#  - xcore's (iz, ix) indexing (vs JustRelax's (i, j))
#  - W = z-velocity (JustRelax's Vy), U = x-velocity (JustRelax's Vx)
#  - pure-viscous rheology (no shear modulus G·dt term)
#  - cell-centred η at xcore's `(Nz, Nx)`; corner-staggered `etaco` at
#    `(Nz+1, Nx+1)`. JustRelax's `ηv` (corner) = xcore's `etaco`.
#  - γ_eff is cell-centred, sized `(Nz, Nx)`.
#
# Output: cache.D_W, cache.λmax_W (size (Nz+1, Nx))
#         cache.D_U, cache.λmax_U (size (Nz, Nx+1))
# These are the Jacobi-diagonal preconditioners and per-face λ_max estimates
# used by the inner DR loop and by `update_dτ_α_β!` below.
#
# See `DESIGN.md` §5.1 for the formula derivation. Note: this kernel computes
# the Gershgorin row-sum bound on the velocity-block (KV) Jacobian after
# Schur-complement reduction (i.e. with γ_eff penalty injected).

# x-face stencils (U direction) — iz=1..Nz, ix=1..Nx+1
# `iner_coef` = (a1 + γ)/dt scales the BD2 inertial diagonal a1·ρu/dt (the
# (a1+γ) factor matches fluidmech.jl's U-momentum inertial term).
@kernel function compute_dyrel_gershgorin_U_kernel!(D_U, λmax_U, @Const(η), @Const(etaco),
                                              @Const(γ_eff), @Const(rhou), invh2, invh,
                                              cn, cc, cs, iner_coef)
    iz, ix = @index(Global, NTuple)
    @inbounds begin
        # The U-face at (iz, ix) sits between cell columns (ix-1, iz) and (ix, iz)
        # (with periodic wrap at ix=1 and ix=Nx+1). Boundary x-faces use the
        # repeat-boundary cell (we trust the caller to fill ghost columns of
        # u_h or to mask the boundary entries downstream).
        Nz, Nx = size(η)
        # Cell-centred η on the left/right of this U-face (periodic in x)
        ixL = ix == 1     ? Nx : ix - 1
        ixR = ix == Nx+1  ? 1  : ix
        η_W = η[iz, ixL]    # left cell
        η_E = η[iz, ixR]    # right cell
        # Penalty γ on left/right cells (cell-centred)
        γ_W = γ_eff[iz, ixL]
        γ_E = γ_eff[iz, ixR]
        # Corner-staggered η at the four corners surrounding this U-face.
        # etaco is (Nz+1, Nx+1): rows = z-faces, cols = x-faces. The 4 corners
        # of cell (iz, ix-1/2) at (iz, ix) U-face are (iz, ix), (iz+1, ix).
        η_N = etaco[iz + 1, ix]    # top-corner viscosity above the face
        η_S = etaco[iz,     ix]    # bottom-corner viscosity below the face

        # Gershgorin row sums with the 2× viscous margin (see file header):
        # cn/cc/cs are double the true-diagonal values, γ_eff keeps coefficient 1.
        # Cxx = self-coupling on U-velocity through τxx and τxy
        # Cxy = cross-coupling onto neighbouring W-velocities via off-diagonal stress
        Cxx = cs * (η_N + η_S) * invh2 +
              (γ_E + cn * η_E) * invh2 +
              (γ_W + cn * η_W) * invh2 +
              (cs * (η_N + η_S) * invh + (γ_E + γ_W + cn * (η_E + η_W)) * invh) * invh
        Cxy = ((γ_E - cc * η_E + cs * η_N) + (γ_E - cc * η_E + cs * η_S)) * invh2 +
              ((γ_W - cc * η_W + cs * η_N) + (γ_W - cc * η_W + cs * η_S)) * invh2
        # BD2 inertial diagonal — positive, regularises the operator (raises D,
        # lowers λmax → larger dτ). Drunken-sailor is left out of the bound
        # (small ± stabilization that could otherwise make D non-positive).
        δ = iner_coef * rhou[iz, ix]
        # Diagonal entry (Jacobi preconditioner) + inertia
        D = (cs * (η_N + η_S) * invh + (γ_E + γ_W + cn * (η_E + η_W)) * invh) * invh + δ
        D_U[iz, ix]    = D
        λmax_U[iz, ix] = (Cxx + Cxy + δ) / D
    end
end

# z-face stencils (W direction) — iz=1..Nz+1, ix=1..Nx
# `iner_coef` = a1/dt scales the BD2 inertial diagonal a1·ρw/dt.
@kernel function compute_dyrel_gershgorin_W_kernel!(D_W, λmax_W, @Const(η), @Const(etaco),
                                              @Const(γ_eff), @Const(rhow), invh2, invh,
                                              cn, cc, cs, iner_coef)
    iz, ix = @index(Global, NTuple)
    @inbounds begin
        Nz, Nx = size(η)
        # The W-face at (iz, ix) sits between cell rows (iz-1, ix) and (iz, ix).
        # Boundary z-faces (iz=1, iz=Nz+1) hit the repeat-boundary cell.
        izN = iz == 1      ? 1  : iz - 1   # cell above (top boundary repeats)
        izS = iz == Nz+1   ? Nz : iz       # cell below (bottom boundary repeats)
        η_N = η[izN, ix]   # cell above
        η_S = η[izS, ix]
        γ_N = γ_eff[izN, ix]
        γ_S = γ_eff[izS, ix]
        # Corner-staggered viscosities surrounding this W-face. etaco at
        # (iz, ix) and (iz, ix+1) are the two corners flanking it horizontally.
        η_W = etaco[iz, ix]        # west-corner
        η_E = etaco[iz, ix + 1]    # east-corner

        # 2× viscous margin (see file header); γ_eff keeps coefficient 1.
        Cyy = cs * (η_E + η_W) * invh2 +
              (γ_N + cn * η_N) * invh2 +
              (γ_S + cn * η_S) * invh2 +
              ((γ_N + γ_S + cn * (η_N + η_S)) * invh + cs * (η_E + η_W) * invh) * invh
        Cyx = ((γ_N + cs * η_E - cc * η_N) + (γ_N - cc * η_N + cs * η_W)) * invh2 +
              ((γ_S + cs * η_E - cc * η_S) + (γ_S - cc * η_S + cs * η_W)) * invh2
        δ = iner_coef * rhow[iz, ix]       # BD2 inertial diagonal
        D = ((γ_N + γ_S + cn * (η_N + η_S)) * invh + cs * (η_E + η_W) * invh) * invh + δ
        D_W[iz, ix]    = D
        λmax_W[iz, ix] = (Cyy + Cyx + δ) / D
    end
end

"""
    gershgorin!(cache, fluid, h; a1, gamma, dt) -> nothing

Recompute the Gershgorin row-sum eigenvalue bound and Jacobi preconditioner
diagonals for both velocity directions, including the BD2 inertial diagonal
(`a1·ρw/dt` for W, `(a1+γ)·ρu/dt` for U). Called once per `fluidmech_dyrel!`
invocation and every `par.n_tune_PT` DR iterations. Pass `dt = Inf` for the
steady-state limit (inertia drops out).
"""
function gershgorin!(cache::DyrelCache{T}, fluid::FluidState{T}, h::Real;
                     a1::Real, gamma::Real, dt::Real) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(fluid.eta)
    invh    = T(inv(h))
    invh2   = invh * invh
    # # 2× the true-diagonal viscous coefficients (true: 2/3, 1/3, 1/2 — used by
    # # src/dyrel). The over-estimate is a deliberate smoother stability margin
    # # the V-cycle requires at N≥256; see the file header before "fixing" this.
    # cn      = T(4//3)
    # cc      = T(2//3)
    # cs      = T(1)
    cn      = T(2//3)
    cc      = T(1//3)
    cs      = T(1//2)
    inv_dt  = isfinite(dt) ? T(inv(dt)) : zero(T)
    iner_U  = (T(a1) + T(gamma)) * inv_dt
    iner_W  = T(a1) * inv_dt

    compute_dyrel_gershgorin_U_kernel!(backend)(cache.D_U, cache.λmax_U,
                                                    fluid.eta, fluid.etaco, cache.γ_eff,
                                                    fluid.rhou, invh2, invh, cn, cc, cs, iner_U;
                                                    ndrange = size(cache.D_U))
    compute_dyrel_gershgorin_W_kernel!(backend)(cache.D_W, cache.λmax_W,
                                                    fluid.eta, fluid.etaco, cache.γ_eff,
                                                    fluid.rhow, invh2, invh, cn, cc, cs, iner_W;
                                                    ndrange = size(cache.D_W))
    # No synchronize: KA keeps same-backend launches ordered; sync happens at
    # the next host read (the reductions / convergence checks).
    return nothing
end

# ---------------------------------------------------------------------------
# Schur-complement diagonal for the Powell-Hestenes pressure update.
#
# The pressure P couples to cell-i continuity through the momentum response:
# a unit ∂P drives the four face velocities by ~1/D_face (Jacobi/Gershgorin
# diagonal), which change ∇·(ρv)/ρ. The resulting self-coupling (Schur diagonal)
# is  s_P[i] = (1/(h²ρ_i)) Σ_faces ρ_face/D_face.  The locally-optimal
# (Newton/Uzawa) multiplier update is then  P_i += ω · R_P_i / s_P[i]  — which
# adapts to the inertia-laden diagonal and the grid spacing automatically,
# instead of the hand-tuned constant γ_eff. D_W/D_U already include the BD2
# inertia term, so s_P captures the Re>1 regime that forced γfact≈1e3.
# ---------------------------------------------------------------------------

@kernel function compute_schur_diag_kernel!(s_P, @Const(D_W), @Const(D_U),
                                            @Const(rhow), @Const(rhou), @Const(rho), invh2)
    iz, ix = @index(Global, NTuple)
    @inbounds begin
        sfac = rhou[iz, ix]     / D_U[iz, ix]     + rhou[iz, ix + 1] / D_U[iz, ix + 1] +
               rhow[iz, ix]     / D_W[iz, ix]     + rhow[iz + 1, ix] / D_W[iz + 1, ix]
        s_P[iz, ix] = sfac * invh2 / rho[iz, ix]
    end
end

"""
    compute_schur_diag!(cache, fluid, h) -> nothing

Fill `cache.s_P` (cell-centred Schur-complement diagonal) from the current
Gershgorin momentum diagonals `D_W`/`D_U` and face densities. Call after
`gershgorin!` (setup + each retune), since it depends on `D_W`/`D_U`.
"""
function compute_schur_diag!(cache::DyrelCache{T}, fluid::FluidState{T},
                             h::Real) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(cache.s_P)
    invh2 = T(inv(h)^2)
    compute_schur_diag_kernel!(backend)(cache.s_P, cache.D_W, cache.D_U,
                                                  fluid.rhow, fluid.rhou, fluid.rho, invh2;
                                                  ndrange = size(cache.s_P))
    # No synchronize: KA keeps same-backend launches ordered; sync happens at
    # the next host read (the reductions / convergence checks).
    return nothing
end

# ---------------------------------------------------------------------------
# dτ / α / β update — combines per-face λ_max (from Gershgorin) with global
# damping c (from Rayleigh quotient). Mirrors JustRelax's
# `_update_dτV_α_β!` / `update_α_β!`.
# ---------------------------------------------------------------------------

@kernel function compute_dyrel_update_dτ_α_β_kernel!(dτ, α, β, @Const(λmax), c, CFL_PT)
    iz, ix = @index(Global, NTuple)
    @inbounds begin
        # Per-face pseudo-time step (DESIGN §5.3)
        dτ_ij = 2 * CFL_PT / sqrt(λmax[iz, ix])
        dτ[iz, ix] = dτ_ij
        # Damping / acceleration coefficients share the denominator (2 + c·dτ):
        # invert it once and multiply (two divisions → one inv + two muls).
        cdτ = c * dτ_ij
        idn = inv(2 + cdτ)
        β[iz, ix] = 2 * dτ_ij * idn
        α[iz, ix] = (2 - cdτ) * idn
    end
end

"""
    update_dτ_α_β!(cache, CFL_PT) -> nothing

Recompute per-face `dτ_W`, `dτ_U`, plus damping coefficients `α_*`, `β_*`
from the current per-face `λmax_*` and the global damping scalar `cache.c`.
Call after each `gershgorin!` (and after each Rayleigh-quotient update of
`cache.c`).
"""
function update_dτ_α_β!(cache::DyrelCache{T}, CFL_PT::Real) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(cache.D_W)
    cval    = cache.c[]
    cflv    = T(CFL_PT)
    compute_dyrel_update_dτ_α_β_kernel!(backend)(cache.dτ_W, cache.α_W, cache.β_W,
                                                     cache.λmax_W, cval, cflv;
                                                     ndrange = size(cache.D_W))
    compute_dyrel_update_dτ_α_β_kernel!(backend)(cache.dτ_U, cache.α_U, cache.β_U,
                                                     cache.λmax_U, cval, cflv;
                                                     ndrange = size(cache.D_U))
    # No synchronize: KA keeps same-backend launches ordered; sync happens at
    # the next host read (the reductions / convergence checks).
    return nothing
end
