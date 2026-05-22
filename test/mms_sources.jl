using xcore
using Statistics

# Test-only helper. Mirrors src/mms.m: build the analytic manufactured solution
# from the same sinusoidal forms, hand-derive the Stokes residuals, and
# evaluate them on the staggered MMS coordinate grids that match
# usr/run_bnchm_VP.m's expectations.
#
# Hand-deriving (rather than calling Symbolics.jl) keeps the package light and
# the math inspectable. The decomposition into four basis functions
# C, S, CC, SS keeps the residual expressions short.

# basis functions: C = cos(kx) sin(kz), S = sin(kx) cos(kz),
#                  CC = cos(kx) cos(kz), SS = sin(kx) sin(kz)
@inline _basis_C(k, x, z)  = cos(k * x) * sin(k * z)
@inline _basis_S(k, x, z)  = sin(k * x) * cos(k * z)
@inline _basis_CC(k, x, z) = cos(k * x) * cos(k * z)
@inline _basis_SS(k, x, z) = sin(k * x) * sin(k * z)

"""
    mms_sources(::Type{T}, Nz, Nx, L) -> NamedTuple

Build the full MMS dataset for one resolution:

* `W_exact`, `U_exact`, `P_exact` — analytic fields evaluated on staggered
  grids (sized to match `FluidState`'s W, U, P).
* `src_W`, `src_U`, `src_P`       — Stokes residual source terms evaluated on
  the same staggered grids; these drive `fluidmech!`'s `bnchm_data` RHS.
* `eta`, `etaco`, `rho`, `rhow`, `rhou`, `Drho`, `MFS` — material/source fields
  evaluated analytically on the appropriate grids (no constitutive blend in
  MMS — they go straight into `FluidState`).

Coefficients match `src/mms.m` exactly. Coordinate convention follows the
MATLAB `x_mms = -h/2 : h : L + h/2` ghosted layout.
"""
function mms_sources(::Type{T}, Nz::Integer, Nx::Integer, L::Real) where {T<:AbstractFloat}
    L_t = T(L)
    h   = L_t / Nz                                        # square cells; Nz == Nx in MMS
    @assert Nz == Nx "mms_sources: MMS expects a square grid (Nz == Nx)"

    # coefficients (mirror src/mms.m lines 10–17)
    aW  = T(5e-5);  aU = T(4e-5);  aP  = T(-3e3)
    eta0 = T(1e3);  aeta = T(-9e2)
    rho0 = T(3e3);  arho = T(-5e1)
    asrc = T(-1e3); rhoref = T(3e3); g0 = T(10)
    k = T(4) * T(π) / L_t

    # ghosted coordinate arrays (matching mms.m lines 62–65)
    x_mms  = collect(T(-h/2) .+ (0:(Nx + 1)) .* h)        # length Nx+2
    z_mms  = collect(T(-h/2) .+ (0:(Nz + 1)) .* h)        # length Nz+2
    xu_mms = (x_mms[1:end-1] .+ x_mms[2:end]) ./ T(2)     # length Nx+1
    zw_mms = (z_mms[1:end-1] .+ z_mms[2:end]) ./ T(2)     # length Nz+1

    # --- analytic field closures (capture k, coefficients) -----------------
    W_(x, z) = aW * _basis_C(k, x, z)
    U_(x, z) = aU * _basis_S(k, x, z)
    P_(x, z) = aP * _basis_CC(k, x, z)
    η_(x, z) = eta0 + aeta * _basis_C(k, x, z)
    ρ_(x, z) = rho0 + arho * _basis_C(k, x, z)
    src_(x, z) = asrc * _basis_C(k, x, z)

    # --- strain rates and stresses (hand-derived from the analytic forms) --
    # exx = ∂xU - divV/3 = k CC (2aU - aW)/3
    # ezz = ∂zW - divV/3 = k CC (2aW - aU)/3
    # exz = (∂zU + ∂xW)/2 = -(aW + aU) k SS / 2
    a_exx = (T(2) * aU - aW) / T(3)
    a_ezz = (T(2) * aW - aU) / T(3)
    a_exz = -(aW + aU) / T(2)
    exx_(x, z) = k * _basis_CC(k, x, z) * a_exx
    ezz_(x, z) = k * _basis_CC(k, x, z) * a_ezz
    exz_(x, z) = k * _basis_SS(k, x, z) * a_exz

    # --- derivatives of η on which Stokes residual depends ----------------
    η_x_(x, z) = -aeta * k * _basis_SS(k, x, z)
    η_z_(x, z) =  aeta * k * _basis_CC(k, x, z)

    # --- second derivatives of the strain rates ---------------------------
    # exx_x = -k^2 S a_exx ;  exx_z = -k^2 C a_exx
    # ezz_z = -k^2 C a_ezz ;  ezz_x = -k^2 S a_ezz
    # exz_x = -k^2 C ((aW+aU)/2) ;  exz_z = -k^2 S ((aW+aU)/2)  with sign
    exx_x_(x, z) = -k^2 * _basis_S(k, x, z)  * a_exx
    exx_z_(x, z) = -k^2 * _basis_C(k, x, z)  * a_exx
    ezz_x_(x, z) = -k^2 * _basis_S(k, x, z)  * a_ezz
    ezz_z_(x, z) = -k^2 * _basis_C(k, x, z)  * a_ezz
    exz_x_(x, z) =  k^2 * _basis_C(k, x, z)  * a_exz
    exz_z_(x, z) =  k^2 * _basis_S(k, x, z)  * a_exz

    # --- stress divergence components: ∂α τβγ = η_α eβγ + η eβγ_α ---------
    dx_txx_(x, z) = η_x_(x, z) * exx_(x, z) + η_(x, z) * exx_x_(x, z)
    dz_txx_(x, z) = η_z_(x, z) * exx_(x, z) + η_(x, z) * exx_z_(x, z)
    dx_tzz_(x, z) = η_x_(x, z) * ezz_(x, z) + η_(x, z) * ezz_x_(x, z)
    dz_tzz_(x, z) = η_z_(x, z) * ezz_(x, z) + η_(x, z) * ezz_z_(x, z)
    dx_txz_(x, z) = η_x_(x, z) * exz_(x, z) + η_(x, z) * exz_x_(x, z)
    dz_txz_(x, z) = η_z_(x, z) * exz_(x, z) + η_(x, z) * exz_z_(x, z)

    # --- pressure gradients ----------------------------------------------
    Px_(x, z) = -aP * k * _basis_S(k, x, z)
    Pz_(x, z) = -aP * k * _basis_C(k, x, z)

    # --- Stokes residuals (= MMS source terms) ---------------------------
    # res_W = -(∂z τzz + ∂x τxz) + ∂z P - (ρ - ρref) g0
    res_W_(x, z) = -(dz_tzz_(x, z) + dx_txz_(x, z)) + Pz_(x, z) -
                   (ρ_(x, z) - rhoref) * g0
    # res_U = -(∂x τxx + ∂z τxz) + ∂x P
    res_U_(x, z) = -(dx_txx_(x, z) + dz_txz_(x, z)) + Px_(x, z)
    # res_P = ∂z(ρW) + ∂x(ρU) - src
    # ∂z(ρW) = (∂z ρ) W + ρ (∂z W);    ∂x(ρU) = (∂x ρ) U + ρ (∂x U)
    ρz_(x, z) =  arho * k * _basis_CC(k, x, z)
    ρx_(x, z) = -arho * k * _basis_SS(k, x, z)
    Wz_(x, z) =   aW * k * _basis_CC(k, x, z)
    Ux_(x, z) =   aU * k * _basis_CC(k, x, z)
    res_P_(x, z) = ρz_(x, z) * W_(x, z) + ρ_(x, z) * Wz_(x, z) +
                   ρx_(x, z) * U_(x, z) + ρ_(x, z) * Ux_(x, z) -
                   src_(x, z)

    # --- evaluate on staggered grids -------------------------------------
    eval_grid(f, zs, xs) = T[f(x, z) for z in zs, x in xs]

    W_exact = eval_grid(W_, zw_mms, x_mms)                 # (Nz+1, Nx+2)
    U_exact = eval_grid(U_, z_mms,  xu_mms)                # (Nz+2, Nx+1)
    P_exact = eval_grid(P_, z_mms,  x_mms)                 # (Nz+2, Nx+2)
    src_W   = eval_grid(res_W_, zw_mms, x_mms)
    src_U   = eval_grid(res_U_, z_mms,  xu_mms)
    src_P   = eval_grid(res_P_, z_mms,  x_mms)

    eta_full   = eval_grid(η_, z_mms,  x_mms)              # (Nz+2, Nx+2)
    eta_int    = eta_full[2:end-1, 2:end-1]                # (Nz,   Nx)
    etaco      = eval_grid(η_, zw_mms, xu_mms)             # (Nz+1, Nx+1)
    rho_full   = eval_grid(ρ_, z_mms,  x_mms)
    rho_int    = rho_full[2:end-1, 2:end-1]                # (Nz,   Nx)
    rhow_full  = eval_grid(ρ_, zw_mms, x_mms)              # (Nz+1, Nx+2)
    rhow       = rhow_full[:, 2:end-1]                     # (Nz+1, Nx)
    rhou_full  = eval_grid(ρ_, z_mms, xu_mms)              # (Nz+2, Nx+1)
    rhou       = rhou_full[2:end-1, :]                     # (Nz,   Nx+1)
    MFS_full   = eval_grid(src_, z_mms, x_mms)
    MFS        = MFS_full[2:end-1, 2:end-1]                # (Nz, Nx)

    Drho = rhow .- mean(rhow, dims = 2)                    # (Nz+1, Nx)

    return (; W_exact, U_exact, P_exact,
              src_W,   src_U,   src_P,
              eta = eta_int, etaco, rho = rho_int, rhow, rhou, Drho, MFS,
              L = L_t, h, k, rhoref, g0,
              x_mms, z_mms, xu_mms, zw_mms)
end
