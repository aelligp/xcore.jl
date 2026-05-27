using KernelAbstractions

# Conservative advection: div(v * f) on a cell-centred field with staggered
# face velocities. Two schemes implemented here (the ones the MMS test and
# production run_D1 need): :centr (centred, halo=1) and :weno5 (5th-order
# WENO, halo=3). Faithful translations of the corresponding branches in
# src/advect.m, restructured for KA's branch-free kernel pattern.

# -------------------- centred (halo = 1) -----------------------------------

@kernel function _advect_centr_kernel!(adv, @Const(f), @Const(u), @Const(w),
                                       invh, halo, half)
    iz, ix = @index(Global, NTuple)
    j = iz + halo
    i = ix + halo
    @inbounds begin
        fc  = f[j, i]
        fmx = f[j, i - 1];  fpx = f[j, i + 1]
        fmz = f[j - 1, i];  fpz = f[j + 1, i]

        ul = u[iz, ix];     ur = u[iz, ix + 1]
        wt = w[iz, ix];     wb = w[iz + 1, ix]

        qxr = ur * (fc + fpx) * half
        qxl = ul * (fc + fmx) * half
        qzb = wb * (fc + fpz) * half
        qzt = wt * (fc + fmz) * half

        adv[iz, ix] = (qxr - qxl + qzb - qzt) * invh
    end
end

# -------------------- WENO5 helper (Jiang & Shu, 1996) ---------------------

@inline function _weno5_poly(fmm::T, fm::T, fc::T, fp::T, fpp::T, eps_val::T) where {T<:AbstractFloat}
    sixth   = T(1//6)
    quarter = T(1//4)
    thirteen_twelfths = T(13//12)

    p1 = sixth * ( T(2)*fmm - T(7)*fm + T(11)*fc)
    p2 = sixth * (    -fm    + T(5)*fc + T(2)*fp)
    p3 = sixth * ( T(2)*fc   + T(5)*fp -    fpp )

    b1 = thirteen_twelfths*(fmm - T(2)*fm + fc )^2 + quarter*(fmm - T(4)*fm + T(3)*fc )^2
    b2 = thirteen_twelfths*(fm  - T(2)*fc + fp )^2 + quarter*(fm                - fp   )^2
    b3 = thirteen_twelfths*(fc  - T(2)*fp + fpp)^2 + quarter*(T(3)*fc - T(4)*fp +   fpp)^2

    w1 = T(1//10) / (b1 + eps_val)^2
    w2 = T(6//10) / (b2 + eps_val)^2
    w3 = T(3//10) / (b3 + eps_val)^2

    return (w1*p1 + w2*p2 + w3*p3) / (w1 + w2 + w3)
end

# -------------------- WENO5 kernel (halo = 3) ------------------------------

@kernel function _advect_weno5_kernel!(adv, @Const(f), @Const(u), @Const(w),
                                       invh, halo, half, eps_val)
    iz, ix = @index(Global, NTuple)
    j = iz + halo
    i = ix + halo
    @inbounds begin
        # x-direction stencil
        fxmmm = f[j, i - 3]; fxmm = f[j, i - 2]; fxm = f[j, i - 1]
        fc    = f[j, i    ]
        fxp   = f[j, i + 1]; fxpp = f[j, i + 2]; fxppp = f[j, i + 3]

        fxppos = _weno5_poly(fxmm,  fxm,  fc,  fxp,  fxpp, eps_val)
        fxpneg = _weno5_poly(fxppp, fxpp, fxp, fc,   fxm,  eps_val)
        fxmpos = _weno5_poly(fxmmm, fxmm, fxm, fc,   fxp,  eps_val)
        fxmneg = _weno5_poly(fxpp,  fxp,  fc,  fxm,  fxmm, eps_val)

        ul = u[iz, ix];     ur = u[iz, ix + 1]
        ulpos = (ul + abs(ul)) * half;  ulneg = (ul - abs(ul)) * half
        urpos = (ur + abs(ur)) * half;  urneg = (ur - abs(ur)) * half
        qxr = urpos * fxppos + urneg * fxpneg
        qxl = ulpos * fxmpos + ulneg * fxmneg

        # z-direction stencil
        fzmmm = f[j - 3, i]; fzmm = f[j - 2, i]; fzm = f[j - 1, i]
        fzp   = f[j + 1, i]; fzpp = f[j + 2, i]; fzppp = f[j + 3, i]

        fzppos = _weno5_poly(fzmm,  fzm,  fc,  fzp,  fzpp, eps_val)
        fzpneg = _weno5_poly(fzppp, fzpp, fzp, fc,   fzm,  eps_val)
        fzmpos = _weno5_poly(fzmmm, fzmm, fzm, fc,   fzp,  eps_val)
        fzmneg = _weno5_poly(fzpp,  fzp,  fc,  fzm,  fzmm, eps_val)

        wt = w[iz, ix];     wb = w[iz + 1, ix]
        wtpos = (wt + abs(wt)) * half;  wtneg = (wt - abs(wt)) * half
        wbpos = (wb + abs(wb)) * half;  wbneg = (wb - abs(wb)) * half
        qzb = wbpos * fzppos + wbneg * fzpneg
        qzt = wtpos * fzmpos + wtneg * fzmneg

        adv[iz, ix] = (qxr - qxl + qzb - qzt) * invh
    end
end

# -------------------- upwind 1st-order (halo = 1) --------------------------

@kernel function _advect_upwd1_kernel!(adv, @Const(f), @Const(u), @Const(w),
                                       invh, halo, half)
    iz, ix = @index(Global, NTuple)
    j = iz + halo
    i = ix + halo
    @inbounds begin
        fc  = f[j, i]
        fmx = f[j, i - 1];  fpx = f[j, i + 1]
        fmz = f[j - 1, i];  fpz = f[j + 1, i]

        ul = u[iz, ix];     ur = u[iz, ix + 1]
        wt = w[iz, ix];     wb = w[iz + 1, ix]

        ulpos = (ul + abs(ul)) * half;  ulneg = (ul - abs(ul)) * half
        urpos = (ur + abs(ur)) * half;  urneg = (ur - abs(ur)) * half
        wtpos = (wt + abs(wt)) * half;  wtneg = (wt - abs(wt)) * half
        wbpos = (wb + abs(wb)) * half;  wbneg = (wb - abs(wb)) * half

        # upwd1: fppos=fc, fpneg=fpx, fmpos=fmx, fmneg=fc (advect.m:82-85)
        qxr = urpos * fc  + urneg * fpx
        qxl = ulpos * fmx + ulneg * fc
        qzb = wbpos * fc  + wbneg * fpz
        qzt = wtpos * fmz + wtneg * fc

        adv[iz, ix] = (qxr - qxl + qzb - qzt) * invh
    end
end

# -------------------- QUICK 3rd-order upwind (halo = 2) --------------------

@kernel function _advect_quick_kernel!(adv, @Const(f), @Const(u), @Const(w),
                                       invh, halo, half, sixth)
    iz, ix = @index(Global, NTuple)
    j = iz + halo
    i = ix + halo
    @inbounds begin
        # x-direction 5-cell stencil [i-2, i-1, i, i+1, i+2]
        fxmm = f[j, i - 2]; fxm = f[j, i - 1]
        fc   = f[j, i    ]
        fxp  = f[j, i + 1]; fxpp = f[j, i + 2]

        # QUICK polynomials at faces i±1/2 (advect.m:93-96)
        fxppos = sixth * (2 * fxp  + 5 * fc  - fxm )    # face i+1/2 from left
        fxpneg = sixth * (2 * fc   + 5 * fxp - fxpp)    # face i+1/2 from right
        fxmpos = sixth * (2 * fc   + 5 * fxm - fxmm)    # face i-1/2 from left
        fxmneg = sixth * (2 * fxm  + 5 * fc  - fxp )    # face i-1/2 from right

        # z-direction 5-cell stencil [j-2, j-1, j, j+1, j+2]
        fzmm = f[j - 2, i]; fzm = f[j - 1, i]
        fzp  = f[j + 1, i]; fzpp = f[j + 2, i]

        fzppos = sixth * (2 * fzp  + 5 * fc  - fzm )
        fzpneg = sixth * (2 * fc   + 5 * fzp - fzpp)
        fzmpos = sixth * (2 * fc   + 5 * fzm - fzmm)
        fzmneg = sixth * (2 * fzm  + 5 * fc  - fzp )

        ul = u[iz, ix];     ur = u[iz, ix + 1]
        wt = w[iz, ix];     wb = w[iz + 1, ix]
        ulpos = (ul + abs(ul)) * half;  ulneg = (ul - abs(ul)) * half
        urpos = (ur + abs(ur)) * half;  urneg = (ur - abs(ur)) * half
        wtpos = (wt + abs(wt)) * half;  wtneg = (wt - abs(wt)) * half
        wbpos = (wb + abs(wb)) * half;  wbneg = (wb - abs(wb)) * half

        qxr = urpos * fxppos + urneg * fxpneg
        qxl = ulpos * fxmpos + ulneg * fxmneg
        qzb = wbpos * fzppos + wbneg * fzpneg
        qzt = wtpos * fzmpos + wtneg * fzmneg

        adv[iz, ix] = (qxr - qxl + qzb - qzt) * invh
    end
end

# -------------------- Fromm (halo = 2) -------------------------------------

@kernel function _advect_fromm_kernel!(adv, @Const(f), @Const(u), @Const(w),
                                       invh, halo, half, quarter)
    iz, ix = @index(Global, NTuple)
    j = iz + halo
    i = ix + halo
    @inbounds begin
        fxmm = f[j, i - 2]; fxm = f[j, i - 1]
        fc   = f[j, i    ]
        fxp  = f[j, i + 1]; fxpp = f[j, i + 2]

        # Fromm polynomials (advect.m:103-106)
        fxppos = fc  + quarter * (fxp  - fxm )
        fxpneg = fxp + quarter * (fc   - fxpp)
        fxmpos = fxm + quarter * (fc   - fxmm)
        fxmneg = fc  + quarter * (fxm  - fxp )

        fzmm = f[j - 2, i]; fzm = f[j - 1, i]
        fzp  = f[j + 1, i]; fzpp = f[j + 2, i]

        fzppos = fc  + quarter * (fzp  - fzm )
        fzpneg = fzp + quarter * (fc   - fzpp)
        fzmpos = fzm + quarter * (fc   - fzmm)
        fzmneg = fc  + quarter * (fzm  - fzp )

        ul = u[iz, ix];     ur = u[iz, ix + 1]
        wt = w[iz, ix];     wb = w[iz + 1, ix]
        ulpos = (ul + abs(ul)) * half;  ulneg = (ul - abs(ul)) * half
        urpos = (ur + abs(ur)) * half;  urneg = (ur - abs(ur)) * half
        wtpos = (wt + abs(wt)) * half;  wtneg = (wt - abs(wt)) * half
        wbpos = (wb + abs(wb)) * half;  wbneg = (wb - abs(wb)) * half

        qxr = urpos * fxppos + urneg * fxpneg
        qxl = ulpos * fxmpos + ulneg * fxmneg
        qzb = wbpos * fzppos + wbneg * fzpneg
        qzt = wtpos * fzmpos + wtneg * fzmneg

        adv[iz, ix] = (qxr - qxl + qzb - qzt) * invh
    end
end

# -------------------- WENO3 helper + kernel (halo = 2) ---------------------

@inline function _weno3_poly(fm::T, fc::T, fp::T, eps_val::T) where {T<:AbstractFloat}
    half = T(0.5)
    p1 = half * (fc + fp)
    p2 = half * (T(3) * fc - fm)

    b1 = (fp - fc)^2
    b2 = (fc - fm)^2

    # NOTE: weno3 uses fixed 1e-6 regulariser (advect.m:318), not machine eps
    w1 = T(1//3) / (b1 + eps_val)
    w2 = T(2//3) / (b2 + eps_val)

    return (w1 * p1 + w2 * p2) / (w1 + w2)
end

@kernel function _advect_weno3_kernel!(adv, @Const(f), @Const(u), @Const(w),
                                       invh, halo, half, eps_val)
    iz, ix = @index(Global, NTuple)
    j = iz + halo
    i = ix + halo
    @inbounds begin
        fxmm = f[j, i - 2]; fxm = f[j, i - 1]
        fc   = f[j, i    ]
        fxp  = f[j, i + 1]; fxpp = f[j, i + 2]

        # +flux (left-biased) and -flux (right-biased) reconstructions
        fxppos = _weno3_poly(fxm,  fc,  fxp,  eps_val)
        fxpneg = _weno3_poly(fxpp, fxp, fc,   eps_val)
        fxmpos = _weno3_poly(fxmm, fxm, fc,   eps_val)
        fxmneg = _weno3_poly(fxp,  fc,  fxm,  eps_val)

        fzmm = f[j - 2, i]; fzm = f[j - 1, i]
        fzp  = f[j + 1, i]; fzpp = f[j + 2, i]

        fzppos = _weno3_poly(fzm,  fc,  fzp,  eps_val)
        fzpneg = _weno3_poly(fzpp, fzp, fc,   eps_val)
        fzmpos = _weno3_poly(fzmm, fzm, fc,   eps_val)
        fzmneg = _weno3_poly(fzp,  fc,  fzm,  eps_val)

        ul = u[iz, ix];     ur = u[iz, ix + 1]
        wt = w[iz, ix];     wb = w[iz + 1, ix]
        ulpos = (ul + abs(ul)) * half;  ulneg = (ul - abs(ul)) * half
        urpos = (ur + abs(ur)) * half;  urneg = (ur - abs(ur)) * half
        wtpos = (wt + abs(wt)) * half;  wtneg = (wt - abs(wt)) * half
        wbpos = (wb + abs(wb)) * half;  wbneg = (wb - abs(wb)) * half

        qxr = urpos * fxppos + urneg * fxpneg
        qxl = ulpos * fxmpos + ulneg * fxmneg
        qzb = wbpos * fzppos + wbneg * fzpneg
        qzt = wtpos * fzmpos + wtneg * fzmneg

        adv[iz, ix] = (qxr - qxl + qzb - qzt) * invh
    end
end

# -------------------- TVD with superbee limiter (halo = 2) -----------------
# Velocity needs one extra face on each side. We pass halo'd u_h (Nz, Nx+3)
# and w_h (Nz+3, Nx) constructed by `advect_centered!` for this scheme.

@inline function _tvd_flux(fm::T, fc::T, fp::T, vm::T, vp::T) where {T<:AbstractFloat}
    # superbee limiter on the flux ratio (advect.m:393-400)
    num = vm * (fc - fm)
    den = vp * (fp - fc)
    # safe division to avoid 0/0; if den ≈ 0 then R is effectively 0 (no slope change)
    R = num / (den + copysign(eps(T), den))
    l = max(zero(T), max(min(one(T), T(2) * R), min(T(2), R)))
    return fc + T(0.5) * l * (fp - fc)
end

@kernel function _advect_tvdim_kernel!(adv, @Const(f), @Const(u_h), @Const(w_h),
                                       invh, halo, half)
    iz, ix = @index(Global, NTuple)
    j = iz + halo
    i = ix + halo
    @inbounds begin
        # cell stencil at (iz, ix)
        fxmm = f[j, i - 2]; fxm = f[j, i - 1]
        fc   = f[j, i    ]
        fxp  = f[j, i + 1]; fxpp = f[j, i + 2]
        fzmm = f[j - 2, i]; fzm = f[j - 1, i]
        fzp  = f[j + 1, i]; fzpp = f[j + 2, i]

        # u_h is sized (Nz, Nx+3); face indices are offset by 1: u_h[:, ix+1] = u[:, ix]
        ull = u_h[iz, ix    ]    # face i-3/2 (one face left of left face)
        ul  = u_h[iz, ix + 1]    # face i-1/2 (left face of cell)
        ur  = u_h[iz, ix + 2]    # face i+1/2 (right face of cell)
        urr = u_h[iz, ix + 3]    # face i+3/2 (one face right of right face)
        ullpos = (ull + abs(ull)) * half
        ulpos  = (ul  + abs(ul )) * half;  ulneg  = (ul  - abs(ul )) * half
        urpos  = (ur  + abs(ur )) * half;  urneg  = (ur  - abs(ur )) * half
        urrneg = (urr - abs(urr)) * half

        # w_h is sized (Nz+3, Nx)
        wtt = w_h[iz,     ix]
        wt  = w_h[iz + 1, ix]
        wb  = w_h[iz + 2, ix]
        wbb = w_h[iz + 3, ix]
        wttpos = (wtt + abs(wtt)) * half
        wtpos  = (wt  + abs(wt )) * half;  wtneg  = (wt  - abs(wt )) * half
        wbpos  = (wb  + abs(wb )) * half;  wbneg  = (wb  - abs(wb )) * half
        wbbneg = (wbb - abs(wbb)) * half

        # i+1/2 face: +flux uses (fxm, fc, fxp) with (ulpos, urpos); -flux uses (fxpp, fxp, fc) with (urrneg, urneg)
        fxppos = _tvd_flux(fxm,  fc,  fxp,  ulpos,  urpos)
        fxpneg = _tvd_flux(fxpp, fxp, fc,   urrneg, urneg)
        # i-1/2 face: +flux uses (fxmm, fxm, fc) with (ullpos, ulpos); -flux uses (fxp, fc, fxm) with (urneg, ulneg)
        fxmpos = _tvd_flux(fxmm, fxm, fc,   ullpos, ulpos)
        fxmneg = _tvd_flux(fxp,  fc,  fxm,  urneg,  ulneg)

        fzppos = _tvd_flux(fzm,  fc,  fzp,  wtpos,  wbpos)
        fzpneg = _tvd_flux(fzpp, fzp, fc,   wbbneg, wbneg)
        fzmpos = _tvd_flux(fzmm, fzm, fc,   wttpos, wtpos)
        fzmneg = _tvd_flux(fzp,  fc,  fzm,  wbneg,  wtneg)

        qxr = urpos * fxppos + urneg * fxpneg
        qxl = ulpos * fxmpos + ulneg * fxmneg
        qzb = wbpos * fzppos + wbneg * fzpneg
        qzt = wtpos * fzmpos + wtneg * fzmneg

        adv[iz, ix] = (qxr - qxl + qzb - qzt) * invh
    end
end

# -------------------- dispatch front-end -----------------------------------

"""
    scheme_halo(scheme::Symbol) -> Int

Halo width required for cell-centred field `f_halo` by each advection scheme.
"""
function scheme_halo(scheme::Symbol)
    scheme === :centr ? 1 :
    scheme === :upwd1 ? 1 :
    scheme === :quick ? 2 :
    scheme === :fromm ? 2 :
    scheme === :weno3 ? 2 :
    scheme === :weno5 ? 3 :
    scheme === :tvdim ? 2 :
    throw(ArgumentError("scheme_halo: unsupported scheme = :$scheme"))
end

"""
    advect!(adv, f_halo, u, w, h, scheme::Symbol) -> adv

Conservative advection `div(v f)` on a cell-centred field, with face-staggered
velocities. `scheme ∈ {:centr, :upwd1, :quick, :fromm, :weno3, :weno5, :tvdim}`
mirrors `src/advect.m`.

Sizing convention:
* `adv`     `(Nz, Nx)`              — interior output
* `f_halo`  `(Nz + 2g, Nx + 2g)`    — cell-centred scalar with halo `g`
* `u`       `(Nz, Nx + 1)`          — x-face velocity (or `(Nz, Nx + 3)` for `:tvdim`)
* `w`       `(Nz + 1, Nx)`          — z-face velocity (or `(Nz + 3, Nx)` for `:tvdim`)

Caller fills ghosts on `f_halo` (typically via `fill_ghosts!`) before each call.
For `:tvdim`, caller must also supply halo'd velocities — `advect_centered!`
handles this automatically.
"""
function advect!(adv::AbstractMatrix, f_halo::AbstractMatrix,
                 u::AbstractMatrix, w::AbstractMatrix, h::Real, scheme::Symbol)
    halo = scheme_halo(scheme)
    Nz = size(adv, 1);  Nx = size(adv, 2)
    @assert size(f_halo) == (Nz + 2halo, Nx + 2halo)  "advect!: f_halo size doesn't match adv + 2*halo"
    if scheme === :tvdim
        @assert size(u) == (Nz, Nx + 3)               "advect!: tvdim requires halo'd u of size (Nz, Nx+3)"
        @assert size(w) == (Nz + 3, Nx)               "advect!: tvdim requires halo'd w of size (Nz+3, Nx)"
    else
        @assert size(u) == (Nz, Nx + 1)               "advect!: u must be (Nz, Nx+1)"
        @assert size(w) == (Nz + 1, Nx)               "advect!: w must be (Nz+1, Nx)"
    end

    backend = KernelAbstractions.get_backend(adv)
    T = eltype(adv)
    invh = T(inv(h))
    half = T(0.5)

    if scheme === :centr
        _advect_centr_kernel!(backend, (16, 16))(adv, f_halo, u, w,
                                                 invh, halo, half;
                                                 ndrange = size(adv))
    elseif scheme === :upwd1
        _advect_upwd1_kernel!(backend, (16, 16))(adv, f_halo, u, w,
                                                 invh, halo, half;
                                                 ndrange = size(adv))
    elseif scheme === :quick
        _advect_quick_kernel!(backend, (16, 16))(adv, f_halo, u, w,
                                                 invh, halo, half, T(1//6);
                                                 ndrange = size(adv))
    elseif scheme === :fromm
        _advect_fromm_kernel!(backend, (16, 16))(adv, f_halo, u, w,
                                                 invh, halo, half, T(0.25);
                                                 ndrange = size(adv))
    elseif scheme === :weno3
        _advect_weno3_kernel!(backend, (16, 16))(adv, f_halo, u, w,
                                                 invh, halo, half, T(1e-6);
                                                 ndrange = size(adv))
    elseif scheme === :weno5
        _advect_weno5_kernel!(backend, (16, 16))(adv, f_halo, u, w,
                                                 invh, halo, half, eps(T);
                                                 ndrange = size(adv))
    elseif scheme === :tvdim
        _advect_tvdim_kernel!(backend, (16, 16))(adv, f_halo, u, w,
                                                 invh, halo, half;
                                                 ndrange = size(adv))
    else
        throw(ArgumentError("advect!: unsupported scheme = :$scheme"))
    end
    KernelAbstractions.synchronize(backend)
    return adv
end

# ============================================================================
# Per-face flux kernels — write face-centred fluxes using the same algebraic
# formula as the divergence kernels above, evaluated once per face. The
# discrete divergence in `_advect_*_kernel!` IS the divergence of these fluxes
# (every face value matches bitwise), so mass conservation holds exactly when
# the same f_halo and face velocities are used.
#
# Sizing:
#   qx_int  (Nz,   Nx+1)   — x-face fluxes at face ix ∈ 1..Nx+1
#   qz_int  (Nz+1, Nx  )   — z-face fluxes at face iz ∈ 1..Nz+1
#
# These are the interior face-flux buffers. `advect_with_flux!` pads them out
# to MATLAB's (Nz+2, Nx+1) / (Nz+1, Nx+2) layout per `src/advect.m:144-161`.
# ============================================================================

# -------------------- centred ----------------------------------------------

@kernel function _flux_centr_x_kernel!(qx_int, @Const(f), @Const(u), halo, half)
    iz, ix = @index(Global, NTuple)
    j = iz + halo
    @inbounds begin
        fL = f[j, ix + halo - 1]
        fR = f[j, ix + halo    ]
        qx_int[iz, ix] = u[iz, ix] * (fL + fR) * half
    end
end

@kernel function _flux_centr_z_kernel!(qz_int, @Const(f), @Const(w), halo, half)
    iz, ix = @index(Global, NTuple)
    i = ix + halo
    @inbounds begin
        fT = f[iz + halo - 1, i]
        fB = f[iz + halo,     i]
        qz_int[iz, ix] = w[iz, ix] * (fT + fB) * half
    end
end

# -------------------- upwind 1 --------------------------------------------

@kernel function _flux_upwd1_x_kernel!(qx_int, @Const(f), @Const(u), halo, half)
    iz, ix = @index(Global, NTuple)
    j = iz + halo
    @inbounds begin
        fL = f[j, ix + halo - 1]
        fR = f[j, ix + halo    ]
        uv = u[iz, ix]
        upos = (uv + abs(uv)) * half;  uneg = (uv - abs(uv)) * half
        qx_int[iz, ix] = upos * fL + uneg * fR
    end
end

@kernel function _flux_upwd1_z_kernel!(qz_int, @Const(f), @Const(w), halo, half)
    iz, ix = @index(Global, NTuple)
    i = ix + halo
    @inbounds begin
        fT = f[iz + halo - 1, i]
        fB = f[iz + halo,     i]
        wv = w[iz, ix]
        wpos = (wv + abs(wv)) * half;  wneg = (wv - abs(wv)) * half
        qz_int[iz, ix] = wpos * fT + wneg * fB
    end
end

# -------------------- QUICK -----------------------------------------------
# At face ix (between cells ix-1 and ix), in face-centred naming:
#   fmm = f[ix-2], fm = f[ix-1] (left cell), fc = f[ix] (right cell), fp = f[ix+1]
#   fppos = (2*fc + 5*fm - fmm)/6      (left-biased — for +flux)
#   fpneg = (2*fm + 5*fc - fp )/6      (right-biased — for -flux)

@kernel function _flux_quick_x_kernel!(qx_int, @Const(f), @Const(u), halo, half, sixth)
    iz, ix = @index(Global, NTuple)
    j = iz + halo
    @inbounds begin
        fmm = f[j, ix + halo - 2]
        fm  = f[j, ix + halo - 1]
        fc  = f[j, ix + halo    ]
        fp  = f[j, ix + halo + 1]
        fppos = sixth * (2 * fc + 5 * fm - fmm)
        fpneg = sixth * (2 * fm + 5 * fc - fp )
        uv = u[iz, ix]
        upos = (uv + abs(uv)) * half;  uneg = (uv - abs(uv)) * half
        qx_int[iz, ix] = upos * fppos + uneg * fpneg
    end
end

@kernel function _flux_quick_z_kernel!(qz_int, @Const(f), @Const(w), halo, half, sixth)
    iz, ix = @index(Global, NTuple)
    i = ix + halo
    @inbounds begin
        fmm = f[iz + halo - 2, i]
        fm  = f[iz + halo - 1, i]
        fc  = f[iz + halo,     i]
        fp  = f[iz + halo + 1, i]
        fppos = sixth * (2 * fc + 5 * fm - fmm)
        fpneg = sixth * (2 * fm + 5 * fc - fp )
        wv = w[iz, ix]
        wpos = (wv + abs(wv)) * half;  wneg = (wv - abs(wv)) * half
        qz_int[iz, ix] = wpos * fppos + wneg * fpneg
    end
end

# -------------------- Fromm -----------------------------------------------
# fppos = fm + (fc - fmm)/4    fpneg = fc + (fm - fp)/4

@kernel function _flux_fromm_x_kernel!(qx_int, @Const(f), @Const(u), halo, half, quarter)
    iz, ix = @index(Global, NTuple)
    j = iz + halo
    @inbounds begin
        fmm = f[j, ix + halo - 2]
        fm  = f[j, ix + halo - 1]
        fc  = f[j, ix + halo    ]
        fp  = f[j, ix + halo + 1]
        fppos = fm + quarter * (fc - fmm)
        fpneg = fc + quarter * (fm - fp )
        uv = u[iz, ix]
        upos = (uv + abs(uv)) * half;  uneg = (uv - abs(uv)) * half
        qx_int[iz, ix] = upos * fppos + uneg * fpneg
    end
end

@kernel function _flux_fromm_z_kernel!(qz_int, @Const(f), @Const(w), halo, half, quarter)
    iz, ix = @index(Global, NTuple)
    i = ix + halo
    @inbounds begin
        fmm = f[iz + halo - 2, i]
        fm  = f[iz + halo - 1, i]
        fc  = f[iz + halo,     i]
        fp  = f[iz + halo + 1, i]
        fppos = fm + quarter * (fc - fmm)
        fpneg = fc + quarter * (fm - fp )
        wv = w[iz, ix]
        wpos = (wv + abs(wv)) * half;  wneg = (wv - abs(wv)) * half
        qz_int[iz, ix] = wpos * fppos + wneg * fpneg
    end
end

# -------------------- WENO3 -----------------------------------------------
# fppos = weno3_poly(fmm, fm, fc)    fpneg = weno3_poly(fp, fc, fm)

@kernel function _flux_weno3_x_kernel!(qx_int, @Const(f), @Const(u), halo, half, eps_val)
    iz, ix = @index(Global, NTuple)
    j = iz + halo
    @inbounds begin
        fmm = f[j, ix + halo - 2]
        fm  = f[j, ix + halo - 1]
        fc  = f[j, ix + halo    ]
        fp  = f[j, ix + halo + 1]
        fppos = _weno3_poly(fmm, fm, fc, eps_val)
        fpneg = _weno3_poly(fp,  fc, fm, eps_val)
        uv = u[iz, ix]
        upos = (uv + abs(uv)) * half;  uneg = (uv - abs(uv)) * half
        qx_int[iz, ix] = upos * fppos + uneg * fpneg
    end
end

@kernel function _flux_weno3_z_kernel!(qz_int, @Const(f), @Const(w), halo, half, eps_val)
    iz, ix = @index(Global, NTuple)
    i = ix + halo
    @inbounds begin
        fmm = f[iz + halo - 2, i]
        fm  = f[iz + halo - 1, i]
        fc  = f[iz + halo,     i]
        fp  = f[iz + halo + 1, i]
        fppos = _weno3_poly(fmm, fm, fc, eps_val)
        fpneg = _weno3_poly(fp,  fc, fm, eps_val)
        wv = w[iz, ix]
        wpos = (wv + abs(wv)) * half;  wneg = (wv - abs(wv)) * half
        qz_int[iz, ix] = wpos * fppos + wneg * fpneg
    end
end

# -------------------- WENO5 -----------------------------------------------
# At face ix: cells used are (ix-3, ix-2, ix-1, ix, ix+1, ix+2)
#   fppos = weno5(f[ix-3], f[ix-2], f[ix-1], f[ix], f[ix+1])     (left-biased)
#   fpneg = weno5(f[ix+2], f[ix+1], f[ix], f[ix-1], f[ix-2])     (right-biased)

@kernel function _flux_weno5_x_kernel!(qx_int, @Const(f), @Const(u), halo, half, eps_val)
    iz, ix = @index(Global, NTuple)
    j = iz + halo
    @inbounds begin
        f0 = f[j, ix + halo - 3]
        f1 = f[j, ix + halo - 2]
        f2 = f[j, ix + halo - 1]
        f3 = f[j, ix + halo    ]
        f4 = f[j, ix + halo + 1]
        f5 = f[j, ix + halo + 2]
        fppos = _weno5_poly(f0, f1, f2, f3, f4, eps_val)
        fpneg = _weno5_poly(f5, f4, f3, f2, f1, eps_val)
        uv = u[iz, ix]
        upos = (uv + abs(uv)) * half;  uneg = (uv - abs(uv)) * half
        qx_int[iz, ix] = upos * fppos + uneg * fpneg
    end
end

@kernel function _flux_weno5_z_kernel!(qz_int, @Const(f), @Const(w), halo, half, eps_val)
    iz, ix = @index(Global, NTuple)
    i = ix + halo
    @inbounds begin
        f0 = f[iz + halo - 3, i]
        f1 = f[iz + halo - 2, i]
        f2 = f[iz + halo - 1, i]
        f3 = f[iz + halo,     i]
        f4 = f[iz + halo + 1, i]
        f5 = f[iz + halo + 2, i]
        fppos = _weno5_poly(f0, f1, f2, f3, f4, eps_val)
        fpneg = _weno5_poly(f5, f4, f3, f2, f1, eps_val)
        wv = w[iz, ix]
        wpos = (wv + abs(wv)) * half;  wneg = (wv - abs(wv)) * half
        qz_int[iz, ix] = wpos * fppos + wneg * fpneg
    end
end

# -------------------- TVD (superbee) --------------------------------------
# Needs velocity halo: u_h sized (Nz, Nx+3), w_h sized (Nz+3, Nx).
# u_h[iz, k+1] = u[iz, k] for k=1..Nx+1.
#
# At face ix (between cells ix-1 and ix) in face naming:
#   fppos = _tvd_flux(fmm=f[ix-2], fm=f[ix-1], fc=f[ix],    vm=upos[face ix-1], vp=upos[face ix])
#   fpneg = _tvd_flux(fmm=f[ix+1], fm=f[ix],   fc=f[ix-1],  vm=uneg[face ix+1], vp=uneg[face ix])
# In u_h indices: face ix-1 → u_h[iz, ix], face ix → u_h[iz, ix+1], face ix+1 → u_h[iz, ix+2].

@kernel function _flux_tvdim_x_kernel!(qx_int, @Const(f), @Const(u_h), halo, half)
    iz, ix = @index(Global, NTuple)
    j = iz + halo
    @inbounds begin
        fmm_p = f[j, ix + halo - 2]    # for fppos
        fm_p  = f[j, ix + halo - 1]
        fc_p  = f[j, ix + halo    ]
        # for fpneg: stencil starts at f[ix+1] going toward f[ix-1]
        fmm_n = f[j, ix + halo + 1]
        fm_n  = f[j, ix + halo    ]
        fc_n  = f[j, ix + halo - 1]

        ul = u_h[iz, ix    ]    # face ix-1
        uc = u_h[iz, ix + 1]    # face ix
        ur = u_h[iz, ix + 2]    # face ix+1
        ulpos = (ul + abs(ul)) * half
        ucpos = (uc + abs(uc)) * half;  ucneg = (uc - abs(uc)) * half
        urneg = (ur - abs(ur)) * half

        fppos = _tvd_flux(fmm_p, fm_p, fc_p, ulpos, ucpos)
        fpneg = _tvd_flux(fmm_n, fm_n, fc_n, urneg, ucneg)
        qx_int[iz, ix] = ucpos * fppos + ucneg * fpneg
    end
end

@kernel function _flux_tvdim_z_kernel!(qz_int, @Const(f), @Const(w_h), halo, half)
    iz, ix = @index(Global, NTuple)
    i = ix + halo
    @inbounds begin
        fmm_p = f[iz + halo - 2, i]
        fm_p  = f[iz + halo - 1, i]
        fc_p  = f[iz + halo,     i]
        fmm_n = f[iz + halo + 1, i]
        fm_n  = f[iz + halo,     i]
        fc_n  = f[iz + halo - 1, i]

        wt = w_h[iz,     ix]
        wc = w_h[iz + 1, ix]
        wb = w_h[iz + 2, ix]
        wtpos = (wt + abs(wt)) * half
        wcpos = (wc + abs(wc)) * half;  wcneg = (wc - abs(wc)) * half
        wbneg = (wb - abs(wb)) * half

        fppos = _tvd_flux(fmm_p, fm_p, fc_p, wtpos, wcpos)
        fpneg = _tvd_flux(fmm_n, fm_n, fc_n, wbneg, wcneg)
        qz_int[iz, ix] = wcpos * fppos + wcneg * fpneg
    end
end

# ----------------------------------------------------------------------------
# advect_with_flux! — writes adv AND the face-flux arrays qx/qz in
# MATLAB sizing: qx (Nz+2, Nx+1), qz (Nz+1, Nx+2). Boundary rows/cols filled
# per src/advect.m:144-161 using the specified BCs.
# ----------------------------------------------------------------------------

"""
    advect_with_flux!(adv, qx, qz, f_halo, u, w, h, scheme; xBC, zBC) -> adv

Same as `advect!` plus emits face-centred fluxes:
* `qx` size `(Nz+2, Nx+1)` — x-face fluxes with z-boundary padding (rows 1, Nz+2)
* `qz` size `(Nz+1, Nx+2)` — z-face fluxes with x-boundary padding (cols 1, Nx+2)

Boundary rows/columns are filled per MATLAB `advect.m:144-161`:
* periodic ⇒ wrap (e.g. `qz[:,1]=qz[:,end-1]`, `qz[:,end]=qz[:,2]`)
* otherwise ⇒ repeat (e.g. `qz[:,1]=qz[:,2]`, `qz[:,end]=qz[:,end-1]`)

Mass conservation: every interior face value in `qx`, `qz` equals the
algebraic flux the divergence kernel uses, so `adv ≡ div(q)/h` bitwise.
"""
function advect_with_flux!(adv::AbstractMatrix, qx::AbstractMatrix, qz::AbstractMatrix,
                           f_halo::AbstractMatrix,
                           u::AbstractMatrix, w::AbstractMatrix,
                           h::Real, scheme::Symbol;
                           xBC::Symbol = :periodic,
                           zBC::Symbol = :closed)
    Nz = size(adv, 1);  Nx = size(adv, 2)
    @assert size(qx) == (Nz + 2, Nx + 1)  "advect_with_flux!: qx must be (Nz+2, Nx+1)"
    @assert size(qz) == (Nz + 1, Nx + 2)  "advect_with_flux!: qz must be (Nz+1, Nx+2)"

    # First do the cell-centred adv (kernel checks halo/velocity sizes itself)
    advect!(adv, f_halo, u, w, h, scheme)

    halo = scheme_halo(scheme)
    backend = KernelAbstractions.get_backend(adv)
    T = eltype(adv)
    half = T(0.5)

    # Interior face-flux views: qx_int is (Nz, Nx+1), qz_int is (Nz+1, Nx)
    qx_int = view(qx, 2:Nz+1, 1:Nx+1)
    qz_int = view(qz, 1:Nz+1, 2:Nx+1)

    if scheme === :centr
        _flux_centr_x_kernel!(backend, (16, 16))(qx_int, f_halo, u, halo, half;
                                                  ndrange = size(qx_int))
        _flux_centr_z_kernel!(backend, (16, 16))(qz_int, f_halo, w, halo, half;
                                                  ndrange = size(qz_int))
    elseif scheme === :upwd1
        _flux_upwd1_x_kernel!(backend, (16, 16))(qx_int, f_halo, u, halo, half;
                                                  ndrange = size(qx_int))
        _flux_upwd1_z_kernel!(backend, (16, 16))(qz_int, f_halo, w, halo, half;
                                                  ndrange = size(qz_int))
    elseif scheme === :quick
        _flux_quick_x_kernel!(backend, (16, 16))(qx_int, f_halo, u, halo, half, T(1//6);
                                                  ndrange = size(qx_int))
        _flux_quick_z_kernel!(backend, (16, 16))(qz_int, f_halo, w, halo, half, T(1//6);
                                                  ndrange = size(qz_int))
    elseif scheme === :fromm
        _flux_fromm_x_kernel!(backend, (16, 16))(qx_int, f_halo, u, halo, half, T(0.25);
                                                  ndrange = size(qx_int))
        _flux_fromm_z_kernel!(backend, (16, 16))(qz_int, f_halo, w, halo, half, T(0.25);
                                                  ndrange = size(qz_int))
    elseif scheme === :weno3
        _flux_weno3_x_kernel!(backend, (16, 16))(qx_int, f_halo, u, halo, half, T(1e-6);
                                                  ndrange = size(qx_int))
        _flux_weno3_z_kernel!(backend, (16, 16))(qz_int, f_halo, w, halo, half, T(1e-6);
                                                  ndrange = size(qz_int))
    elseif scheme === :weno5
        _flux_weno5_x_kernel!(backend, (16, 16))(qx_int, f_halo, u, halo, half, eps(T);
                                                  ndrange = size(qx_int))
        _flux_weno5_z_kernel!(backend, (16, 16))(qz_int, f_halo, w, halo, half, eps(T);
                                                  ndrange = size(qz_int))
    elseif scheme === :tvdim
        _flux_tvdim_x_kernel!(backend, (16, 16))(qx_int, f_halo, u, halo, half;
                                                  ndrange = size(qx_int))
        _flux_tvdim_z_kernel!(backend, (16, 16))(qz_int, f_halo, w, halo, half;
                                                  ndrange = size(qz_int))
    else
        throw(ArgumentError("advect_with_flux!: unsupported scheme = :$scheme"))
    end
    KernelAbstractions.synchronize(backend)

    # ------------- MATLAB advect.m:144-161 boundary fills --------------------
    # qz boundary cols (1 and Nx+2) — periodic wrap or repeat
    if xBC === :periodic && Nx > 1
        @views qz[:, 1]     .= qz[:, Nx + 1]
        @views qz[:, Nx + 2] .= qz[:, 2]
    else
        @views qz[:, 1]     .= qz[:, 2]
        @views qz[:, Nx + 2] .= qz[:, Nx + 1]
    end
    # qx boundary rows (1 and Nz+2)
    if zBC === :periodic && Nz > 1
        @views qx[1, :]     .= qx[Nz + 1, :]
        @views qx[Nz + 2, :] .= qx[2, :]
    else
        @views qx[1, :]     .= qx[2, :]
        @views qx[Nz + 2, :] .= qx[Nz + 1, :]
    end

    return adv
end
