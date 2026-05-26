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
