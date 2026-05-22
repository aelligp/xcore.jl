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

# -------------------- dispatch front-end -----------------------------------

"""
    advect!(adv, f_halo, u, w, h, scheme::Symbol) -> adv

Conservative advection `div(v f)` on a cell-centred field, with face-staggered
velocities. `scheme` is `:centr` (halo=1) or `:weno5` (halo=3) — matching the
MATLAB advection-scheme names. Other schemes from `src/advect.m` (`:upwd1`,
`:quick`, `:fromm`, `:weno3`, `:tvdim`) can be added as additional kernels.

Sizing convention:
* `adv`     `(Nz, Nx)`              — interior output
* `f_halo`  `(Nz + 2g, Nx + 2g)`    — cell-centred scalar with halo `g`
* `u`       `(Nz, Nx + 1)`          — x-face velocity
* `w`       `(Nz + 1, Nx)`          — z-face velocity

Caller fills ghosts on `f_halo` (typically via `fill_ghosts!`) before each call.
"""
function advect!(adv::AbstractMatrix, f_halo::AbstractMatrix,
                 u::AbstractMatrix, w::AbstractMatrix, h::Real, scheme::Symbol)
    halo = scheme === :centr ? 1 : scheme === :weno5 ? 3 :
        throw(ArgumentError("advect!: unsupported scheme = :$scheme (use :centr or :weno5)"))

    Nz = size(adv, 1);  Nx = size(adv, 2)
    @assert size(f_halo) == (Nz + 2halo, Nx + 2halo)  "advect!: f_halo size doesn't match adv + 2*halo"
    @assert size(u) == (Nz, Nx + 1)                   "advect!: u must be (Nz, Nx+1)"
    @assert size(w) == (Nz + 1, Nx)                   "advect!: w must be (Nz+1, Nx)"

    backend = KernelAbstractions.get_backend(adv)
    T = eltype(adv)
    invh = T(inv(h))
    half = T(0.5)

    if scheme === :centr
        _advect_centr_kernel!(backend, (16, 16))(adv, f_halo, u, w,
                                                 invh, halo, half;
                                                 ndrange = size(adv))
    else # :weno5
        _advect_weno5_kernel!(backend, (16, 16))(adv, f_halo, u, w,
                                                 invh, halo, half, eps(T);
                                                 ndrange = size(adv))
    end
    KernelAbstractions.synchronize(backend)
    return adv
end
