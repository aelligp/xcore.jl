using KernelAbstractions

# Conservative advection div(v f) of a cell field f with face-normal velocities
# v[k], in flux form. Each scheme is a one-dimensional face reconstruction
# (`face_fluxes`, translations of the branches of src/advect.m) applied along every
# axis. Neighbors outside the domain follow the ghost rule of their axis:
# periodic wrap, otherwise repeat of the edge cell (or edge face, for the TVD
# velocities). Sums over axes run horizontal axes first, then z.

const ADVN_SCHEMES = (:centr, :upwd1, :quick, :fromm, :weno3, :weno5, :tvdim)

# -------------------- reconstructions --------------------------------------

# WENO5 (Jiang & Shu, 1996)
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

# WENO3; the regularizer is a fixed 1e-6 (advect.m:318), not machine eps
@inline function _weno3_poly(fm::T, fc::T, fp::T, eps_val::T) where {T<:AbstractFloat}
    half = T(0.5)
    p1 = half * (fc + fp)
    p2 = half * (T(3) * fc - fm)

    b1 = (fp - fc)^2
    b2 = (fc - fm)^2

    w1 = T(1//3) / (b1 + eps_val)
    w2 = T(2//3) / (b2 + eps_val)

    return (w1 * p1 + w2 * p2) / (w1 + w2)
end

# TVD with the superbee limiter on the flux ratio (advect.m:393-400)
@inline function _tvd_flux(fm::T, fc::T, fp::T, vm::T, vp::T) where {T<:AbstractFloat}
    num = vm * (fc - fm)
    den = vp * (fp - fc)
    # den ≈ 0 ⇒ R ≈ 0 (no slope change), without dividing 0/0
    R = num / (den + copysign(eps(T), den))
    l = max(zero(T), max(min(one(T), T(2) * R), min(T(2), R)))
    return fc + T(0.5) * l * (fp - fc)
end

@inline _pos(u, half) = (u + abs(u)) * half
@inline _neg(u, half) = (u - abs(u)) * half

# One-dimensional face fluxes (q_hi, q_lo) of a cell along an axis: `F(m)` is the
# advected field m cells from the current one; `vl`, `vh` the velocities of the
# low/high faces, `vll`, `vhh` those one face further out (TVD only).
@inline function face_fluxes(::Val{:centr}, F, vl, vh, vll, vhh, half)
    fc = F(0)
    return vh * (fc + F(1)) * half, vl * (fc + F(-1)) * half
end
@inline function face_fluxes(::Val{:upwd1}, F, vl, vh, vll, vhh, half)
    fc = F(0);  fm = F(-1);  fp = F(1)
    return _pos(vh, half) * fc + _neg(vh, half) * fp, _pos(vl, half) * fm + _neg(vl, half) * fc
end
@inline function face_fluxes(::Val{:quick}, F, vl, vh, vll, vhh, half)
    T = typeof(half);  sixth = T(1//6)
    fmm = F(-2);  fm = F(-1);  fc = F(0);  fp = F(1);  fpp = F(2)
    fppos = sixth * (2 * fp + 5 * fc - fm)
    fpneg = sixth * (2 * fc + 5 * fp - fpp)
    fmpos = sixth * (2 * fc + 5 * fm - fmm)
    fmneg = sixth * (2 * fm + 5 * fc - fp)
    return _pos(vh, half) * fppos + _neg(vh, half) * fpneg,
           _pos(vl, half) * fmpos + _neg(vl, half) * fmneg
end
@inline function face_fluxes(::Val{:fromm}, F, vl, vh, vll, vhh, half)
    T = typeof(half);  quarter = T(0.25)
    fmm = F(-2);  fm = F(-1);  fc = F(0);  fp = F(1);  fpp = F(2)
    fppos = fc + quarter * (fp - fm)
    fpneg = fp + quarter * (fc - fpp)
    fmpos = fm + quarter * (fc - fmm)
    fmneg = fc + quarter * (fm - fp)
    return _pos(vh, half) * fppos + _neg(vh, half) * fpneg,
           _pos(vl, half) * fmpos + _neg(vl, half) * fmneg
end
@inline function face_fluxes(::Val{:weno3}, F, vl, vh, vll, vhh, half)
    T = typeof(half);  ε = T(1e-6)
    fmm = F(-2);  fm = F(-1);  fc = F(0);  fp = F(1);  fpp = F(2)
    fppos = _weno3_poly(fm, fc, fp, ε)
    fpneg = _weno3_poly(fpp, fp, fc, ε)
    fmpos = _weno3_poly(fmm, fm, fc, ε)
    fmneg = _weno3_poly(fp, fc, fm, ε)
    return _pos(vh, half) * fppos + _neg(vh, half) * fpneg,
           _pos(vl, half) * fmpos + _neg(vl, half) * fmneg
end
@inline function face_fluxes(::Val{:weno5}, F, vl, vh, vll, vhh, half)
    ε = eps(typeof(half))
    fmmm = F(-3);  fmm = F(-2);  fm = F(-1);  fc = F(0);  fp = F(1);  fpp = F(2);  fppp = F(3)
    fppos = _weno5_poly(fmm, fm, fc, fp, fpp, ε)
    fpneg = _weno5_poly(fppp, fpp, fp, fc, fm, ε)
    fmpos = _weno5_poly(fmmm, fmm, fm, fc, fp, ε)
    fmneg = _weno5_poly(fpp, fp, fc, fm, fmm, ε)
    return _pos(vh, half) * fppos + _neg(vh, half) * fpneg,
           _pos(vl, half) * fmpos + _neg(vl, half) * fmneg
end
@inline function face_fluxes(::Val{:tvdim}, F, vl, vh, vll, vhh, half)
    fmm = F(-2);  fm = F(-1);  fc = F(0);  fp = F(1);  fpp = F(2)
    lpos = _pos(vl, half);  lneg = _neg(vl, half)
    hpos = _pos(vh, half);  hneg = _neg(vh, half)
    fppos = _tvd_flux(fm, fc, fp, lpos, hpos)
    fpneg = _tvd_flux(fpp, fp, fc, _neg(vhh, half), hneg)
    fmpos = _tvd_flux(fmm, fm, fc, _pos(vll, half), lpos)
    fmneg = _tvd_flux(fp, fc, fm, hneg, lneg)
    return hpos * fppos + hneg * fpneg, lpos * fmpos + lneg * fmneg
end

# ghost rule of a row with n cells: periodic wrap, otherwise edge repeat
@inline ghost_index(i, n, periodic) = periodic ? mod1(i, n) : clamp(i, 1, n)
# face index outside 1..n+1 (one face further out, TVD only)
@inline ghost_face_index(i, n, periodic) = i < 1 ? (periodic ? n : 1) : (i > n + 1 ? (periodic ? 2 : n + 1) : i)

# fluxes (q_hi, q_lo) of cell c of `f` along axis k; c[k] = n + 1 is the ghost
# cell past the high boundary, whose q_lo is the flux through the last face
Base.@propagate_inbounds @inline function _cell_fluxes(scheme, f, vk, c, k, periodic, half)
    n = size(f, k);  p = periodic[k];  i = c[k]
    F(m) = f[replace_index(c, k, ghost_index(i + m, n, p))]
    vel(j) = vk[replace_index(c, k, ghost_face_index(j, n, p))]
    tvd = scheme isa Val{:tvdim}
    return face_fluxes(scheme, F, vel(i), vel(i + 1),
                   tvd ? vel(i - 1) : zero(half), tvd ? vel(i + 2) : zero(half), half)
end

# -------------------- kernels ----------------------------------------------

@kernel function _advect_kernel!(adv, @Const(f), v, scheme, periodic, invh, half)
    c = @index(Global, Cartesian)
    @inbounds begin
        ax = hz_vaxes(Val(length(c)))
        qk(a) = (k = unval(a); _cell_fluxes(scheme, f, v[k], c, k, periodic, half))
        q1 = qk(ax[1])
        acc = tuple_foldl((s, a) -> (q = qk(a); s + q[1] - q[2]), q1[1] - q1[2], Base.tail(ax))
        adv[c] = acc * invh
    end
end

# q: fluxes through the faces normal to `axis`, the face I between cells I - e and I
@kernel function _advective_flux_kernel!(q, @Const(f), @Const(vk), axis, scheme, periodic, half)
    I = @index(Global, Cartesian)
    @inbounds q[I] = _cell_fluxes(scheme, f, vk, I, unval(axis), periodic, half)[2]
end

@kernel function _flux_divergence_kernel!(adv, q, invh)
    c = @index(Global, Cartesian)
    @inbounds begin
        ax = hz_vaxes(Val(length(c)))
        hi(a) = (k = unval(a); q[k][c + unit_index(c, k)]);  lo(a) = q[unval(a)][c]
        adv[c] = tuple_foldl((s, a) -> s + hi(a) - lo(a), hi(ax[1]) - lo(ax[1]), Base.tail(ax)) * invh
    end
end

# -------------------- front-end --------------------------------------------

function advection_setup(name, adv, f, v, scheme, xBC, yBC, zBC)
    scheme in ADVN_SCHEMES || throw(ArgumentError("$name: unsupported scheme = :$scheme"))
    N = size(f)
    size(adv) == N && length(v) == length(N) &&
        all(d -> size(v[d]) == face_shape(N, d), eachindex(v)) ||
        throw(DimensionMismatch("$name: need adv and f of size N, v[d] of size N .+ e_d; " *
                                "got $(size(adv)), $(size(f)), $(map(size, v))"))
    Base.require_one_based_indexing(adv, f, v...)
    return Val(scheme), periodic_axes(name, Val(ndims(f)), xBC, yBC, zBC)
end

"""
    advect!(adv, f, v, h, scheme; xBC=:periodic, yBC=:periodic, zBC=:closed) -> adv

Conservative advection `div(v f)` of the cell field `f` (size `N`, axes
`(z, x[, y])`) with the face velocities `v = (w, u[, v_y])`, `v[d]` of size
`N .+ e_d`; scheme `∈ (:centr, :upwd1, :quick, :fromm, :weno3, :weno5, :tvdim)`
as in `src/advect.m`. Ghost cells follow `xBC`/`yBC`/`zBC` (`:periodic` wrap or
`:closed` edge repeat); `yBC` is read only in 3-D.
"""
function advect!(adv::AbstractArray, f::AbstractArray, v::Tuple, h::Real, scheme::Symbol;
                 xBC::Symbol = :periodic, yBC::Symbol = :periodic, zBC::Symbol = :closed)
    s, periodic = advection_setup("advect!", adv, f, v, scheme, xBC, yBC, zBC)
    T = eltype(adv);  backend = KernelAbstractions.get_backend(adv)
    _advect_kernel!(backend)(adv, f, v, s, periodic, T(inv(h)), T(0.5); ndrange = size(adv))
    KernelAbstractions.synchronize(backend)
    return adv
end

"""
    advect_with_flux!(adv, q, f, v, h, scheme; xBC, yBC, zBC) -> adv

`advect!` that also returns the face fluxes `q = (q_z, q_x[, q_y])`, with `adv`
their exact divergence. `q[d]` has size `N .+ 2 .- e_d`: the faces normal to
axis d plus one padding layer along every other axis, filled as in
`advect.m:144-161` (periodic ⇒ wrap, otherwise ⇒ repeat). In 2-D this is
`q_z` `(Nz+1, Nx+2)` and `q_x` `(Nz+2, Nx+1)`.
"""
function advect_with_flux!(adv::AbstractArray, q::Tuple, f::AbstractArray, v::Tuple,
                           h::Real, scheme::Symbol;
                           xBC::Symbol = :periodic, yBC::Symbol = :periodic, zBC::Symbol = :closed)
    s, periodic = advection_setup("advect_with_flux!", adv, f, v, scheme, xBC, yBC, zBC)
    check_padded_fluxes("advect_with_flux!", q, size(f))
    T = eltype(adv);  backend = KernelAbstractions.get_backend(adv)
    qi = ntuple(d -> face_interior(q[d], d), length(q))
    for d in eachindex(qi)
        _advective_flux_kernel!(backend)(qi[d], f, v[d], Val(d), s, periodic, T(0.5); ndrange = size(qi[d]))
    end
    _flux_divergence_kernel!(backend)(adv, qi, T(inv(h)); ndrange = size(adv))
    KernelAbstractions.synchronize(backend)
    fill_face_ghosts!(q, periodic)
    return adv
end
