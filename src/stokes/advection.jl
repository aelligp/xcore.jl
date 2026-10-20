# Momentum advection div(v·ρv_d), frozen at the velocity a solve starts from:
# the outer Picard loop updates it by calling the solver again, which is the
# treatment `fluidmech!` applies, and it keeps per-iteration work free of the
# wide-stencil advection.
#
# Each component's momentum ρ_d·v_d is advected as a cell field on the control
# volumes centered on its faces, by `advect!`'s flux-form schemes (generic in D).
# Along its own axis d:
#   * bounded (always z): the control volumes are the interior faces 2..N_d,
#     advected by v_d interpolated to cell centres; boundary faces get zero.
#   * periodic: all N_d + 1 faces (the two boundary faces are the same physical
#     face), advected by v_d at the midpoints between faces with periodic wrap;
#     the two boundary values are averaged afterwards so they agree.
# Along every other axis k the control volumes follow the cell grid, advected
# by v_k averaged along d onto the control-volume faces.

# One-dimensional face fluxes (q_hi, q_lo) along an axis: `f` is the halo field,
# `c` the cell, `e` the unit offset; `vl`, `vh` the velocities of the low/high
# faces, `vll`, `vhh` those one face further out (TVD only).
@inline _pos(u, half) = (u + abs(u)) * half
@inline _neg(u, half) = (u - abs(u)) * half

Base.@propagate_inbounds function _fluxes(::Val{:centr}, f, c, e, vl, vh, vll, vhh, half)
    fc = f[c]
    return vh * (fc + f[c + e]) * half, vl * (fc + f[c - e]) * half
end
Base.@propagate_inbounds function _fluxes(::Val{:upwd1}, f, c, e, vl, vh, vll, vhh, half)
    fc = f[c];  fm = f[c - e];  fp = f[c + e]
    return _pos(vh, half) * fc + _neg(vh, half) * fp, _pos(vl, half) * fm + _neg(vl, half) * fc
end
Base.@propagate_inbounds function _fluxes(::Val{:quick}, f, c, e, vl, vh, vll, vhh, half)
    T = typeof(half);  sixth = T(1//6)
    fmm = f[c - 2e];  fm = f[c - e];  fc = f[c];  fp = f[c + e];  fpp = f[c + 2e]
    fppos = sixth * (2 * fp + 5 * fc - fm)
    fpneg = sixth * (2 * fc + 5 * fp - fpp)
    fmpos = sixth * (2 * fc + 5 * fm - fmm)
    fmneg = sixth * (2 * fm + 5 * fc - fp)
    return _pos(vh, half) * fppos + _neg(vh, half) * fpneg,
           _pos(vl, half) * fmpos + _neg(vl, half) * fmneg
end
Base.@propagate_inbounds function _fluxes(::Val{:fromm}, f, c, e, vl, vh, vll, vhh, half)
    T = typeof(half);  quarter = T(0.25)
    fmm = f[c - 2e];  fm = f[c - e];  fc = f[c];  fp = f[c + e];  fpp = f[c + 2e]
    fppos = fc + quarter * (fp - fm)
    fpneg = fp + quarter * (fc - fpp)
    fmpos = fm + quarter * (fc - fmm)
    fmneg = fc + quarter * (fm - fp)
    return _pos(vh, half) * fppos + _neg(vh, half) * fpneg,
           _pos(vl, half) * fmpos + _neg(vl, half) * fmneg
end
Base.@propagate_inbounds function _fluxes(::Val{:weno3}, f, c, e, vl, vh, vll, vhh, half)
    T = typeof(half);  ε = T(1e-6)
    fmm = f[c - 2e];  fm = f[c - e];  fc = f[c];  fp = f[c + e];  fpp = f[c + 2e]
    fppos = _weno3_poly(fm, fc, fp, ε)
    fpneg = _weno3_poly(fpp, fp, fc, ε)
    fmpos = _weno3_poly(fmm, fm, fc, ε)
    fmneg = _weno3_poly(fp, fc, fm, ε)
    return _pos(vh, half) * fppos + _neg(vh, half) * fpneg,
           _pos(vl, half) * fmpos + _neg(vl, half) * fmneg
end
Base.@propagate_inbounds function _fluxes(::Val{:weno5}, f, c, e, vl, vh, vll, vhh, half)
    ε = eps(typeof(half))
    fmmm = f[c - 3e];  fmm = f[c - 2e];  fm = f[c - e];  fc = f[c]
    fp = f[c + e];  fpp = f[c + 2e];  fppp = f[c + 3e]
    fppos = _weno5_poly(fmm, fm, fc, fp, fpp, ε)
    fpneg = _weno5_poly(fppp, fpp, fp, fc, fm, ε)
    fmpos = _weno5_poly(fmmm, fmm, fm, fc, fp, ε)
    fmneg = _weno5_poly(fpp, fp, fc, fm, fmm, ε)
    return _pos(vh, half) * fppos + _neg(vh, half) * fpneg,
           _pos(vl, half) * fmpos + _neg(vl, half) * fmneg
end
Base.@propagate_inbounds function _fluxes(::Val{:tvdim}, f, c, e, vl, vh, vll, vhh, half)
    fmm = f[c - 2e];  fm = f[c - e];  fc = f[c];  fp = f[c + e];  fpp = f[c + 2e]
    lpos = _pos(vl, half);  lneg = _neg(vl, half)
    hpos = _pos(vh, half);  hneg = _neg(vh, half)
    fppos = _tvd_flux(fm, fc, fp, lpos, hpos)
    fpneg = _tvd_flux(fpp, fp, fc, _neg(vhh, half), hneg)
    fmpos = _tvd_flux(fmm, fm, fc, _pos(vll, half), lpos)
    fmneg = _tvd_flux(fp, fc, fm, hneg, lneg)
    return hpos * fppos + hneg * fpneg, lpos * fmpos + lneg * fmneg
end

# `vel[k]` has n_k + 1 + 2pad faces along k; face j of the cell grid is at j + pad.
@kernel function _advect_kernel!(adv, @Const(f), @Const(vel), scheme, g, pad, invh, half)
    I = @index(Global, Cartesian)
    @inbounds begin
        D = length(I)
        c = I + g * _ones(I)
        ax = _hz_axes(Val(D))
        q(k) = @inbounds (ek = _unit(I, k);  vk = vel[k];  P = pad * ek;
                          _fluxes(scheme, f, c, ek, vk[I + P], vk[I + ek + P],
                                  vk[I + P - pad * ek], vk[I + ek + P + pad * ek], half))
        q1 = q(ax[1])
        acc = _tfoldl((s, k) -> (qk = q(k); s + qk[1] - qk[2]), q1[1] - q1[2], Base.tail(ax))
        adv[I] = acc * invh
    end
end

# Ghost layers of width g: periodic wrap or zero-gradient repeat per axis.
function _fill_ghosts!(f::AbstractArray{<:Any, D}, g, periodic::NTuple{D, Bool}) where {D}
    for k in (ntuple(k -> k + 1, Val(D - 1))..., 1)
        n = size(f, k) - 2g
        sel(r) = view(f, ntuple(m -> m == k ? r : Colon(), Val(D))...)
        if periodic[k]
            sel(1:g) .= sel(n + 1:n + g)
            sel(n + g + 1:n + 2g) .= sel(g + 1:2g)
        else
            for m in 1:g
                sel(m:m) .= sel(g + 1:g + 1)
                sel(n + g + m:n + g + m) .= sel(n + g:n + g)
            end
        end
    end
    return f
end

"""
    advect_nd!(adv, f, vel, h, scheme, periodic) -> adv

Flux-form `div(v f)` of the cell field `f` (shape of `adv`), with `vel[k]` the
velocity on the faces normal to axis k (n_k + 1 along k, cell-shaped
elsewhere) and `periodic[k]` the ghost rule of each axis (otherwise repeat).
`scheme` as in `advect!`.
"""
function advect_nd!(adv::AbstractArray{T, D}, f, vel::NTuple{D}, h, scheme::Symbol,
                    periodic::NTuple{D, Bool}) where {T, D}
    g = scheme_halo(scheme)
    fh = similar(f, size(f) .+ 2g)
    fill!(fh, zero(T))
    view(fh, ntuple(k -> g + 1:g + size(f, k), Val(D))...) .= f
    _fill_ghosts!(fh, g, periodic)
    pad = scheme === :tvdim ? 1 : 0
    if pad == 1
        vel = ntuple(Val(D)) do k
            u = vel[k];  n = size(u, k) - 1
            up = similar(u, size(u) .+ 2 .* ntuple(m -> Int(m == k), Val(D)))
            sel(A, r) = view(A, ntuple(m -> m == k ? r : Colon(), Val(D))...)
            sel(up, 2:n + 2) .= u
            if periodic[k]
                sel(up, 1:1) .= sel(u, n:n);  sel(up, n + 3:n + 3) .= sel(u, 2:2)
            else
                sel(up, 1:1) .= sel(u, 1:1);  sel(up, n + 3:n + 3) .= sel(u, n + 1:n + 1)
            end
            up
        end
    end
    backend = KernelAbstractions.get_backend(adv)
    _advect_kernel!(backend)(adv, fh, vel, Val(scheme), g, pad, T(inv(h)), T(0.5);
                             ndrange = size(adv))
    return adv
end

"""
    momentum_advection!(cache, fluid, grid, par, op) -> nothing

Fill `cache.advn[d]` with `div(v·ρ_d v_d)` for every component from the current
velocity and face densities, scheme `par.ADVN` (see the file header).
"""
function momentum_advection!(cache::StokesCache{D, T}, fluid, grid, par, op::StokesOp{D}) where {D, T}
    v = velocity(fluid);  ρf = face_density(fluid)
    N = cells(grid);  h = grid.h
    for d in 1:D
        own_periodic = op.periodic[d]
        Nd = N[d]
        # control volumes along d: face index range (compact) and count
        cv = own_periodic ? (1:Nd + 1) : (2:Nd)
        # cell-shaped selection on other axes: compact 1:N, ghosted 2:N+1
        sel(A, rd, ghosted) = view(A, ntuple(k -> k == d ? rd : (ghosted ? (2:N[k] + 1) : (1:N[k])), Val(D))...)
        f = sel(ρf[d], cv, false) .* sel(v[d], cv, true)
        vel = ntuple(Val(D)) do k
            if k == d
                if own_periodic
                    # midpoints between consecutive faces, wrapping across the boundary
                    vd = v[d];  n = Nd + 1
                    sl(r) = sel(vd, r, true)
                    out = similar(f, ntuple(m -> m == d ? n + 1 : size(f, m), Val(D)))
                    so(r) = view(out, ntuple(m -> m == d ? r : Colon(), Val(D))...)
                    so(2:n) .= (sl(1:n - 1) .+ sl(2:n)) ./ T(2)
                    so(1:1) .= (sl(Nd:Nd) .+ sl(1:1)) ./ T(2)
                    so(n + 1:n + 1) .= (sl(n:n) .+ sl(2:2)) ./ T(2)
                    out
                else
                    (sel(v[d], 1:Nd, true) .+ sel(v[d], 2:Nd + 1, true)) ./ T(2)
                end
            else
                # v_k averaged along d onto the control-volume faces, all faces along k
                rd0, rd1 = own_periodic ? (1:Nd + 1, 2:Nd + 2) : (2:Nd, 3:Nd + 1)
                vk(rd) = view(v[k], ntuple(m -> m == d ? rd : (m == k ? (1:N[k] + 1) : (2:N[m] + 1)), Val(D))...)
                (vk(rd0) .+ vk(rd1)) ./ T(2)
            end
        end
        if own_periodic
            out = cache.advn[d]
            advect_nd!(out, f, vel, h, par.ADVN, op.periodic)
            sd(r) = view(out, ntuple(m -> m == d ? r : Colon(), Val(D))...)
            avg = (sd(1:1) .+ sd(Nd + 1:Nd + 1)) ./ T(2)
            sd(1:1) .= avg
            sd(Nd + 1:Nd + 1) .= avg
        else
            fill!(cache.advn[d], zero(T))
            out = view(cache.advn[d], ntuple(m -> m == d ? (2:Nd) : Colon(), Val(D))...)
            advect_nd!(out, f, vel, h, par.ADVN, op.periodic)
        end
    end
    return nothing
end
