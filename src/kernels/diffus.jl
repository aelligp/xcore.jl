using KernelAbstractions

# Two-point centred diffusion, faithful to src/diffus.m:
#   q   = −k_face·∂f      (k_face = arithmetic average of the two cells)
#   dff = −div(q)         (returned positive for use as a source term)
# Neighbors outside the domain follow the ghost rule of advection (`_ghost`):
# periodic wrap, otherwise edge repeat. `k` may be any indexable cell field,
# e.g. a lazy product `Broadcast.instantiate(Broadcast.broadcasted(*, X, kx))`.

# cell value A[c + m·e_k] under the ghost rule of axis k
Base.@propagate_inbounds @inline _ghosted(A, c, k, m, periodic) =
    A[_setidx(c, k, _ghost(c[k] + m, length(axes(A, k)), periodic[k]))]

@kernel function _diffus_kernel!(dff, @Const(f), k, periodic, invh2, half)
    c = @index(Global, Cartesian)
    @inbounds begin
        ax = _hz_vaxes(Val(length(c)))
        # face-averaged k times the face gradient on the high and low faces;
        # the factor 1/2 of the face average is folded into `half * invh2`
        function flux(a)
            d = _val(a)
            fc = f[c];  kc = k[c]
            hi = (_ghosted(k, c, d, 1, periodic) + kc) * (_ghosted(f, c, d, 1, periodic) - fc)
            lo = (kc + _ghosted(k, c, d, -1, periodic)) * (fc - _ghosted(f, c, d, -1, periodic))
            return hi, lo
        end
        q1 = flux(ax[1])
        acc = _tfoldl((s, a) -> (q = flux(a); s + q[1] - q[2]), q1[1] - q1[2], Base.tail(ax))
        dff[c] = half * invh2 * acc
    end
end

# q: fluxes through the faces normal to `axis`, the face I between cells I − e and I
@kernel function _diffus_flux_kernel!(q, @Const(f), k, axis, periodic, invh, half)
    I = @index(Global, Cartesian)
    @inbounds begin
        d = _val(axis)
        fL = _ghosted(f, I, d, -1, periodic);  fR = _ghosted(f, I, d, 0, periodic)
        kL = _ghosted(k, I, d, -1, periodic);  kR = _ghosted(k, I, d, 0, periodic)
        q[I] = -(kL + kR) * half * (fR - fL) * invh
    end
end

function _diffusion_setup(xBC, zBC)
    for bc in (xBC, zBC)
        bc in (:periodic, :closed) || throw(ArgumentError("diffus!: unsupported BC = :$bc"))
    end
    return (zBC === :periodic, xBC === :periodic)
end

"""
    diffus!(dff, f, k, h; xBC=:periodic, zBC=:closed) -> dff

In-place diffusion operator: writes `-div(-k grad f)` to `dff` for the cell
fields `f` and `k` (all `(Nz, Nx)`). Returns positive values where `f` is
locally concave — the "diffusion as a positive source" sign convention of
`src/diffus.m` (`dffn_X` enters `dXdt` with a `+` sign). Ghost cells follow
`xBC`/`zBC` (`:periodic` wrap or `:closed` edge repeat).
"""
function diffus!(dff::AbstractMatrix, f::AbstractMatrix, k, h::Real;
                 xBC::Symbol = :periodic, zBC::Symbol = :closed)
    size(dff) == size(f) == size(k) ||
        throw(DimensionMismatch("diffus!: dff, f and k must match; got $(size(dff)), $(size(f)), $(size(k))"))
    periodic = _diffusion_setup(xBC, zBC)
    T = eltype(dff);  backend = KernelAbstractions.get_backend(dff)
    _diffus_kernel!(backend)(dff, f, k, periodic, T(inv(h)^2), T(0.5); ndrange = size(dff))
    KernelAbstractions.synchronize(backend)
    return dff
end

"""
    diffus_with_flux!(dff, qx, qz, f, k, h; xBC, zBC) -> dff

`diffus!` that also returns the MATLAB-sized face fluxes:
* `qx` size `(Nz+2, Nx+1)` — `q_x = -(k_face)·∂_x f`
* `qz` size `(Nz+1, Nx+2)` — `q_z = -(k_face)·∂_z f`

The padding is filled as in `diffus.m:70-81`: periodic ⇒ wrap, otherwise ⇒
repeat. The stored flux carries the minus sign, so `dff = -div(q)`.
"""
function diffus_with_flux!(dff::AbstractMatrix, qx::AbstractMatrix, qz::AbstractMatrix,
                           f::AbstractMatrix, k, h::Real;
                           xBC::Symbol = :periodic, zBC::Symbol = :closed)
    Nz, Nx = size(f)
    size(qx) == (Nz + 2, Nx + 1) && size(qz) == (Nz + 1, Nx + 2) ||
        throw(DimensionMismatch("diffus_with_flux!: qx must be (Nz+2, Nx+1), qz (Nz+1, Nx+2)"))
    diffus!(dff, f, k, h; xBC, zBC)
    periodic = _diffusion_setup(xBC, zBC)
    T = eltype(dff);  backend = KernelAbstractions.get_backend(dff)
    for (axis, q) in ((Val(1), view(qz, 1:Nz+1, 2:Nx+1)), (Val(2), view(qx, 2:Nz+1, 1:Nx+1)))
        _diffus_flux_kernel!(backend)(q, f, k, axis, periodic, T(inv(h)), T(0.5); ndrange = size(q))
    end
    KernelAbstractions.synchronize(backend)

    if xBC === :periodic && Nx > 1
        @views qz[:, 1]      .= qz[:, Nx + 1]
        @views qz[:, Nx + 2] .= qz[:, 2]
    else
        @views qz[:, 1]      .= qz[:, 2]
        @views qz[:, Nx + 2] .= qz[:, Nx + 1]
    end
    if zBC === :periodic && Nz > 1
        @views qx[1, :]      .= qx[Nz + 1, :]
        @views qx[Nz + 2, :] .= qx[2, :]
    else
        @views qx[1, :]      .= qx[2, :]
        @views qx[Nz + 2, :] .= qx[Nz + 1, :]
    end
    return dff
end
