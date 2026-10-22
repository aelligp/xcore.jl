using KernelAbstractions

# Two-point centred diffusion, faithful to src/diffus.m:
#   q   = −k_face·∂f      (k_face = arithmetic average of the two cells)
#   dff = −div(q)         (returned positive for use as a source term)
# Neighbors outside the domain follow the ghost rule of advection (`ghost_index`):
# periodic wrap, otherwise edge repeat. `k` may be any indexable cell field,
# e.g. a lazy product `Broadcast.instantiate(Broadcast.broadcasted(*, X, kx))`.

# cell value A[c + m·e_k] under the ghost rule of axis k
@inline ghosted(A, c, k, m, periodic) =
    A[replace_index(c, k, ghost_index(c[k] + m, length(axes(A, k)), periodic[k]))]

@kernel function diffus_kernel!(dff, @Const(f), k, periodic, invh2, half)
    c = @index(Global, Cartesian)
    ax = hz_vaxes(Val(length(c)))
    # face-averaged k times the face gradient on the high and low faces;
    # the factor 1/2 of the face average is folded into `half * invh2`
    function flux(a)
        d = unval(a)
        fc = f[c];  kc = k[c]
        hi = (ghosted(k, c, d, 1, periodic) + kc) * (ghosted(f, c, d, 1, periodic) - fc)
        lo = (kc + ghosted(k, c, d, -1, periodic)) * (fc - ghosted(f, c, d, -1, periodic))
        (hi, lo)
    end
    q1 = flux(ax[1])
    acc = tuple_foldl((s, a) -> (q = flux(a); s + q[1] - q[2]), q1[1] - q1[2], Base.tail(ax))
    dff[c] = half * invh2 * acc
end

# q: fluxes through the faces normal to `axis`, the face I between cells I − e and I
@kernel function diffus_flux_kernel!(q, @Const(f), k, axis, periodic, invh, half)
    I = @index(Global, Cartesian)
    d = unval(axis)
    fL = ghosted(f, I, d, -1, periodic);  fR = ghosted(f, I, d, 0, periodic)
    kL = ghosted(k, I, d, -1, periodic);  kR = ghosted(k, I, d, 0, periodic)
    q[I] = -(kL + kR) * half * (fR - fL) * invh
end

function diffusion_setup(name, dff, f, k)
    size(dff) == size(f) == size(k) ||
        throw(DimensionMismatch("$name: dff, f and k must match; got $(size(dff)), $(size(f)), $(size(k))"))
    Base.require_one_based_indexing(dff, f)
    return nothing
end

"""
    diffus!(dff, f, k, h; xBC=:periodic, yBC=:periodic, zBC=:closed) -> dff

In-place diffusion operator: writes `-div(-k grad f)` to `dff` for the cell
fields `f` and `k` (all of size `N`, axes `(z, x[, y])`). Returns positive
values where `f` is locally concave — the "diffusion as a positive source" sign
convention of `src/diffus.m` (`dffn_X` enters `dXdt` with a `+` sign). Ghost
cells follow `xBC`/`yBC`/`zBC` (`:periodic` wrap or `:closed` edge repeat);
`yBC` is read only in 3-D.
"""
function diffus!(dff::AbstractArray, f::AbstractArray, k, h::Real;
                 xBC::Symbol = :periodic, yBC::Symbol = :periodic, zBC::Symbol = :closed)
    diffusion_setup("diffus!", dff, f, k)
    periodic = periodic_axes("diffus!", Val(ndims(f)), xBC, yBC, zBC)
    T = eltype(dff);  backend = KernelAbstractions.get_backend(dff)
    diffus_kernel!(backend)(dff, f, k, periodic, T(inv(h)^2), T(0.5); ndrange = size(dff))
    KernelAbstractions.synchronize(backend)
    return dff
end

"""
    diffus_with_flux!(dff, q, f, k, h; xBC, yBC, zBC) -> dff

`diffus!` that also returns the face fluxes `q = (q_z, q_x[, q_y])`,
`q_d = -(k_face)·∂_d f`, sized and padded as in `advect_with_flux!`
(`diffus.m:70-81`). The stored flux carries the minus sign, so `dff = -div(q)`.
"""
function diffus_with_flux!(dff::AbstractArray, q::Tuple, f::AbstractArray, k, h::Real;
                           xBC::Symbol = :periodic, yBC::Symbol = :periodic, zBC::Symbol = :closed)
    check_padded_fluxes("diffus_with_flux!", q, size(f))
    diffus!(dff, f, k, h; xBC, yBC, zBC)
    periodic = periodic_axes("diffus_with_flux!", Val(ndims(f)), xBC, yBC, zBC)
    T = eltype(dff);  backend = KernelAbstractions.get_backend(dff)
    for d in eachindex(q)
        qd = face_interior(q[d], d)
        diffus_flux_kernel!(backend)(qd, f, k, Val(d), periodic, T(inv(h)), T(0.5); ndrange = size(qd))
    end
    KernelAbstractions.synchronize(backend)
    fill_face_ghosts!(q, periodic)
    return dff
end
