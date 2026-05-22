using KernelAbstractions

# Two-point centred diffusion operator, faithful to src/diffus.m:
#   q  = - (k_face) ⋅ grad(f)              (k_face = arithmetic average to face)
#   dff = - div(q)                         (returned positive for use as a source term)
#
# Requires a 1-cell halo on both `f_halo` and `k_halo`; call `fill_ghosts!` first.

@kernel function _diffus_kernel!(dff, @Const(f), @Const(k), invh2, halo, half)
    iz, ix = @index(Global, NTuple)
    j = iz + halo
    i = ix + halo
    @inbounds begin
        fc  = f[j, i]
        fmx = f[j, i - 1];   fpx = f[j, i + 1]
        fmz = f[j - 1, i];   fpz = f[j + 1, i]
        kc  = k[j, i]
        kmx = k[j, i - 1];   kpx = k[j, i + 1]
        kmz = k[j - 1, i];   kpz = k[j + 1, i]

        # face-averaged k times the face gradient; the factor 1/2 from the
        # face average is folded into `half * invh2` below.
        flux_xr = (kpx + kc) * (fpx - fc)
        flux_xl = (kc + kmx) * (fc - fmx)
        flux_zb = (kpz + kc) * (fpz - fc)
        flux_zt = (kc + kmz) * (fc - fmz)

        dff[iz, ix] = half * invh2 *
                      (flux_xr - flux_xl + flux_zb - flux_zt)
    end
end

"""
    diffus!(dff, f_halo, k_halo, h; halo=1) -> dff

In-place diffusion operator: writes `-div(-k grad f)` to `dff` (size `(Nz, Nx)`),
reading from halo'd `f_halo` and `k_halo` (size `(Nz + 2halo, Nx + 2halo)`).
Returns positive values where `f` is locally concave — i.e. the standard
"diffusion as a positive source" sign convention used in `src/diffus.m` and
throughout the MATLAB code (`dffn_X` enters `dXdt` with a `+` sign).

The caller is responsible for filling ghost cells with the correct BC values
beforehand (typically via `fill_ghosts!`).
"""
function diffus!(dff::AbstractMatrix, f_halo::AbstractMatrix, k_halo::AbstractMatrix,
                 h::Real; halo::Integer = 1)
    @assert size(f_halo) == size(k_halo)                "diffus!: f and k must have matching halo sizes"
    @assert size(dff, 1) == size(f_halo, 1) - 2halo     "diffus!: output row count must match f interior"
    @assert size(dff, 2) == size(f_halo, 2) - 2halo     "diffus!: output col count must match f interior"
    backend = KernelAbstractions.get_backend(dff)
    T = eltype(dff)
    _diffus_kernel!(backend, (16, 16))(dff, f_halo, k_halo,
                                       T(inv(h)^2), halo, T(0.5);
                                       ndrange = size(dff))
    KernelAbstractions.synchronize(backend)
    return dff
end
