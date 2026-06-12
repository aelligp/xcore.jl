using KernelAbstractions

# ----------------------------------------------------------------------------
# Bare first-difference operators (no halo required — just shrink by one in
# the differentiated direction). Faithful to MATLAB's ddx.m/ddz.m, which use
# `diff(a,1,dim)./h` for constant grid spacing.
# ----------------------------------------------------------------------------

@kernel function _ddx_kernel!(out, @Const(f), invh)
    iz, ix = @index(Global, NTuple)
    @inbounds out[iz, ix] = (f[iz, ix + 1] - f[iz, ix]) * invh
end

@kernel function _ddz_kernel!(out, @Const(f), invh)
    iz, ix = @index(Global, NTuple)
    @inbounds out[iz, ix] = (f[iz + 1, ix] - f[iz, ix]) * invh
end

"""
    ddx!(out, f, h) -> out

In-place x-direction finite difference: `out[j,i] = (f[j,i+1] - f[j,i]) / h`.
`size(out) == (size(f,1), size(f,2)-1)`. Backend-agnostic via
`KernelAbstractions.get_backend(out)`.
"""
function ddx!(out::AbstractMatrix, f::AbstractMatrix, h::Real)
    @assert size(out, 1) == size(f, 1)            "ddx!: row count mismatch"
    @assert size(out, 2) == size(f, 2) - 1        "ddx!: output must be one col narrower than input"
    backend = KernelAbstractions.get_backend(out)
    T = eltype(out)
    _ddx_kernel!(backend)(out, f, T(inv(h)); ndrange = size(out))
    KernelAbstractions.synchronize(backend)
    return out
end

"""
    ddz!(out, f, h) -> out

In-place z-direction finite difference: `out[j,i] = (f[j+1,i] - f[j,i]) / h`.
`size(out) == (size(f,1)-1, size(f,2))`.
"""
function ddz!(out::AbstractMatrix, f::AbstractMatrix, h::Real)
    @assert size(out, 1) == size(f, 1) - 1        "ddz!: output must be one row shorter than input"
    @assert size(out, 2) == size(f, 2)            "ddz!: column count mismatch"
    backend = KernelAbstractions.get_backend(out)
    T = eltype(out)
    _ddz_kernel!(backend)(out, f, T(inv(h)); ndrange = size(out))
    KernelAbstractions.synchronize(backend)
    return out
end
