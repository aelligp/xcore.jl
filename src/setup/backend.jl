using KernelAbstractions

"""
    xcore_zeros(backend, T, dims...) -> array

Zero-filled array on `backend`, a wrapper around `KernelAbstractions.zeros`
that checks the backend first.

Pick `backend = CPU()` for host arrays; pass a GPU backend (e.g. `CUDABackend()`)
for device arrays. Throws if the backend reports itself non-functional (broken
or missing driver), or if `T` is double precision and the backend lacks Float64
support (e.g. `MetalBackend()`; use `Float32` there).
"""
function xcore_zeros(backend, ::Type{T}, dims::Integer...) where {T}
    KernelAbstractions.functional(backend) === false &&
        error("Backend $backend is loaded but not functional; check its driver " *
              "(e.g. `Metal.versioninfo()` or `CUDA.versioninfo()`).")
    T <: Union{Float64, ComplexF64} && !KernelAbstractions.supports_float64(backend) &&
        error("Backend $backend does not support Float64; use Float32 instead.")
    return KernelAbstractions.zeros(backend, T, dims...)
end

"""
    adapt_backend(backend, A::AbstractArray) -> array

Copy the host array `A` onto `backend`, preserving the element type (including
complex). For setup-time constants such as the noise FFT filters, which are
built on the CPU and transferred once. For `backend = CPU()` this is a copy.
"""
function adapt_backend(backend, A::AbstractArray{S}) where {S}
    d = KernelAbstractions.allocate(backend, S, size(A))
    copyto!(d, A)
    return d
end

"""
    interior(f::AbstractArray, halo::Integer = 1) -> SubArray

View into the interior of a halo'd field, excluding `halo` rings on every
side.
"""
interior(f::AbstractArray{<:Any, D}, halo::Integer = 1) where {D} =
    view(f, ntuple(k -> (halo + 1):(size(f, k) - halo), Val(D))...)
