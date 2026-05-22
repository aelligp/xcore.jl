using KernelAbstractions

"""
    xcore_zeros(backend, T, dims...) -> array

Backend-aware zero allocation. Thin wrapper around `KernelAbstractions.zeros`
so call sites in the package don't need to import KA directly.

Pick `backend = CPU()` for host arrays; pass a GPU backend (e.g. `CUDABackend()`
loaded via a weak-dep extension) for device arrays.
"""
xcore_zeros(backend, ::Type{T}, dims::Integer...) where {T} =
    KernelAbstractions.zeros(backend, T, dims...)

"""
    interior(f::AbstractArray, halo::Integer) -> SubArray

View into the interior of a halo'd 2D field, excluding `halo` rings on every
side. Used both to read interior values for comparison and to assign into the
interior from MATLAB-sized arrays.
"""
interior(f::AbstractArray{<:Any,2}, halo::Integer) =
    view(f, (halo+1):(size(f,1)-halo), (halo+1):(size(f,2)-halo))
