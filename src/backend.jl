using KernelAbstractions
import Metal

"""
    xcore_backend(which::Symbol = :cpu) -> KernelAbstractions.Backend

Single entry point for backend selection, avoiding any `using CUDA` /
`using KernelAbstractions` name juggling in run scripts:

    be = xcore_backend()         # CPU
    be = xcore_backend(:metal)   # Metal — works out of the box on macOS (F32 only!)
    be = xcore_backend(:cuda)    # CUDA  — requires `using CUDA` in the script
                                 #         (weak-dep extension ext/xcoreCUDAExt.jl)

Metal is a regular dependency (zero setup on Apple hardware); CUDA is a weak
dependency so Linux clusters opt in by adding CUDA to their environment and
loading it. Both paths check `functional()` so a broken driver fails loudly
here rather than at the first kernel launch.
"""
xcore_backend(which::Symbol = :cpu) = _backend_impl(Val(which))

_backend_impl(::Val{:cpu}) = CPU()

function _backend_impl(::Val{:metal})
    Metal.functional() ||
        error("xcore_backend(:metal): Metal is not functional on this machine " *
              "(Apple GPU required). Check `Metal.versioninfo()`.")
    return Metal.MetalBackend()
end

_backend_impl(::Val{S}) where {S} =
    error("xcore_backend(:$S) is not available. For :cuda add `using CUDA` to your " *
          "script (and `CUDA` to your environment). " *
          "Available without extras: :cpu, :metal.")

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
    adapt_backend(backend, A::AbstractArray) -> array

Move a host array `A` onto `backend`, preserving element type (incl. complex).
Used for setup-time constants (e.g. the noise Gaussian FFT filters) that are
cheaper to build on the CPU and then transfer once. For `backend = CPU()` this
is a copy; for a GPU backend it allocates a device array and copies into it.
"""
function adapt_backend(backend, A::AbstractArray{S}) where {S}
    d = KernelAbstractions.allocate(backend, S, size(A))
    copyto!(d, A)
    return d
end

"""
    interior(f::AbstractArray, halo::Integer) -> SubArray

View into the interior of a halo'd 2D field, excluding `halo` rings on every
side. Used both to read interior values for comparison and to assign into the
interior from MATLAB-sized arrays.
"""
interior(f::AbstractArray{<:Any,2}, halo::Integer) =
    view(f, (halo+1):(size(f,1)-halo), (halo+1):(size(f,2)-halo))
