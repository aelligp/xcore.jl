# Weak-dep extension: activates when both xcore and CUDA are loaded.
# Provides `xcore_backend(:cuda)` so run scripts never need to touch
# `CUDA.CUDABackend` (or resolve KernelAbstractions/CUDA name clashes) directly.
module xcoreCUDAExt

using xcore, CUDA

function xcore._backend_impl(::Val{:cuda})
    CUDA.functional() ||
        error("xcore_backend(:cuda): CUDA.jl is loaded but not functional " *
              "(no device / driver problem). Check `CUDA.versioninfo()`.")
    return CUDA.CUDABackend()
end

end # module
