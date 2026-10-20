# Weak-dep extension: activates when both xcore and Metal are loaded.
# Provides `xcore_backend(:metal)` so run scripts never need to touch
# `Metal.MetalBackend` (or resolve KernelAbstractions/Metal name clashes) directly.
# Apple GPUs are Float32 only — drive the solver in F32 on this backend.
module xcoreMetalExt

using xcore, Metal

function xcore._backend_impl(::Val{:metal})
    Metal.functional() ||
        error("xcore_backend(:metal): Metal.jl is loaded but not functional " *
              "(Apple GPU required). Check `Metal.versioninfo()`.")
    return Metal.MetalBackend()
end

end # module
