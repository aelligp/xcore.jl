# Weak-dep extension: activates when both xcore and AMDGPU are loaded.
# Provides `xcore_backend(:amdgpu)` so run scripts never need to touch
# `AMDGPU.ROCBackend` (or resolve KernelAbstractions/AMDGPU name clashes) directly.
module xcoreAMDGPUExt

using xcore, AMDGPU

function xcore._backend_impl(::Val{:amdgpu})
    AMDGPU.functional() ||
        error("xcore_backend(:amdgpu): AMDGPU.jl is loaded but not functional " *
              "(no ROCm device / driver problem). Check `AMDGPU.versioninfo()`.")
    return AMDGPU.ROCBackend()
end

end # module
