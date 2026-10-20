using Test
using xcore

using KernelAbstractions: CPU

@testset "xcore" begin
    include("test_parameters.jl")
    include("test_grid.jl")
    include("test_scales.jl")
    include("test_stencils.jl")
    include("test_diffus.jl")
    include("test_advect.jl")
    include("test_mms.jl")
    include("test_update.jl")
    include("test_noise.jl")
    include("test_history.jl")
    include("test_diagnose.jl")
    include("test_run.jl")
    include("test_dyrel.jl")
    include("test_gmg_direct.jl")
    include("test_state3d.jl")
    include("test_schur3d.jl")
end
