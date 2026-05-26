#=
Visualize the MMS benchmark result.

Run from the repo root:

    julia --project=julia examples/visualize_mms.jl

Requires CairoMakie (for PNG output) or GLMakie (for interactive). Install with:

    julia --project=julia -e 'using Pkg; Pkg.add("CairoMakie")'

Writes PNGs into `xcore/out/visualize/`.
=#

using xcore
using KernelAbstractions: CPU
using CairoMakie                                         # activates XCoreMakieExt
using LinearAlgebra

# pull the MMS source generator (test-only helper, not part of the package)
include(joinpath(@__DIR__, "..", "test", "mms_sources.jl"))

const OUT = joinpath(@__DIR__, "..", "..", "out", "visualize")
mkpath(OUT)

T = Float64
L = T(10)

for N in (64, 128,256)
    println("\n=== solving MMS at N = $N ===")
    mms = mms_sources(T, N, N, L)

    par  = Parameters(T; D = L, N = N, L = L, etam0 = 1e3, g0 = 10.0,
                        rhom0 = 2700, rhox0 = 3200, d0 = 1e-2,
                        Da = 0.0, Xi = 0.0, gamma = 0.0)
    grid = Grid(par)
    state = FluidState(T, CPU(), grid.Nz, grid.Nx)

    # inject analytic material fields
    state.eta   .= mms.eta
    state.etaco .= mms.etaco
    state.rho   .= mms.rho
    state.rhow  .= mms.rhow
    state.rhou  .= mms.rhou
    state.Drho  .= mms.Drho
    state.MFS   .= mms.MFS

    bnchm_data = BnchmData{T, typeof(mms.src_W)}(
        mms.src_W, mms.src_U, mms.src_P,
        mms.W_exact, mms.U_exact, mms.P_exact,
    )

    fluidmech!(state, grid, par;
               bnchm_data = bnchm_data,
               sds = -1, top_cnv = -1, bot_cnv = -1,
               open_cnv = false,
               dt = T(1e32),
               a1 = T(1), a2 = T(1), a3 = T(0))

    plot_state(state, grid;
               title = "MMS solve, N = $N",
               path  = joinpath(OUT, "mms_state_N$(N).png"))
    plot_mms_comparison(state, mms, grid;
                        path = joinpath(OUT, "mms_compare_N$(N).png"))

    @info "errors" N EW = norm(state.W .- mms.W_exact) / norm(mms.W_exact) EU = norm(state.U .- mms.U_exact) / norm(mms.U_exact) EP = norm(state.P .- mms.P_exact) / norm(mms.P_exact)
end

println("\nPNGs written to $OUT")
