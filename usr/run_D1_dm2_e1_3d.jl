using xcore, KernelAbstractions
# Backend and float type. For GPU, pass the GPU package's backend instead:
#   using CUDA;  be = CUDABackend()                 # Float64 OK
#   using Metal; be = MetalBackend(); T = Float32   # Metal has no Float64
be = CPU()
T  = Float64

# 3-D version of usr/run_D1_dm2_e1.jl: D = 1e1 m, d0 = 1e-2 m, etam0 = 1e1 Pa·s.
# The grid is N × 1.5N × 1.5N cells (z, x, y), periodic in x and y. Output is
# ParaView frames (`<runID>.pvd`); the Makie figures are 2-D only.

par = Parameters(T;
    runID   = "D1_dm2_e1_3d_N32",
    outdir  = joinpath(@__DIR__, "..", "out"),
    save_op = true,
    plot_op = false,
    vtk_op  = true,
    nop     = 10,           # write a frame and checkpoint every nop steps
    nrh     = 1,            # record history every step
    restart = 0,            # frame number to restart from; -1 for the most recent

    # domain
    D  = 1e1,
    N  = 32,
    L  = 1e1 * 1.5,         # 1.5 × D

    # timing
    t0end = 2.0,            # stop at 2 dimensionless time units

    # physics
    d0    = 1e-2,
    etam0 = 1e1,
    L0    = 1e1 / 100,      # D/100
    l0    = 1e-2 * 10,      # d0×10
    Da    = 0.01,
    Xi    = 0.5,

    # :dyrel runs at any N; :vmg needs every side even with half ≥ mg_minlvl
    solver = :vmg,
    atol   = 1e-6,
)

grid   = Grid(par, Val(3))
scales = compute_scales(par, grid)
print_scales(scales, par)

fluid  = FluidState(T, be, grid)
phase  = PhaseState(T, be, grid)
ns     = NoiseState(T, be, grid, scales)
hst    = History(T)

time0, dt0, step0 = initialize!(phase, fluid, ns, hst, grid, par, scales)

final_time, final_dt = run!(phase, fluid, ns, hst, grid, par, scales;
                            time = time0, dt = dt0, step = step0, verbose = true)
println("\nDone. final_time = $final_time s  ($(round(final_time/scales.t0, sigdigits=4)) t₀)")
