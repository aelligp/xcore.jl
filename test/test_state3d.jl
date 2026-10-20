# Phase-1 smoke test for the 3-D GMG scaffold: the real `FluidState3D` /
# `DyrelCache3D` structs must be field-compatible with the validated
# `src/dyrel3D/schur.jl` kernel wrappers (which the standalone test
# `test_schur3d.jl` checks for correctness on NamedTuples). Here we confirm the
# struct field names/shapes line up and the wrappers run end-to-end producing
# finite, correctly-shaped output.
using xcore
using KernelAbstractions: CPU
using Random
Random.seed!(7)

const GMG3D = xcore.GMG3D
be = CPU()
T  = Float64
Nz, Nx, Ny = 8, 6, 4

f = FluidState3D(T, be, Nz, Nx, Ny)
c = DyrelCache3D(T, be, Nz, Nx, Ny)

# random, well-conditioned fill (viscosities/densities/diagonals strictly > 0)
randfill!(A)  = (A .= randn(size(A)...); A)
posfill!(A)   = (A .= rand(size(A)...) .+ 1.0; A)
for fld in (f.W, f.U, f.V, f.P, f.Drho, f.MFS,
            f.rhoWo, f.rhoWoo, f.rhoUo, f.rhoUoo, f.rhoVo, f.rhoVoo)
    randfill!(fld)
end
for fld in (f.eta, f.etaxz, f.etayz, f.etaxy, f.rho, f.rhow, f.rhou, f.rhov)
    posfill!(fld)
end
for fld in (c.D_W, c.D_U, c.D_V, c.γ_eff, c.η_b)
    posfill!(fld)
end
for fld in (c.bext_W, c.bext_U, c.bext_V, c.bext_P, c.advn_mz, c.advn_mx, c.advn_my)
    randfill!(fld)
end
P0 = randn(Nz + 2, Nx + 2, Ny + 2)

# shapes the kernels expect (sanity on the struct constructors)
@assert size(f.W)   == (Nz + 1, Nx + 2, Ny + 2)
@assert size(f.U)   == (Nz + 2, Nx + 1, Ny + 2)
@assert size(f.V)   == (Nz + 2, Nx + 2, Ny + 1)
@assert size(c.R_W) == (Nz + 1, Nx, Ny)
@assert size(c.R_U) == (Nz, Nx + 1, Ny)
@assert size(c.R_V) == (Nz, Nx, Ny + 1)
@assert size(c.s_P) == (Nz, Nx, Ny)
println("OK  FluidState3D / DyrelCache3D shapes match the schur.jl header table")

h, g0, dt = 0.37, 9.81, 0.7
for dr in (false, true)
    GMG3D.dyrel3d_residual_P!(c, f, h, P0, dt)
    GMG3D.dyrel3d_residual_V!(c, f, h, g0; dr, a1 = 1.5, a2 = -0.6, a3 = 0.1,
                              gamma = 1e-3, dt)
    for (name, A) in (("R_W", c.R_W), ("R_U", c.R_U), ("R_V", c.R_V),
                      ("R_P", c.R_P), ("P_num", c.P_num))
        all(isfinite, A) || error("FAIL $name dr=$dr has non-finite entries")
    end
    println("OK  dyrel3d residuals run on real structs (dr=$dr, all finite)")
end

GMG3D.compute_schur_diag3d!(c, f, h)
all(isfinite, c.s_P) && all(c.s_P .> 0) || error("FAIL s_P not finite/positive")
println("OK  compute_schur_diag3d! on real structs")

Pbefore = copy(f.P)
GMG3D.dyrel3d_update_pressure!(c, f; ω = 0.8)
expect = Pbefore[2:end-1, 2:end-1, 2:end-1] .+ 0.8 .* c.γ_eff .* c.R_P
maximum(abs.(f.P[2:end-1, 2:end-1, 2:end-1] .- expect)) < 1e-13 ||
    error("FAIL P update interior")
println("OK  dyrel3d_update_pressure! on real structs")

# coarsening geometry
g  = Grid3D(xcore.Parameters(T; N = 32, solver = :gmg))
gc = coarsen_grid3d(g)
@assert (gc.Nz, gc.Nx, gc.Ny) == (g.Nz ÷ 2, g.Nx ÷ 2, g.Ny ÷ 2)
@assert gc.h ≈ 2g.h
println("OK  Grid3D + coarsen_grid3d")

println("\nAll Phase-1 3-D scaffold checks passed.")
