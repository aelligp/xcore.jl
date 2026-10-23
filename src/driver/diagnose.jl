using Printf
using LinearAlgebra: norm

# Port of src/report.m + src/diagnose.m.
#
# Two levels of output:
#   report_iter   — one line per Picard iteration (residual convergence)
#   print_step!   — full per-step diagnostic block printed after each outer step

# One "min / mean / max" line with numeric format `fmt` (a printf spec); the
# centre statistic is `center(A)`, all three are divided by `scale`, `unit` is
# shown in brackets, and `blank` adds a trailing empty line to close a group.
function print_stats(name, A, fmt, center = geomean; scale = 1, unit = "", blank = false)
    f = Printf.Format("    min %-4s= $fmt;  mean %-4s= $fmt;  max %-4s= $fmt")
    Printf.format(stdout, f, name, minimum(A) / scale, name, center(A) / scale, name, maximum(A) / scale)
    println(isempty(unit) ? "" : "  [$unit]", blank ? "\n" : "")
    return nothing
end

"""
    StepResidual

Scratch workspace for tracking Picard-iteration convergence.  Allocate once
per run via `StepResidual(grid, fluid)`, then call `snapshot!` at the start of
each iteration to snapshot the primary unknowns, and `compute_resnorm` after
each iteration to measure the change.
"""
struct StepResidual{T, D, V<:NTuple{D, AbstractArray{T, D}}, A<:AbstractArray{T, D}}
    vo::V;  Po::A;  MFSo::A
end

# Buffers are copies of the fluid fields, so they live on the same backend and
# `snapshot!`'s `.=` stays a device-side copy.
StepResidual(grid::Grid{T, D}, fluid::FluidState{T, D}) where {T<:AbstractFloat, D} =
    StepResidual(map(similar, velocity(fluid)), similar(fluid.P), similar(fluid.MFS))

"""
    snapshot!(res, fluid) -> nothing

Capture the current velocity components, P and MFS into the `res` buffers.
Call at the *start* of each Picard iteration, before the sweeps modify the fields.
"""
function snapshot!(res::StepResidual, fluid::FluidState)
    foreach((o, v) -> o .= v, res.vo, velocity(fluid))
    res.Po   .= fluid.P
    res.MFSo .= fluid.MFS
    return nothing
end

"""
    compute_resnorm(res, fluid, dt, norm_upd_X) -> (resnorm, res_mass, res_mmnt)

Relative Picard residual: norm of the change in each primary field divided by
the norm of the field itself (mirrors `src/report.m`; in 3-D the momentum term
also sums the y-velocity). The `X` term uses `norm_upd_X`, the unclamped update
returned by [`phsevo!`](@ref).
"""
function compute_resnorm(res::StepResidual, fluid::FluidState, dt::Real, norm_upd_X::Real)
    ep = eps(Float64)
    # ‖a − b‖₂ as one on-device reduction, without a temporary a .- b
    normdiff(a, b) = sqrt(sum(abs2, Broadcast.instantiate(Broadcast.broadcasted(-, a, b))))
    relchange(a, b) = normdiff(a, b) / (norm(a) + ep)
    nrho = Float64(norm(fluid.rho))
    res_mass = norm_upd_X                    / (nrho            + ep) +
               normdiff(fluid.MFS, res.MFSo) / (nrho / dt       + ep)
    res_mmnt = sum(map(relchange, velocity(fluid), res.vo)) + relchange(fluid.P, res.Po)
    resnorm = res_mass + res_mmnt
    return resnorm, res_mass, res_mmnt
end

"""
    report_iter(iter, resnorm, resnorm0, res_mass, res_mmnt) -> nothing

Print one convergence line per Picard iteration (mirrors `src/report.m`).
"""
function report_iter(iter::Int, resnorm::Float64, resnorm0::Float64,
                     res_mass::Float64, res_mmnt::Float64)
    isnan(resnorm) && error("Solver diverged with NaN at iter $iter")
    @printf("    ---  iter = %3d;  abs = %1.2e;  rel = %1.2e;  mass = %1.2e;  mmnt = %1.2e\n",
            iter, resnorm, resnorm / (resnorm0 + 1e-32), res_mass, res_mmnt)
end

"""
    print_step!(step, time, dt, phase, fluid, ns, scales; elapsed=nothing) -> nothing

Print a full per-step diagnostic block (mirrors `src/diagnose.m`): timings,
field statistics, dimensionless numbers — all in the scale-normalised units
given by `scales`.
"""
function print_step!(step::Int, time::Real, dt::Real,
                     phase::PhaseState, fluid::FluidState, ns::NoiseState,
                     scales::Scales;
                     elapsed::Union{Nothing,Real} = nothing,
                     t_phs::Union{Nothing,Real} = nothing,
                     t_fm::Union{Nothing,Real}  = nothing,
                     t_upd::Union{Nothing,Real} = nothing,
                     iter::Union{Nothing,Integer} = nothing)

    W0  = Float64(scales.W0);   w0  = Float64(scales.w0)
    ke0 = Float64(scales.ke0);  ks0 = Float64(scales.ks0);  kx0 = Float64(scales.kx0)
    r0  = Float64(scales.rho0)
    E0  = Float64(scales.eta0)

    println("\n")
    @printf("=== step %d | t = %.4e s | dt = %.4e s", step, Float64(time), Float64(dt))
    elapsed !== nothing && @printf(" | T2S = %.2f s", Float64(elapsed))
    println("\n")

    # per-component timing (diagnose.m:3-5). The "/iter" form mirrors MATLAB's
    # `FMtime/(iter-1)`: average wall time per Picard sweep.
    if t_phs !== nothing && t_fm !== nothing && t_upd !== nothing
        n = max(1, Int(something(iter, 1)))
        @printf("    fluid-mechanics solve = %1.3e s/iter\n", Float64(t_fm)  / n)
        @printf("    phase evolution solve = %1.3e s/iter\n", Float64(t_phs) / n)
        @printf("    coefficients update   = %1.3e s/iter\n\n", Float64(t_upd) / n)
    end

    # crystallinity & melt fraction (arithmetic means)
    print_stats("x", phase.x, "%.6f", mean)
    print_stats("m", phase.m, "%.6f", mean; blank = true)

    # diffusivities (each normalised by its own scale)
    print_stats("ks", phase.ks, "%1.2e"; scale = ks0, unit = "ks0")
    print_stats("kx", phase.kx, "%1.2e"; scale = kx0, unit = "kx0")
    print_stats("ke", phase.ke, "%1.2e"; scale = ke0, unit = "ke0", blank = true)

    # density and viscosity
    print_stats("rho", fluid.rho, "%.4f";  scale = r0, unit = "r0")
    print_stats("eta", fluid.eta, "%1.2e"; scale = E0, unit = "E0", blank = true)

    # velocities
    print_stats("V",  phase.V,  "%1.2e"; scale = W0, unit = "W0")
    print_stats("vx", phase.vx, "%1.2e"; scale = w0, unit = "w0")
    print_stats("vm", phase.vm, "%1.2e"; scale = w0, unit = "w0", blank = true)

    # noise
    print_stats("xie", ns.xie, "%1.2e"; scale = W0, unit = "W0")
    print_stats("xix", ns.xix, "%1.2e"; scale = W0, unit = "W0")
    print_stats("xis", ns.xis, "%1.2e"; scale = W0, unit = "W0", blank = true)

    # dimensionless numbers
    print_stats("Rc",  phase.Rc,  "%1.2e"; unit = "-")
    print_stats("Ra",  phase.Ra,  "%1.2e"; unit = "-")
    print_stats("ReD", phase.ReD, "%1.2e"; unit = "-")
    print_stats("Red", phase.Red, "%1.2e"; unit = "-", blank = true)

    return nothing
end
