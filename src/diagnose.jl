using Printf
using LinearAlgebra: norm

# Port of src/report.m + src/diagnose.m.
#
# Two levels of output:
#   report_iter   — one line per Picard iteration (residual convergence)
#   print_step!   — full per-step diagnostic block printed after each outer step

_gm(A) = exp(sum(log, max.(vec(Float64.(A)), 1e-300)) / length(A))

"""
    StepResidual

Scratch workspace for tracking Picard-iteration convergence.  Allocate once
per run via `StepResidual(grid)`, then call `update_residual!` at the start of
each iteration to snapshot the primary unknowns, and `compute_resnorm` after
each iteration to measure the change.
"""
mutable struct StepResidual{T}
    Wo::Matrix{T};  Uo::Matrix{T};  Po::Matrix{T}
    Xo::Matrix{T};  MFSo::Matrix{T}
    resnorm0::Float64
end

function StepResidual(grid::Grid{T}) where {T<:AbstractFloat}
    Nz, Nx = grid.Nz, grid.Nx
    z(s...) = zeros(T, s...)
    StepResidual{T}(
        z(Nz+1, Nx+2), z(Nz+2, Nx+1), z(Nz+2, Nx+2),
        z(Nz,   Nx),   z(Nz,   Nx),
        1.0,
    )
end

"""
    snapshot!(res, phase, fluid) -> nothing

Capture current W / U / P / X / MFS into `res.*o` buffers.  Call at the
*start* of each Picard iteration, before the sweeps modify the fields.
"""
function snapshot!(res::StepResidual, phase::PhaseState, fluid::FluidState)
    res.Wo   .= fluid.W
    res.Uo   .= fluid.U
    res.Po   .= fluid.P
    res.Xo   .= phase.X
    res.MFSo .= fluid.MFS
    return nothing
end

"""
    compute_resnorm(res, phase, fluid, dt) -> (resnorm, res_mass, res_mmnt)

Relative Picard residual: norm of the change in each primary field divided by
the norm of the field itself (mirrors `src/report.m`).
"""
function compute_resnorm(res::StepResidual, phase::PhaseState,
                          fluid::FluidState, dt::Real)
    ep = eps(Float64)
    nrho = Float64(norm(fluid.rho))
    res_mass = norm(phase.X   .- res.Xo)   / (nrho            + ep) +
               norm(fluid.MFS .- res.MFSo) / (nrho / dt       + ep)
    res_mmnt = norm(fluid.W   .- res.Wo)   / (norm(fluid.W)   + ep) +
               norm(fluid.U   .- res.Uo)   / (norm(fluid.U)   + ep) +
               norm(fluid.P   .- res.Po)   / (norm(fluid.P)   + ep)
    resnorm = res_mass + res_mmnt
    if res.resnorm0 < resnorm
        res.resnorm0 = resnorm + 1e-32
    end
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

    # unpack scales for readability
    W0  = Float64(scales.W0);   w0  = Float64(scales.w0)
    ke0 = Float64(scales.ke0);  ks0 = Float64(scales.ks0);  kx0 = Float64(scales.kx0)
    r0  = Float64(scales.rho0)
    E0  = Float64(scales.eta0)

    println("\n")
    @printf("=== step %d | t = %.4e s | dt = %.4e s", step, Float64(time), Float64(dt))
    elapsed !== nothing && @printf(" | T2S = %.2f s", Float64(elapsed))
    println("\n")

    # per-component timing (matches MATLAB diagnose.m:3-5). The "/iter" form
    # mirrors MATLAB's `FMtime/(iter-1)`: average wall time per Picard sweep.
    if t_phs !== nothing && t_fm !== nothing && t_upd !== nothing
        n = max(1, Int(something(iter, 1)))
        @printf("    fluid-mechanics solve = %1.3e s/iter\n", Float64(t_fm)  / n)
        @printf("    phase evolution solve = %1.3e s/iter\n", Float64(t_phs) / n)
        @printf("    coefficients update   = %1.3e s/iter\n\n", Float64(t_upd) / n)
    end

    # crystallinity & melt fraction
    xv = Float64.(phase.x);  mv = Float64.(phase.m)
    @printf("    min x   = %.6f;  mean x   = %.6f;  max x   = %.6f\n",
            minimum(xv), mean(xv), maximum(xv))
    @printf("    min m   = %.6f;  mean m   = %.6f;  max m   = %.6f\n\n",
            minimum(mv), mean(mv), maximum(mv))

    # diffusivities (each normalised by its own scale)
    ks = Float64.(phase.ks) ./ ks0
    kx = Float64.(phase.kx) ./ kx0
    ke = Float64.(phase.ke) ./ ke0
    @printf("    min ks  = %1.2e;  mean ks  = %1.2e;  max ks  = %1.2e  [ks0]\n",
            minimum(ks), _gm(ks), maximum(ks))
    @printf("    min kx  = %1.2e;  mean kx  = %1.2e;  max kx  = %1.2e  [kx0]\n",
            minimum(kx), _gm(kx), maximum(kx))
    @printf("    min ke  = %1.2e;  mean ke  = %1.2e;  max ke  = %1.2e  [ke0]\n\n",
            minimum(ke), _gm(ke), maximum(ke))

    # density and viscosity
    rho = Float64.(fluid.rho) ./ r0
    eta = Float64.(fluid.eta) ./ E0
    @printf("    min rho = %.4f;  mean rho = %.4f;  max rho = %.4f  [r0]\n",
            minimum(rho), _gm(rho), maximum(rho))
    @printf("    min eta = %1.2e;  mean eta = %1.2e;  max eta = %1.2e  [E0]\n\n",
            minimum(eta), _gm(eta), maximum(eta))

    # velocities
    V  = Float64.(phase.V)  ./ W0
    vx = Float64.(phase.vx) ./ w0
    vm = Float64.(phase.vm) ./ w0
    @printf("    min V   = %1.2e;  mean V   = %1.2e;  max V   = %1.2e  [W0]\n",
            minimum(V), _gm(V), maximum(V))
    @printf("    min vx  = %1.2e;  mean vx  = %1.2e;  max vx  = %1.2e  [w0]\n",
            minimum(vx), _gm(vx), maximum(vx))
    @printf("    min vm  = %1.2e;  mean vm  = %1.2e;  max vm  = %1.2e  [w0]\n\n",
            minimum(vm), _gm(vm), maximum(vm))

    # noise
    xie = Float64.(ns.xie) ./ W0
    xix = Float64.(ns.xix) ./ W0
    xis = Float64.(ns.xis) ./ W0
    @printf("    min xie = %1.2e;  mean xie = %1.2e;  max xie = %1.2e  [W0]\n",
            minimum(xie), _gm(xie), maximum(xie))
    @printf("    min xix = %1.2e;  mean xix = %1.2e;  max xix = %1.2e  [W0]\n",
            minimum(xix), _gm(xix), maximum(xix))
    @printf("    min xis = %1.2e;  mean xis = %1.2e;  max xis = %1.2e  [W0]\n\n",
            minimum(xis), _gm(xis), maximum(xis))

    # dimensionless numbers
    Rc  = Float64.(phase.Rc);   Ra  = Float64.(phase.Ra)
    ReD = Float64.(phase.ReD);  Red = Float64.(phase.Red)
    @printf("    min Rc  = %1.2e;  mean Rc  = %1.2e;  max Rc  = %1.2e  [-]\n",
            minimum(Rc),  _gm(Rc),  maximum(Rc))
    @printf("    min Ra  = %1.2e;  mean Ra  = %1.2e;  max Ra  = %1.2e  [-]\n",
            minimum(Ra),  _gm(Ra),  maximum(Ra))
    @printf("    min ReD = %1.2e;  mean ReD = %1.2e;  max ReD = %1.2e  [-]\n",
            minimum(ReD), _gm(ReD), maximum(ReD))
    @printf("    min Red = %1.2e;  mean Red = %1.2e;  max Red = %1.2e  [-]\n\n",
            minimum(Red), _gm(Red), maximum(Red))

    return nothing
end
