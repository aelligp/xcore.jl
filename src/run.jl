using Printf
using Dates
# Top-level time-loop orchestrator. Mirrors src/main.m's outer structure:
# pick time-stepping coefficients, snapshot the previous solution, run a
# fixed number of nonlinear Picard iterations, then march on.
#
# Differences from main.m for now:
#  - no iterative-convergence test on resnorm; we just do `n_iter` Picard sweeps
#  - no history / diagnose / output hooks — caller passes a `callback`
#  - noise (xi*) is still pending — only segregation is wired into Wx, Wm
#  - boundary crystallisation (Gx) is zero (Da-driven bndshape pending)

"""
    compute_bndtaperw!(phase, grid, par, scales) -> nothing

Fill `phase.bndtaperw` (sized `(Nz+1, Nx+2)`) according to `init.m` line 142:

    bndtaperw = 1 - (exp(-z/l0h) + exp(-(D-z)/l0h)) · (1 - open_sgr)

With closed segregation boundaries (`open_sgr = false`), the taper is ≈ 0 at
both z = 0 and z = D and ≈ 1 in the interior. Multiplied into `wx` it zeroes
the segregation speed at impermeable boundaries.
"""
function compute_bndtaperw!(phase::PhaseState{T}, grid::Grid{T},
                            par::Parameters{T}, scales::Scales{T}) where {T<:AbstractFloat}
    Nz1, Nx2 = size(phase.bndtaperw)
    h = grid.h;  D = grid.D;  l0h = scales.l0h
    osgr = par.open_sgr ? one(T) : zero(T)
    @inbounds for j in 1:Nz1, i in 1:Nx2
        z = T(j - 1) * h
        phase.bndtaperw[j, i] = one(T) -
            (exp(-z / l0h) + exp(-(D - z) / l0h)) * (one(T) - osgr)
    end
    return nothing
end

"""
    compute_bndshape!(phase, grid, scales) -> nothing

Fill `phase.bndshape` (sized `(Nz, Nx)`) per `init.m` line 141:

    bndshape = exp((-z + h/2) / bnd_w)

Top-localised exponential profile that selects where boundary
crystallisation `Gx = G0·(1-x)·bndshape` is active.
"""
function compute_bndshape!(phase::PhaseState{T}, grid::Grid{T},
                           scales::Scales{T}) where {T<:AbstractFloat}
    h = grid.h;  bnd_w = scales.bnd_w
    @inbounds for j in 1:size(phase.bndshape, 1), i in 1:size(phase.bndshape, 2)
        zc = (T(j) - T(0.5)) * h
        phase.bndshape[j, i] = exp((-zc + h / T(2)) / bnd_w)
    end
    return nothing
end

"""
    initialize!(phase, fluid, par; perturbation_kind=:gaussian, perturbation_amp=0.1,
                seed=15) -> nothing

Set up an initial state on a fresh `(phase, fluid)` pair. Seeds `x = x0` with a
Gaussian or random perturbation (`dxr` / `dxg` from `par`), sets `m = 1 - x`,
computes the initial volume fractions and densities. Mirrors the relevant
lines of `src/init.m`.
"""
function initialize!(phase::PhaseState{T}, fluid::FluidState{T}, ns::NoiseState{T},
                     grid::Grid{T}, par::Parameters{T}, scales::Scales{T};
                     xBC::Symbol = :periodic,
                     zBC::Symbol = :closed) where {T<:AbstractFloat}
    Nz = grid.Nz;  Nx = grid.Nx
    # Gaussian + random perturbation (init.m: x = xin * (1 + dxr*rp + dxg*gp))
    rng_seed = par.seed
    rp = randn(MersenneTwister(rng_seed), T, Nz, Nx)
    rp .= (rp .- sum(rp) / length(rp)) ./ std(rp)
    @inbounds for j in 1:Nz, i in 1:Nx
        xc = (T(i) - T(0.5)) * grid.h / grid.L
        zc = (T(j) - T(0.5)) * grid.h / grid.D
        gp = exp(-((xc - T(0.5)) / T(0.125))^2) *
             exp(-((zc - T(0.5)) / T(0.125))^2)
        phase.x[j, i] = max(par.x0, par.x0 * (one(T) + par.dxr * rp[j, i] + par.dxg * gp))
    end
    phase.m .= one(T) .- phase.x

    # seed eta to the melt viscosity so the Picard blend in update_viscosity!
    # has something sensible to relax against
    fill!(fluid.eta,   par.etam0)
    fill!(fluid.etaco, par.etam0)
    fill!(phase.etas,  par.etam0)

    # boundary taper for segregation speed (closed top/bot by default)
    compute_bndtaperw!(phase, grid, par, scales)
    # top-localised reaction shape for Gx = G0·(1-x)·bndshape
    compute_bndshape!(phase, grid, scales)

    update!(phase, fluid, grid, par, scales; xBC, zBC)
    noise!(ns, phase, grid, par, scales, scales.dt0; first_iter=true)
    update_phase_velocities!(phase, fluid, ns, grid, par; xBC, zBC)

    # seed history buffers with the current state — first BD2 step uses BE
    # coefficients so this is consistent.
    phase.Xo  .= phase.X;  phase.Xoo  .= phase.X
    phase.Mo  .= phase.M;  phase.Moo  .= phase.M
    fluid.rhoo .= fluid.rho;  fluid.rhooo .= fluid.rho
    store_noise!(ns)

    return nothing
end

"""
    run!(phase, fluid, ns, grid, par, scales; nsteps, dt=scales.dt0,
         ADVN=:weno5, xBC=:periodic, zBC=:closed,
         sds=-1, top_cnv=1, bot_cnv=1, open_cnv=false,
         verbose=true, callback=nothing)
        -> (final_time, final_dt)

Drive the coupled phase + Stokes system for `nsteps` outer time steps.

Each outer step runs a Picard loop that mirrors MATLAB `main.m`: at least 3
sweeps of `phsevo! → fluidmech! → update! → update_phase_velocities!`, then
exits when `resnorm/resnorm0 < par.rtol` or `resnorm < par.atol`, up to
`par.maxit` iterations. After each step the time-step is refreshed via
`update_dt`.

When `verbose = true` (default) the per-iteration convergence line and a full
end-of-step diagnostic block are printed via `report_iter` / `print_step!`.

`callback(step, time, dt, phase, fluid, ns)` is called after each step if
provided; use it for output, history recording, etc.
"""
function run!(phase::PhaseState{T}, fluid::FluidState{T}, ns::NoiseState{T},
              grid::Grid{T}, par::Parameters{T}, scales::Scales{T};
              nsteps::Integer,
              dt::Real = scales.dt0,
              ADVN::Symbol = :weno5,
              xBC::Symbol = :periodic,
              zBC::Symbol = :closed,
              sds::Integer = -1,
              top_cnv::Integer = 1,
              bot_cnv::Integer = 1,
              open_cnv::Bool = false,
              verbose::Bool = true,
              callback = nothing) where {T<:AbstractFloat}

    println("\n\n")
    println("****************************************************************\n")
    println("********** RUN XCORE.jl MODEL | $(now()) ********\n")
    println("****************************************************************\n")
    println("\n run ID: %s\n", par.runID)

    time = T(0)
    dt   = T(dt)
    res  = StepResidual(grid)

    for step in 1:nsteps
        (a1, a2, a3, b1, b2, b3) = time_coefs(T, par.TINT, step)
        store_previous!(fluid, phase)
        store_noise!(ns)

        resnorm  = T(1)
        resnorm0 = T(1)
        iter     = 0

        elapsed = @elapsed begin
            # mirror MATLAB: at least 3 sweeps, then exit on convergence or maxit
            while (resnorm / resnorm0 >= T(par.rtol) &&
                   resnorm             >= T(par.atol) &&
                   iter                <  par.maxit) || iter < 3

                iter += 1
                snapshot!(res, phase, fluid)

                phsevo!(phase, fluid, grid, par, scales;
                        ADVN, xBC, zBC,
                        a1, a2, a3, b1, b2, b3, dt)
                fluidmech!(fluid, grid, par;
                           phase = phase,
                           sds, top_cnv, bot_cnv, open_cnv,
                           xBC, zBC,
                           dt, a1, a2, a3, b1, b2, b3)
                # noise + phase velocities before update! — mirrors MATLAB main.m:
                # fluidmech sets wx/wm/Wx/Ux/Wm/Um, THEN update uses them
                # (update.m lines 26-29, 73-104 require current Wx/Wm)
                noise!(ns, phase, grid, par, scales, dt; first_iter = (iter ≤ 1))
                update_phase_velocities!(phase, fluid, ns, grid, par; xBC, zBC)
                update!(phase, fluid, grid, par, scales; xBC, zBC)

                resnorm, rm, rp = compute_resnorm(res, phase, fluid, dt)
                # set reference on first sweep, or reset if residual increased
                if iter == 1 || resnorm > resnorm0
                    resnorm0 = resnorm + T(1e-32)
                end

                verbose && report_iter(iter, Float64(resnorm), Float64(resnorm0), Float64(rm), Float64(rp))
            end
        end

        time += dt
        verbose && print_step!(step, time, dt, phase, fluid, ns, scales; elapsed)
        callback !== nothing && callback(step, time, dt, phase, fluid, ns)
        dt = update_dt(phase, fluid, grid, par, dt)
    end
    return time, dt
end
