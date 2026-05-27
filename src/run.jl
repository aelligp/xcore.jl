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
    initialize!(phase, fluid, ns, hst, grid, par, scales;
                xBC=:periodic, zBC=:closed) -> (time, dt, step)

Set up an initial state on a fresh `(phase, fluid, ns, hst)` quartet, then
record the t=0 entry into `hst` so the conservation baseline `HST.sumB[1]` is
the true initial total mass (matches `src/init.m:264-267`'s
`store; history; output;` block at t=0).

If `par.restart != 0`, load from the corresponding checkpoint
(`par.restart < 0` ⇒ most recent `_cont.jld2`; `par.restart > 0` ⇒
`_<restart>.jld2`) and return `(time, dt, step)` to resume the time loop
mid-run. Otherwise the fresh-init path returns `(0, scales.dt0, 0)`.
"""
function initialize!(phase::PhaseState{T}, fluid::FluidState{T}, ns::NoiseState{T},
                     hst::History{T},
                     grid::Grid{T}, par::Parameters{T}, scales::Scales{T};
                     xBC::Symbol = :periodic,
                     zBC::Symbol = :closed) where {T<:AbstractFloat}
    Nz = grid.Nz;  Nx = grid.Nx

    # ---- restart: short-circuit fresh init when par.restart != 0 ----------
    if par.restart != 0
        path = resolve_restart(par.outdir, par.runID, par.restart)
        meta = load_checkpoint!(path, phase, fluid, ns, hst)
        # rebuild the geometric shape functions (deterministic from grid; not
        # checkpointed because they're cheap and might evolve with code changes)
        compute_bndtaperw!(phase, grid, par, scales)
        compute_bndshape!(phase, grid, scales)
        return (T(meta.time), T(meta.dt), Int(meta.step))
    end

    # boundary taper for segregation speed (closed top/bot by default)
    compute_bndtaperw!(phase, grid, par, scales)
    # top-localised reaction shape for Gx = G0·(1-x)·bndshape — needed below
    # to build the boundary-modulated base crystallinity `xin`
    compute_bndshape!(phase, grid, scales)

    # Gaussian + random perturbation (init.m:157-159):
    #   xin = x0 + (Da - x0) * bndshape
    #   x   = xin * (1 + dxr * rp + dxg * gp)
    rng_seed = par.seed
    rp = randn(MersenneTwister(rng_seed), T, Nz, Nx)
    rp .= (rp .- sum(rp) / length(rp)) ./ std(rp)
    @inbounds for j in 1:Nz, i in 1:Nx
        xc = (T(i) - T(0.5)) * grid.h / grid.L
        zc = (T(j) - T(0.5)) * grid.h / grid.D
        gp = exp(-((xc - T(0.5)) / T(0.125))^2) *
             exp(-((zc - T(0.5)) / T(0.125))^2)
        xin = par.x0 + (par.Da - par.x0) * phase.bndshape[j, i]
        phase.x[j, i] = xin * (one(T) + par.dxr * rp[j, i] + par.dxg * gp)
    end
    phase.m .= one(T) .- phase.x

    # seed eta to the melt viscosity so the Picard blend in update_viscosity!
    # has something sensible to relax against
    fill!(fluid.eta,   par.etam0)
    fill!(fluid.etaco, par.etam0)
    fill!(phase.etas,  par.etam0)

    update!(phase, fluid, grid, par, scales; xBC, zBC)
    # seed phase densities X = rho·x, M = rho·m once (init.m:198-199). From
    # here on `X`, `M` are owned by phsevo! — update! only refreshes rho/chi/mu.
    @. phase.X = fluid.rho * phase.x
    @. phase.M = fluid.rho * phase.m
    noise!(ns, phase, grid, par, scales, scales.dt0; first_iter=true)
    update_phase_velocities!(phase, fluid, ns, grid, par; xBC, zBC)

    # seed history buffers with the current state — first BD2 step uses BE
    # coefficients so this is consistent.
    phase.Xo  .= phase.X;  phase.Xoo  .= phase.X
    phase.Mo  .= phase.M;  phase.Moo  .= phase.M
    fluid.rhoo .= fluid.rho;  fluid.rhooo .= fluid.rho
    store_noise!(ns)

    # t=0 record so `hst.sumB[1]` is the true initial baseline used by
    # `EB/EM/EX` (mirrors src/init.m:264-267 `store; history; output;`).
    # Use BE coefficients — no history exists yet, so dsumBdto/dsumBdtoo are
    # zero and DB[1] becomes zero by design.
    record_history!(hst, T(0), T(scales.dt0), phase, fluid, ns, grid,
                    T(1), T(1), T(0), T(1), T(0), T(0))

    return (T(0), T(scales.dt0), 0)
end

"""
    run!(phase, fluid, ns, hst, grid, par, scales; nsteps=par.Nt, dt=scales.dt0,
         ADVN=:weno5, xBC=:periodic, zBC=:closed,
         sds=-1, top_cnv=1, bot_cnv=1, open_cnv=false,
         verbose=true)
        -> (final_time, final_dt)

Drive the coupled phase + Stokes system. Each outer step runs a Picard loop
that mirrors MATLAB `main.m`: at least 3 sweeps of
`phsevo! → fluidmech! → update! → update_phase_velocities!`, then exits when
`resnorm/resnorm0 < par.rtol` or `resnorm < par.atol`, up to `par.maxit`
iterations. After each step the time-step is refreshed via `update_dt`.

Termination — mirrors MATLAB main.m: stops on the first of
* `step > nsteps`,
* `time > par.tend` (dimensional cutoff),
* `time/scales.t0 > par.t0end` (dimensionless cutoff).

Per-step hooks (no callbacks — everything is parameter-driven):
* `record_history!(hst, ...)` runs every `par.nrh` steps.
* `save_output(phase, fluid, ns, hst, ...)` runs every `par.nop` steps when
  `par.save_op` is true. PNGs land in `joinpath(par.outdir, par.runID)/`;
  a JLD2 checkpoint is written alongside.
* `print_step!` runs every step when `verbose=true`.
"""
function run!(phase::PhaseState{T}, fluid::FluidState{T}, ns::NoiseState{T},
              hst::History{T},
              grid::Grid{T}, par::Parameters{T}, scales::Scales{T};
              nsteps::Integer = par.Nt,
              time::Real = 0,
              dt::Real = scales.dt0,
              step::Integer = 0,
              ADVN::Symbol = :weno5,
              xBC::Symbol = :periodic,
              zBC::Symbol = :closed,
              sds::Integer = -1,
              top_cnv::Integer = 1,
              bot_cnv::Integer = 1,
              open_cnv::Bool = false,
              verbose::Bool = true) where {T<:AbstractFloat}

    nsteps0 = ceil(Int, (par.t0end * scales.t0) / scales.dt0)

    println("\n\n")
    println("****************************************************************\n")
    println("********** RUN XCORE.jl MODEL | $(now()) ********\n")
    println("****************************************************************\n")
    printstyled("\n run ID: $(par.runID)  for approx. $nsteps0 timesteps \n"; bold=true)

    time = T(time)
    dt   = T(dt)
    step = Int(step)
    res  = StepResidual(grid)

    tend_dim = T(par.tend)
    tend_t0  = T(par.t0end) * T(scales.t0)
    xend     = T(par.xend)

    # x-criterion: most recent mean crystallinity (HST.x(end,2) in main.m).
    # If no history exists yet, assume value < xend so the loop enters.
    x_mean_last() = isempty(hst.x_mean) ? T(0) : hst.x_mean[end]

    while step < nsteps && time < tend_dim && time < tend_t0 && x_mean_last() < xend
        step += 1
        (a1, a2, a3, b1, b2, b3) = time_coefs(T, par.TINT, step)
        store_previous!(fluid, phase)
        store_noise!(ns)

        resnorm  = T(1)
        resnorm0 = T(1)
        iter     = 0

        # per-component timing accumulators (mirrors FMtime/XEtime/UDtime in
        # MATLAB src/timing.m + diagnose.m). Reset every outer step.
        t_phs = 0.0;  t_fm = 0.0;  t_upd = 0.0

        elapsed = @elapsed begin
            # mirror MATLAB: at least 3 sweeps, then exit on convergence or maxit
            while (resnorm / resnorm0 >= T(par.rtol) &&
                   resnorm             >= T(par.atol) &&
                   iter                <  par.maxit) || iter < 3

                iter += 1
                snapshot!(res, phase, fluid)

                t_phs += @elapsed phsevo!(phase, fluid, grid, par, scales;
                        ADVN, xBC, zBC,
                        a1, a2, a3, b1, b2, b3, dt)
                t_fm  += @elapsed fluidmech!(fluid, grid, par;
                           phase = phase,
                           sds, top_cnv, bot_cnv, open_cnv,
                           xBC, zBC,
                           dt, a1, a2, a3, b1, b2, b3)
                # noise + phase velocities before update! — mirrors MATLAB main.m:
                # fluidmech sets wx/wm/Wx/Ux/Wm/Um, THEN update uses them
                # (update.m lines 26-29, 73-104 require current Wx/Wm)

                t_upd += @elapsed begin
                    noise!(ns, phase, grid, par, scales, dt; first_iter = (iter ≤ 1))
                    update_phase_velocities!(phase, fluid, ns, grid, par; xBC, zBC)
                    update!(phase, fluid, grid, par, scales; xBC, zBC)
                end

                resnorm, rm, rp = compute_resnorm(res, phase, fluid, dt)
                # set reference on first sweep, or reset if residual increased
                if iter == 1 || resnorm > resnorm0
                    resnorm0 = resnorm + T(1e-32)
                end

                verbose && report_iter(iter, Float64(resnorm), Float64(resnorm0), Float64(rm), Float64(rp))
            end
        end

        time += dt
        verbose && print_step!(step, time, dt, phase, fluid, ns, scales;
                               elapsed, t_phs, t_fm, t_upd, iter)

        # history — every par.nrh steps (mirrors main.m line 41)
        if step % par.nrh == 0
            record_history!(hst, time, dt, phase, fluid, ns, grid,
                            a1, a2, a3, b1, b2, b3)
        end

        # output (figures + JLD2 checkpoint) — every par.nop steps when enabled
        # (mirrors main.m line 47)
        if par.save_op && step % par.nop == 0
            frame = step ÷ par.nop
            save_output(phase, fluid, ns, hst, grid, par, scales, time;
                        outdir = par.outdir, runID = par.runID,
                        frame, dt, step)
        end

        dt = update_dt(phase, fluid, grid, par, dt)
    end
    return time, dt
end
