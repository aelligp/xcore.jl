using Printf
using Dates
# Top-level time loop, following src/main.m: pick time-stepping coefficients,
# snapshot the previous solution, iterate Picard sweeps to convergence, record
# history and output, then march on.

"""
    compute_bndtaperw!(phase, grid, par, scales) -> nothing

Fill `phase.bndtaperw` (ghosted z-face shape `(Nz+1, Nx+2[, Ny+2])`) according to
`init.m` line 142:

    bndtaperw = 1 - (exp(-z/l0h) + exp(-(D-z)/l0h)) · (1 - open_sgr)

With closed segregation boundaries (`open_sgr = false`), the taper is ≈ 0 at
both z = 0 and z = D and ≈ 1 in the interior. Multiplied into `wx` it zeroes
the segregation speed at impermeable boundaries.
"""
@kernel function bndtaperw_kernel!(bndtaperw, h, D, l0h, osgr)
    I = @index(Global, Cartesian)
    z = (I[1] - one(h)) * h
    bndtaperw[I] = one(h) - (exp(-z / l0h) + exp(-(D - z) / l0h)) * (one(h) - osgr)
end

function compute_bndtaperw!(phase::PhaseState{T}, grid::Grid{T},
                            par::Parameters{T}, scales::Scales{T}) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(phase.bndtaperw)
    h = T(grid.h);  D = T(grid.li[1]);  l0h = T(scales.l0h)
    osgr = par.open_sgr ? one(T) : zero(T)
    bndtaperw_kernel!(backend)(phase.bndtaperw, h, D, l0h, osgr; ndrange = size(phase.bndtaperw))
    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    compute_bndshape!(phase, grid, scales) -> nothing

Fill `phase.bndshape` (cell shape `(Nz, Nx[, Ny])`) per `init.m` line 141:

    bndshape = exp((-z + h/2) / bnd_w)

Top-localised exponential profile that selects where boundary
crystallisation `Gx = G0·(1-x)·bndshape` is active.
"""
@kernel function bndshape_kernel!(bndshape, h, bnd_w)
    I = @index(Global, Cartesian)
    zc = (I[1] - oftype(h, 0.5)) * h
    bndshape[I] = exp((-zc + h / oftype(h, 2)) / bnd_w)
end

# Initial-condition fill for the crystallinity field `x` (init.m:157-159).
# `rp` (normalised host random field) is already on-device; `gp` is the analytic
# Gaussian bump centered in the domain of lengths `li` (a ball in 3-D). Runs on
# the backend of `x`.
@kernel function init_x_kernel!(x, @Const(rp), @Const(bndshape), h, li, x0, Da, dxr, dxg)
    I = @index(Global, Cartesian)
    half = oftype(h, 0.5)
    c  = oftype(h, 0.125)
    gp = prod(map((i, l) -> exp(-(((i - half) * h / l - half) / c)^2), Tuple(I), li))
    xin = x0 + (Da - x0) * bndshape[I]
    x[I] = xin * (one(h) + dxr * rp[I] + dxg * gp)
end

function compute_bndshape!(phase::PhaseState{T}, grid::Grid{T},
                           scales::Scales{T}) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(phase.bndshape)
    h = T(grid.h);  bnd_w = T(scales.bnd_w)
    bndshape_kernel!(backend)(phase.bndshape, h, bnd_w; ndrange = size(phase.bndshape))
    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    initialize!(phase, fluid, ns, hst, grid, par, scales;
                xBC=:periodic, yBC=:periodic, zBC=:closed) -> (time, dt, step)

Set up an initial state on a fresh `(phase, fluid, ns, hst)` quartet (`src/init.m`).
`hst` stays empty: the first history record, and the conservation baseline
`hst.sumB[1]`, is the state after step 1, as in the MATLAB code.

If `par.restart != 0`, load from the corresponding checkpoint
(`par.restart < 0` ⇒ most recent `_cont.jld2`; `par.restart > 0` ⇒
`_<restart>.jld2`) and return `(time, dt, step)` to resume the time loop
mid-run. Otherwise the fresh-init path returns `(0, scales.dt0/10, 0)`.
"""
function initialize!(phase::PhaseState{T}, fluid::FluidState{T}, ns::NoiseState{T},
                     hst::History{T},
                     grid::Grid{T}, par::Parameters{T}, scales::Scales{T};
                     xBC::Symbol = :periodic,
                     yBC::Symbol = :periodic,
                     zBC::Symbol = :closed) where {T<:AbstractFloat}
    N = cells(grid)

    # ---- restart: short-circuit fresh init when par.restart != 0 ----------
    if par.restart != 0
        path = resolve_restart(par.outdir, par.runID, par.restart)
        # Unlike init.m, no update/store follows: the checkpoint holds every
        # field, including the previous time levels, so the run continues exactly.
        meta = load_checkpoint!(path, phase, fluid, ns, hst)
        # rebuild the geometric shape functions (deterministic from grid; not
        # checkpointed because they're cheap and might evolve with code changes)
        compute_bndtaperw!(phase, grid, par, scales)
        compute_bndshape!(phase, grid, scales)
        # the checkpoint holds the saved step's start time (init.m:250)
        return (T(meta.time + meta.dt), T(meta.dt), Int(meta.step))
    end

    # boundary taper for segregation speed (closed top/bot by default)
    compute_bndtaperw!(phase, grid, par, scales)
    # top-localised reaction shape for Gx = G0·(1-x)·bndshape — needed below
    # to build the boundary-modulated base crystallinity `xin`
    compute_bndshape!(phase, grid, scales)

    # Gaussian + random perturbation (init.m:157-159):
    #   xin = x0 + (Da - x0) * bndshape
    #   x   = xin * (1 + dxr * rp + dxg * gp)
    # The random field `rp` is drawn + normalised on the host (seeded
    # MersenneTwister ⇒ reproducible across backends), transferred to the device
    # once, then the per-cell assignment runs as a kernel on whatever backend
    # `phase.x` lives on. `gp` is analytic in the cell position.
    backend = KernelAbstractions.get_backend(phase.x)
    # smoothed with the Gaussian Gkrp = exp(-bnd_w²/2·|k|²) (init.m:103,114)
    rp_h = randn(MersenneTwister(par.seed), T, N...)
    Gkrp = gaussian_filter(T, N, grid.h, scales.bnd_w / sqrt(T(2)))
    rp_h .= real.(ifft(Gkrp .* fft(rp_h)))
    rp_h .= (rp_h .- sum(rp_h) / length(rp_h)) ./ std(rp_h)
    rp = adapt_backend(backend, rp_h)
    init_x_kernel!(backend)(phase.x, rp, phase.bndshape, T(grid.h), map(T, grid.li),
                            T(par.x0), T(par.Da), T(par.dxr), T(par.dxg); ndrange = N)
    KernelAbstractions.synchronize(backend)
    phase.m .= one(T) .- phase.x

    # seed eta to the melt viscosity so the Picard blend in update_viscosity!
    # has something sensible to relax against, and the Re ramp factors to their
    # scale values (init.m:182-189)
    fill!(fluid.eta,   par.etam0)
    foreach(A -> fill!(A, par.etam0), edge_viscosity(fluid))
    fill!(phase.etas,  par.etam0)
    fill!(phase.fReL,  scales.fReL0)
    fill!(phase.fRel,  scales.fRel0)

    # seed phase densities X = rho·x, M = rho·m from the initial bulk density,
    # then update (init.m:198-201). From here on `X`, `M` are owned by phsevo! —
    # update! only refreshes rho/chi/mu. Velocities, segregation speeds and noise
    # potentials start at zero, as in init.m.
    update_volume_fractions!(phase, fluid, par)
    @. phase.X = fluid.rho * phase.x
    @. phase.M = fluid.rho * phase.m
    update!(phase, fluid, grid, par, scales; xBC, yBC, zBC)

    # seed history buffers with the current state — first BD2 step uses BE
    # coefficients so this is consistent.
    phase.Xo  .= phase.X;  phase.Xoo  .= phase.X
    phase.Mo  .= phase.M;  phase.Moo  .= phase.M
    fluid.rhoo .= fluid.rho;  fluid.rhooo .= fluid.rho
    store_noise!(ns)

    # No history record at t = 0: init.m records one, but step 1 overwrites it
    # (history.m: stp = max(1, step/nrh)), so the first record and the mass
    # baseline of EB/EM/EX are the state after step 1.
    return (T(0), T(scales.dt0) / 10, 0)   # dt = dt0/10 (init.m:75)
end

"""
    run!(phase, fluid, ns, hst, grid, par, scales; nsteps=par.Nt, dt=scales.dt0,
         ADVN=:weno5, xBC=:periodic, yBC=:periodic, zBC=:closed,
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
              dt::Real = scales.dt0 / 10,
              step::Integer = 0,
              ADVN::Symbol = par.ADVN,
              xBC::Symbol = :periodic,
              yBC::Symbol = :periodic,
              zBC::Symbol = :closed,
              sds::Integer = -1,
              open_cnv::Bool = par.open_cnv,
              top_cnv::Integer = 1,
              bot_cnv::Integer = open_cnv ? -1 : 1,   # open bottom is free slip (init.m:148)
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
    res  = StepResidual(grid, fluid)

    # Stokes/Navier-Stokes solver strategy (:direct | :dyrel | :gmg | :vmg).
    # Persistent state (caches, MG hierarchy) allocated once, reused across all
    # Picard sweeps and time steps. Dispatch via `solve_fluidmech!` below.
    solver = make_solver(par, fluid, phase, grid)

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
        dto = dt

        resnorm  = T(1)
        resnorm0 = T(1)
        resmin   = T(Inf)
        iter     = 0
        lin_ok   = true      # did the last fluidmech solve meet its own tolerance?

        # per-component timing accumulators (mirrors FMtime/XEtime/UDtime in
        # MATLAB src/timing.m + diagnose.m). Reset every outer step.
        t_phs = 0.0;  t_fm = 0.0;  t_upd = 0.0

        elapsed = @elapsed begin
            # At least 3 sweeps, then exit on convergence or maxit. A fluidmech
            # solve that missed its own tolerance cannot end the loop: `resnorm`
            # measures the CHANGE between sweeps, and an under-solved linear
            # system reproduces the previous fields — small change, wrong answer.
            while ((resnorm / resnorm0 >= T(par.rtol) &&
                    resnorm             >= T(par.atol)) || !lin_ok) &&
                  iter < par.maxit || iter < 3

                iter += 1
                snapshot!(res, fluid)

                t_phs += @elapsed nupd_X = phsevo!(phase, fluid, grid, par, scales;
                        ADVN, xBC, yBC, zBC,
                        a1, a2, a3, b1, b2, b3, dt)
                t_fm  += @elapsed lin_ok = solve_fluidmech!(solver, fluid, phase, grid, par, scales;
                           dt, a1, a2, a3, b1, b2, b3,
                           sds, top_cnv, bot_cnv, open_cnv, xBC, yBC, zBC)
                # noise + phase velocities before update! — mirrors MATLAB main.m:
                # fluidmech sets wx/wm/Wx/Ux/Wm/Um, THEN update uses them
                # (update.m lines 26-29, 73-104 require current Wx/Wm)

                t_upd += @elapsed begin
                    noise!(ns, phase, grid, par, scales, dt; first_iter = (iter ≤ 1), xBC, yBC)
                    update_phase_velocities!(phase, fluid, ns, grid, par; xBC, yBC, zBC)
                    update!(phase, fluid, grid, par, scales; xBC, yBC, zBC)
                    # every sweep adapts dt to the current iterate (update.m:159-164)
                    dt = update_dt(phase, fluid, grid, par, dto)
                end

                resnorm, rm, rp = compute_resnorm(res, fluid, dt, nupd_X)
                # The reference is the FIRST sweep's residual and stays fixed.
                # Re-baselining it whenever the residual rises pins the reported
                # ratio at 1 and makes a growing residual indistinguishable from
                # a converging one.
                iter == 1 && (resnorm0 = resnorm + T(1e-32))
                resnorm > resmin * T(1e3) && error(
                    "Picard iteration diverged at step $step, sweep $iter: " *
                    "resnorm = $resnorm, best so far = $resmin")
                resmin = min(resmin, resnorm)

                verbose && report_iter(iter, Float64(resnorm), Float64(resnorm0), Float64(rm), Float64(rp))
            end
        end

        # An exhausted budget means the step's fields do not solve the equations;
        # continuing would march that error forward silently.
        (resnorm / resnorm0 < T(par.rtol) || resnorm < T(par.atol)) && lin_ok || error(
            "Picard iteration failed to converge at step $step after $iter sweeps: " *
            "resnorm = $resnorm (atol = $(par.atol)), rel = $(resnorm / resnorm0) " *
            "(rtol = $(par.rtol)), last fluidmech solve converged = $lin_ok")

        # diagnostics, history and output carry the step's start time (main.m:41-50)
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
        time += dt
    end
    return time, dt
end
