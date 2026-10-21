# Output every `par.nop` steps: figures (Makie extension) and a JLD2 checkpoint.

# ---------------------------------------------------------------------------
# Convenience: save all panels at once
# ---------------------------------------------------------------------------

"""
    save_output(phase, fluid, ns, hst, grid, par, scales, time;
                outdir="out", runID="run", frame=0,
                dt=0, step=0, checkpoint=true) -> nothing

With `par.plot_op`, compute visualization scales from `par` + `scales` +
`grid` and write all six figure panels to `outdir/runID/` as PNG files; this
requires a loaded Makie backend (`using CairoMakie`). When `checkpoint=true`
(default), also writes a JLD2 restart file `<runID>_<frame>.jld2` plus the
rolling `<runID>_cont.jld2`, mirroring MATLAB's per-frame + continuation
file pattern (`output.m`).

`dt` and `step` are saved into the checkpoint so a restart can resume the
time loop exactly. Pass them through from the run callback.
"""
function save_output(phase::PhaseState, fluid::FluidState,
                     ns::NoiseState, hst::History,
                     grid::Grid, par::Parameters, scales::Scales,
                     time::Real;
                     outdir::AbstractString = "out",
                     runID::AbstractString  = "run",
                     frame::Integer         = 0,
                     dt::Real               = 0,
                     step::Integer          = 0,
                     checkpoint::Bool       = true)
    dir = joinpath(outdir, runID)
    mkpath(dir)
    tag = @sprintf("%s_%04d", runID, frame)

    if par.plot_op
        Base.get_extension(@__MODULE__, :xcoreMakieExt) === nothing && error(
            "save_output: par.plot_op = true needs a Makie backend; " *
            "load one (`using CairoMakie`) or set plot_op = false")
        vis = compute_vis_scales(par, scales, grid)
        plot_fluid(        phase, fluid,     grid, vis; time, path = joinpath(dir, "$(tag)_cnv.png"))
        plot_phase(        phase,            grid, vis; time, path = joinpath(dir, "$(tag)_sgr.png"))
        plot_diffuse(      phase,     ns,    grid, vis; time, path = joinpath(dir, "$(tag)_dff.png"))
        plot_dimensionless(phase,     ns,    grid, vis; time, path = joinpath(dir, "$(tag)_ndn.png"))
        plot_profiles(     phase, fluid,     grid, vis; time, path = joinpath(dir, "$(tag)_prf.png"))
        plot_history(      hst,              vis;        scales, path = joinpath(dir, "$(runID)_hst.png"))
    end

    if checkpoint
        # per-frame snapshot + rolling continuation file (mirrors MATLAB's
        # `_<frame>.mat` + `_cont.mat` pattern in output.m)
        frame_path = restart_path(outdir, runID; frame)
        cont_path  = restart_path(outdir, runID; frame = -1)
        save_checkpoint(frame_path, phase, fluid, ns, hst; time, dt, step)
        cp(frame_path, cont_path; force = true)
    end

    return nothing
end
