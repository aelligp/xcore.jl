# Output every `par.nop` steps: figures (Makie extension), ParaView frames and a
# JLD2 checkpoint.

"""
    save_output(phase, fluid, ns, hst, grid, par, scales, time;
                outdir="out", runID="run", frame=0,
                dt=0, step=0, checkpoint=true) -> nothing

With `par.plot_op`, compute visualization scales from `par` + `scales` +
`grid` and write all six figure panels to `outdir/runID/` as PNG files; this
requires a loaded Makie backend (`using CairoMakie`) and a 2-D grid. With
`par.vtk_op`, write the ParaView frame `<runID>_<frame>.vtr` (see [`save_vtk`](@ref))
to the same directory and add it to the time series `<runID>.pvd`. With
`checkpoint = true`, also write the restart file `<runID>_<frame>.jld2` and
copy it to the rolling `<runID>_cont.jld2`.

`dt` and `step` are saved into the checkpoint so a restart can resume the
time loop exactly. Pass them through from the run loop.
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
        grid isa Grid{<:Any, 2} ||
            error("save_output: the figures of par.plot_op = true are 2-D; set plot_op = false and vtk_op = true for a 3-D grid")
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

    par.vtk_op && save_vtk(joinpath(dir, tag), phase, fluid, ns, grid; t = time, pvd = joinpath(dir, runID))

    if checkpoint
        # per-frame snapshot plus rolling continuation file (output.m)
        frame_path = restart_path(outdir, runID; frame)
        cont_path  = restart_path(outdir, runID; frame = -1)
        save_checkpoint(frame_path, phase, fluid, ns, hst; time, dt, step)
        cp(frame_path, cont_path; force = true)
    end

    return nothing
end
