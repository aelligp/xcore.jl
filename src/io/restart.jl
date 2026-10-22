using JLD2

# Restart / checkpointing — mirrors the `if restart` block in src/init.m
# (lines 231-270). MATLAB stores per-frame `.mat` files plus a rolling
# `_cont.mat`; we use `.jld2` so Julia's RNG and complex-valued precomputed
# filters round-trip cleanly. Loading restores arrays in place into the
# user-allocated `phase`, `fluid`, `ns`, `hst` so call sites stay symmetric
# with `initialize!`.
#
# What is saved:
#   * every array field of `FluidState`, `PhaseState` and `NoiseState`, and
#     the noise RNG state;
#   * the full `History`;
#   * scalars: `time`, `dt`, `step`.
#
# What is NOT saved:
#   * `Parameters`, `Grid`, `Scales` — the run script reconstructs these
#     from source so a checkpoint can't silently drift with code changes.
#   * solver caches (`FluidState.solver`, the matrix-free `StokesCache`): they
#     are rebuilt on the first solve, so a matrix-free restart warm-starts from
#     zero and agrees with the uninterrupted run to the solver tolerance.

# The `(group, state)` pairs whose array fields a checkpoint holds.
checkpoint_states(phase, fluid, ns) = (("fluid", fluid), ("phase", phase), ("noise", ns))

"""
    restart_path(outdir, runID; frame = -1) -> String

Resolve the path to a checkpoint file. `frame < 0` returns the rolling
continuation file `<runID>_cont.jld2`; `frame ≥ 0` returns the per-frame
file `<runID>_<frame>.jld2`. Mirrors the MATLAB convention in `init.m:233-237`.
"""
function restart_path(outdir::AbstractString, runID::AbstractString;
                      frame::Integer = -1)
    dir = joinpath(outdir, runID)
    if frame < 0
        return joinpath(dir, "$(runID)_cont.jld2")
    else
        return joinpath(dir, @sprintf("%s_%04d.jld2", runID, frame))
    end
end

"""
    save_checkpoint(path, phase, fluid, ns, hst; time, dt, step) -> nothing

Write the full restart-relevant state to `path` (creating parent directories
as needed). Use `restart_path` to build the canonical file name.
"""
function save_checkpoint(path::AbstractString,
                         phase::PhaseState, fluid::FluidState,
                         ns::NoiseState, hst::History;
                         time::Real, dt::Real, step::Integer)
    mkpath(dirname(path))
    jldopen(path, "w") do f
        for (group, state) in checkpoint_states(phase, fluid, ns)
            for name in fieldnames(typeof(state))
                val = getfield(state, name)
                val isa AbstractArray && (f["$(group)/$(name)"] = val)
            end
        end
        f["noise/rng"] = ns.rng
        # ---- history (vectors and scalars; History is mutable so scalars
        # round-trip as plain values)
        for name in fieldnames(typeof(hst))
            f["hst/$(name)"] = getfield(hst, name)
        end
        # ---- scalars
        f["meta/time"] = time
        f["meta/dt"]   = dt
        f["meta/step"] = step
    end
    return nothing
end

"""
    load_checkpoint!(path, phase, fluid, ns, hst) -> (time, dt, step)

Read a checkpoint written by `save_checkpoint` and overwrite the arrays of
`phase`, `fluid`, `ns`, and `hst` in place. The pre-allocated containers
must have matching sizes (same `(Nz, Nx)` as when the checkpoint was made).

Returns `(time, dt, step)` so the caller can resume the time loop.
"""
function load_checkpoint!(path::AbstractString,
                          phase::PhaseState, fluid::FluidState,
                          ns::NoiseState, hst::History)
    isfile(path) || error("load_checkpoint!: file not found: $(path)")
    jldopen(path, "r") do f
        for (group, state) in checkpoint_states(phase, fluid, ns)
            for name in fieldnames(typeof(state))
                arr = getfield(state, name)
                arr isa AbstractArray || continue
                key = "$(group)/$(name)"
                haskey(f, key) || error("load_checkpoint!: $(path) has no `$(key)`; it was written by an older version of xcore.jl")
                arr .= f[key]
            end
        end
        copy!(ns.rng, f["noise/rng"])
        # ---- history: vector fields get empty!+append!; scalar fields go
        # through setfield! (History is `mutable struct`)
        for name in fieldnames(typeof(hst))
            cur = getfield(hst, name)
            val = f["hst/$(name)"]
            if cur isa AbstractVector
                empty!(cur);  append!(cur, val)
            else
                setfield!(hst, name, val)
            end
        end
        return (time = f["meta/time"], dt = f["meta/dt"], step = f["meta/step"])
    end
end

"""
    resolve_restart(outdir, runID, restart) -> String | Nothing

MATLAB-compatible restart-flag resolver (init.m:231-261):
* `restart == 0`  → `nothing` (no restart)
* `restart < 0`   → most recent `_cont.jld2`
* `restart > 0`   → specific `_<frame>.jld2` (frame number)

Returns the resolved path, or `nothing` if `restart == 0`. Errors if the
requested file does not exist.
"""
function resolve_restart(outdir::AbstractString, runID::AbstractString,
                         restart::Integer)
    restart == 0 && return nothing
    path = restart_path(outdir, runID; frame = restart < 0 ? -1 : restart)
    isfile(path) || error("resolve_restart: checkpoint not found: $(path)")
    return path
end
