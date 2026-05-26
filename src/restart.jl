using JLD2

# Restart / checkpointing — mirrors the `if restart` block in src/init.m
# (lines 231-270). MATLAB stores per-frame `.mat` files plus a rolling
# `_cont.mat`; we use `.jld2` so Julia's RNG and complex-valued precomputed
# filters round-trip cleanly. Loading restores arrays in place into the
# user-allocated `phase`, `fluid`, `ns`, `hst` so call sites stay symmetric
# with `initialize!`.
#
# What is saved:
#   * every matrix field of `FluidState` and `PhaseState`
#   * the time-evolving fields of `NoiseState` (ψ, ξ, RNG state) — the
#     precomputed filter kernels `Gkpe`, `Gkps`, `padL0`, `padl0`, `fL`, `fl`
#     are deterministic from (grid, scales) and rebuilt by the constructor.
#   * the full `History` (every Vector field).
#   * scalars: `time`, `dt`, `step`.
#
# What is NOT saved:
#   * `Parameters`, `Grid`, `Scales` — the run script reconstructs these
#     from source so a checkpoint can't silently drift with code changes.

# Time-evolving fields of NoiseState that must be checkpointed. Everything
# else in NoiseState (Gkpe, Gkps, padL0, padl0, fL, fl) is deterministic
# from (grid, scales) and rebuilt by the constructor.
noise_dynamic_fields() = (:psie, :psix, :psis, :psieo, :psixo, :psiso,
                          :re, :rs,
                          :xiew, :xieu, :xixw, :xixu, :xisw, :xisu,
                          :xie,  :xix,  :xis)

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
        # ---- fluid arrays
        for name in fieldnames(typeof(fluid))
            f["fluid/$(name)"] = getfield(fluid, name)
        end
        # ---- phase arrays (hasx/hasm get recomputed by update!, but cheap to save)
        for name in fieldnames(typeof(phase))
            f["phase/$(name)"] = getfield(phase, name)
        end
        # ---- noise dynamic state + RNG; static filter kernels are skipped
        for name in noise_dynamic_fields()
            f["noise/$(name)"] = getfield(ns, name)
        end
        f["noise/rng"] = ns.rng
        # ---- history
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
        # ---- fluid
        for name in fieldnames(typeof(fluid))
            getfield(fluid, name) .= f["fluid/$(name)"]
        end
        # ---- phase: skip bndtaperw/bndshape (recomputed in initialize!) only if
        # sizes don't match; otherwise overwrite for byte-exact restart.
        for name in fieldnames(typeof(phase))
            arr = getfield(phase, name)
            arr .= f["phase/$(name)"]
        end
        # ---- noise dynamic state + RNG (copy! preserves the immutable wrapper)
        for name in noise_dynamic_fields()
            getfield(ns, name) .= f["noise/$(name)"]
        end
        copy!(ns.rng, f["noise/rng"])
        # ---- history: empty + repopulate so the mutable struct's vector fields
        # become byte-equal to the checkpoint
        for name in fieldnames(typeof(hst))
            v = getfield(hst, name)
            empty!(v)
            append!(v, f["hst/$(name)"])
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
