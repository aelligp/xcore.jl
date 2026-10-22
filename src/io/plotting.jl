# Plotting front-end. The methods live in `ext/xcoreMakieExt.jl` and load
# with any Makie backend (`using CairoMakie` for PNGs, `using GLMakie` for
# interactive windows).

"""
    plot_fluid(phase, fluid, grid, vis; time=0, path=nothing) -> Figure

Six-panel heatmap: -W, U, P, ρ, log₁₀η, MFS normalised by `vis`. Mirrors fh1.
"""
function plot_fluid end

"""
    plot_phase(phase, grid, vis; time=0, path=nothing) -> Figure

Four-panel: crystallinity, segregation viscosity, segregation speeds. fh2.
"""
function plot_phase end

"""
    plot_diffuse(phase, ns, grid, vis; time=0, path=nothing) -> Figure

Six-panel: log₁₀ diffusivities / characteristic scale (top) and noise speeds
normalised by their scales (bottom). Mirrors fh3.
"""
function plot_diffuse end

"""
    plot_dimensionless(phase, ns, grid, vis; time=0, path=nothing) -> Figure

Six-panel log₁₀ heatmaps of ReD, Ra, Rc, Noe, Nox, Nos normalised by their
characteristic scales. Nox/Nos colorranges are clipped to avoid outlier cells.
Mirrors fh4.
"""
function plot_dimensionless end

"""
    plot_profiles(phase, fluid, grid, vis; time=0, path=nothing) -> Figure

Four-panel 1-D depth profiles of x, velocities, diffusivities, and
viscosities. Profile centre line uses rms for velocities and geomean for
diffusivities/viscosities (matching MATLAB fh13). Min/max shown as shaded band.
"""
function plot_profiles end

"""
    plot_history(hst, vis; scales=nothing, path=nothing) -> Figure

Three-panel timeseries of crystallinity, flow speeds, and dimensionless
numbers, all normalised by `vis`. Time axis uses dynamic units (s/hr/yr/kyr)
matching MATLAB output.m. Mirrors fh14.

When `scales` is provided, end-time scatter markers are added to the
dimensionless-numbers panel showing the characteristic-scale predictions
`Ra0`, `ReD0`, `Red0`, `Rc0` (output.m:303-307) — a visual target for how
close the run is to its predicted scaling.
"""
function plot_history end

"""
    plot_state(fluid, grid; phase=nothing, path=nothing, title="xcore state") -> Figure

Heatmaps of the fluid-mechanics fields (`W`, `U`, `P`, `eta`, `rho`, `etaco`)
in physical coordinates, plus `chi`, `etamix` and `eII` from `phase` if
supplied. Ghost rings are dropped. With `path = "out.png"`, the figure is also
saved to disk.
"""
function plot_state end

"""
    plot_mms_comparison(fluid, mms, grid; path=nothing, title="MMS Stokes comparison") -> Figure

3-row × 3-col grid for the MMS benchmark: numerical, analytic, and error
heatmaps for each of `W`, `U`, `P`, with ghost rings dropped. `mms` must
provide `W_exact`, `U_exact` and `P_exact` shaped like `fluid.W`, `fluid.U`
and `fluid.P` (as returned by the `mms_sources` test helper).
"""
function plot_mms_comparison end
