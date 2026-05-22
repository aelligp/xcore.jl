# Plotting front-end. Real method bodies live in `ext/XCoreMakieExt.jl` and
# load automatically when a Makie backend (CairoMakie / GLMakie / WGLMakie)
# is brought into scope. Without one, calls raise an informative error.

"""
    plot_state(fluid, grid; phase=nothing, path=nothing, title="xcore state") -> Figure

Heatmaps of the primary fluid-mechanics fields (`W`, `U`, `P`, `eta`, `rho`)
plus χ from `phase` if supplied. With `path = "out.png"`, the figure is also
saved to disk (PNG via CairoMakie, etc.).

Load a Makie backend to enable: `using CairoMakie` (PNG) or
`using GLMakie` (interactive).
"""
function plot_state end

"""
    plot_mms_comparison(fluid, mms, grid; path=nothing) -> Figure

3-row × 3-col grid for the MMS benchmark: numerical, analytic, and error
heatmaps for each of `W`, `U`, `P`. `mms` is the NamedTuple returned by the
`mms_sources` test helper.
"""
function plot_mms_comparison end
