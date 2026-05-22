using CairoMakie

# Production output figures.  Mirrors the four main figure panels of
# src/output.m (fh1–fh4), the 1-D profile panel (fh13), and the history
# timeseries (fh14).
#
# All plot functions accept a `vis::VisScales` argument that carries the
# normalization factors and unit strings (dimensional or dimensionless mode).
# `save_output` computes VisScales once via `compute_vis_scales` and passes
# it to every sub-panel, matching the `if ndm_op` block in `src/scales.m`.

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Pixel dimensions matching MATLAB's geometry formula
function _figsize(grid::Grid; ncols::Int, nrows::Int)
    D = Float64(grid.D);  L = Float64(grid.L)
    axh = 6.0 * sqrt(D / L)            # cm per panel (height)
    axw = 6.0 * sqrt(L / D) + 1.5      # cm per panel (width, includes colorbar)
    ahs = 0.6;  avs = 0.8              # horizontal / vertical gap between panels
    axb = 1.2;  axt = 1.5             # bottom / top margin
    axl = 1.5;  axr = 0.6             # left / right margin
    ppc = 50.0                         # pixels per cm
    fw  = round(Int, (axl + ncols * axw + (ncols - 1) * ahs + axr) * ppc)
    fh  = round(Int, (axb + nrows * axh + (nrows - 1) * avs + axt) * ppc)
    return fw, fh
end

# log10 with a safe floor to avoid -Inf
function _log10d(A::AbstractMatrix)
    log10.(max.(Float64.(A), 1e-30))
end

# Dynamic time unit selection matching MATLAB output.m lines 20-34
function _time_scale(time::Real)
    yr = 365.25 * 24 * 3600.0
    hr = 3600.0
    t  = Float64(time)
    t < 1e3      && return (1.0,      "s")
    t < 1e3 * hr && return (hr,       "hr")
    t < 1e2 * yr && return (yr,       "yr")
    return (1e3 * yr, "kyr")
end

# Heatmap + colorbar in one call
function _heatmap_panel!(ax, Xc, Zc, d::Matrix{Float64};
                          sym::Bool = false,
                          cmap = :lapaz,
                          colorrange = nothing)
    if sym
        r = max(maximum(abs, d), 1e-30)
        hm = heatmap!(ax, Xc, Zc, d'; colormap = :lapaz, colorrange = (-r, r))
    elseif colorrange !== nothing
        hm = heatmap!(ax, Xc, Zc, d'; colormap = cmap, colorrange = colorrange)
    else
        hm = heatmap!(ax, Xc, Zc, d'; colormap = cmap)
    end
    return hm
end

# One panel: axis + heatmap + colorbar. Returns the Axis.
function _panel!(fig, row, col, title_str, Xc, Zc, d::Matrix{Float64};
                 sym = false, cmap = :lapaz, colorrange = nothing,
                 hide_x = false, hide_y = false,
                 xlabel_str = "Width [m]", ylabel_str = "Depth [m]")
    ax = Axis(fig[row, col]; title = title_str,
              xlabel = hide_x ? "" : xlabel_str,
              ylabel = hide_y ? "" : ylabel_str,
              yreversed = true, aspect = DataAspect())
    hide_x && hidexdecorations!(ax; ticks = false, grid = false)
    hide_y && hideydecorations!(ax; ticks = false, grid = false)
    hm = _heatmap_panel!(ax, Xc, Zc, d; sym, cmap, colorrange)
    Colorbar(fig[row, col + 1], hm; width = 12, ticklabelsize = 9)
    return ax
end

# Convenience: scale data, optionally take log10, then call _panel!
function _plot_panel!(fig, row, col, title_str, Xc, Zc, data;
                      scale = 1.0,
                      log10_scale = false, sym = false, cmap = :lapaz,
                      colorrange = nothing,
                      hide_x = false, hide_y = false,
                      xlabel_str = "Width [m]", ylabel_str = "Depth [m]")
    d = log10_scale ? _log10d(Float64.(data) ./ scale) :
                      Float64.(data) ./ scale
    _panel!(fig, row, col, title_str, Xc, Zc, d;
            sym, cmap, colorrange, hide_x, hide_y, xlabel_str, ylabel_str)
end

# ---------------------------------------------------------------------------
# Figure 1: fluid-mechanics panel  (W, U, P, ρ, log η, MFS)
# ---------------------------------------------------------------------------

"""
    plot_fluid(phase, fluid, grid, vis; time=0, path=nothing) -> Figure

Six-panel heatmap: -W, U, P, ρ, log₁₀η, MFS normalised by `vis`. Mirrors fh1.
"""
function plot_fluid(phase::PhaseState, fluid::FluidState,
                    grid::Grid, vis::VisScales; time::Real = 0,
                    path::Union{Nothing,AbstractString} = nothing)
    Xc = Float64.(grid.Xc);  Zc = Float64.(grid.Zc)
    Xf = Float64.(grid.Xf);  Zf = Float64.(grid.Zf)
    fw, fh = _figsize(grid; ncols = 3, nrows = 2)

    xl = "Width [$(vis.sun)]"; yl = "Depth [$(vis.sun)]"
    Xcs = Xc ./ vis.ssc;  Zcs = Zc ./ vis.ssc
    Xfs = Xf ./ vis.ssc;  Zfs = Zf ./ vis.ssc

    tsc, tun = _time_scale(time)
    fig = Figure(size = (fw, fh))
    Label(fig[0, 1:6],
          @sprintf("t = %.3g [%s]   −W / U / P / ρ / η / MFS", Float64(time)/tsc, tun);
          fontsize = 13, halign = :center)

    _plot_panel!(fig, 1, 1, "-W [$(vis.Wun)]",     Xcs, Zfs, -fluid.W[:, 2:end-1];
                 scale = vis.Wsc,  sym=true, hide_x=true, xlabel_str=xl, ylabel_str=yl)
    _plot_panel!(fig, 1, 3, "U [$(vis.Wun)]",      Xfs, Zcs, fluid.U[2:end-1, :];
                 scale = vis.Wsc,  sym=true, hide_x=true, hide_y=true, xlabel_str=xl, ylabel_str=yl)
    _plot_panel!(fig, 1, 5, "P [$(vis.pun)]",      Xcs, Zcs, fluid.P[2:end-1, 2:end-1];
                 scale = vis.psc,  sym=true, hide_x=true, hide_y=true, xlabel_str=xl, ylabel_str=yl)
    _plot_panel!(fig, 2, 1, "ρ [$(vis.dun)]",      Xcs, Zcs, fluid.rho;
                 scale = vis.rsc,  xlabel_str=xl, ylabel_str=yl)
    _plot_panel!(fig, 2, 3, "log₁₀ η [$(vis.eun)]", Xcs, Zcs,
                 max.(Float64.(fluid.eta) .+ Float64(vis.eesc), 1e-30) ./ (Float64(vis.esc) + Float64(vis.eesc));
                 log10_scale=true, hide_y=true, xlabel_str=xl, ylabel_str=yl)
    _plot_panel!(fig, 2, 5, "∇·ρv [$(vis.MFSun)]", Xcs, Zcs, fluid.MFS;
                 scale = vis.MFSsc, sym=true, hide_y=true, xlabel_str=xl, ylabel_str=yl)

    rowgap!(fig.layout, 6);  colgap!(fig.layout, 4)
    resize_to_layout!(fig)
    path !== nothing && save(path, fig)
    return fig
end

# ---------------------------------------------------------------------------
# Figure 2: phase & segregation panel  (x, log ηs, –wx, –wm)
# ---------------------------------------------------------------------------

"""
    plot_phase(phase, grid, vis; time=0, path=nothing) -> Figure

Four-panel: crystallinity, segregation viscosity, segregation speeds. fh2.
"""
function plot_phase(phase::PhaseState,
                    grid::Grid, vis::VisScales; time::Real = 0,
                    path::Union{Nothing,AbstractString} = nothing)
    Xcs = Float64.(grid.Xc) ./ vis.ssc
    Zcs = Float64.(grid.Zc) ./ vis.ssc
    Zfs = Float64.(grid.Zf) ./ vis.ssc
    fw, fh = _figsize(grid; ncols = 2, nrows = 2)

    xl = "Width [$(vis.sun)]"; yl = "Depth [$(vis.sun)]"
    wx_int = Float64.(phase.wx[2:end-1, 2:end-1])
    wm_int = Float64.(phase.wm[2:end-1, 2:end-1])

    tsc, tun = _time_scale(time)
    fig = Figure(size = (fw, fh))
    Label(fig[0, 1:4],
          @sprintf("t = %.3g [%s]   x / ηs / wx / wm", Float64(time)/tsc, tun);
          fontsize = 13, halign = :center)

    if maximum(phase.x) > 10 * max(minimum(phase.x[phase.x .> 0]), 1e-30)
        _plot_panel!(fig, 1, 1, "log₁₀ x [$(vis.xun)]", Xcs, Zcs, phase.x;
                     scale = vis.xsc, log10_scale=true, hide_x=true, xlabel_str=xl, ylabel_str=yl)
    else
        _plot_panel!(fig, 1, 1, "x [$(vis.xun)]", Xcs, Zcs, phase.x;
                     scale = vis.xsc, hide_x=true, xlabel_str=xl, ylabel_str=yl)
    end

    # ηs / (esc + etsc); protect against esc+etsc ≈ 0 in dimensional mode
    etas_norm = max.(Float64.(phase.etas) .+ Float64(vis.etsc), 1e-30) ./
                max(Float64(vis.esc) + Float64(vis.etsc), 1e-30)
    _plot_panel!(fig, 1, 3, "log₁₀ ηs [$(vis.eun)]", Xcs, Zcs, etas_norm;
                 log10_scale=true, hide_x=true, hide_y=true, xlabel_str=xl, ylabel_str=yl)

    _plot_panel!(fig, 2, 1, "−wx [$(vis.wun)]", Xcs, Zfs[2:end-1], -wx_int;
                 scale = vis.wxsc, sym=true, xlabel_str=xl, ylabel_str=yl)
    _plot_panel!(fig, 2, 3, "−wm [$(vis.wun)]", Xcs, Zfs[2:end-1], -wm_int;
                 scale = vis.wmsc, sym=true, hide_y=true, xlabel_str=xl, ylabel_str=yl)

    rowgap!(fig.layout, 6);  colgap!(fig.layout, 4)
    resize_to_layout!(fig)
    path !== nothing && save(path, fig)
    return fig
end

# ---------------------------------------------------------------------------
# Figure 3: diffusivity & noise panel
# ---------------------------------------------------------------------------

"""
    plot_diffuse(phase, ns, grid, vis; time=0, path=nothing) -> Figure

Six-panel: log₁₀ diffusivities / characteristic scale (top) and noise speeds
normalised by their scales (bottom). Mirrors fh3.
"""
function plot_diffuse(phase::PhaseState, ns::NoiseState,
                      grid::Grid, vis::VisScales; time::Real = 0,
                      path::Union{Nothing,AbstractString} = nothing)
    Xcs = Float64.(grid.Xc) ./ vis.ssc
    Zcs = Float64.(grid.Zc) ./ vis.ssc
    fw, fh = _figsize(grid; ncols = 3, nrows = 2)
    xl = "Width [$(vis.sun)]"; yl = "Depth [$(vis.sun)]"

    tsc, tun = _time_scale(time)
    fig = Figure(size = (fw, fh))
    Label(fig[0, 1:6],
          @sprintf("t = %.3g [%s]   κs/κs₀ / κx/κx₀ / κe/κe₀ / ξe/ξe₀ / ξx/ξx₀ / ξs/ξs₀",
                   Float64(time)/tsc, tun); fontsize = 13, halign = :center)

    _plot_panel!(fig, 1, 1, "log₁₀ κs [$(vis.kun)]", Xcs, Zcs, phase.ks;
                 scale = vis.kssc, log10_scale=true, hide_x=true, xlabel_str=xl, ylabel_str=yl)
    _plot_panel!(fig, 1, 3, "log₁₀ κx [$(vis.kun)]", Xcs, Zcs, phase.kx;
                 scale = vis.kxsc, log10_scale=true, hide_x=true, hide_y=true, xlabel_str=xl, ylabel_str=yl)
    _plot_panel!(fig, 1, 5, "log₁₀ κe [$(vis.kun)]", Xcs, Zcs, phase.ke;
                 scale = vis.kesc, log10_scale=true, hide_x=true, hide_y=true, xlabel_str=xl, ylabel_str=yl)

    _plot_panel!(fig, 2, 1, "ξe [$(vis.xieun)]",  Xcs, Zcs, ns.xie;
                 scale = vis.xiesc, xlabel_str=xl, ylabel_str=yl)
    _plot_panel!(fig, 2, 3, "ξx [$(vis.xixun)]",  Xcs, Zcs, ns.xix;
                 scale = vis.xixsc, hide_y=true, xlabel_str=xl, ylabel_str=yl)
    _plot_panel!(fig, 2, 5, "ξs [$(vis.xisun)]",  Xcs, Zcs, ns.xis;
                 scale = vis.xissc, hide_y=true, xlabel_str=xl, ylabel_str=yl)

    rowgap!(fig.layout, 6);  colgap!(fig.layout, 4)
    resize_to_layout!(fig)
    path !== nothing && save(path, fig)
    return fig
end

# ---------------------------------------------------------------------------
# Figure 4: dimensionless numbers
# ---------------------------------------------------------------------------

"""
    plot_dimensionless(phase, ns, grid, vis; time=0, path=nothing) -> Figure

Six-panel log₁₀ heatmaps of ReD, Ra, Rc, Noe, Nox, Nos normalised by their
characteristic scales. Nox/Nos colorranges are clipped to avoid outlier cells.
Mirrors fh4.
"""
function plot_dimensionless(phase::PhaseState, ns::NoiseState,
                             grid::Grid, vis::VisScales; time::Real = 0,
                             path::Union{Nothing,AbstractString} = nothing)
    Xcs = Float64.(grid.Xc) ./ vis.ssc
    Zcs = Float64.(grid.Zc) ./ vis.ssc
    fw, fh = _figsize(grid; ncols = 3, nrows = 2)
    xl = "Width [$(vis.sun)]"; yl = "Depth [$(vis.sun)]"

    Noe = Float64.(ns.xie) ./ max.(Float64.(phase.V),  1e-30) ./ Float64(vis.Noesc)
    Nox = Float64.(ns.xix) ./ max.(Float64.(phase.vx), 1e-30) ./ Float64(vis.Noxsc)
    Nos = Float64.(ns.xis) ./ max.(Float64.(phase.vx), 1e-30) ./ Float64(vis.Nossc)

    _clip_cr(d) = (-4.0, max(-3.0, maximum(_log10d(max.(d, 1e-30)))))

    tsc, tun = _time_scale(time)
    fig = Figure(size = (fw, fh))
    Label(fig[0, 1:6],
          @sprintf("t = %.3g [%s]   ReD/ReD₀ / Ra/Ra₀ / Rc/Rc₀ / Noe/Noe₀ / Nox/Nox₀ / Nos/Nos₀",
                   Float64(time)/tsc, tun); fontsize = 13, halign = :center)

    _plot_panel!(fig, 1, 1, "log₁₀ ReD/ReD₀", Xcs, Zcs, phase.ReD;
                 scale = vis.ReDsc, log10_scale=true, hide_x=true, xlabel_str=xl, ylabel_str=yl)
    _plot_panel!(fig, 1, 3, "log₁₀ Ra/Ra₀",   Xcs, Zcs, phase.Ra;
                 scale = vis.Rasc,  log10_scale=true, hide_x=true, hide_y=true, xlabel_str=xl, ylabel_str=yl)
    _plot_panel!(fig, 1, 5, "log₁₀ Rc/Rc₀",   Xcs, Zcs, phase.Rc;
                 scale = vis.Rcsc,  log10_scale=true, hide_x=true, hide_y=true, xlabel_str=xl, ylabel_str=yl)

    _plot_panel!(fig, 2, 1, "log₁₀ Noe/Noe₀", Xcs, Zcs, Noe;
                 log10_scale=true, xlabel_str=xl, ylabel_str=yl)
    _plot_panel!(fig, 2, 3, "log₁₀ Nox/Nox₀", Xcs, Zcs, Nox;
                 log10_scale=true, colorrange=_clip_cr(Nox), hide_y=true, xlabel_str=xl, ylabel_str=yl)
    _plot_panel!(fig, 2, 5, "log₁₀ Nos/Nos₀", Xcs, Zcs, Nos;
                 log10_scale=true, colorrange=_clip_cr(Nos), hide_y=true, xlabel_str=xl, ylabel_str=yl)

    rowgap!(fig.layout, 6);  colgap!(fig.layout, 4)
    resize_to_layout!(fig)
    path !== nothing && save(path, fig)
    return fig
end

# ---------------------------------------------------------------------------
# Figure 5: 1-D horizontal-average profiles
# ---------------------------------------------------------------------------

"""
    plot_profiles(phase, fluid, grid, vis; time=0, path=nothing) -> Figure

Four-panel 1-D depth profiles of x, velocities, diffusivities, and
viscosities. Profile centre line uses rms for velocities and geomean for
diffusivities/viscosities (matching MATLAB fh13). Min/max shown as shaded band.
"""
function plot_profiles(phase::PhaseState, fluid::FluidState,
                       grid::Grid, vis::VisScales; time::Real = 0,
                       path::Union{Nothing,AbstractString} = nothing)
    Zcs = Float64.(grid.Zc) ./ Float64(vis.ssc)
    D_s = Float64(grid.D)   / Float64(vis.ssc)
    yl  = "Depth [$(vis.sun)]"

    tsc, tun = _time_scale(time)
    fig = Figure(size = (1200, 500))
    Label(fig[0, 1:4], @sprintf("t = %.3g [%s]   1-D profiles", Float64(time)/tsc, tun);
          fontsize = 13, halign = :center)

    # Shade min-max envelope and draw the profile centre line.
    # stat = :mean | :rms | :geomean  — mirrors MATLAB's choice per panel.
    function add_profile!(ax, data, Zcs, color; label = "", scale = 1.0, stat = :mean)
        d   = Float64.(data)
        mn  = vec(minimum(d, dims = 2)) ./ scale
        mx  = vec(maximum(d, dims = 2)) ./ scale
        avg = if stat == :rms
            vec(sqrt.(mean(d .^ 2, dims = 2))) ./ scale
        elseif stat == :geomean
            vec(exp.(mean(log.(max.(d, 1e-100)), dims = 2))) ./ scale
        else
            vec(mean(d, dims = 2)) ./ scale
        end
        xs = vcat(mn, reverse(mx))
        ys = vcat(Zcs, reverse(Zcs))
        poly!(ax, Point2f.(xs, ys); color = (color, 0.2), strokewidth = 0)
        lines!(ax, avg, Zcs; color, linewidth = 1.5, label)
    end

    ax1 = Axis(fig[1, 1]; xlabel = "x [$(vis.xun)]", ylabel = yl,
               yreversed = true, title = "Crystallinity [$(vis.xun)]")
    add_profile!(ax1, phase.x, Zcs, :steelblue; label = "mean ± range",
                 scale = Float64(vis.xsc), stat = :mean)
    ylims!(ax1, 0, D_s)
    axislegend(ax1; position = :rb)

    ax2 = Axis(fig[1, 2]; xlabel = "Speed [$(vis.wpun)]", ylabel = yl,
               yreversed = true, title = "Velocity [$(vis.wpun)]")
    Vd = sqrt.(((fluid.W[1:end-1, 2:end-1] .+ fluid.W[2:end, 2:end-1]) ./ 2).^2 .+
               ((fluid.U[2:end-1, 1:end-1] .+ fluid.U[2:end-1, 2:end]) ./ 2).^2)
    sc_w = Float64(vis.wmpsc)
    add_profile!(ax2, Vd,       Zcs, :steelblue; label = "|v|",   scale = sc_w, stat = :rms)
    add_profile!(ax2, phase.vx, Zcs, :tomato;    label = "|vx|",  scale = sc_w, stat = :rms)
    add_profile!(ax2, phase.vm, Zcs, :seagreen;  label = "|vm|",  scale = sc_w, stat = :rms)
    ylims!(ax2, 0, D_s)
    axislegend(ax2; position = :rb)

    ax3 = Axis(fig[1, 3]; xlabel = "κ [$(vis.kun)]", ylabel = yl,
               yreversed = true, title = "Diffusivity [$(vis.kun)]", xscale = log10)
    add_profile!(ax3, phase.ke, Zcs, :steelblue; label = "κe", scale = Float64(vis.kesc), stat = :geomean)
    add_profile!(ax3, phase.ks, Zcs, :tomato;    label = "κs", scale = Float64(vis.kssc), stat = :geomean)
    add_profile!(ax3, phase.kx, Zcs, :seagreen;  label = "κx", scale = Float64(vis.kxsc), stat = :geomean)
    ylims!(ax3, 0, D_s)
    axislegend(ax3; position = :rb)

    ax4 = Axis(fig[1, 4]; xlabel = "η [$(vis.eun)]", ylabel = yl,
               yreversed = true, title = "Viscosity [$(vis.eun)]", xscale = log10)
    esc_f  = max(Float64(vis.esc) + Float64(vis.eesc), 1e-30)
    etsc_f = max(Float64(vis.esc) + Float64(vis.etsc), 1e-30)
    eta_n    = max.(Float64.(fluid.eta)    .+ Float64(vis.eesc), 1e-30) ./ esc_f
    etas_n   = max.(Float64.(phase.etas)   .+ Float64(vis.etsc), 1e-30) ./ etsc_f
    etamix_n = max.(Float64.(phase.etamix),               1e-30) ./ Float64(vis.esc)
    add_profile!(ax4, eta_n,    Zcs, :steelblue; label = "η",  stat = :geomean)
    add_profile!(ax4, etas_n,   Zcs, :tomato;    label = "ηs", stat = :geomean)
    add_profile!(ax4, etamix_n, Zcs, :seagreen;  label = "η̄",  stat = :geomean)
    ylims!(ax4, 0, D_s)
    axislegend(ax4; position = :rb)

    rowgap!(fig.layout, 6)
    resize_to_layout!(fig)
    path !== nothing && save(path, fig)
    return fig
end

# ---------------------------------------------------------------------------
# Figure 6: history timeseries
# ---------------------------------------------------------------------------

"""
    plot_history(hst, vis; path=nothing) -> Figure

Three-panel timeseries of crystallinity, flow speeds, and dimensionless
numbers, all normalised by `vis`. Time axis uses dynamic units (s/hr/yr/kyr)
matching MATLAB output.m. Mirrors fh14.
"""
function plot_history(hst::History, vis::VisScales;
                      path::Union{Nothing,AbstractString} = nothing)
    isempty(hst.time) && return Figure()

    tmax = maximum(hst.time)
    tsc, tun = _time_scale(tmax)
    t = Float64.(hst.time) ./ tsc

    fig = Figure(size = (1200, 900))

    ax1 = Axis(fig[1, 1]; xlabel = "Time [$tun]", ylabel = "x [$(vis.xun)]",
               title = "Crystallinity [$(vis.xun)]")
    band!(ax1, t, Float64.(hst.x_min) ./ vis.xsc, Float64.(hst.x_max) ./ vis.xsc;
          color = (:steelblue, 0.2))
    lines!(ax1, t, Float64.(hst.x_mean) ./ vis.xsc; color = :steelblue, linewidth = 1.5, label = "mean")
    lines!(ax1, t, (Float64.(hst.x_mean) .+ Float64.(hst.x_std)) ./ vis.xsc;
           color = :steelblue, linewidth = 0.75, linestyle = :dash, label = "mean+std")
    axislegend(ax1; position = :rb)

    # convection: Wsc (W0 unit);  settling: whsc (= Wsc dim, = w0 ndm)
    ax2 = Axis(fig[2, 1]; xlabel = "Time [$tun]", ylabel = "Speed [$(vis.Wun)]",
               title = "Flow speeds [$(vis.Wun)]", yscale = log10)
    lines!(ax2, t, Float64.(hst.V_rms)   ./ vis.Wsc;  color = :steelblue, linewidth = 1.5, label = "convection")
    lines!(ax2, t, Float64.(hst.vx_rms)  ./ vis.whsc; color = :tomato,    linewidth = 1.5, label = "settling")
    lines!(ax2, t, Float64.(hst.xie_rms) ./ vis.xiesc; color = :grey50,   linewidth = 1.5, label = "ξe")
    lines!(ax2, t, Float64.(hst.xis_rms) ./ vis.xissc; color = :grey50,   linewidth = 1.5, linestyle = :dash, label = "ξs")
    lines!(ax2, t, Float64.(hst.xix_rms) ./ vis.xixsc; color = :grey50,   linewidth = 1.5, linestyle = :dot,  label = "ξx")
    hlines!(ax2, [1.0]; color = :black, linewidth = 0.75, linestyle = :dot)
    axislegend(ax2; position = :rb)

    ax3 = Axis(fig[3, 1]; xlabel = "Time [$tun]", ylabel = "[-]",
               title = "Dimensionless numbers", yscale = log10)
    lines!(ax3, t, Float64.(hst.Ra_gm)  ./ vis.Rasc;  color = :steelblue, linewidth = 1.5, label = "Ra")
    lines!(ax3, t, Float64.(hst.ReD_gm) ./ vis.ReDsc; color = :steelblue, linewidth = 1.5, linestyle = :dash, label = "ReD")
    lines!(ax3, t, Float64.(hst.Red_gm) ./ vis.Redsc; color = :tomato,    linewidth = 1.5, linestyle = :dot,  label = "Red")
    lines!(ax3, t, Float64.(hst.Rc_gm)  ./ vis.Rcsc;  color = :tomato,    linewidth = 1.5, label = "Rc")
    hlines!(ax3, [1.0]; color = :black, linewidth = 0.75, linestyle = :dot)
    axislegend(ax3; position = :rb)

    rowgap!(fig.layout, 8)
    resize_to_layout!(fig)
    path !== nothing && save(path, fig)
    return fig
end

# ---------------------------------------------------------------------------
# Convenience: save all panels at once
# ---------------------------------------------------------------------------

"""
    save_output(phase, fluid, ns, hst, grid, par, scales, time;
                outdir="out", runID="run", frame=0) -> nothing

Compute visualization scales from `par` + `scales` + `grid`, then write all
six figure panels to `outdir/runID/` as PNG files.
"""
function save_output(phase::PhaseState, fluid::FluidState,
                     ns::NoiseState, hst::History,
                     grid::Grid, par::Parameters, scales::Scales,
                     time::Real;
                     outdir::AbstractString = "out",
                     runID::AbstractString  = "run",
                     frame::Integer         = 0)
    dir = joinpath(outdir, runID)
    mkpath(dir)
    tag = @sprintf("%s_%04d", runID, frame)

    vis = compute_vis_scales(par, scales, grid)

    plot_fluid(        phase, fluid,     grid, vis; time, path = joinpath(dir, "$(tag)_cnv.png"))
    plot_phase(        phase,            grid, vis; time, path = joinpath(dir, "$(tag)_sgr.png"))
    plot_diffuse(      phase,     ns,    grid, vis; time, path = joinpath(dir, "$(tag)_dff.png"))
    plot_dimensionless(phase,     ns,    grid, vis; time, path = joinpath(dir, "$(tag)_ndn.png"))
    plot_profiles(     phase, fluid,     grid, vis; time, path = joinpath(dir, "$(tag)_prf.png"))
    plot_history(      hst,              vis;        path = joinpath(dir, "$(runID)_hst.png"))

    return nothing
end
