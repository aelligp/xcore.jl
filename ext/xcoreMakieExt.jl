module xcoreMakieExt

using xcore
using xcore: FluidState, PhaseState, Grid
using Makie

# helper: largest absolute value, with a tiny floor so zero-valued fields
# don't blow up the colorrange computation.
maxabs(A) = max(maximum(abs, A), eps(Float64))

# place a heatmap of `data` into a Makie cell with its own colorbar; `sym`
# toggles symmetric divergent colormap centred on zero.
function placepanel!(fig, row, col, name, data; sym = false, cmap = :lapaz)
    ax = Axis(fig[row, col]; title = name, xlabel = "x", ylabel = "z",
              yreversed = true, aspect = DataAspect())
    if sym
        r = maxabs(data)
        hm = heatmap!(ax, permutedims(Float64.(data));
                      colormap = :balance, colorrange = (-r, r))
    else
        hm = heatmap!(ax, permutedims(Float64.(data)); colormap = cmap)
    end
    Colorbar(fig[row, col + 1], hm)
    return ax, hm
end

function xcore.plot_state(fluid::FluidState, grid::Grid;
                          phase::Union{Nothing, PhaseState} = nothing,
                          path::Union{Nothing, AbstractString} = nothing,
                          title::AbstractString = "xcore state")
    # interior-only views drop the ghost rings so the plots show only
    # physically meaningful cells (matching the MATLAB visualisation).
    Wp = @view fluid.W[:, 2:end-1]            # (Nz+1, Nx)
    Up = @view fluid.U[2:end-1, :]            # (Nz, Nx+1)
    Pp = @view fluid.P[2:end-1, 2:end-1]      # (Nz, Nx)

    nrows = phase === nothing ? 2 : 3
    fig   = Figure(size = (1400, 380 * nrows))
    Label(fig[0, 1:6], title; fontsize = 18, halign = :center)

    placepanel!(fig, 1, 1, "W (z-vel)",   Wp;  sym = true)
    placepanel!(fig, 1, 3, "U (x-vel)",   Up;  sym = true)
    placepanel!(fig, 1, 5, "P",           Pp;  sym = true)
    placepanel!(fig, 2, 1, "η",           fluid.eta;  cmap = :lapaz)
    placepanel!(fig, 2, 3, "ρ",           fluid.rho;  cmap = :lapaz)
    placepanel!(fig, 2, 5, "η_corner",    fluid.etaco; cmap = :lapaz)

    if phase !== nothing
        placepanel!(fig, 3, 1, "χ (xtal)",   phase.chi;    cmap = :lapaz)
        placepanel!(fig, 3, 3, "ηmix",       phase.etamix; cmap = :lapaz)
        placepanel!(fig, 3, 5, "eII",        phase.eII;    cmap = :lapaz)
    end

    if path !== nothing
        mkpath(dirname(abspath(path)))
        save(path, fig)
    end
    return fig
end

function xcore.plot_mms_comparison(fluid::FluidState, mms::NamedTuple, grid::Grid;
                                   path::Union{Nothing, AbstractString} = nothing,
                                   title::AbstractString = "MMS Stokes comparison")
    fig = Figure(size = (1600, 1200))
    Label(fig[0, 1:6], "$(title) — N = $(grid.Nz)"; fontsize = 18, halign = :center)

    # row pattern: (name, numerical, exact)
    rows = (
        ("W", fluid.W, mms.W_exact),
        ("U", fluid.U, mms.U_exact),
        ("P", fluid.P, mms.P_exact),
    )

    for (irow, (name, num, exact)) in enumerate(rows)
        r_field = maxabs(exact)
        r_err   = max(maxabs(num .- exact), eps(Float64))

        ax1 = Axis(fig[irow, 1]; title = "$name numerical", yreversed = true, aspect = DataAspect())
        hm1 = heatmap!(ax1, permutedims(Float64.(num));
                       colormap = :balance, colorrange = (-r_field, r_field))
        Colorbar(fig[irow, 2], hm1)

        ax2 = Axis(fig[irow, 3]; title = "$name exact", yreversed = true, aspect = DataAspect())
        hm2 = heatmap!(ax2, permutedims(Float64.(exact));
                       colormap = :balance, colorrange = (-r_field, r_field))
        Colorbar(fig[irow, 4], hm2)

        ax3 = Axis(fig[irow, 5]; title = "$name error", yreversed = true, aspect = DataAspect())
        hm3 = heatmap!(ax3, permutedims(Float64.(num .- exact));
                       colormap = :balance, colorrange = (-r_err, r_err))
        Colorbar(fig[irow, 6], hm3)
    end

    if path !== nothing
        mkpath(dirname(abspath(path)))
        save(path, fig)
    end
    return fig
end

end # module XCoreMakieExt
