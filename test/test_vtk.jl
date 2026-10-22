# ParaView output: fields read back with ReadVTK must land at the matching
# (x[, y], elevation) location, with elevation = -z, as cell or point data.

using Test
using xcore
using KernelAbstractions: CPU
import ReadVTK

# VTK index of the xcore index I = (iz, ix[, iy]) on an axis of n entries in z.
vtk_index(I, n) = (Tuple(I)[2:end]..., n + 1 - I[1])
reshaped(data, name; cell) = ReadVTK.get_data_reshaped(data[name]; cell_data = cell)

@testset "save_vtk on a vertex grid" begin
    dir = mktempdir()
    xvi = (collect(0:0.5:1.5), collect(-1.0:0.25:0.0))
    xci = map(x -> (x[1:end-1] .+ x[2:end]) ./ 2, xvi)
    A = reshape(collect(1.0:prod(length.(xci))), length.(xci))
    B = reshape(collect(1.0:prod(length.(xvi))), length.(xvi))
    vel = (B, -B)
    for (k, t) in enumerate((0.0, 0.5))
        save_vtk(joinpath(dir, "f$k"), xvi, xci, (; B), (; A), vel; t, pvd = joinpath(dir, "series"))
    end
    vtk = ReadVTK.VTKFile(joinpath(dir, "f2.vtr"))
    @test ReadVTK.get_coordinates(vtk)[1:2] == xvi
    @test reshaped(ReadVTK.get_cell_data(vtk), "A"; cell = true)[:, :, 1] == A
    points = ReadVTK.get_point_data(vtk)
    @test reshaped(points, "B"; cell = false)[:, :, 1] == B
    V = reshaped(points, "Velocity"; cell = false)
    @test V[1, :, :, 1] == B && V[2, :, :, 1] == -B && all(iszero, V[3, :, :, :])
    @test ReadVTK.get_data(ReadVTK.get_field_data(vtk)["TimeValue"]) == [0.5]
    pvd = ReadVTK.PVDFile(joinpath(dir, "series.pvd"))
    @test pvd.timesteps == [0.0, 0.5] && pvd.vtk_filenames == ["f1.vtr", "f2.vtr"]

    @test_throws "matches neither" save_vtk(joinpath(dir, "bad"), xvi, xci, (;), (; A = A[1:end-1, :]), vel)
    @test_throws "velocity must be given on the vertices" save_vtk(joinpath(dir, "bad"), xvi, xci, (;), (; A), (A, A))
    @test !isfile(joinpath(dir, "bad.vtr"))
end

@testset "save_vtk of the model state" begin
    T = Float64
    par = Parameters(T; N = 4, L = 1.5, D = 1.0)
    scales = compute_scales(par, Grid(par))
    dir = mktempdir()
    for grid in (Grid(par), Grid(par, Val(3)))
        N = cells(grid);  D = length(N)
        fluid = FluidState(T, CPU(), grid);  phase = PhaseState(T, CPU(), grid)
        ns = NoiseState(T, CPU(), grid, scales)
        # Face velocity d is its face coordinate along d, constant along the
        # other axes (ghosts included), so its vertex average is that coordinate.
        for (d, v) in enumerate(xcore.velocity(fluid))
            v .= reshape(grid.xvi[d], ntuple(k -> k == d ? N[d] + 1 : 1, D))
        end
        phase.wx .= fluid.W;  phase.wm .= 2 .* fluid.W
        fluid.rho .= reshape(1:prod(N), N)
        save_vtk(joinpath(dir, "s$(D)d"), phase, fluid, ns, grid; t = 1.5)

        vtk = ReadVTK.VTKFile(joinpath(dir, "s$(D)d.vtr"))
        coords = ReadVTK.get_coordinates(vtk)
        @test coords[1] ≈ grid.xvi[2] && coords[D] ≈ -reverse(grid.xvi[1])
        D == 3 && @test coords[2] ≈ grid.xvi[3]
        cells_ = ReadVTK.get_cell_data(vtk);  points = ReadVTK.get_point_data(vtk)
        @test issubset(["P", "rho", "eta", "MFS", "x", "etas", "ks", "kx", "ke",
                        "xie", "xix", "xis", "ReD", "Ra", "Rc"], keys(cells_))
        rho = reshaped(cells_, "rho"; cell = true)
        V = reshaped(points, "Velocity"; cell = false)
        wx = reshaped(points, "wx"; cell = false);  wm = reshaped(points, "wm"; cell = false)
        for I in CartesianIndices(N)
            @test rho[vtk_index(I, N[1])..., ntuple(_ -> 1, 3 - D)...] == fluid.rho[I]
        end
        for I in CartesianIndices(N .+ 1)
            J = (vtk_index(I, N[1] + 1)..., ntuple(_ -> 1, 3 - D)...)
            @test V[1:D-1, J...] ≈ [grid.xvi[d][I[d]] for d in 2:D]
            @test V[D, J...] ≈ -grid.xvi[1][I[1]]
            D == 2 && @test V[3, J...] == 0
            @test wx[J...] ≈ -grid.xvi[1][I[1]] && wm[J...] ≈ -2grid.xvi[1][I[1]]
        end
    end
end

@testset "save_output writes a ParaView time series" begin
    T = Float64
    dir = mktempdir()
    par = Parameters(T; N = 4, L = 1.0, D = 1.0, plot_op = false, vtk_op = true)
    scales = compute_scales(par, Grid(par))
    for grid in (Grid(par), Grid(par, Val(3)))
        D = length(cells(grid));  runID = "r$(D)d"
        fluid = FluidState(T, CPU(), grid);  phase = PhaseState(T, CPU(), grid)
        ns = NoiseState(T, CPU(), grid, scales);  hst = History(T)
        for (frame, time) in enumerate((0.25, 0.75))
            save_output(phase, fluid, ns, hst, grid, par, scales, time;
                        outdir = dir, runID, frame, checkpoint = false)
        end
        pvd = ReadVTK.PVDFile(joinpath(dir, runID, "$runID.pvd"))
        @test pvd.timesteps == [0.25, 0.75]
        @test pvd.vtk_filenames == ["$(runID)_0001.vtr", "$(runID)_0002.vtr"]
        D == 3 && @test_throws "figures of par.plot_op = true are 2-D" save_output(
            phase, fluid, ns, hst, grid, Parameters(T; N = 4, L = 1.0, D = 1.0, plot_op = true),
            scales, 0.0; outdir = dir, runID, checkpoint = false)
    end
end
