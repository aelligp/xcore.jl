@testset "Grid" begin
    par = Parameters{Float64}(D = 10.0, N = 100, L = 10.0)
    g = Grid(par)

    @test g.Nz == 100
    @test g.Nx == 100
    @test g.h  == 0.1

    # cell centres at h/2, 3h/2, ..., (N-1/2)h; matches MATLAB Xc(2:end-1) range
    @test g.Xc[1]   ≈ g.h/2
    @test g.Xc[end] ≈ g.L - g.h/2

    # meshgrid: leading axis z, trailing x — pulled to verify Pa/MATLAB convention
    @test size(g.XX) == (100, 100)
    @test g.XX[1, end] ≈ g.Xc[end]
    @test g.ZZ[end, 1] ≈ g.Zc[end]

    # ghost indices reproduce MATLAB wrap-around
    @test g.icx[1]   == 100 && g.icx[end] == 1
    @test g.icz[1]   == 1   && g.icz[end] == 100   # closed-z (top/bot)
    @test length(g.icx) == 102
    @test length(g.ifx) == 103

    # Float32 grid
    par32 = Parameters{Float32}(D = 10, N = 50, L = 15)
    g32 = Grid(par32)
    @test g32.h isa Float32
    @test g32.Nx == 75
end
