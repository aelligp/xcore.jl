@testset "Grid" begin
    par = Parameters{Float64}(D = 10.0, N = 100, L = 10.0)
    g = Grid(par)

    @test g.ni[1] == 100
    @test g.ni[2] == 100
    @test g.h  == 0.1

    # cell centres at h/2, 3h/2, ..., (N-1/2)h; matches MATLAB Xc(2:end-1) range
    @test g.xci[2][1]   ≈ g.h/2
    @test g.xci[2][end] ≈ g.li[2] - g.h/2

    @test length(g.ni) == length(g.li) == length(g.xci) == length(g.xvi) == 2
    @test cells(g) == (100, 100)

    # Float32 grid
    par32 = Parameters{Float32}(D = 10, N = 50, L = 15)
    g32 = Grid(par32)
    @test g32.h isa Float32
    @test g32.ni[2] == 75

    # 3-D grid: square horizontal footprint, y-coordinates as x
    g3 = Grid(par32, Val(3))
    @test g3 isa Grid{Float32, 3}
    @test cells(g3) == (50, 75, 75)
    @test g3.xci[3] == g3.xci[2] && g3.xvi[3] == g3.xvi[2]

    # JustRelax-style constructor: the dimension is the tuple length
    gn = Grid((4, 6, 2), (1, 1.5, 0.5))
    @test gn isa Grid{Float64, 3} && gn.h == 0.25 && gn.li == (1.0, 1.5, 0.5)
    @test map(length, gn.xci) == (4, 6, 2) && map(length, gn.xvi) == (5, 7, 3)
    @test_throws "cells must be square" Grid((4, 4), (1.0, 2.0))
    @test_throws "need 2 or 3 dimensions" Grid((4,), (1.0,))
end
