@testset "ddx, ddz" begin
    for T in (Float64, Float32)
        L = T(2π)

        # ---- ddx on a sinusoid: cell-centred difference of sin(x) at faces
        # gives cos(xc)*sin(h/2)/(h/2) → cos(xc) with O(h^2) error.
        errs_x = T[]
        for Nx in (32, 64, 128)
            h  = L / Nx
            xf = collect(T(0):h:L)                          # length Nx+1
            xc = (xf[1:end-1] .+ xf[2:end]) ./ T(2)
            f  = reshape(sin.(xf), 1, Nx + 1)               # one z-row
            out = similar(f, 1, Nx)
            ddx!(out, f, h)
            push!(errs_x, sqrt(sum((vec(out) .- cos.(xc)).^2) / Nx))
        end
        # observed convergence order: log2(err[1]/err[2]) ≈ 2
        @test log2(errs_x[1] / errs_x[2]) > 1.9
        @test log2(errs_x[2] / errs_x[3]) > 1.9

        # ---- ddz mirror test
        errs_z = T[]
        for Nz in (32, 64, 128)
            h  = L / Nz
            zf = collect(T(0):h:L)
            zc = (zf[1:end-1] .+ zf[2:end]) ./ T(2)
            f  = reshape(sin.(zf), Nz + 1, 1)
            out = similar(f, Nz, 1)
            ddz!(out, f, h)
            push!(errs_z, sqrt(sum((vec(out) .- cos.(zc)).^2) / Nz))
        end
        @test log2(errs_z[1] / errs_z[2]) > 1.9
        @test log2(errs_z[2] / errs_z[3]) > 1.9
    end
end

@testset "fill_ghosts! periodic+closed" begin
    f = Float64[
        1 2 3 4 ;
        5 6 7 8 ;
        9 10 11 12 ;
        13 14 15 16 ;
    ]                                                       # 4x4 interior
    halo = 1
    fh = zeros(Float64, 4 + 2halo, 4 + 2halo)
    embed_interior!(fh, f, halo)
    fill_ghosts!(fh, halo; zBC = :closed, xBC = :periodic)

    # periodic in x: leftmost ghost column equals rightmost interior column
    @test fh[2:5, 1]   == fh[2:5, 5]                        # column 1 ← column 5 (interior end)
    @test fh[2:5, end] == fh[2:5, 2]                        # column end ← column 2 (interior start)

    # closed in z: top/bot ghost rows repeat the first/last interior row
    @test fh[1, 2:5]   == fh[2, 2:5]
    @test fh[end, 2:5] == fh[end-1, 2:5]
end
