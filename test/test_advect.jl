@testset "advect! :centr — 2nd-order convergence on a cosine" begin
    # f(x,z) = cos(k x) cos(k z), constant velocity (U0, W0).
    # div(v f) = -U0 k sin(k x) cos(k z) - W0 k cos(k x) sin(k z).
    for T in (Float64, Float32)
        L = T(2π); k = T(1)
        U0 = T(0.7);  W0 = T(-0.3)
        halo = 1
        errs = T[]
        for N in (32, 64, 128)
            h  = L / N
            xc = collect(((1:N) .- T(0.5)) .* h)
            zc = collect(((1:N) .- T(0.5)) .* h)
            xf = collect((0:N) .* h)
            zf = collect((0:N) .* h)
            XX = repeat(reshape(xc, 1, N), N, 1)
            ZZ = repeat(reshape(zc, N, 1), 1, N)

            f_int  = cos.(k .* XX) .* cos.(k .* ZZ)
            f_halo = zeros(T, N + 2halo, N + 2halo)
            embed_interior!(f_halo, f_int, halo)
            fill_ghosts!(f_halo, halo; zBC = :periodic, xBC = :periodic)

            u = fill(U0, N, N + 1)                          # constant U on x-faces
            w = fill(W0, N + 1, N)                          # constant W on z-faces

            adv = zeros(T, N, N)
            advect!(adv, f_halo, u, w, h, :centr)

            exact = .-U0 .* k .* sin.(k .* XX) .* cos.(k .* ZZ) .+
                     W0 .* k .* cos.(k .* XX) .* (.-sin.(k .* ZZ))
            push!(errs, sqrt(sum((adv .- exact).^2) / (N*N)))
        end
        @test log2(errs[1] / errs[2]) > 1.9
        @test log2(errs[2] / errs[3]) > 1.9
    end
end

@testset "advect! :weno5 — high-order convergence on a smooth cosine" begin
    # Same setup; WENO5 should hit ~5th order on smooth, well-resolved data.
    for T in (Float64, Float32)
        L = T(2π); k = T(1)
        U0 = T(0.7);  W0 = T(-0.3)
        halo = 3
        errs = T[]
        for N in (32, 64, 128)
            h  = L / N
            xc = collect(((1:N) .- T(0.5)) .* h)
            zc = collect(((1:N) .- T(0.5)) .* h)
            XX = repeat(reshape(xc, 1, N), N, 1)
            ZZ = repeat(reshape(zc, N, 1), 1, N)

            f_int  = cos.(k .* XX) .* cos.(k .* ZZ)
            f_halo = zeros(T, N + 2halo, N + 2halo)
            embed_interior!(f_halo, f_int, halo)
            fill_ghosts!(f_halo, halo; zBC = :periodic, xBC = :periodic)

            u = fill(U0, N, N + 1)
            w = fill(W0, N + 1, N)

            adv = zeros(T, N, N)
            advect!(adv, f_halo, u, w, h, :weno5)

            exact = .-U0 .* k .* sin.(k .* XX) .* cos.(k .* ZZ) .+
                     W0 .* k .* cos.(k .* XX) .* (.-sin.(k .* ZZ))
            push!(errs, sqrt(sum((adv .- exact).^2) / (N*N)))
        end
        # WENO5 converges fast on smooth data. For Float32 the error saturates
        # near machine precision by N=128, so the 64→128 ratio breaks down;
        # we assert order in the unsaturated regime and absolute magnitude at
        # the finest grid instead.
        @test log2(errs[1] / errs[2]) > 4.0
        if T === Float64
            @test log2(errs[2] / errs[3]) > 3.5
        else
            @test errs[3] < T(1e-5)
        end
    end
end
