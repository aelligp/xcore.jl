@testset "advect! :centr — 2nd-order convergence on a cosine" begin
    # f(x,z) = cos(k x) cos(k z), constant velocity (U0, W0).
    # div(v f) = -U0 k sin(k x) cos(k z) - W0 k cos(k x) sin(k z).
    for T in (Float64, Float32)
        L = T(2π); k = T(1)
        U0 = T(0.7);  W0 = T(-0.3)
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

            u = fill(U0, N, N + 1)                          # constant U on x-faces
            w = fill(W0, N + 1, N)                          # constant W on z-faces

            adv = zeros(T, N, N)
            advect!(adv, f_int, (w, u), h, :centr; xBC = :periodic, zBC = :periodic)

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
        errs = T[]
        for N in (32, 64, 128)
            h  = L / N
            xc = collect(((1:N) .- T(0.5)) .* h)
            zc = collect(((1:N) .- T(0.5)) .* h)
            XX = repeat(reshape(xc, 1, N), N, 1)
            ZZ = repeat(reshape(zc, N, 1), 1, N)

            f_int  = cos.(k .* XX) .* cos.(k .* ZZ)

            u = fill(U0, N, N + 1)
            w = fill(W0, N + 1, N)

            adv = zeros(T, N, N)
            advect!(adv, f_int, (w, u), h, :weno5; xBC = :periodic, zBC = :periodic)

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

# 3-D fields constant along one horizontal axis, with zero velocity along it,
# reproduce the 2-D result on every slice: along y ⇒ the (z, x) problem, along
# x ⇒ the (z, y) problem.

@testset "advect_with_flux! 3-D reduces to 2-D on invariant fields" begin
    lift_y(A, n) = repeat(A, 1, 1, n)
    lift_x(A, n) = repeat(reshape(A, size(A, 1), 1, size(A, 2)), 1, n, 1)
    Nz, Nh, Nt = 9, 12, 7                     # z, varying horizontal, invariant axis
    f = rand(Nz, Nh);  w = randn(Nz + 1, Nh);  u = randn(Nz, Nh + 1);  h = 0.13
    for s in xcore.ADVN_SCHEMES, hBC in (:periodic, :closed), tBC in (:periodic, :closed),
        zBC in (:periodic, :closed)
        adv2 = zeros(Nz, Nh);  q2 = (zeros(Nz + 1, Nh + 2), zeros(Nz + 2, Nh + 1))
        advect_with_flux!(adv2, q2, f, (w, u), h, s; xBC = hBC, zBC)

        q3 = (zeros(Nz + 1, Nh + 2, Nt + 2), zeros(Nz + 2, Nh + 1, Nt + 2), zeros(Nz + 2, Nh + 2, Nt + 1))
        adv3 = zeros(Nz, Nh, Nt)
        advect_with_flux!(adv3, q3, lift_y(f, Nt), (lift_y(w, Nt), lift_y(u, Nt), zeros(Nz, Nh, Nt + 1)),
                          h, s; xBC = hBC, yBC = tBC, zBC)
        @test adv3 == lift_y(adv2, Nt)
        @test q3[1] == lift_y(q2[1], Nt + 2) && q3[2] == lift_y(q2[2], Nt + 2) && all(iszero, q3[3])

        q3 = (zeros(Nz + 1, Nt + 2, Nh + 2), zeros(Nz + 2, Nt + 1, Nh + 2), zeros(Nz + 2, Nt + 2, Nh + 1))
        adv3 = zeros(Nz, Nt, Nh)
        advect_with_flux!(adv3, q3, lift_x(f, Nt), (lift_x(w, Nt), zeros(Nz, Nt + 1, Nh), lift_x(u, Nt)),
                          h, s; xBC = tBC, yBC = hBC, zBC)
        @test adv3 == lift_x(adv2, Nt)
        @test q3[1] == lift_x(q2[1], Nt + 2) && all(iszero, q3[2]) && q3[3] == lift_x(q2[2], Nt + 2)
    end
end

@testset "advect! 3-D convergence on cos·cos·cos" begin
    # div(v f) for constant v: Σ_d v_d ∂_d f
    k = 1.0;  v0 = (-0.3, 0.7, 0.5)
    for (s, order) in ((:centr, 1.9), (:weno5, 3.5))
        errs = Float64[]
        for N in (16, 32, 64)
            h = 2π / N;  c = ((1:N) .- 0.5) .* h
            Z = reshape(c, N, 1, 1);  X = reshape(c, 1, N, 1);  Y = reshape(c, 1, 1, N)
            f = cos.(k .* Z) .* cos.(k .* X) .* cos.(k .* Y)
            v = ntuple(d -> fill(v0[d], ntuple(m -> N + (m == d), 3)), 3)
            adv = zeros(N, N, N)
            advect!(adv, f, v, h, s; xBC = :periodic, yBC = :periodic, zBC = :periodic)
            exact = -k .* (v0[1] .* sin.(k .* Z) .* cos.(k .* X) .* cos.(k .* Y) .+
                           v0[2] .* cos.(k .* Z) .* sin.(k .* X) .* cos.(k .* Y) .+
                           v0[3] .* cos.(k .* Z) .* cos.(k .* X) .* sin.(k .* Y))
            push!(errs, sqrt(sum(abs2, adv .- exact) / N^3))
        end
        @test log2(errs[1] / errs[2]) > order
        @test log2(errs[2] / errs[3]) > order
    end
end

@testset "advect! rejects mismatched shapes and BCs" begin
    f = zeros(4, 5, 6);  adv = similar(f)
    v = (zeros(5, 5, 6), zeros(4, 6, 6), zeros(4, 5, 7))
    @test_throws "v[d] of size N .+ e_d" advect!(adv, f, v[1:2], 1.0, :centr)
    @test_throws "unsupported BC = :open" advect!(adv, f, v, 1.0, :centr; yBC = :open)
    @test_throws "unsupported scheme" advect!(adv, f, v, 1.0, :lax)
end
