@testset "diffus! 2nd-order convergence on sin(kx)sin(kz)" begin
    # f(x,z) = sin(k x) sin(k z), -Δf = 2 k^2 f
    for T in (Float64, Float32)
        L = T(2π)
        k = T(1)                                            # k = 1 => one period over [0, 2π]
        errs = T[]
        for N in (32, 64, 128)
            h  = L / N
            xc = collect(((1:N) .- T(0.5)) .* h)
            zc = collect(((1:N) .- T(0.5)) .* h)
            XX = repeat(reshape(xc, 1, N), N, 1)
            ZZ = repeat(reshape(zc, N, 1), 1, N)

            f_int  = sin.(k .* XX) .* sin.(k .* ZZ)
            k_int  = ones(T, N, N)

            dff = zeros(T, N, N)
            diffus!(dff, f_int, k_int, h; xBC = :periodic, zBC = :periodic)

            # diffus! returns +(-div(-k grad f)) = + Δf for constant k=1
            # so it should approximate Δf = -2 k^2 f.
            exact = -T(2) * k^2 .* f_int
            push!(errs, sqrt(sum((dff .- exact).^2) / (N*N)))
        end
        @test log2(errs[1] / errs[2]) > 1.9
        @test log2(errs[2] / errs[3]) > 1.9
    end
end

@testset "diffus_with_flux! conserves mass and dff = -div(q)" begin
    # Edge-repeat ghosts make the closed boundary faces flux-free, and periodic
    # faces cancel, so the total diffusive source vanishes for every BC pair.
    f = rand(Float64, 12, 16);  k = rand(Float64, 12, 16);  h = 0.1
    for (xBC, zBC) in ((:periodic, :closed), (:closed, :closed), (:periodic, :periodic))
        dff = zeros(12, 16);  qx = zeros(14, 17);  qz = zeros(13, 18)
        diffus_with_flux!(dff, qx, qz, f, k, h; xBC, zBC)
        @test abs(sum(dff)) < 1e-10 * sum(abs, dff)
        divq = (diff(qx[2:end-1, :]; dims = 2) .+ diff(qz[:, 2:end-1]; dims = 1)) ./ h
        @test dff ≈ -divq
        zBC === :closed && @test all(iszero, qz[[1, end], :])
        xBC === :closed && @test all(iszero, qx[:, [1, end]])
    end
end
