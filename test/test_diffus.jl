@testset "diffus! 2nd-order convergence on sin(kx)sin(kz)" begin
    # f(x,z) = sin(k x) sin(k z), -Δf = 2 k^2 f
    for T in (Float64, Float32)
        L = T(2π)
        k = T(1)                                            # k = 1 => one period over [0, 2π]
        halo = 1
        errs = T[]
        for N in (32, 64, 128)
            h  = L / N
            xc = collect(((1:N) .- T(0.5)) .* h)
            zc = collect(((1:N) .- T(0.5)) .* h)
            XX = repeat(reshape(xc, 1, N), N, 1)
            ZZ = repeat(reshape(zc, N, 1), 1, N)

            f_int  = sin.(k .* XX) .* sin.(k .* ZZ)
            k_int  = ones(T, N, N)
            f_halo = zeros(T, N + 2halo, N + 2halo)
            k_halo = zeros(T, N + 2halo, N + 2halo)
            embed_interior!(f_halo, f_int, halo)
            embed_interior!(k_halo, k_int, halo)
            fill_ghosts!(f_halo, halo; zBC = :periodic, xBC = :periodic)
            fill_ghosts!(k_halo, halo; zBC = :periodic, xBC = :periodic)

            dff = zeros(T, N, N)
            diffus!(dff, f_halo, k_halo, h; halo = halo)

            # diffus! returns +(-div(-k grad f)) = + Δf for constant k=1
            # so it should approximate Δf = -2 k^2 f.
            exact = -T(2) * k^2 .* f_int
            push!(errs, sqrt(sum((dff .- exact).^2) / (N*N)))
        end
        @test log2(errs[1] / errs[2]) > 1.9
        @test log2(errs[2] / errs[3]) > 1.9
    end
end
