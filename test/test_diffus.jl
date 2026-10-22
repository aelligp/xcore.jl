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
        diffus_with_flux!(dff, (qz, qx), f, k, h; xBC, zBC)
        @test abs(sum(dff)) < 1e-10 * sum(abs, dff)
        divq = (diff(qx[2:end-1, :]; dims = 2) .+ diff(qz[:, 2:end-1]; dims = 1)) ./ h
        @test dff ≈ -divq
        zBC === :closed && @test all(iszero, qz[[1, end], :])
        xBC === :closed && @test all(iszero, qx[:, [1, end]])
    end
end

# 3-D fields constant along one horizontal axis reproduce the 2-D result on
# every slice (see the advection counterpart)
@testset "diffus_with_flux! 3-D reduces to 2-D on invariant fields" begin
    Nz, Nh, Nt = 9, 12, 7
    lift_y(A, n) = repeat(A, 1, 1, n)
    lift_x(A, n) = repeat(reshape(A, size(A, 1), 1, size(A, 2)), 1, n, 1)
    f = rand(Nz, Nh);  k = rand(Nz, Nh) .+ 0.5;  h = 0.13
    for hBC in (:periodic, :closed), tBC in (:periodic, :closed), zBC in (:periodic, :closed)
        dff2 = zeros(Nz, Nh);  q2 = (zeros(Nz + 1, Nh + 2), zeros(Nz + 2, Nh + 1))
        diffus_with_flux!(dff2, q2, f, k, h; xBC = hBC, zBC)

        q3 = (zeros(Nz + 1, Nh + 2, Nt + 2), zeros(Nz + 2, Nh + 1, Nt + 2), zeros(Nz + 2, Nh + 2, Nt + 1))
        dff3 = zeros(Nz, Nh, Nt)
        diffus_with_flux!(dff3, q3, lift_y(f, Nt), lift_y(k, Nt), h; xBC = hBC, yBC = tBC, zBC)
        @test dff3 == lift_y(dff2, Nt)
        @test q3[1] == lift_y(q2[1], Nt + 2) && q3[2] == lift_y(q2[2], Nt + 2) && all(iszero, q3[3])

        q3 = (zeros(Nz + 1, Nt + 2, Nh + 2), zeros(Nz + 2, Nt + 1, Nh + 2), zeros(Nz + 2, Nt + 2, Nh + 1))
        dff3 = zeros(Nz, Nt, Nh)
        diffus_with_flux!(dff3, q3, lift_x(f, Nt), lift_x(k, Nt), h; xBC = tBC, yBC = hBC, zBC)
        @test dff3 == lift_x(dff2, Nt)
        @test q3[1] == lift_x(q2[1], Nt + 2) && all(iszero, q3[2]) && q3[3] == lift_x(q2[2], Nt + 2)
    end
end

@testset "diffus! 3-D convergence and conservation" begin
    # f = sin(z) sin(x) sin(y), k = 1 ⇒ Δf = -3f
    errs = Float64[]
    for N in (16, 32, 64)
        h = 2π / N;  c = ((1:N) .- 0.5) .* h
        f = sin.(reshape(c, N, 1, 1)) .* sin.(reshape(c, 1, N, 1)) .* sin.(reshape(c, 1, 1, N))
        dff = zeros(N, N, N)
        diffus!(dff, f, ones(N, N, N), h; xBC = :periodic, yBC = :periodic, zBC = :periodic)
        push!(errs, sqrt(sum(abs2, dff .+ 3 .* f) / N^3))
    end
    @test log2(errs[1] / errs[2]) > 1.9
    @test log2(errs[2] / errs[3]) > 1.9

    Nz, Nx, Ny = 6, 8, 5;  h = 0.1
    f = rand(Nz, Nx, Ny);  k = rand(Nz, Nx, Ny)
    for (xBC, yBC) in ((:periodic, :periodic), (:closed, :periodic), (:closed, :closed))
        dff = zeros(Nz, Nx, Ny)
        q = (zeros(Nz + 1, Nx + 2, Ny + 2), zeros(Nz + 2, Nx + 1, Ny + 2), zeros(Nz + 2, Nx + 2, Ny + 1))
        diffus_with_flux!(dff, q, f, k, h; xBC, yBC, zBC = :closed)
        @test abs(sum(dff)) < 1e-10 * sum(abs, dff)
        qi = ntuple(d -> view(q[d], ntuple(m -> m == d ? Colon() : 2:size(q[d], m) - 1, 3)...), 3)
        @test dff ≈ -sum(d -> diff(qi[d]; dims = d), 1:3) ./ h
        @test all(iszero, selectdim(q[1], 1, 1)) && all(iszero, selectdim(q[1], 1, Nz + 1))
        xBC === :closed && @test all(iszero, selectdim(q[2], 2, 1)) && all(iszero, selectdim(q[2], 2, Nx + 1))
        yBC === :closed && @test all(iszero, selectdim(q[3], 3, 1)) && all(iszero, selectdim(q[3], 3, Ny + 1))
    end
end
