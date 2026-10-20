using Test, Random, LinearAlgebra
using xcore
import xcore as XC
using KernelAbstractions: CPU

# Three-dimensional instance of the dimension-generic matrix-free solvers.

const S3 = XC.Stokes

function _stokes3d_base(N; atol_PH = 1.0e-7)
    T = Float64
    par = Parameters(T; D = 1.0, N, L = 1.0, etam0 = 1.0, g0 = 1.0, alpha = 0.0,
                     gamma = 0.0, atol_PH)
    grid = Grid3D(par)
    fluid = FluidState3D(T, CPU(), grid)
    return fluid, PhaseState3D(T, CPU(), grid), grid, par, compute_scales(par, Grid(par))
end

# Discretely divergence-free flow: v = curl of two edge potentials that vanish at
# z = 0 and z = D. Lagged momentum forcing built from its residual makes
# (v, P = c) the exact discrete solution for any constant c.
function _stokes3d_mms(N, c; dt = 10.0)
    fluid, phase, grid, par, scales = _stokes3d_base(N)
    Nz, Nx, Ny, h = grid.Nz, grid.Nx, grid.Ny, grid.h
    fluid.eta .= [exp(0.5 * sinpi(2 * (i - 0.5) * h) * cospi((j - 0.5) * h))
                  for j in 1:Nz, i in 1:Nx, k in 1:Ny]
    fill!(fluid.rho, 1)
    kw = (; dt, a1 = 1, a2 = 1, a3 = 0, gamma = 0, top_cnv = -1, bot_cnv = -1)
    op = S3.StokesOp(Float64, Val(3), par; kw...)
    S3.derived_coefficients!(fluid, op)
    fill!(fluid.P, c)
    ψ1 = [1.0e-4 * sinpi((j - 1) * h) * sinpi(2 * (i - 1) * h) * cospi(2 * (k - 0.5) * h)
          for j in 1:Nz+1, i in 1:Nx+1, k in 1:Ny]
    ψ2 = [1.0e-4 * sinpi((j - 1) * h) * cospi(2 * (i - 0.5) * h) * sinpi(2 * (k - 1) * h)
          for j in 1:Nz+1, i in 1:Nx, k in 1:Ny+1]
    fluid.W[:, 2:end-1, 2:end-1] .= diff(ψ1; dims = 2) ./ h .+ diff(ψ2; dims = 3) ./ h
    fluid.U[2:end-1, :, 2:end-1] .= .-diff(ψ1; dims = 1) ./ h
    fluid.V[2:end-1, 2:end-1, :] .= .-diff(ψ2; dims = 1) ./ h
    S3.apply_velocity_bcs!(fluid, grid, op)
    cache = S3.StokesCache(Float64, CPU(), grid)
    S3.residual_momentum!(cache, fluid, grid, op; dr = false)
    fluid.rhoWo .= .-cache.Rv[1] .* dt
    fluid.rhoUo .= .-cache.Rv[2] .* dt
    fluid.rhoVo .= .-cache.Rv[3] .* dt
    vex = copy.((fluid.W, fluid.U, fluid.V))
    foreach(A -> fill!(A, 0), (fluid.W, fluid.U, fluid.V))
    return fluid, phase, grid, par, scales, vex, kw
end

function _stokes3d_solve(kind, fluid, phase, grid, par, scales; kw...)
    if kind === :dyrel
        return S3.fluidmech_dyrel!(fluid, phase, S3.StokesCache(Float64, CPU(), grid),
                                   grid, par, scales; kw...)
    elseif kind === :gmg
        return S3.fluidmech_gmg!(fluid, phase, S3.build_hierarchy(fluid, grid, par),
                                 grid, par, scales; kw...)
    end
    hier = S3.build_hierarchy(fluid, grid, par; inner = 16, npre = 1, npost = 1)
    return S3.fluidmech_vmg!(fluid, phase, hier, grid, par, scales; cc = true, kw...)
end

@testset "3-D kernels reduce to 2-D and are symmetric under x ↔ y" begin
    Random.seed!(3)
    T = Float64;  Nz, Nx, Ny = 8, 6, 4
    par = Parameters(T; D = 1.0, N = Nz, L = Nx / Nz, g0 = 9.81)
    g2 = Grid(par);  h = g2.h
    g3 = XC._grid3d_at(T, Nz, Nx, Ny, h, 1.0, 1.0)
    kw = (; a1 = 1.3, a2 = -0.4, a3 = 0.1, gamma = 0.7, dt = 0.05, top_cnv = -1, bot_cnv = 1)
    rep(A, n) = repeat(A, 1, 1, n)
    for xBC in (:periodic, :closed)
        op2 = S3.StokesOp(T, Val(2), par; kw..., xBC, sds = 1)
        op3 = S3.StokesOp(T, Val(3), par; kw..., xBC, yBC = :periodic, sds = 1)
        # a y-uniform 3-D state with V = 0 is the 2-D state on every y-slice
        f2 = FluidState(T, CPU(), Nz, Nx)
        for A in (f2.W, f2.U, f2.P, f2.Drho, f2.MFS, f2.rhoWo, f2.rhoUo); A .= randn(T, size(A)); end
        f2.eta .= exp.(randn(T, Nz, Nx));  f2.rho .= 1 .+ 0.3 .* rand(T, Nz, Nx)
        S3.derived_coefficients!(f2, op2)
        f3 = FluidState3D(T, CPU(), Nz, Nx, Ny)
        f3.W .= rep(f2.W, Ny + 2);  f3.U .= rep(f2.U, Ny + 2);  f3.P .= rep(f2.P, Ny + 2)
        for n in (:Drho, :MFS, :rhoWo, :eta, :rho); getfield(f3, n) .= rep(getfield(f2, n), Ny); end
        f3.rhoUo .= rep(f2.rhoUo, Ny)
        S3.derived_coefficients!(f3, op3)
        c2 = S3.StokesCache(T, CPU(), g2);  c3 = S3.StokesCache(T, CPU(), g3)
        c2.γ_eff .= 1 .+ rand(T, Nz, Nx);  c3.γ_eff .= rep(c2.γ_eff, Ny)
        c2.η_b .= 1;  c3.η_b .= 1
        S3.apply_velocity_bcs!(f2, g2, op2);  S3.apply_velocity_bcs!(f3, g3, op3)
        S3.apply_pressure_bcs!(f2, op2);  S3.apply_pressure_bcs!(f3, op3)
        @test all(f3.W[:, :, k] == f2.W && f3.U[:, :, k] == f2.U && f3.P[:, :, k] == f2.P for k in 1:Ny + 1)
        S3.gershgorin!(c2, f2, g2, op2);  S3.gershgorin!(c3, f3, g3, op3)
        S3.residual_continuity!(c2, f2, g2, f2.P, 0.3);  S3.residual_continuity!(c3, f3, g3, f3.P, 0.3)
        # b − A·v: the Jacobi-preconditioned form divides by a diagonal that carries
        # the y-shear terms in 3-D, so only the unpreconditioned residual is shared
        S3.residual_momentum!(c2, f2, g2, op2; dr = false);  S3.residual_momentum!(c3, f3, g3, op3; dr = false)
        for k in 1:Ny
            @test c3.R_P[:, :, k] ≈ c2.R_P rtol = 1.0e-14
            @test c3.Rv[1][:, :, k] ≈ c2.Rv[1] rtol = 1.0e-14
            @test c3.Rv[2][:, :, k] ≈ c2.Rv[2] rtol = 1.0e-14
        end
        @test all(iszero, c3.Rv[3])
        S3.momentum_advection!(c2, f2, g2, par, op2);  S3.momentum_advection!(c3, f3, g3, par, op3)
        @test c3.advn[1][:, :, 2] ≈ c2.advn[1] rtol = 1.0e-14
        @test c3.advn[2][:, :, 2] ≈ c2.advn[2] rtol = 1.0e-14
    end
    # x ↔ y: on a random periodic state the V-component is the U-component of
    # the axis-swapped state
    σ(A) = permutedims(A, (1, 3, 2))
    f = FluidState3D(T, CPU(), Nz, Nx, Ny)
    for A in (f.W, f.U, f.V, f.P, f.Drho, f.MFS); A .= randn(T, size(A)); end
    f.eta .= exp.(randn(T, size(f.eta)));  f.rho .= 1 .+ 0.3 .* rand(T, size(f.rho))
    op = S3.StokesOp(T, Val(3), par; kw..., xBC = :periodic, yBC = :periodic)
    S3.derived_coefficients!(f, op)
    fs = FluidState3D(T, CPU(), Nz, Ny, Nx)
    for (a, b) in ((:W, :W), (:U, :V), (:V, :U), (:P, :P), (:eta, :eta), (:rho, :rho), (:rhow, :rhow),
                   (:rhou, :rhov), (:rhov, :rhou), (:etaxz, :etayz), (:etayz, :etaxz), (:etaxy, :etaxy),
                   (:Drho, :Drho), (:MFS, :MFS))
        getfield(fs, a) .= σ(getfield(f, b))
    end
    g = XC._grid3d_at(T, Nz, Nx, Ny, h, 1.0, 1.0);  gs = XC._grid3d_at(T, Nz, Ny, Nx, h, 1.0, 1.0)
    c = S3.StokesCache(T, CPU(), g);  cs = S3.StokesCache(T, CPU(), gs)
    c.γ_eff .= 1 .+ rand(T, Nz, Nx, Ny);  cs.γ_eff .= σ(c.γ_eff)
    c.η_b .= 1;  cs.η_b .= 1
    for (cc, ff, gg) in ((c, f, g), (cs, fs, gs))
        S3.gershgorin!(cc, ff, gg, op)
        S3.residual_continuity!(cc, ff, gg, ff.P, Inf)
        S3.residual_momentum!(cc, ff, gg, op; dr = true)
    end
    @test σ(cs.Dv[2]) == c.Dv[3] && σ(cs.λmax[2]) == c.λmax[3]
    @test σ(cs.Rv[2]) ≈ c.Rv[3] rtol = 1.0e-14
    @test σ(cs.Rv[1]) ≈ c.Rv[1] rtol = 1.0e-14
end

@testset "3-D Gershgorin bounds the penalized operator under density contrast" begin
    # Row sums of the finite-difference Jacobian of the penalized momentum
    # operator must stay below λmax·D, and its diagonal must equal D up to the
    # drunken-sailor term D carries by its positive part only.
    Random.seed!(5)
    T = Float64;  Nz, Nx, Ny = 6, 4, 4;  h = 0.3
    par = Parameters(T; D = 1.0, N = 6, L = 1.0, g0 = 9.81)
    grid = XC._grid3d_at(T, Nz, Nx, Ny, h, 1.0, 1.0)
    for xBC in (:periodic, :closed)
        f = FluidState3D(T, CPU(), Nz, Nx, Ny)
        f.eta .= exp.(randn(T, size(f.eta)));  f.rho .= 1 .+ 0.3 .* rand(T, size(f.rho))
        for A in (f.W, f.U, f.V); A .= randn(T, size(A)); end
        op = S3.StokesOp(T, Val(3), par; a1 = 1.3, a2 = 0, a3 = 0, gamma = 0.7, dt = 0.05,
                         xBC, sds = -1, top_cnv = 1, bot_cnv = -1)
        S3.derived_coefficients!(f, op)
        c = S3.StokesCache(T, CPU(), grid)
        S3.penalty!(c, f, grid, op; γfact = 7.0)
        S3.apply_velocity_bcs!(f, grid, op)
        v = S3.velocity(f)
        unknowns = [(d, J) for d in 1:3 for J in CartesianIndices(c.Rv[d])
                    if !S3._dirichlet_face(J, d, size(c.Rv[d], d), op.periodic) &&
                       !(op.periodic[d] && J[d] == size(c.Rv[d], d))]
        function Av()
            S3.residual_continuity!(c, f, grid, f.P, Inf)
            S3.residual_momentum!(c, f, grid, op; dr = true)
            return [-c.Rv[d][J] * c.Dv[d][J] for (d, J) in unknowns]
        end
        r0 = Av();  A = zeros(T, length(unknowns), length(unknowns))
        for (i, (d, J)) in enumerate(unknowns)
            vi = J + CartesianIndex(1, 1, 1) - S3._unit(J, d)
            v[d][vi] += 1;  S3.apply_velocity_bcs!(f, grid, op)
            A[:, i] .= Av() .- r0
            v[d][vi] -= 1;  S3.apply_velocity_bcs!(f, grid, op)
        end
        bound = [c.λmax[d][J] * c.Dv[d][J] for (d, J) in unknowns]
        @test all(vec(sum(abs, A; dims = 2)) .<= bound .* (1 + 1.0e-12))
        σ = map(unknowns) do (d, J)
            n = size(f.rho, d);  e = S3._unit(J, d)
            lo = J[d] == 1     ? (d == 1 ? J : S3._setidx(J, d, n)) : J - e
            hi = J[d] == n + 1 ? (d == 1 ? J - e : S3._setidx(J, d, 1)) : J
            (f.rho[hi] - f.rho[lo]) / h * par.g0 * 0.05
        end
        # rows next to a reflected ghost carry the reflection in their diagonal
        inner = [!(d >= 2 && J[1] in (1, Nz)) && !(xBC === :closed && d != 2 && J[2] in (1, Nx))
                 for (d, J) in unknowns]
        Dg = [c.Dv[d][J] for (d, J) in unknowns]
        @test maximum(abs, (diag(A) .- (Dg .- max.(σ, 0) .+ σ))[inner]) < 1.0e-10 * maximum(Dg)
    end
end

@testset "3-D manufactured solution, every solver, invariant under a pressure offset" begin
    for (kind, N) in ((:dyrel, 16), (:gmg, 32), (:vmg, 32))
        results = map((0.0, -1.0e3)) do c
            fluid, phase, grid, par, scales, vex, kw = _stokes3d_mms(N, c)
            res = _stokes3d_solve(kind, fluid, phase, grid, par, scales; kw...)
            verr = maximum(maximum(abs, a .- b) for (a, b) in zip((fluid.W, fluid.U, fluid.V), vex))
            (; res, verr = verr / maximum(abs, vex[1]),
               perr = maximum(abs, S3._interior(fluid.P) .- c))
        end
        r0, r1 = results
        its(r) = haskey(r, :ncyc) ? r.ncyc : r.PH_iters
        @test r0.res.converged && r1.res.converged
        @test its(r0.res) == its(r1.res)
        @test r0.verr < 1.0e-5 && r1.verr < 1.0e-5
        @test isapprox(r0.perr, r1.perr; atol = 1.0e-9)
    end
end

@testset "3-D solver entry points infer concrete return types" begin
    function inferred_solves(fluid, phase, grid, par, scales, kw)
        cache = S3.StokesCache(Float64, CPU(), grid)
        hier = @inferred S3.build_hierarchy(fluid, grid, par)
        @inferred S3.fluidmech_dyrel!(fluid, phase, cache, grid, par, scales; kw...)
        @inferred S3.fluidmech_gmg!(fluid, phase, hier, grid, par, scales; kw...)
        @inferred S3.fluidmech_vmg!(fluid, phase, hier, grid, par, scales; cc = true, kw...)
        return true
    end
    fluid, phase, grid, par, scales, _, kw = _stokes3d_mms(32, 0.0)
    @test inferred_solves(fluid, phase, grid, par, scales, kw)
end

@testset "3-D inertia-dominated buoyant blob (vmg)" begin
    fluid, phase, grid, par, scales = _stokes3d_base(32)
    h = grid.h
    fill!(fluid.eta, 1);  fill!(fluid.rho, 1.0e3)
    kw = (; dt = 1.0e-4, a1 = 1, a2 = 0, a3 = 0, gamma = 0, top_cnv = -1, bot_cnv = -1)
    S3.derived_coefficients!(fluid, S3.StokesOp(Float64, Val(3), par; kw...))
    fluid.Drho .= [exp(-(((i - 0.5) * h - 0.5)^2 + ((k - 0.5) * h - 0.5)^2 + ((j - 1) * h - 0.5)^2) / 0.01)
                   for j in 1:grid.Nz+1, i in 1:grid.Nx, k in 1:grid.Ny]
    res = _stokes3d_solve(:vmg, fluid, phase, grid, par, scales; kw...)
    @test res.converged
    @test res.PH_iters <= 12
    @test maximum(abs, fluid.W) > 0
end
