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

@testset "3-D residuals match the validated Schur kernels" begin
    Random.seed!(3)
    T = Float64;  Nz, Nx, Ny = 8, 6, 4;  h = 0.37
    par = Parameters(T; D = 1.0, N = 8, L = 1.0, g0 = 9.81)
    grid = XC._grid3d_at(T, Nz, Nx, Ny, h, 1.0, 1.0)
    f = FluidState3D(T, CPU(), Nz, Nx, Ny)
    for A in (f.W, f.U, f.V, f.P, f.Drho, f.MFS, f.rhoWo, f.rhoUo, f.rhoVo)
        A .= randn(T, size(A))
    end
    for A in (f.eta, f.etaxz, f.etayz, f.etaxy); A .= exp.(randn(T, size(A))); end
    for A in (f.rho, f.rhow, f.rhou, f.rhov); A .= 1 .+ 0.3 .* rand(T, size(A)); end
    c = S3.StokesCache(T, CPU(), grid)
    for d in 1:3
        c.Dv[d] .= 1 .+ rand(T, size(c.Dv[d]));  c.bext[d] .= randn(T, size(c.bext[d]))
        c.advn[d] .= randn(T, size(c.advn[d]))
    end
    c.advn[1][[1, Nz + 1], :, :] .= 0
    c.γ_eff .= 1 .+ rand(T, Nz, Nx, Ny);  c.η_b .= 1 .+ rand(T, Nz, Nx, Ny)
    kw = (; a1 = 1.3, a2 = -0.4, a3 = 0.1, gamma = 0.7, dt = 0.05)
    op = S3.StokesOp(T, Val(3), par; kw..., top_cnv = 1, bot_cnv = -1)
    ref = (; R_W = similar(c.Rv[1]), R_U = similar(c.Rv[2]), R_V = similar(c.Rv[3]),
             R_P = similar(c.R_P), P_num = similar(c.R_P), D_W = c.Dv[1], D_U = c.Dv[2],
             D_V = c.Dv[3], bext_W = c.bext[1], bext_U = c.bext[2], bext_V = c.bext[3],
             bext_P = c.bext_P, advn_mz = c.advn[1][2:Nz, :, :], advn_mx = c.advn[2],
             advn_my = c.advn[3], γ_eff = c.γ_eff, η_b = c.η_b)
    P0 = randn(T, size(f.P))
    rel(a, b) = maximum(abs, a .- b) / maximum(abs, b)
    for dr in (false, true)
        XC.GMG3D.dyrel3d_residual_P!(ref, f, h, P0, 0.3)
        S3.residual_continuity!(c, f, grid, P0, 0.3)
        @test rel(c.R_P, ref.R_P) < 1.0e-14
        XC.GMG3D.dyrel3d_residual_V!(ref, f, h, par.g0; dr, kw...)
        S3.residual_momentum!(c, f, grid, op; dr)
        @test rel(c.Rv[1], ref.R_W) < 1.0e-14
        @test rel(c.Rv[2], ref.R_U) < 1.0e-14
        @test rel(c.Rv[3], ref.R_V) < 1.0e-14
    end
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
