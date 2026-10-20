# Verification of src/dyrel3D/schur.jl against the validated 2-D GMG kernels.
#  Test 1: y-uniform 3-D state (V=0, all fields constant along y) — the 3-D
#          W/U/P kernels must reproduce the 2-D kernels slice-for-slice.
#  Test 2: x↔y permutation symmetry — the V kernel on a fully RANDOM state must
#          equal the U kernel applied to the axis-swapped state.
#  Test 3: Schur diagonal vs brute-force; pressure-update kernel vs broadcast.
using xcore
using KernelAbstractions
using KernelAbstractions: CPU
using Random
Random.seed!(42)

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "src/dyrel3D/schur.jl"))

Nz, Nx, Ny = 8, 6, 4
h  = 0.37
g0 = 9.81
a1, a2, a3 = 1.5, -0.6, 0.1
gamma  = 1e-3
dt     = 0.7
invh   = 1 / h
inv_dt = 1 / dt
third  = 1 / 3
half   = 1 / 2
be = CPU()

σ(A) = permutedims(A, (1, 3, 2))                 # x↔y axis swap
rep(A, n) = repeat(A, 1, 1, n)                   # replicate 2-D array along y

function checkslices(A3, A2, name)
    scale = max(1.0, maximum(abs.(A2)))
    for iy in axes(A3, 3)
        err = maximum(abs.(A3[:, :, iy] .- A2))
        err < 1e-10 * scale || error("FAIL $name slice iy=$iy: max abs diff = $err")
    end
    println("OK  $name  (y-uniform 3D == 2D, all $(size(A3,3)) slices)")
end

# ============================ 2-D random state ==============================
W2     = randn(Nz+1, Nx+2);  U2     = randn(Nz+2, Nx+1);  P2  = randn(Nz+2, Nx+2)
P02    = randn(Nz+2, Nx+2);  MFS2   = randn(Nz, Nx)
eta2   = rand(Nz, Nx) .+ 0.5;  etaco2 = rand(Nz+1, Nx+1) .+ 0.5
rho2   = rand(Nz, Nx) .+ 1.0;  rhow2  = rand(Nz+1, Nx) .+ 1.0;  rhou2 = rand(Nz, Nx+1) .+ 1.0
Drho2  = randn(Nz+1, Nx)
rhoWo2 = randn(Nz+1, Nx);  rhoWoo2 = randn(Nz+1, Nx)
rhoUo2 = randn(Nz, Nx+1);  rhoUoo2 = randn(Nz, Nx+1)
advn_mz2 = randn(Nz-1, Nx);  advn_mx2 = randn(Nz, Nx+1)
D_W2   = rand(Nz+1, Nx) .+ 1.0;  D_U2 = rand(Nz, Nx+1) .+ 1.0
γ2     = rand(Nz, Nx) .+ 1.0;  ηb2 = rand(Nz, Nx) .+ 1.0
bW2    = randn(Nz+1, Nx);  bU2 = randn(Nz, Nx+1);  bP2 = randn(Nz, Nx)
R_W2   = zeros(Nz+1, Nx);  R_U2 = zeros(Nz, Nx+1);  R_P2 = zeros(Nz, Nx);  Pn2 = zeros(Nz, Nx)

# ===================== 3-D y-uniform replication of it ======================
f3 = (; W = rep(W2, Ny+2), U = rep(U2, Ny+2), V = zeros(Nz+2, Nx+2, Ny+1),
        P = rep(P2, Ny+2), eta = rep(eta2, Ny),
        etaxz = rep(etaco2, Ny),
        etayz = rand(Nz+1, Nx, Ny+1) .+ 0.5,        # arbitrary: must not matter
        etaxy = rand(Nz, Nx+1, Ny+1) .+ 0.5,        # arbitrary: must not matter
        Drho = rep(Drho2, Ny), rho = rep(rho2, Ny),
        rhow = rep(rhow2, Ny), rhou = rep(rhou2, Ny),
        rhov = rand(Nz, Nx, Ny+1) .+ 1.0,           # multiplies V=0 only
        rhoWo = rep(rhoWo2, Ny), rhoWoo = rep(rhoWoo2, Ny),
        rhoUo = rep(rhoUo2, Ny), rhoUoo = rep(rhoUoo2, Ny),
        rhoVo = zeros(Nz, Nx, Ny+1), rhoVoo = zeros(Nz, Nx, Ny+1),
        MFS = rep(MFS2, Ny))
c3 = (; R_W = zeros(Nz+1, Nx, Ny), R_U = zeros(Nz, Nx+1, Ny), R_V = zeros(Nz, Nx, Ny+1),
        R_P = zeros(Nz, Nx, Ny), P_num = zeros(Nz, Nx, Ny),
        D_W = rep(D_W2, Ny), D_U = rep(D_U2, Ny), D_V = rand(Nz, Nx, Ny+1) .+ 1.0,
        bext_W = rep(bW2, Ny), bext_U = rep(bU2, Ny), bext_V = zeros(Nz, Nx, Ny+1),
        bext_P = rep(bP2, Ny),
        advn_mz = rep(advn_mz2, Ny), advn_mx = rep(advn_mx2, Ny),
        advn_my = zeros(Nz, Nx, Ny+1),
        γ_eff = rep(γ2, Ny), η_b = rep(ηb2, Ny), s_P = zeros(Nz, Nx, Ny))
P03 = rep(P02, Ny+2)

for dr in (false, true)
    # 2-D reference (the validated single-grid kernels; P first to fill P_num)
    xcore.compute_dyrel_residual_P_kernel!(be)(
        R_P2, Pn2, W2, U2, P2, P02, MFS2, rho2, rhow2, rhou2, ηb2, bP2, γ2,
        invh, inv_dt, 1.0; ndrange = size(R_P2))
    xcore.compute_dyrel_residual_W_kernel!(be)(
        R_W2, W2, U2, eta2, etaco2, P2, Pn2, Drho2, rho2, rhow2, rhoWo2, rhoWoo2,
        advn_mz2, D_W2, bW2, invh, g0, third, half, a1, a2, a3, inv_dt, dt, 1.0, dr;
        ndrange = size(R_W2))
    xcore.compute_dyrel_residual_U_kernel!(be)(
        R_U2, U2, W2, eta2, etaco2, P2, Pn2, rho2, rhou2, rhoUo2, rhoUoo2,
        advn_mx2, D_U2, bU2, invh, g0, third, half, a1, a2, a3, gamma, inv_dt, dt, 1.0, dr;
        ndrange = size(R_U2))
    # 3-D under test
    dyrel3d_residual_P!(c3, f3, h, P03, dt)
    dyrel3d_residual_V!(c3, f3, h, g0; dr, a1, a2, a3, gamma, dt)
    checkslices(c3.R_P,   R_P2, "R_P   dr=$dr")
    checkslices(c3.P_num, Pn2,  "P_num dr=$dr")
    checkslices(c3.R_W,   R_W2, "R_W   dr=$dr")
    checkslices(c3.R_U,   R_U2, "R_U   dr=$dr")
end

# ================== x↔y permutation symmetry: V vs U kernel =================
# Fully random 3-D state (no uniformity, random bext, nonzero V/rhoVo/advn_my).
fr = (; W = randn(Nz+1, Nx+2, Ny+2), U = randn(Nz+2, Nx+1, Ny+2), V = randn(Nz+2, Nx+2, Ny+1),
        P = randn(Nz+2, Nx+2, Ny+2), eta = rand(Nz, Nx, Ny) .+ 0.5,
        etaxz = rand(Nz+1, Nx+1, Ny) .+ 0.5, etayz = rand(Nz+1, Nx, Ny+1) .+ 0.5,
        etaxy = rand(Nz, Nx+1, Ny+1) .+ 0.5,
        Drho = randn(Nz+1, Nx, Ny), rho = rand(Nz, Nx, Ny) .+ 1.0,
        rhow = rand(Nz+1, Nx, Ny) .+ 1.0, rhou = rand(Nz, Nx+1, Ny) .+ 1.0,
        rhov = rand(Nz, Nx, Ny+1) .+ 1.0,
        rhoWo = randn(Nz+1, Nx, Ny), rhoWoo = randn(Nz+1, Nx, Ny),
        rhoUo = randn(Nz, Nx+1, Ny), rhoUoo = randn(Nz, Nx+1, Ny),
        rhoVo = randn(Nz, Nx, Ny+1), rhoVoo = randn(Nz, Nx, Ny+1),
        MFS = randn(Nz, Nx, Ny))
cr = (; R_W = zeros(Nz+1, Nx, Ny), R_U = zeros(Nz, Nx+1, Ny), R_V = zeros(Nz, Nx, Ny+1),
        R_P = zeros(Nz, Nx, Ny), P_num = zeros(Nz, Nx, Ny),
        D_W = rand(Nz+1, Nx, Ny) .+ 1.0, D_U = rand(Nz, Nx+1, Ny) .+ 1.0,
        D_V = rand(Nz, Nx, Ny+1) .+ 1.0,
        bext_W = randn(Nz+1, Nx, Ny), bext_U = randn(Nz, Nx+1, Ny),
        bext_V = randn(Nz, Nx, Ny+1), bext_P = randn(Nz, Nx, Ny),
        advn_mz = randn(Nz-1, Nx, Ny), advn_mx = randn(Nz, Nx+1, Ny),
        advn_my = randn(Nz, Nx, Ny+1),
        γ_eff = rand(Nz, Nx, Ny) .+ 1.0, η_b = rand(Nz, Nx, Ny) .+ 1.0,
        s_P = zeros(Nz, Nx, Ny))
P0r = randn(Nz+2, Nx+2, Ny+2)

for dr in (false, true)
    dyrel3d_residual_P!(cr, fr, h, P0r, dt)
    dyrel3d_residual_V!(cr, fr, h, g0; dr, a1, a2, a3, gamma, dt)
    # U kernel in the axis-swapped world should reproduce the V residual
    R_U_sw = zeros(Nz, Ny+1, Nx)
    compute_dyrel3d_residual_U_kernel!(be)(
        R_U_sw, σ(fr.V), σ(fr.W), σ(fr.U), σ(fr.eta), σ(fr.etayz), σ(fr.etaxy),
        σ(fr.P), σ(cr.P_num), σ(fr.rho), σ(fr.rhov), σ(fr.rhoVo), σ(fr.rhoVoo),
        σ(cr.advn_my), σ(cr.D_V), σ(cr.bext_V),
        invh, g0, third, half, a1, a2, a3, gamma, inv_dt, dt, 1.0, dr;
        ndrange = size(R_U_sw))
    err = maximum(abs.(σ(R_U_sw) .- cr.R_V))
    err < 1e-10 * max(1.0, maximum(abs.(cr.R_V))) ||
        error("FAIL V↔U permutation symmetry dr=$dr: max abs diff = $err")
    println("OK  R_V dr=$dr  (x↔y permutation symmetry vs U kernel)")
end

# =================== Schur diagonal + pressure update =======================
compute_schur_diag3d!(cr, fr, h)
sref = [ (fr.rhou[iz,ix,iy]/cr.D_U[iz,ix,iy] + fr.rhou[iz,ix+1,iy]/cr.D_U[iz,ix+1,iy] +
          fr.rhov[iz,ix,iy]/cr.D_V[iz,ix,iy] + fr.rhov[iz,ix,iy+1]/cr.D_V[iz,ix,iy+1] +
          fr.rhow[iz,ix,iy]/cr.D_W[iz,ix,iy] + fr.rhow[iz+1,ix,iy]/cr.D_W[iz+1,ix,iy]) /
          h^2 / fr.rho[iz,ix,iy]
         for iz in 1:Nz, ix in 1:Nx, iy in 1:Ny ]
maximum(abs.(cr.s_P .- sref)) < 1e-12 * maximum(abs.(sref)) || error("FAIL s_P vs brute force")
println("OK  s_P  (3D Schur diagonal vs brute force)")

Pbefore = copy(fr.P)
dyrel3d_update_pressure!(cr, fr; ω = 0.8)
expect = Pbefore[2:end-1, 2:end-1, 2:end-1] .+ 0.8 .* cr.γ_eff .* cr.R_P
maximum(abs.(fr.P[2:end-1, 2:end-1, 2:end-1] .- expect)) < 1e-13 || error("FAIL P update interior")
fr.P[2:end-1, 2:end-1, 2:end-1] .= Pbefore[2:end-1, 2:end-1, 2:end-1]
fr.P ≈ Pbefore || error("FAIL P update touched the ghost ring")
println("OK  P += ω·γ_eff·R_P  (interior only, ghosts untouched)")

println("\nAll 3-D Schur-complement kernel checks passed.")
