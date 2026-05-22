using LinearAlgebra
include("mms_sources.jl")

@testset "MMS Stokes convergence (run_bnchm_VP analogue)" begin
    # sweep precisions; only Float64 active so far (Float32 saddle-point
    # solves through UMFPACK need extra care — TODO once we add it)
    for T in (Float64,)
        L = T(10)
        # collect per-resolution L2 errors so we can compute observed orders
        errs_W = T[];  errs_U = T[];  errs_P = T[]

        # resolution sweep: match run_bnchm_VP.m's NN = [100, 200, 400] scaling
        # (smaller N here to keep the test runtime tight; ratio is what matters)
        for N in (32, 64, 128)
            # ------ build analytic manufactured solution for this resolution
            mms = mms_sources(T, N, N, L)

            # ------ build the package state at the matching grid size
            par = Parameters(T; D = L, N = N, L = L, etam0 = 1e3, g0 = 10.0,
                                rhom0 = 2700, rhox0 = 3200, d0 = 1e-2,
                                Da = 0.0, Xi = 0.0, gamma = 0.0)
            grid = Grid(par)
            state = FluidState(T, CPU(), grid.Nz, grid.Nx)

            # ------ inject material fields directly from the analytic MMS;
            # bypasses update! since the rheology is fixed by construction
            state.eta   .= mms.eta
            state.etaco .= mms.etaco
            state.rho   .= mms.rho
            state.rhow  .= mms.rhow
            state.rhou  .= mms.rhou
            state.Drho  .= mms.Drho
            state.MFS   .= mms.MFS
            # history flux: zero in MMS (dt → ∞ kills the inertial term)

            # ------ bundle the MMS RHS sources + exact-solution pin values
            bnchm_data = BnchmData{T, typeof(mms.src_W)}(
                mms.src_W, mms.src_U, mms.src_P,
                mms.W_exact, mms.U_exact, mms.P_exact,
            )

            # ------ solve once with free-slip top/bot, periodic sides
            # (matches the BC setup at the bottom of src/mms.m)
            fluidmech!(state, grid, par;
                       bnchm_data = bnchm_data,
                       sds = -1, top_cnv = -1, bot_cnv = -1,
                       open_cnv = false,
                       dt = T(1e32),
                       a1 = T(1), a2 = T(1), a3 = T(0))

            # ------ relative L2 error against the analytic solution
            push!(errs_W, norm(state.W .- mms.W_exact) / norm(mms.W_exact))
            push!(errs_U, norm(state.U .- mms.U_exact) / norm(mms.U_exact))
            push!(errs_P, norm(state.P .- mms.P_exact) / norm(mms.P_exact))
        end

        # log raw errors + observed convergence orders for regression triage
        @info "MMS errors (T = $T)" errs_W errs_U errs_P
        @info "MMS observed orders" W64 = log2(errs_W[1] / errs_W[2]) U64 = log2(errs_U[1] / errs_U[2]) P64 = log2(errs_P[1] / errs_P[2]) W128 = log2(errs_W[2] / errs_W[3]) U128 = log2(errs_U[2] / errs_U[3]) P128 = log2(errs_P[2] / errs_P[3])

        # MATLAB benchmark expects quadratic convergence on W, U, P.
        # Floor the assertion at >= 1.8 to absorb small constants from the
        # 32→64 step (the 64→128 step usually shows cleaner ≥ 2.0).
        @test log2(errs_W[1] / errs_W[2]) > 1.8
        @test log2(errs_U[1] / errs_U[2]) > 1.8
        @test log2(errs_P[1] / errs_P[2]) > 1.8
        @test log2(errs_W[2] / errs_W[3]) > 1.8
        @test log2(errs_U[2] / errs_U[3]) > 1.8
        @test log2(errs_P[2] / errs_P[3]) > 1.8
    end
end
