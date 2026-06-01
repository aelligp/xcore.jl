using SparseArrays
using LinearAlgebra
using Statistics

# Faithful port of src/fluidmech.m: build the coupled [W; U; P] saddle-point
# system as a sparse matrix on the host, scale it with a Jacobi-like diagonal
# preconditioner, factor with sparse LU, and update the primary unknowns by
# the resulting Newton-style correction.
#
# Sparse assembly is host-CPU only — SparseMatrixCSC + UMFPACK in Julia stdlib.
# State fields are pulled to host via `Array(...)` if they happen to be on a
# GPU backend; for the current pure-CPU path this is a no-op view.

# FluidmechCache definition + constructor live in state.jl so FluidState can
# include the field. Helpers (`_compute_nzmap`, `_scatter!`) are defined below.

"""
    _compute_nzmap(I, J, M) -> Vector{Int}

For each triplet `k`, return the linear index into `M.nzval` where
`M[I[k], J[k]]` lives. Uses the CSC invariant that `M.rowval` is sorted
within each column.
"""
function _compute_nzmap(I::Vector{Int}, J::Vector{Int}, M::SparseMatrixCSC{T,Int}) where {T}
    nz_map = Vector{Int}(undef, length(I))
    @inbounds for k in eachindex(I)
        col = J[k];  row = I[k]
        lo = M.colptr[col];  hi = M.colptr[col + 1] - 1
        # rowval[lo:hi] is sorted ascending
        pos = searchsortedfirst(view(M.rowval, lo:hi), row) + lo - 1
        nz_map[k] = pos
    end
    return nz_map
end

"""
    _scatter!(M, nz_map, vals) -> M

Zero `M.nzval` then accumulate `vals[k]` into `M.nzval[nz_map[k]]` for every
`k`. Equivalent in effect to `sparse(I, J, vals, m, n)` but skips the sort.
"""
function _scatter!(M::SparseMatrixCSC{T,Int}, nz_map::Vector{Int}, vals::Vector{T}) where {T}
    fill!(M.nzval, zero(T))
    @inbounds for k in eachindex(vals)
        M.nzval[nz_map[k]] += vals[k]
    end
    return M
end

"""
    fluidmech!(state, grid, par;
               bnchm_data = nothing,
               sds = -1, top_cnv = 1, bot_cnv = 1, open_cnv = false,
               dt = T(1e32),
               a1 = T(1), a2 = T(1), a3 = T(0)) -> state

Solve the linear momentum + continuity system for `(W, U, P)` and write the
solution increment back into `state` in place. With `bnchm_data ≠ nothing`
the manufactured-solution branch is taken: RHS gains `src_*` forcing and one
`P` and one `U` cell are pinned to the exact solution values.

Boundary flags follow the MATLAB convention:
* `sds == -1` (default) → periodic in x for the velocity equations.
* `top_cnv`, `bot_cnv` ∈ {-1, +1} encode free-slip (-1) vs no-slip (+1) at
  top/bottom z-boundaries for the x-velocity equation.
* `open_cnv = true` opens the bottom z-boundary for the z-velocity equation.

Time-integration coefficients `a1, a2, a3` come from `src/timing.m`. In MMS
mode the inertial term is annihilated by `dt → ∞`.
"""
function fluidmech!(state::FluidState{T}, grid::Grid{T}, par::Parameters{T};
                    phase::Union{Nothing, PhaseState{T}} = nothing,
                    bnchm_data::Union{Nothing, BnchmData{T}} = nothing,
                    sds::Int = -1, top_cnv::Int = 1, bot_cnv::Int = 1,
                    open_cnv::Bool = false,
                    xBC::Symbol = :periodic,
                    zBC::Symbol = :closed,
                    dt::Real = T(1e32),
                    a1::Real = T(1), a2::Real = T(1), a3::Real = T(0),
                    b1::Real = T(1), b2::Real = T(0), b3::Real = T(0)) where {T<:AbstractFloat}
    bnchm = bnchm_data !== nothing
    Nz = grid.Nz;  Nx = grid.Nx;  h = grid.h
    g0 = par.g0;   gamma = par.gamma
    invh  = inv(h);  invh2 = inv(h)^2

    # ---------------- solver cache (built lazily on first call) -------------
    cache = state.solver
    # Invalidate if grid size or MMS flag changed (different sparsity).
    if cache.initialized && (cache.sig_Nz != Nz || cache.sig_Nx != Nx || cache.sig_bnchm != bnchm)
        cache.initialized = false
        cache.lu_F = nothing
    end

    # ---------------- map arrays + flat sizes (matches MATLAB MapP/MapW/MapU)
    NP = (Nz + 2) * (Nx + 2)
    NW = (Nz + 1) * (Nx + 2)
    NU = (Nz + 2) * (Nx + 1)
    MapP = reshape(1:NP, Nz + 2, Nx + 2)
    MapW = reshape(1:NW, Nz + 1, Nx + 2)
    MapU = reshape(1:NU, Nz + 2, Nx + 1) .+ NW

    # local aliases — pulled to plain CPU arrays once. For Array-backed state
    # this is a no-op view (`convert(Array, ::Array) === arr`).
    W = Array(state.W);  U = Array(state.U);  P = Array(state.P)
    eta = Array(state.eta);  etaco = Array(state.etaco)
    rho = Array(state.rho);  rhow = Array(state.rhow);  rhou = Array(state.rhou)
    Drho = Array(state.Drho);  MFS = Array(state.MFS)
    rhoWo = Array(state.rhoWo);  rhoWoo = Array(state.rhoWoo)
    rhoUo = Array(state.rhoUo);  rhoUoo = Array(state.rhoUoo)

    # ---------------- MFS update from mass-flux residual (non-bnchm)
    # Mirrors fluidmech.m lines 9-23. `phase.advn_rho` is produced by phsevo!
    # one or more Picard iterations earlier; we accumulate a correction into
    # MFS so the divergence equation enforces ∂_t ρ = -∇·(ρ v).
    if !bnchm
        rhoo   = Array(state.rhoo);   rhooo  = Array(state.rhooo)
        drhodt = Array(state.drhodt); drhodto = Array(state.drhodto); drhodtoo = Array(state.drhodtoo)
        advn_rho = Array(phase.advn_rho)
        @. drhodt = -advn_rho
        res_rho = (T(a1) .* rho .- T(a2) .* rhoo .- T(a3) .* rhooo) ./ T(dt) .-
                  (T(b1) .* drhodt .+ T(b2) .* drhodto .+ T(b3) .* drhodtoo)
        @. MFS = MFS - par.alpha * res_rho
        # write back the updated buffers
        state.drhodt .= drhodt
        state.MFS    .= MFS
    end
    MFSmean = T(mean(MFS))


    # ghost-index arrays — periodic-x always; periodic-z when `bnchm` (per
    # mms.m), reflect otherwise.
    icx = [Nx; collect(1:Nx); 1]
    icz = bnchm ? [Nz; collect(1:Nz); 1] : [1; collect(1:Nz); Nz]
    ifx = [Nx; collect(1:(Nx+1)); 2]

    # ----------------- z-stress divergence block (KV rows for W eqns) -------
    IIL = Int[];       # equation indeces into L
    JJL = Int[];       # variable indeces into L
    AAL = T[];         # coefficients for L
    IIR = Int[];       # equation indeces into R
    AAR = T[];         # forcing entries for R

    ## Assemble coefficients of the z-stess divergence

    # left boundary (i = 1, all rows): W[:,1] + sds * W[:,end-1] = 0
    let ii  = vec(MapW[:, 1]),
        jj1 = ii,
        jj2 = vec(MapW[:, end - 1]),
        aa   = length(ii)
        append!(IIL, ii); append!(JJL, jj1); append!(AAL, fill(T(1),   aa))
        append!(IIL, ii); append!(JJL, jj2); append!(AAL, fill(T(sds), aa))
        append!(IIR, ii); append!(AAR, zeros(T, aa))
    end
    # right boundary (i = Nx+2)
    let ii  = vec(MapW[:, end]),
        jj1 = vec(MapW[:, end]),
        jj2 = vec(MapW[:, 2]),
        aa   = length(ii)
        append!(IIL, ii); append!(JJL, jj1); append!(AAL, fill(T(1),   aa))
        append!(IIL, ii); append!(JJL, jj2); append!(AAL, fill(T(sds), aa))
        append!(IIR, ii); append!(AAR, zeros(T, aa))
    end
    # top boundary (z = 0): W = 0 (Dirichlet)
    let ii = vec(MapW[1, 2:end-1]),
        aa  = length(ii)
        append!(IIL, ii); append!(JJL, ii); append!(AAL, fill(T(1), aa))
        append!(IIR, ii); append!(AAR, zeros(T, aa))
    end
    # bottom boundary: closed (W=0) or open (∂W/∂z = -∂U/∂x via density)
    let ii  = vec(MapW[end, 2:end-1]),
        jj1 = ii,
        jj2 = vec(MapW[end - 1, 2:end-1]),
        jj3 = vec(MapU[end - 1, 2:end]),
        jj4 = vec(MapU[end - 1, 1:end-1]),
        rho1 = vec(rhow[end, :]),
        rho2 = vec(rhow[end - 1, :]),
        rho3 = vec(rhou[end, 2:end]),
        rho4 = vec(rhou[end, 1:end-1]),
        # n   = length(ii),
        oc  = open_cnv ? T(1) : T(0)
        append!(IIL, ii); append!(JJL, jj1); append!(AAL,  rho1 .* invh)
        append!(IIL, ii); append!(JJL, jj2); append!(AAL, -oc .* rho2 .* invh)
        append!(IIL, ii); append!(JJL, jj3); append!(AAL,  oc .* rho3 .* invh)
        append!(IIL, ii); append!(JJL, jj4); append!(AAL, -oc .* rho4 .* invh)
        # MFBG = MFSmean · ZZw — for the closed-bottom case enters here as
        # MFSmean · D / h (constant in x). In MMS MFSmean = 0 so this term
        # vanishes; in production it carries the depth-integrated divergence.
        mfbg_bot = MFSmean * grid.D / h
        aa_bot = oc .* vec(MFS[end, :]) .+ (T(1) - oc) * mfbg_bot
        append!(IIR, ii); append!(AAR, aa_bot)
    end

    # internal W points
    let ii    = vec(MapW[2:end-1, 2:end-1]),
        EtaC1 = vec(etaco[2:end-1, 1:end-1]),
        EtaC2 = vec(etaco[2:end-1, 2:end]),
        EtaP1 = vec(eta[1:end-1, :]),
        EtaP2 = vec(eta[2:end,   :])
        # coefficients multiplying z-velocities W
        jj1 = vec(MapW[1:end-2, 2:end-1])    # top
        jj2 = vec(MapW[3:end,   2:end-1])    # bottom
        jj3 = vec(MapW[2:end-1, 1:end-2])    # left
        jj4 = vec(MapW[2:end-1, 3:end])      # right

        # inertial term  rhow .* a1 / dt
        aa = a1 .* vec(rhow[2:end-1, :]) ./ T(dt)
        append!(IIL, ii); append!(JJL, ii); append!(AAL, aa) # inertial term

        # diagonal stress coefficient
        aa = T(2/3) .* (EtaP1 .+ EtaP2) .* invh2 .+ T(1/2) .* (EtaC1 .+ EtaC2) .* invh2
        append!(IIL, ii); append!(JJL, ii); append!(AAL,  aa)                           # W on stencil center
        append!(IIL, ii); append!(JJL, jj1); append!(AAL, -T(2/3) .* EtaP1 .* invh2)    # W one above
        append!(IIL, ii); append!(JJL, jj2); append!(AAL, -T(2/3) .* EtaP2 .* invh2)    # W one below
        append!(IIL, ii); append!(JJL, jj3); append!(AAL, -T(1/2) .* EtaC1 .* invh2)    # W one to the left
        append!(IIL, ii); append!(JJL, jj4); append!(AAL, -T(1/2) .* EtaC2 .* invh2)    # W one to the right

        # what shall we do with the drunken sailor...
        if !bnchm
            ddz_rho = (rho[2:end, :] .- rho[1:end-1, :]) .* invh
            aa = vec(ddz_rho) .* g0 .* T(dt)
            append!(IIL, ii); append!(JJL, ii); append!(AAL, aa)
        end

        # coefficients multiplying x-velocities U
        jU1 = vec(MapU[2:end-2, 1:end-1])   # top left
        jU2 = vec(MapU[3:end-1, 1:end-1])   # bottom left
        jU3 = vec(MapU[2:end-2, 2:end])     # top right
        jU4 = vec(MapU[3:end-1, 2:end])     # bottom right

        append!(IIL, ii); append!(JJL, jU1); append!(AAL, -(T(1/2) .* EtaC1 .- T(1/3) .* EtaP1) .* invh2)   # U one to the top and left
        append!(IIL, ii); append!(JJL, jU2); append!(AAL, +(T(1/2) .* EtaC1 .- T(1/3) .* EtaP2) .* invh2)   # U one to the bottom and left
        append!(IIL, ii); append!(JJL, jU3); append!(AAL, +(T(1/2) .* EtaC2 .- T(1/3) .* EtaP1) .* invh2)   # U one to the top and right
        append!(IIL, ii); append!(JJL, jU4); append!(AAL, -(T(1/2) .* EtaC2 .- T(1/3) .* EtaP2) .* invh2)   # U one to the bottom and right

        # z - RHS vector
        f_mz    = rhow[2:end-1, :] .* W[2:end-1, 2:end-1]
        u_mz    = (U[2:end-2, :] .+ U[3:end-1, :]) ./ T(2)
        w_mz    = (W[1:end-1, 2:end-1] .+ W[2:end, 2:end-1]) ./ T(2)
        advn_mz = similar(f_mz)
        advect_centered!(advn_mz, f_mz, u_mz, w_mz, h, par.ADVN; xBC, zBC)
        rr =  + Drho[2:end-1, :] .* g0 .+ (a2 .* rhoWo[2:end-1, :] .+ a3 .* rhoWoo[2:end-1, :]) ./ T(dt) .- advn_mz

        if bnchm
            rr .+= bnchm_data.src_W[2:end-1, 2:end-1]
        end
        append!(IIR, ii); append!(AAR, vec(rr))
    end

    # ----------------- x-stress divergence block (KV rows for U eqns) -------
    # assemble coefficients of x-stress divergence
    # top boundary (z = 0): U + top_cnv * U[2, :] = 0  (free/no-slip)
    let ii  = vec(MapU[1, :]),
        jj2 = vec(MapU[2, :]),
        aa   = length(ii)
        append!(IIL, ii); append!(JJL, ii);  append!(AAL, fill(T(1),       aa))
        append!(IIL, ii); append!(JJL, jj2); append!(AAL, fill(T(top_cnv), aa))
        append!(IIR, ii); append!(AAR, zeros(T, aa))
    end

    # bottom boundary (z = D)
    let ii  = vec(MapU[end, :]),
        jj2 = vec(MapU[end - 1, :]),
        n   = length(ii)
        append!(IIL, ii); append!(JJL, ii);  append!(AAL, fill(T(1),       n))
        append!(IIL, ii); append!(JJL, jj2); append!(AAL, fill(T(bot_cnv), n))
        append!(IIR, ii); append!(AAR, zeros(T, n))
    end

    # internal points
    let ii    = vec(MapU[2:end-1, :]),
        EtaC1 = vec(etaco[1:end-1, :]),
        EtaC2 = vec(etaco[2:end,   :]),
        EtaP1 = vec(eta[:, icx[1:end-1]]),
        EtaP2 = vec(eta[:, icx[2:end]])

        jj1 = vec(MapU[2:end-1, ifx[1:end-2]])    # left
        jj2 = vec(MapU[2:end-1, ifx[3:end]])      # right
        jj3 = vec(MapU[1:end-2, ifx[2:end-1]])    # top
        jj4 = vec(MapU[3:end,   ifx[2:end-1]])    # bottom

        # inertial term — note (a1 + gamma) here matches MATLAB exactly
        aa = (a1 + gamma) .* rhou ./ T(dt)
        append!(IIL, ii); append!(JJL, ii); append!(AAL, vec(aa))

        aa = T(2/3) .* (EtaP1 .+ EtaP2) .* invh2 .+ T(1/2) .* (EtaC1 .+ EtaC2) .* invh2
        append!(IIL, ii); append!(JJL, ii);  append!(AAL,  aa)
        append!(IIL, ii); append!(JJL, jj1); append!(AAL, -T(2/3) .* EtaP1 .* invh2)
        append!(IIL, ii); append!(JJL, jj2); append!(AAL, -T(2/3) .* EtaP2 .* invh2)
        append!(IIL, ii); append!(JJL, jj3); append!(AAL, -T(1/2) .* EtaC1 .* invh2)
        append!(IIL, ii); append!(JJL, jj4); append!(AAL, -T(1/2) .* EtaC2 .* invh2)

        # what shall we do with the drunken sailor...
        if !bnchm
            ddx_rho = (rho[:, icx[2:end]] .- rho[:, icx[1:end-1]]) .* invh
            aa = vec(ddx_rho) .* g0 .* T(dt)
            append!(IIL, ii); append!(JJL, ii); append!(AAL, aa)
        end

        # coefficients multiplying z-velocities W
        jW1 = vec(MapW[1:end-1, 1:end-1])    # top left
        jW2 = vec(MapW[1:end-1, 2:end])      # top right
        jW3 = vec(MapW[2:end,   1:end-1])    # bottom left
        jW4 = vec(MapW[2:end,   2:end])      # bottom right
        append!(IIL, ii); append!(JJL, jW1); append!(AAL, -(T(1/2) .* EtaC1 .- T(1/3) .* EtaP1) .* invh2)   # W one to the top and left
        append!(IIL, ii); append!(JJL, jW2); append!(AAL, +(T(1/2) .* EtaC1 .- T(1/3) .* EtaP2) .* invh2)   # W one to the top and right
        append!(IIL, ii); append!(JJL, jW3); append!(AAL, +(T(1/2) .* EtaC2 .- T(1/3) .* EtaP1) .* invh2)   # W one to the bottom and left
        append!(IIL, ii); append!(JJL, jW4); append!(AAL, -(T(1/2) .* EtaC2 .- T(1/3) .* EtaP2) .* invh2)   # W one to the bottom and right

        # x-RHS vector
        u_mx    = (U[2:end-1, ifx[1:end-1]] .+ U[2:end-1, ifx[2:end]]) ./ T(2)
        w_mx    = (W[:, 1:end-1] .+ W[:, 2:end]) ./ T(2)
        f_mx    = rhou .* U[2:end-1, :]
        advn_mx = similar(f_mx)
        advect_centered!(advn_mx, f_mx, u_mx, w_mx, h, par.ADVN; xBC, zBC)
        # average the periodic-equivalent boundary columns (fluidmech.m:164)
        col_avg = (advn_mx[:, 1] .+ advn_mx[:, end]) ./ T(2)
        advn_mx[:, 1]   .= col_avg
        advn_mx[:, end] .= col_avg
        rr = (a2 .* rhoUo .+ a3 .* rhoUoo) ./ T(dt) .- advn_mx
        if bnchm
            rr .+= bnchm_data.src_U[2:end-1, :]
        end
        append!(IIR, ii); append!(AAR, vec(rr))
    end

    # assemble coefficient matrix & right-hand side vector — first call builds
    # the sparsity, subsequent calls only scatter the new AAL values into the
    # cached `cache.KV.nzval`.
    if cache.initialized
        _scatter!(cache.KV, cache.nz_KV, AAL)
        KV = cache.KV
    else
        KV = sparse(IIL, JJL, AAL, NW + NU, NW + NU)
        cache.KV = KV
        cache.nz_KV = _compute_nzmap(IIL, JJL, KV)
    end
    RV = sparsevec(IIR, AAR, NW + NU)

    # assemble coefficients for gradient operator
    IIL_g = Int[];  # equation indeces into A
    JJL_g = Int[];  # variable indeces into A
    AAL_g = T[]     # coefficients for A
    let ii  = vec(MapW[2:end-1, 2:end-1]),  # coefficients for z-gradient
        jj1 = vec(MapP[2:end-2, 2:end-1]),  # top
        jj2 = vec(MapP[3:end-1, 2:end-1]),  # bottom
        aa   = length(ii)
        append!(IIL_g, ii); append!(JJL_g, jj1); append!(AAL_g, fill(-invh, aa))    # one to the top
        append!(IIL_g, ii); append!(JJL_g, jj2); append!(AAL_g, fill(+invh, aa))    # one to the bottom
    end
    let ii  = vec(MapU[2:end-1, :]),        # coefficients for x-gradient
        jj1 = vec(MapP[2:end-1, 1:end-1]),  # left
        jj2 = vec(MapP[2:end-1, 2:end]),    # right
        aa   = length(ii)
        append!(IIL_g, ii); append!(JJL_g, jj1); append!(AAL_g, fill(-invh, aa))    # one to the left
        append!(IIL_g, ii); append!(JJL_g, jj2); append!(AAL_g, fill(+invh, aa))    # one to the right
    end
    # assemble coefficient matrix
    if cache.initialized
        _scatter!(cache.GG, cache.nz_GG, AAL_g)
        GG = cache.GG
    else
        GG = sparse(IIL_g, JJL_g, AAL_g, NW + NU, NP)
        cache.GG = GG
        cache.nz_GG = _compute_nzmap(IIL_g, JJL_g, GG)
    end

    # assemble coefficients for divergence of matrix mass flux (DM)

    IIL_d = Int[];  # equation indeces into A
    JJL_d = Int[];  # variable indeces into A
    AAL_d = T[]     # coefficients for A
    # internal points
    let ii  = vec(MapP[2:end-1, 2:end-1]),
        jUL = vec(MapU[2:end-1, 1:end-1]),      # left U
        jUR = vec(MapU[2:end-1, 2:end]),        # right U
        jWT = vec(MapW[1:end-1, 2:end-1]),      # top W
        jWB = vec(MapW[2:end,   2:end-1]),      # bottom W
        r1  = vec(rhou[:, 1:end-1]),
        r2  = vec(rhou[:, 2:end]),
        r3  = vec(rhow[1:end-1, :]),
        r4  = vec(rhow[2:end,   :])
        append!(IIL_d, ii); append!(JJL_d, jUL); append!(AAL_d, -r1 .* invh)    # U one to the left
        append!(IIL_d, ii); append!(JJL_d, jUR); append!(AAL_d, +r2 .* invh)    # U one to the right
        append!(IIL_d, ii); append!(JJL_d, jWT); append!(AAL_d, -r3 .* invh)    # W one above
        append!(IIL_d, ii); append!(JJL_d, jWB); append!(AAL_d, +r4 .* invh)    # W one below
    end
    # Assemble coefficient matrix
    if cache.initialized
        _scatter!(cache.DM, cache.nz_DM, AAL_d)
        DM = cache.DM
    else
        DM = sparse(IIL_d, JJL_d, AAL_d, NP, NW + NU)
        cache.DM = DM
        cache.nz_DM = _compute_nzmap(IIL_d, JJL_d, DM)
    end

    # assemble coefficients for matrix pressure diagonal and right-hand side

    IIL_p = Int[];  # equation indeces into A
    JJL_p = Int[];  # variable indeces into A
    AAL_p = T[]     # coefficients for A

    # top & bottom rows: P[1,:] = P[2,:] and P[end,:] = P[end-1,:]
    # boundary points
    let ii  = vcat(vec(MapP[1, :]),   vec(MapP[end,   :])),
        jj2 = vcat(vec(MapP[2, :]),   vec(MapP[end-1, :])),
        aa   = length(ii)
        append!(IIL_p, ii); append!(JJL_p, ii);  append!(AAL_p, fill(T(1),  aa))
        append!(IIL_p, ii); append!(JJL_p, jj2); append!(AAL_p, fill(T(-1), aa))
    end
    # left & right cols
    let ii  = vcat(vec(MapP[2:end-1, 1]),     vec(MapP[2:end-1, end])),
        jj2 = vcat(vec(MapP[2:end-1, end-1]), vec(MapP[2:end-1, 2])),
        aa   = length(ii)
        append!(IIL_p, ii); append!(JJL_p, ii);  append!(AAL_p, fill(T(1),  aa))
        append!(IIL_p, ii); append!(JJL_p, jj2); append!(AAL_p, fill(T(-1), aa))
    end
    # Pre-add the pin diagonal entries (value 0) so the subsequent
    # `KP[np0, np0] = T(1)` writes into an existing nzval slot instead of
    # restructuring KP — required for `_scatter!` reuse on later calls.
    if bnchm
        nzp_pin = round(Int, (Nz + 2) * 3 / 8);  nxp_pin = round(Int, (Nx + 2) / 2)
        push!(IIL_p, MapP[nzp_pin, nxp_pin])
        push!(JJL_p, MapP[nzp_pin, nxp_pin])
        push!(AAL_p, zero(T))
    elseif open_cnv
        for np0 in vec(MapP[Nz + 1, 1:(Nx + 2)])
            push!(IIL_p, np0); push!(JJL_p, np0); push!(AAL_p, zero(T))
        end
    else
        np0 = MapP[round(Int, Nz / 2), round(Int, Nx / 2)]
        push!(IIL_p, np0); push!(JJL_p, np0); push!(AAL_p, zero(T))
    end
    if cache.initialized
        _scatter!(cache.KP, cache.nz_KP, AAL_p)
        KP = cache.KP
    else
        KP = sparse(IIL_p, JJL_p, AAL_p, NP, NP)
        cache.KP = KP
        cache.nz_KP = _compute_nzmap(IIL_p, JJL_p, KP)
        # mark cache initialized after all four block patterns are built
        cache.initialized = true
        cache.sig_Nz = Nz;  cache.sig_Nx = Nx;  cache.sig_bnchm = bnchm
    end

    # RHS for pressure: mass-flux source
    # MATLAB: RP = sparse(IIR, ones(size(IIR)), AAR, NP, 1) — scatter AAR into
    # a length-NP sparse vector at positions IIR. We use a dense scratch vector
    # so the bnchm pin branch below can mutate a single entry cheaply.
    IIR = vec(MapP[2:end-1, 2:end-1])
    rr_p = copy(MFS)
    if bnchm
        rr_p .+= bnchm_data.src_P[2:end-1, 2:end-1]
    end
    AAR = vec(rr_p)

    RP_vec = zeros(T, NP)
    RP_vec[IIR] .= AAR
    RP = sparsevec(1:NP, RP_vec, NP)

    # ----------------- pin one P (and one U in MMS) -------------------------
    if bnchm
        nzp = round(Int, (Nz + 2) * 3 / 8);  nxp = round(Int, (Nx + 2) / 2)
        np0 = MapP[nzp, nxp]
        _zero_row!(KP, np0);
        KP[np0, np0] = T(1);
        _zero_row!(DM, np0);
        RP_vec[np0] = bnchm_data.P_exact[nzp, nxp];
        RP = sparsevec(1:NP, RP_vec, NP)

        nzu = round(Int, (Nz + 2) / 2);  nxu = round(Int, (Nx + 2) / 2)
        nu0 = MapU[nzu, nxu]
        _zero_row!(KV, nu0);  KV[nu0, nu0] = T(1)
        _zero_row!(GG, nu0)
        RV_vec = Vector(RV);  RV_vec[nu0] = bnchm_data.U_exact[nzu, nxu]
        RV = sparsevec(1:(NW+NU), RV_vec, NW + NU)
    else
        if open_cnv
            nzp = Nz + 1;  nxp = 1:(Nx + 2)
        else
            nzp = round(Int, Nz / 2);  nxp = [round(Int, Nx / 2)]
        end
        np0s = vec(MapP[nzp, nxp])
        for np0 in np0s
            _zero_row!(KP, np0);
            KP[np0, np0] = T(1)
        end
    end

    # assemble and scale global coefficient matrix and right-hand side vector
    LL = [KV GG; DM KP]
    RR = sparsevec(1:(NW + NU + NP),
                   vcat(Vector(RV), Vector(RP)), NW + NU + NP)

    # Jacobi-style diagonal preconditioner (matches MATLAB scaling step)
    diagLL = abs.(diag(LL))
    scl_p  = ones(T, Nz + 2, Nx + 2)
    @views scl_p[2:end-1, 2:end-1] .= rho ./ eta
    extra  = vcat(zeros(T, NU + NW), sqrt.(T(1) ./ vec(scl_p)))
    scl    = T(1) ./ (sqrt.(diagLL) .+ extra)
    SCL    = Diagonal(scl)

    SOL = vcat(vec(W), vec(U), vec(P))
    # collect to dense Vector — UMFPACK \ doesn't accept SparseVector RHS,
    # and the scaling step can leak the sparse type through `Diagonal *` ops.
    FF::Vector{T}  = collect(SCL * (LL * SOL .- Vector(RR)))
    LLs::SparseMatrixCSC{T,Int} = SCL * LL * SCL

    # UMFPACK numeric-refactor reuse: keep one UmfpackLU instance, refactor
    # in-place via `lu!(F, LLs)` on subsequent calls (symbolic stays valid
    # because LLs has the same sparsity each call).
    if cache.lu_F === nothing
        cache.lu_F = lu(LLs)
    else
        try
            lu!(cache.lu_F, LLs)
        catch err
            err isa SparseArrays.UMFPACK.UMFPACKException || rethrow()
            cache.lu_F = lu(LLs)
        end
    end
    UPD_perm = cache.lu_F \ FF
    UPD      = scl .* UPD_perm

    # decode update (matching MATLAB sign convention)
    upd_W = -reshape(UPD[vec(MapW)],           Nz + 1, Nx + 2)
    upd_U = -reshape(UPD[vec(MapU)],           Nz + 2, Nx + 1)
    upd_P = -reshape(UPD[vec(MapP) .+ (NW + NU)], Nz + 2, Nx + 2)

    state.W .= W .+ upd_W
    state.U .= U .+ upd_U
    state.P .= P .+ upd_P
    return state
end

# zero a single row in a SparseMatrixCSC in-place. Cheap helper for the
# constraint-pinning logic. Mutates `A.colptr/nzval` via direct setindex!.
function _zero_row!(A::SparseMatrixCSC{T}, row::Integer) where {T}
    for col in 1:size(A, 2)
        @inbounds for k in A.colptr[col]:(A.colptr[col + 1] - 1)
            if A.rowval[k] == row
                A.nzval[k] = zero(T)
            end
        end
    end
    return A
end
