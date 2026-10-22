using SparseArrays
using LinearAlgebra
using Statistics

# Port of fluidmech.m: assemble the coupled [W; U; P] saddle-point system
# `LL = [KV GG; DM KP]` as a host sparse matrix, apply a symmetric Jacobi
# scaling, factor with sparse LU (UMFPACK), and add the resulting correction to
# the primary unknowns. State fields are copied to the host with `Array`.
#
# The sparsity pattern depends only on (Nz, Nx, bnchm, xBC). The first call
# builds the patterns and the position maps stored in `FluidmechCache`
# (state.jl); later calls only rewrite the stored values and refactor numerically.

"""
    mround(x) -> Int

Round half away from zero, as MATLAB's `round` does. `round(Int, x)` rounds half
to even, which picks a different cell than `fluidmech.m` whenever the argument
lands exactly on a half (`(N+2)/2` at odd `N`, `(N+2)*3/8` at `N = 10, 18, 26, …`).
"""
mround(x::Real) = floor(Int, x + 1//2)

"""
    compute_nzmap(I, J, M) -> Vector{Int}

For each triplet `k`, return the linear index into `M.nzval` where
`M[I[k], J[k]]` lives. Uses the CSC invariant that `M.rowval` is sorted
within each column.
"""
function compute_nzmap(I::Vector{Int}, J::Vector{Int}, M::SparseMatrixCSC{T,Int}) where {T}
    nz_map = Vector{Int}(undef, length(I))
    for k in eachindex(I)
        col = J[k];  row = I[k]
        lo = M.colptr[col];  hi = M.colptr[col + 1] - 1
        pos = searchsortedfirst(view(M.rowval, lo:hi), row) + lo - 1
        nz_map[k] = pos
    end
    return nz_map
end

"""
    scatter_nzval!(M, nz_map, vals) -> M

Zero `M.nzval` then accumulate `vals[k]` into `M.nzval[nz_map[k]]` for every
`k`. Equivalent to `sparse(I, J, vals, m, n)` but skips the sort.
"""
function scatter_nzval!(M::SparseMatrixCSC{T,Int}, nz_map::Vector{Int}, vals::Vector{T}) where {T}
    fill!(M.nzval, zero(T))
    for k in eachindex(vals)
        M.nzval[nz_map[k]] += vals[k]
    end
    return M
end

"""
    block_map(LL, B, roff, coff) -> Vector{Int}

For block `B` placed at global offset `(roff, coff)` inside `LL`, return the
position in `LL.nzval` of each of `B`'s structural nonzeros, in `B.nzval` order.
The four blocks of `[KV GG; DM KP]` occupy disjoint quadrants, so later sweeps
copy each `B.nzval` straight into `LL.nzval` via this map — no sparse concat.
"""
function block_map(LL::SparseMatrixCSC{T,Int}, B::SparseMatrixCSC{T,Int},
                    roff::Int, coff::Int) where {T}
    map = Vector{Int}(undef, nnz(B))
    k = 0
    for j in 1:size(B, 2)
        gcol = j + coff
        lo = LL.colptr[gcol];  hi = LL.colptr[gcol + 1] - 1
        for p in B.colptr[j]:(B.colptr[j + 1] - 1)
            grow = B.rowval[p] + roff
            pos  = searchsortedfirst(view(LL.rowval, lo:hi), grow) + lo - 1
            k += 1;  map[k] = pos
        end
    end
    return map
end

"""Copy `B.nzval` into `LL.nzval` at the positions in `map` (from `block_map`)."""
function fill_block!(LL::SparseMatrixCSC{T,Int}, map::Vector{Int},
                      B::SparseMatrixCSC{T,Int}) where {T}
    nz = B.nzval
    for k in eachindex(nz)
        LL.nzval[map[k]] = nz[k]
    end
    return LL
end

"""Column index of every stored entry of `A`, in `A.nzval` order."""
function col_of_nzval(A::SparseMatrixCSC{T,Int}) where {T}
    col = Vector{Int}(undef, nnz(A))
    for j in 1:size(A, 2), p in A.colptr[j]:(A.colptr[j + 1] - 1)
        col[p] = j
    end
    return col
end

"""
    scale_into!(LLs, LL, Lcol, scl) -> LLs

In-place symmetric Jacobi scaling `LLs = Diagonal(scl)·LL·Diagonal(scl)`,
reusing `LLs`'s storage, which shares `LL`'s structure.
"""
function scale_into!(LLs::SparseMatrixCSC{T,Int}, LL::SparseMatrixCSC{T,Int},
                      Lcol::Vector{Int}, scl::Vector{T}) where {T}
    Lrow = LL.rowval;  Lnz = LL.nzval;  Snz = LLs.nzval
    for k in eachindex(Lnz)
        Snz[k] = scl[Lrow[k]] * Lnz[k] * scl[Lcol[k]]
    end
    return LLs
end


"""
    zero_row!(A, row) -> A

Set every stored entry in row `row` of `A` to zero, keeping the sparsity pattern.
"""
function zero_row!(A::SparseMatrixCSC{T}, row::Integer) where {T}
    for k in eachindex(A.rowval)
        A.rowval[k] == row && (A.nzval[k] = zero(T))
    end
    return A
end

"""
    pressure_pins(MapP, Nz, Nx, bnchm, open_cnv) -> Vector{Int}

Pressure cells replaced by a constraint row, which fixes the pressure gauge:
the cell `(3/8, 1/2)` of the domain in MMS mode, the whole row `Nz + 1` for an
open bottom, otherwise the central cell.
"""
function pressure_pins(MapP, Nz, Nx, bnchm, open_cnv)
    bnchm    && return [MapP[mround((Nz + 2) * 3 / 8), mround((Nx + 2) / 2)]]
    open_cnv && return vec(MapP[Nz + 1, 1:(Nx + 2)])
    return [MapP[mround(Nz / 2), mround(Nx / 2)]]
end

# Triplet storage for one sparse block: equation rows `I`, variable columns `J`,
# coefficients `A`, plus right-hand-side rows `IR` and entries `AR`.
new_triplets(::Type{T}) where {T} = (; I = Int[], J = Int[], A = T[], IR = Int[], AR = T[])

"""
    add_entries!(L, ii, jj, aa)

Append the entries `(ii[k], jj[k]) = aa[k]` to the triplets `L`.
"""
function add_entries!(L, ii, jj, aa)
    append!(L.I, ii);  append!(L.J, jj);  append!(L.A, aa)
    return L
end

"""
    add_rhs!(L, ii, aa)

Append the right-hand-side entries `rhs[ii[k]] = aa[k]` to the triplets `L`.
"""
function add_rhs!(L, ii, aa)
    append!(L.IR, ii);  append!(L.AR, aa)
    return L
end

"""
    add_index_bands!(L, ii, jjs...)

Append the structural indices `(ii, jj)` for each `jj` in `jjs`, in order. They
pair with coefficient bands appended to `L.A` in the same order.
"""
function add_index_bands!(L, ii, jjs...)
    for jj in jjs
        append!(L.I, ii);  append!(L.J, jj)
    end
    return L
end

"""
    stokes_bands(EtaP1, EtaP2, EtaC1, EtaC2, invh2)

The eight viscous stencil bands shared by the z- and x-momentum equations, in
the order center, first/second normal neighbor, first/second tangential
neighbor, then the four cross-derivative bands. `EtaP*` are the cell-centered
and `EtaC*` the edge viscosities adjacent to the row.
"""
function stokes_bands(EtaP1::AbstractVector{T}, EtaP2, EtaC1, EtaC2, invh2) where {T}
    center = T(2/3).*(EtaP1.+EtaP2).*invh2 .+ T(1/2).*(EtaC1.+EtaC2).*invh2
    n1 = -T(2/3) .* EtaP1 .* invh2
    n2 = -T(2/3) .* EtaP2 .* invh2
    t1 = -T(1/2) .* EtaC1 .* invh2
    t2 = -T(1/2) .* EtaC2 .* invh2
    c1 = -(T(1/2).*EtaC1 .- T(1/3).*EtaP1) .* invh2
    c2 = +(T(1/2).*EtaC1 .- T(1/3).*EtaP2) .* invh2
    c3 = +(T(1/2).*EtaC2 .- T(1/3).*EtaP1) .* invh2
    c4 = -(T(1/2).*EtaC2 .- T(1/3).*EtaP2) .* invh2
    return center, n1, n2, t1, t2, c1, c2, c3, c4
end

"""
    update_mfs!(state, phase, par; dt, a1, a2, a3, b1, b2, b3)

Accumulate the mass-flux residual of the density equation into `state.MFS`
(fluidmech.m:9-23), so the divergence equation enforces `∂ₜρ = -∇·(ρv)`.
`phase.advn_rho` is the advective density flux produced by `phsevo!`.
"""
function update_mfs!(state::FluidState{T}, phase, par; dt, a1, a2, a3, b1, b2, b3) where {T}
    rho = Array(state.rho);  rhoo = Array(state.rhoo);  rhooo = Array(state.rhooo)
    drhodt = Array(state.drhodt);  drhodto = Array(state.drhodto);  drhodtoo = Array(state.drhodtoo)
    MFS = Array(state.MFS)
    advn_rho = Array(phase.advn_rho)
    @. drhodt = -advn_rho
    res_rho = (T(a1) .* rho .- T(a2) .* rhoo .- T(a3) .* rhooo) ./ T(dt) .-
              (T(b1) .* drhodt .+ T(b2) .* drhodto .+ T(b3) .* drhodtoo)
    @. MFS = MFS - par.alpha * res_rho
    state.drhodt .= drhodt
    state.MFS    .= MFS
    return state
end

"""
    z_momentum_rows!(L, f, g, par, tc, o, cache, bnchm_data)

Append the W-equation rows of KV to `L`: the side, top, and bottom boundary rows
and the interior z-momentum balance with its right-hand side. `f` holds the host
fields, `g` the grid constants and index maps, `tc` the time-integration
coefficients, and `o` the boundary options.
"""
function z_momentum_rows!(L, f, g, par, tc, o, cache, bnchm_data)
    T = eltype(f.eta)
    (; MapW, MapU, invh, invh2, periodic, bnchm) = g
    (; dt, a1, a2, a3) = tc
    # Side boundaries: W[:,1] + sds * W[:,j] = 0. Periodic pairs the ghost column
    # with the far interior column; a solid wall pairs it with its own neighbor,
    # so sds = -1 gives ∂ₓW = 0 (free slip) and sds = +1 gives W = 0 (no slip).
    for (ii, jj2) in ((vec(MapW[:, 1]),   periodic ? vec(MapW[:, end - 1]) : vec(MapW[:, 2])),
                      (vec(MapW[:, end]), periodic ? vec(MapW[:, 2])       : vec(MapW[:, end - 1])))
        aa = length(ii)
        add_entries!(L, ii, ii,  fill(T(1),     aa))
        add_entries!(L, ii, jj2, fill(T(o.sds), aa))
        add_rhs!(L, ii, zeros(T, aa))
    end
    # Top boundary: W = 0.
    let ii = vec(MapW[1, 2:end-1])
        add_entries!(L, ii, ii, fill(T(1), length(ii)))
        add_rhs!(L, ii, zeros(T, length(ii)))
    end
    # Bottom boundary: closed (W = 0) or open (∂W/∂z = -∂U/∂x, weighted by density).
    let ii  = vec(MapW[end, 2:end-1]),
        jj2 = vec(MapW[end - 1, 2:end-1]),
        jj3 = vec(MapU[end - 1, 2:end]),
        jj4 = vec(MapU[end - 1, 1:end-1]),
        rho1 = vec(f.rhow[end, :]),
        rho2 = vec(f.rhow[end - 1, :]),
        rho3 = vec(f.rhou[end, 2:end]),
        rho4 = vec(f.rhou[end, 1:end-1]),
        oc  = o.open_cnv ? T(1) : T(0)
        add_entries!(L, ii, ii,  rho1 .* invh)
        add_entries!(L, ii, jj2, -oc .* rho2 .* invh)
        add_entries!(L, ii, jj3,  oc .* rho3 .* invh)
        add_entries!(L, ii, jj4, -oc .* rho4 .* invh)
        # Closed bottom: the matrix mass flux enters as the mean mass-flux
        # source times the depth over h; open bottom: the local source.
        add_rhs!(L, ii, oc .* vec(f.MFS[end, :]) .+ (T(1) - oc) * g.mfbg)
    end

    # Interior points. Coefficient bands are appended to `L.A` every sweep in a
    # fixed order; the matching index bands are appended on the first call only.
    @timeit_debug TO "assemble:stencils" begin
        EtaC1 = vec(f.etaco[2:end-1, 1:end-1]);  EtaC2 = vec(f.etaco[2:end-1, 2:end])
        EtaP1 = vec(f.eta[1:end-1, :]);          EtaP2 = vec(f.eta[2:end, :])
        ii    = vec(MapW[2:end-1, 2:end-1])
        center, n1, n2, t1, t2, c1, c2, c3, c4 = stokes_bands(EtaP1, EtaP2, EtaC1, EtaC2, invh2)

        append!(L.A, a1 .* vec(f.rhow[2:end-1, :]) ./ T(dt))          # inertia
        append!(L.A, center)
        append!(L.A, n1);  append!(L.A, n2);  append!(L.A, t1);  append!(L.A, t2)
        if !bnchm                                                      # drunken sailor
            append!(L.A, vec((f.rho[2:end, :] .- f.rho[1:end-1, :]) .* invh) .* par.g0 .* T(dt))
        end
        append!(L.A, c1);  append!(L.A, c2);  append!(L.A, c3);  append!(L.A, c4)

        if g.first
            jj1 = vec(MapW[1:end-2, 2:end-1]); jj2 = vec(MapW[3:end, 2:end-1])
            jj3 = vec(MapW[2:end-1, 1:end-2]); jj4 = vec(MapW[2:end-1, 3:end])
            jU1 = vec(MapU[2:end-2, 1:end-1]); jU2 = vec(MapU[3:end-1, 1:end-1])
            jU3 = vec(MapU[2:end-2, 2:end]);   jU4 = vec(MapU[3:end-1, 2:end])
            add_index_bands!(L, ii, ii, ii, jj1, jj2, jj3, jj4)
            bnchm || add_index_bands!(L, ii, ii)
            add_index_bands!(L, ii, jU1, jU2, jU3, jU4)
        end

        # z-momentum right-hand side, reusing the cached advection buffer.
        f_mz = f.rhow[2:end-1, :] .* f.W[2:end-1, 2:end-1]
        u_mz = (f.U[2:end-2, :] .+ f.U[3:end-1, :]) ./ T(2)
        w_mz = (f.W[1:end-1, 2:end-1] .+ f.W[2:end, 2:end-1]) ./ T(2)
        size(cache.advn_mz) == size(f_mz) || (cache.advn_mz = similar(f_mz))
        advn_mz = cache.advn_mz
        advect!(advn_mz, f_mz, (w_mz, u_mz), g.h, par.ADVN; xBC = o.xBC, zBC = o.zBC)
        rr = f.Drho[2:end-1, :] .* par.g0 .+
             (a2 .* f.rhoWo[2:end-1, :] .+ a3 .* f.rhoWoo[2:end-1, :]) ./ T(dt) .- advn_mz
        bnchm && (rr .+= bnchm_data.src_W[2:end-1, 2:end-1])
        add_rhs!(L, ii, vec(rr))
    end
    return L
end

"""
    x_momentum_rows!(L, f, g, par, tc, o, cache, bnchm_data)

Append the U-equation rows of KV to `L`, as `z_momentum_rows!` does for W. Under
periodic sides the outermost x-faces are interior faces; with solid walls they
are Dirichlet rows and every interior x-slice narrows by one column per side.
"""
function x_momentum_rows!(L, f, g, par, tc, o, cache, bnchm_data)
    T = eltype(f.eta)
    (; MapW, MapU, invh2, periodic, bnchm, Nx) = g
    (; dt, a1, a2, a3) = tc
    # Top and bottom boundaries: U + cnv * U[neighbor] = 0 (free/no slip).
    for (row, nb, cnv) in ((1, 2, o.top_cnv), (size(MapU, 1), size(MapU, 1) - 1, o.bot_cnv))
        ii = vec(MapU[row, :]);  jj2 = vec(MapU[nb, :]);  n = length(ii)
        add_entries!(L, ii, ii,  fill(T(1),   n))
        add_entries!(L, ii, jj2, fill(T(cnv), n))
        add_rhs!(L, ii, zeros(T, n))
    end
    # Solid walls: the wall-normal velocity is zero.
    if !periodic
        ii = vcat(vec(MapU[2:end-1, 1]), vec(MapU[2:end-1, end]))
        add_entries!(L, ii, ii, fill(T(1), length(ii)))
        add_rhs!(L, ii, zeros(T, length(ii)))
    end

    @timeit_debug TO "assemble:stencils" begin
        icx = periodic ? [Nx; collect(1:Nx); 1] : [1; collect(1:Nx); Nx]
        ifx = [Nx; collect(1:(Nx+1)); 2]
        EtaC1 = periodic ? vec(f.etaco[1:end-1, :])    : vec(f.etaco[1:end-1, 2:end-1])
        EtaC2 = periodic ? vec(f.etaco[2:end, :])      : vec(f.etaco[2:end, 2:end-1])
        EtaP1 = periodic ? vec(f.eta[:, icx[1:end-1]]) : vec(f.eta[:, 1:end-1])
        EtaP2 = periodic ? vec(f.eta[:, icx[2:end]])   : vec(f.eta[:, 2:end])
        ii    = periodic ? vec(MapU[2:end-1, :]) : vec(MapU[2:end-1, 2:end-1])
        rhou_i = periodic ? f.rhou : f.rhou[:, 2:end-1]
        center, n1, n2, t1, t2, c1, c2, c3, c4 = stokes_bands(EtaP1, EtaP2, EtaC1, EtaC2, invh2)

        # γ is the artificial horizontal inertia, defined only for periodic sides.
        append!(L.A, vec((a1 + (periodic ? par.gamma : zero(par.gamma))) .* rhou_i ./ T(dt)))
        append!(L.A, center)
        append!(L.A, n1);  append!(L.A, n2);  append!(L.A, t1);  append!(L.A, t2)
        if !bnchm                                                      # drunken sailor
            ddx_rho = periodic ? (f.rho[:, icx[2:end]] .- f.rho[:, icx[1:end-1]]) :
                                 (f.rho[:, 2:end] .- f.rho[:, 1:end-1])
            append!(L.A, vec(ddx_rho .* g.invh) .* par.g0 .* T(dt))
        end
        append!(L.A, c1);  append!(L.A, c2);  append!(L.A, c3);  append!(L.A, c4)

        if g.first
            jj1 = periodic ? vec(MapU[2:end-1, ifx[1:end-2]]) : vec(MapU[2:end-1, 1:end-2])
            jj2 = periodic ? vec(MapU[2:end-1, ifx[3:end]])   : vec(MapU[2:end-1, 3:end])
            jj3 = periodic ? vec(MapU[1:end-2, ifx[2:end-1]]) : vec(MapU[1:end-2, 2:end-1])
            jj4 = periodic ? vec(MapU[3:end,   ifx[2:end-1]]) : vec(MapU[3:end,   2:end-1])
            jW1 = periodic ? vec(MapW[1:end-1, 1:end-1]) : vec(MapW[1:end-1, 2:end-2])
            jW2 = periodic ? vec(MapW[1:end-1, 2:end])   : vec(MapW[1:end-1, 3:end-1])
            jW3 = periodic ? vec(MapW[2:end,   1:end-1]) : vec(MapW[2:end,   2:end-2])
            jW4 = periodic ? vec(MapW[2:end,   2:end])   : vec(MapW[2:end,   3:end-1])
            add_index_bands!(L, ii, ii, ii, jj1, jj2, jj3, jj4)
            bnchm || add_index_bands!(L, ii, ii)
            add_index_bands!(L, ii, jW1, jW2, jW3, jW4)
        end

        # x-momentum right-hand side, reusing the cached advection buffer. Under
        # periodic sides the two periodic-equivalent boundary columns are
        # averaged so they agree (fluidmech.m:164).
        if periodic
            u_mx = (f.U[2:end-1, ifx[1:end-1]] .+ f.U[2:end-1, ifx[2:end]]) ./ T(2)
            w_mx = (f.W[:, 1:end-1] .+ f.W[:, 2:end]) ./ T(2)
            f_mx = f.rhou .* f.U[2:end-1, :]
        else
            u_mx = (f.U[2:end-1, 1:end-1] .+ f.U[2:end-1, 2:end]) ./ T(2)
            w_mx = (f.W[:, 2:end-2] .+ f.W[:, 3:end-1]) ./ T(2)
            f_mx = f.rhou[:, 2:end-1] .* f.U[2:end-1, 2:end-1]
        end
        size(cache.advn_mx) == size(f_mx) || (cache.advn_mx = similar(f_mx))
        advn_mx = cache.advn_mx
        advect!(advn_mx, f_mx, (w_mx, u_mx), g.h, par.ADVN; xBC = o.xBC, zBC = o.zBC)
        if periodic
            col_avg = (advn_mx[:, 1] .+ advn_mx[:, end]) ./ T(2)
            advn_mx[:, 1]   .= col_avg
            advn_mx[:, end] .= col_avg
        end
        rr = periodic ?
            (a2 .* f.rhoUo .+ a3 .* f.rhoUoo) ./ T(dt) .- advn_mx :
            (a2 .* f.rhoUo[:, 2:end-1] .+ a3 .* f.rhoUoo[:, 2:end-1]) ./ T(dt) .- advn_mx
        if bnchm
            rr .+= periodic ? bnchm_data.src_U[2:end-1, :] :
                              bnchm_data.src_U[2:end-1, 2:end-1]
        end
        add_rhs!(L, ii, vec(rr))
    end
    return L
end

"""
    gradient_triplets(g, T)

Triplets of the pressure-gradient block GG, which maps `P` to the W and U
momentum equations.
"""
function gradient_triplets(g, ::Type{T}) where {T}
    (; MapP, MapW, MapU, invh, periodic) = g
    L = new_triplets(T)
    # z-gradient: one cell above and one below
    ii = vec(MapW[2:end-1, 2:end-1])
    add_entries!(L, ii, vec(MapP[2:end-2, 2:end-1]), fill(-invh, length(ii)))
    add_entries!(L, ii, vec(MapP[3:end-1, 2:end-1]), fill(+invh, length(ii)))
    # x-gradient: one cell left and one right
    ii  = periodic ? vec(MapU[2:end-1, :]) : vec(MapU[2:end-1, 2:end-1])
    jj1 = periodic ? vec(MapP[2:end-1, 1:end-1]) : vec(MapP[2:end-1, 2:end-2])
    jj2 = periodic ? vec(MapP[2:end-1, 2:end])   : vec(MapP[2:end-1, 3:end-1])
    add_entries!(L, ii, jj1, fill(-invh, length(ii)))
    add_entries!(L, ii, jj2, fill(+invh, length(ii)))
    return L
end

"""
    divergence_triplets(f, g)

Triplets of the matrix mass-flux divergence block DM, which maps `(W, U)` to the
continuity equation. Index bands are appended on the first call only.
"""
function divergence_triplets(f, g)
    (; MapP, MapW, MapU, invh) = g
    L = new_triplets(eltype(f.eta))
    @timeit_debug TO "assemble:stencils" begin
        append!(L.A, -vec(f.rhou[:, 1:end-1]) .* invh)   # left U
        append!(L.A, +vec(f.rhou[:, 2:end])   .* invh)   # right U
        append!(L.A, -vec(f.rhow[1:end-1, :]) .* invh)   # top W
        append!(L.A, +vec(f.rhow[2:end,   :]) .* invh)   # bottom W
        if g.first
            ii = vec(MapP[2:end-1, 2:end-1])
            add_index_bands!(L, ii, vec(MapU[2:end-1, 1:end-1]), vec(MapU[2:end-1, 2:end]),
                                    vec(MapW[1:end-1, 2:end-1]), vec(MapW[2:end, 2:end-1]))
        end
    end
    return L
end

"""
    pressure_triplets(g, pins, T)

Triplets of the pressure block KP: Neumann ghost rows and columns, plus a zero
diagonal entry at every pin cell so that the later pin write
`KP[p, p] = 1` stays inside the stored pattern.
"""
function pressure_triplets(g, pins, ::Type{T}) where {T}
    (; MapP, periodic) = g
    L = new_triplets(T)
    # top and bottom rows: P[1,:] = P[2,:], P[end,:] = P[end-1,:]
    ii  = vcat(vec(MapP[1, :]), vec(MapP[end, :]))
    jj2 = vcat(vec(MapP[2, :]), vec(MapP[end-1, :]))
    add_entries!(L, ii, ii,  fill(T(1),  length(ii)))
    add_entries!(L, ii, jj2, fill(T(-1), length(ii)))
    # left and right columns: periodic wraps the ghost onto the far interior
    # column, a solid wall repeats its own neighbor.
    ii  = vcat(vec(MapP[2:end-1, 1]), vec(MapP[2:end-1, end]))
    jj2 = periodic ? vcat(vec(MapP[2:end-1, end-1]), vec(MapP[2:end-1, 2])) :
                     vcat(vec(MapP[2:end-1, 2]),     vec(MapP[2:end-1, end-1]))
    add_entries!(L, ii, ii,  fill(T(1),  length(ii)))
    add_entries!(L, ii, jj2, fill(T(-1), length(ii)))
    add_entries!(L, pins, pins, zeros(T, length(pins)))
    return L
end

"""
    assemble_block!(first, M, nz_map, L, m, n) -> (M, nz_map)

First call: build the block from the triplets `L` and its `nzval` position map.
Later calls: scatter the new coefficients `L.A` into the existing `M`.
"""
function assemble_block!(first::Bool, M, nz_map, L, m, n)
    @timeit_debug TO "assemble:build" if first
        M = sparse(L.I, L.J, L.A, m, n)
        nz_map = compute_nzmap(L.I, L.J, M)
    else
        scatter_nzval!(M, nz_map, L.A)
    end
    return M, nz_map
end

"""
    merge_blocks!(cache, KV, GG, DM, KP, nv) -> LL

Return `LL = [KV GG; DM KP]` (`nv` = number of velocity unknowns). The four
blocks occupy disjoint quadrants of fixed sparsity, so after the first call the
block values are copied straight into `cache.LL.nzval`.
"""
function merge_blocks!(cache, KV, GG, DM, KP, nv)
    if cache.LL_initialized
        @timeit_debug TO "merge" begin
            fill_block!(cache.LL, cache.map_KV, KV)
            fill_block!(cache.LL, cache.map_GG, GG)
            fill_block!(cache.LL, cache.map_DM, DM)
            fill_block!(cache.LL, cache.map_KP, KP)
        end
        return cache.LL
    end
    LL = @timeit_debug TO "merge" [KV GG; DM KP]
    cache.LL     = LL
    cache.map_KV = block_map(LL, KV, 0,  0)
    cache.map_GG = block_map(LL, GG, 0,  nv)
    cache.map_DM = block_map(LL, DM, nv, 0)
    cache.map_KP = block_map(LL, KP, nv, nv)
    cache.Lcol   = col_of_nzval(LL)
    cache.LLs    = SparseMatrixCSC(size(LL, 1), size(LL, 2),
                                   copy(LL.colptr), copy(LL.rowval), similar(LL.nzval))
    cache.LL_initialized = true
    return LL
end

"""
    factor!(cache, LLs) -> UmfpackLU

Numeric refactorization of `LLs` into the cached LU object, reusing its symbolic
analysis (the sparsity pattern is identical on every call). A UMFPACK failure
triggers a fresh factorization.
"""
function factor!(cache, LLs)
    @timeit_debug TO "factor" if cache.lu_F === nothing
        cache.lu_F = lu(LLs)
    else
        try
            lu!(cache.lu_F, LLs)
        catch err
            err isa SparseArrays.UMFPACK.UMFPACKException || rethrow()
            cache.lu_F = lu(LLs)
        end
    end
    return cache.lu_F
end

"""
    fluidmech!(state, grid, par;
               phase = nothing, bnchm_data = nothing,
               sds = -1, top_cnv = 1, bot_cnv = 1, open_cnv = false,
               xBC = :periodic, zBC = :closed,
               dt = T(1e32), a1 = T(1), a2 = T(1), a3 = T(0),
               b1 = T(1), b2 = T(0), b3 = T(0)) -> state

Solve the linear momentum + continuity system for `(W, U, P)` and add the
solution increment to `state` in place. With `bnchm_data ≠ nothing` the
manufactured-solution branch is taken: the right-hand sides gain `src_*`
forcing and one `P` and one `U` cell are pinned to the exact solution values.
Otherwise `phase` supplies the density flux for the mass-flux source update.

Boundary flags follow the MATLAB convention:
* `xBC` selects the side boundaries: `:periodic` wraps left onto right,
  anything else makes them solid walls.
* `sds`, `top_cnv`, `bot_cnv` ∈ {-1, +1} encode free-slip (-1) vs no-slip (+1)
  on the sides and on the top/bottom z-boundaries. `sds` must be `-1` under
  `xBC = :periodic`, where it carries the wrap instead.
* `open_cnv = true` opens the bottom z-boundary for the z-velocity equation.

Time-integration coefficients `a1, a2, a3` (momentum and density) and
`b1, b2, b3` (density flux) come from `src/timing.m`. In MMS mode the inertial
term is annihilated by `dt → ∞`.
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
    Nz = grid.ni[1];  Nx = grid.ni[2];  h = grid.h

    # The cached sparsity depends on the grid size, the MMS flag, and the side mode.
    cache = state.solver
    if cache.initialized && (cache.sig_Nz != Nz || cache.sig_Nx != Nx ||
                             cache.sig_bnchm != bnchm || cache.sig_xBC != xBC)
        cache.initialized = false
        cache.LL_initialized = false
        cache.lu_F = nothing
    end

    # Flat unknown numbering (fluidmech.m MapP/MapW/MapU): [W; U; P].
    NP = (Nz + 2) * (Nx + 2)
    NW = (Nz + 1) * (Nx + 2)
    NU = (Nz + 2) * (Nx + 1)
    NV = NW + NU
    MapP = reshape(1:NP, Nz + 2, Nx + 2)
    MapW = reshape(1:NW, Nz + 1, Nx + 2)
    MapU = reshape(1:NU, Nz + 2, Nx + 1) .+ NW

    bnchm || update_mfs!(state, phase, par; dt, a1, a2, a3, b1, b2, b3)

    W = Array(state.W);  U = Array(state.U);  P = Array(state.P)
    f = (; W, U, eta = Array(state.eta), etaco = Array(state.etaco),
         rho = Array(state.rho), rhow = Array(state.rhow), rhou = Array(state.rhou),
         Drho = Array(state.Drho), MFS = Array(state.MFS),
         rhoWo = Array(state.rhoWo), rhoWoo = Array(state.rhoWoo),
         rhoUo = Array(state.rhoUo), rhoUoo = Array(state.rhoUoo))
    MFSmean = T(mean(f.MFS))

    periodic = xBC === :periodic
    invh = inv(h)
    g = (; Nz, Nx, h, invh, invh2 = inv(h)^2, MapP, MapW, MapU, periodic, bnchm,
         first = !cache.initialized, mfbg = MFSmean * grid.li[1] / h)
    tc = (; dt, a1, a2, a3)
    o  = (; sds, top_cnv, bot_cnv, open_cnv, xBC, zBC)

    # ---- assemble KV (momentum), GG (gradient), DM (mass flux), KP (pressure)
    LV = new_triplets(T)
    z_momentum_rows!(LV, f, g, par, tc, o, cache, bnchm_data)
    x_momentum_rows!(LV, f, g, par, tc, o, cache, bnchm_data)
    LG = gradient_triplets(g, T)
    LD = divergence_triplets(f, g)
    pins = pressure_pins(MapP, Nz, Nx, bnchm, open_cnv)
    LP = pressure_triplets(g, pins, T)

    KV, cache.nz_KV = assemble_block!(g.first, cache.KV, cache.nz_KV, LV, NV, NV)
    GG, cache.nz_GG = assemble_block!(g.first, cache.GG, cache.nz_GG, LG, NV, NP)
    DM, cache.nz_DM = assemble_block!(g.first, cache.DM, cache.nz_DM, LD, NP, NV)
    KP, cache.nz_KP = assemble_block!(g.first, cache.KP, cache.nz_KP, LP, NP, NP)
    cache.KV = KV;  cache.GG = GG;  cache.DM = DM;  cache.KP = KP
    if g.first
        cache.initialized = true
        cache.sig_Nz = Nz;  cache.sig_Nx = Nx;  cache.sig_bnchm = bnchm;  cache.sig_xBC = xBC
    end

    # Right-hand sides. Only the stored rows of KV's triplets are nonzero.
    RV = zeros(T, NV)
    RV[LV.IR] .= LV.AR
    RP = zeros(T, NP)
    rr_p = copy(f.MFS)
    bnchm && (rr_p .+= bnchm_data.src_P[2:end-1, 2:end-1])
    RP[vec(MapP[2:end-1, 2:end-1])] .= vec(rr_p)

    # ---- fix the pressure gauge (and, in MMS mode, one U cell)
    for np0 in pins
        zero_row!(KP, np0)
        KP[np0, np0] = T(1)
    end
    if bnchm
        np0 = only(pins)
        zero_row!(DM, np0)
        RP[np0] = bnchm_data.P_exact[mround((Nz + 2) * 3 / 8), mround((Nx + 2) / 2)]

        nzu = mround((Nz + 2) / 2);  nxu = mround((Nx + 2) / 2)
        nu0 = MapU[nzu, nxu]
        zero_row!(KV, nu0);  KV[nu0, nu0] = T(1)
        zero_row!(GG, nu0)
        RV[nu0] = bnchm_data.U_exact[nzu, nxu]
    end

    LL = merge_blocks!(cache, KV, GG, DM, KP, NV)
    RR = vcat(RV, RP)

    # Jacobi-style diagonal scaling (matches the MATLAB scaling step); the
    # pressure rows additionally carry sqrt(η/ρ).
    diagLL = abs.(diag(LL))
    scl_p  = ones(T, Nz + 2, Nx + 2)
    @views scl_p[2:end-1, 2:end-1] .= f.rho ./ f.eta
    extra  = vcat(zeros(T, NV), sqrt.(T(1) ./ vec(scl_p)))
    scl    = collect(T(1) ./ (sqrt.(diagLL) .+ extra))

    # Scaled system: (D·LL·D) y = D·(LL·SOL − RR), with D = Diagonal(scl).
    SOL = vcat(vec(W), vec(U), vec(P))
    LLs::SparseMatrixCSC{T,Int} = @timeit_debug TO "scale" scale_into!(cache.LLs, LL, cache.Lcol, scl)
    FF::Vector{T} = @timeit_debug TO "scale" collect(scl .* (LL * SOL .- RR))

    lu_F = factor!(cache, LLs)
    UPD_perm = @timeit_debug TO "solve" lu_F \ FF
    UPD = scl .* UPD_perm

    # The solve returns the negative increment (MATLAB sign convention).
    state.W .= W .- reshape(UPD[vec(MapW)],            Nz + 1, Nx + 2)
    state.U .= U .- reshape(UPD[vec(MapU)],            Nz + 2, Nx + 1)
    state.P .= P .- reshape(UPD[vec(MapP) .+ NV],      Nz + 2, Nx + 2)
    return state
end
