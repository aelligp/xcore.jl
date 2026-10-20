# Multigrid hierarchy: per level a grid, a fluid state holding the level's
# (v, P) and coarsened coefficients, a `StokesCache`, and a pseudo-transient
# reference pressure `P0`. Level 1 wraps the caller's fluid state.

"""
    coarsen(grid; xBC = :periodic) -> grid

Halve every cell count and double `h`. Errors on a dimension that cannot be halved.
"""
function coarsen(grid::Grid{T}; xBC::Symbol = :periodic) where {T}
    Nz, Nx = grid.Nz, grid.Nx
    (iseven(Nz) && (iseven(Nx) || Nx == 1)) ||
        error("multigrid: cannot coarsen grid $(Nz)×$(Nx) (need even dims)")
    Nz, Nx, h = Nz ÷ 2, (Nx == 1 ? 1 : Nx ÷ 2), grid.h * 2
    Xc = collect(((1:Nx) .- T(0.5)) .* h);  Zc = collect(((1:Nz) .- T(0.5)) .* h)
    Xf = collect((0:Nx) .* h);  Zf = collect((0:Nz) .* h)
    XX = repeat(reshape(Xc, 1, Nx), Nz, 1);  ZZ = repeat(reshape(Zc, Nz, 1), 1, Nx)
    icx = xBC === :periodic ? [Nx; collect(1:Nx); 1] : [1; collect(1:Nx); Nx]
    icz = [1; collect(1:Nz); Nz]
    ifx = [Nx; collect(1:Nx + 1); 2]
    ifz = [2; collect(1:Nz + 1); Nz]
    return Grid{T}(Nx, Nz, h, grid.D, grid.L, Xc, Zc, Xf, Zf, XX, ZZ, icx, icz, ifx, ifz)
end
coarsen(grid::Grid3D; kw...) = coarsen_grid3d(grid)

"""
    n_levels(N; minlvl) -> Int

Number of levels from `N` cells down to the coarsest grid whose dimensions stay
≥ `minlvl` (a horizontal extent of 1 is kept).
"""
function n_levels(N::NTuple{D, Int}; minlvl::Int = 4) where {D}
    lvls = 1
    ok(n, k) = k == 1 ? (iseven(n) && n ÷ 2 ≥ minlvl) :
                        ((iseven(n) || n == 1) && (n == 1 || n ÷ 2 ≥ minlvl))
    while all(ntuple(k -> ok(N[k], k), Val(D)))
        N = ntuple(k -> N[k] == 1 ? 1 : N[k] ÷ 2, Val(D));  lvls += 1
    end
    return lvls
end

struct MGLevel{G, F, C, A}
    grid::G
    fluid::F
    cache::C
    P0::A          # pseudo-transient reference pressure (see `vcycle!`)
end

struct MGHierarchy{T, L}
    levels::Vector{L}     # levels[1] = finest
    minlvl::Int
    npre::Int
    npost::Int
    ncoarse::Int          # max coarsest-level PH steps
    coarse_rtol::T        # coarsest-level relative residual drop
    inner::Int            # DR iterations per smoothing PH step
    γ_mg::T               # coarse-correction weight
end

"""
    build_hierarchy(fluid, grid, par; minlvl, npre, npost, ncoarse, coarse_rtol,
                    inner, γ_mg, xBC) -> MGHierarchy

Level 1 wraps `fluid` with a fresh cache; coarse levels own zero-initialized
states down to the coarsest grid with every dimension ≥ `minlvl`. Coefficients
are filled per solve by `refresh_coefficients!`. Defaults from `par.mg_*`.
Every dimension must be a multiple of 32 (clean coarsening, GPU warps).
"""
function build_hierarchy(fluid, grid, par::Parameters{T};
                         minlvl::Int = par.mg_minlvl, npre::Int = par.mg_npre,
                         npost::Int = par.mg_npost, ncoarse::Int = par.mg_ncoarse,
                         coarse_rtol::Real = par.mg_coarse_rtol, inner::Int = par.mg_inner,
                         γ_mg::Real = par.mg_gamma, xBC::Symbol = :periodic) where {T}
    N = cells(grid)
    all(n -> n % 32 == 0, N) || throw(ArgumentError(
        "multigrid solvers need every grid dimension to be a multiple of 32 for clean " *
        "coarsening and GPU-friendly sizes (got $(join(N, "×"))); e.g. N = 128/192/256 " *
        "(with L = 1.5·D choose N a multiple of 64 so that Nx = 1.5N is a multiple of 32)"))
    backend = KernelAbstractions.get_backend(fluid.P)
    nlev = n_levels(N; minlvl)
    # Level 1's P0 is supplied per solve. Coarse levels keep a permanently zero
    # P0: their P is the pressure correction (started at 0 each cycle), so
    # (P − 0)/(η_b·dt) is the linearized pseudo-transient term the fine operator
    # applies to its own per-cycle increment.
    z(g) = xcore_zeros(backend, T, (cells(g) .+ 2)...)
    levels = [MGLevel(grid, fluid, StokesCache(T, backend, grid), z(grid))]
    g = grid
    for _ in 2:nlev
        g = coarsen(g; xBC)
        push!(levels, MGLevel(g, _similar_fluid(fluid, backend, cells(g)),
                              StokesCache(T, backend, g), z(g)))
    end
    return MGHierarchy{T, eltype(levels)}(levels, minlvl, npre, npost, ncoarse,
                                          T(coarse_rtol), inner, T(γ_mg))
end

# face densities: mean of the two adjacent cells (edge cell repeated at a bound)
@kernel function _face_density_kernel!(ρk, @Const(rho), axis, periodic::Bool)
    J = @index(Global, Cartesian)
    @inbounds begin
        k = _val(axis);  n = size(rho, k)
        lo = J[k] == 1     ? _setidx(J, k, periodic ? n : 1) : J - _unit(J, k)
        hi = J[k] == n + 1 ? _setidx(J, k, periodic ? 1 : n) : J
        ρk[J] = (rho[lo] + rho[hi]) / 2
    end
end

# edge viscosity of the axis pair (a, b): geometric mean of the four cells
# around the edge (edge cell repeated at a bound)
@kernel function _edge_viscosity_kernel!(ηab, @Const(eta), pa, pb, periodic)
    E = @index(Global, Cartesian)
    @inbounds begin
        a = _val(pa);  b = _val(pb)
        cell(k, up) = (n = size(eta, k);
                       up ? (E[k] == n + 1 ? (periodic[k] ? 1 : n) : E[k]) :
                            (E[k] == 1 ? (periodic[k] ? n : 1) : E[k] - 1))
        c00 = _setidx(_setidx(E, a, cell(a, false)), b, cell(b, false))
        c10 = _setidx(c00, a, cell(a, true))
        c01 = _setidx(c00, b, cell(b, true))
        c11 = _setidx(c10, b, cell(b, true))
        ηab[E] = (eta[c00] * eta[c10] * eta[c01] * eta[c11])^oftype(eta[c00], 0.25)
    end
end

"""
    derived_coefficients!(fluid, op) -> nothing

Rebuild the face densities and edge viscosities of `fluid` from its cell
`rho` and `eta` (arithmetic face means, geometric edge means).
"""
function derived_coefficients!(fluid, op::StokesOp{D}) where {D}
    backend = KernelAbstractions.get_backend(fluid.rho)
    ρf = face_density(fluid);  ηe = edge_viscosity(fluid)
    for k in 1:D
        _face_density_kernel!(backend)(ρf[k], fluid.rho, Val(k), op.periodic[k]; ndrange = size(ρf[k]))
    end
    for a in 1:D-1, b in a+1:D
        η = ηe[_pair(a, b, D)]
        _edge_viscosity_kernel!(backend)(η, fluid.eta, Val(a), Val(b), op.periodic; ndrange = size(η))
    end
    return nothing
end

# Fixed per-level smoother damping: a smoother only damps the upper half of its
# level's preconditioned spectrum (λ ≳ λ̄max/4; coarser levels handle the rest),
# whose critical damping is c = 2·√(λ̄max/4) = √λ̄max. The coarsest level's
# adaptive solve overrides c through its own Rayleigh retunes.
function _set_smoother_damping!(cache::StokesCache{D, T}, mg_cfact::Real) where {D, T}
    λ̄ = foldl(+, ntuple(d -> sum(cache.λmax[d]), Val(D))) /
         foldl(+, ntuple(d -> length(cache.λmax[d]), Val(D)))
    cache.c = T(mg_cfact) * sqrt(λ̄)
    return nothing
end

"""
    refresh_coefficients!(hier, par, op; γfact = nothing) -> nothing

Restrict η and ρ down the hierarchy (cell means), rebuild each coarse level's
face densities and edge viscosities from them, and set every level's penalty,
diagonals, smoother damping and dτ/α/β. The finest level's coefficients are
live. Smoothing levels use `par.mg_gfact`; the coarsest level keeps the penalty
factor its own solve learned (`par.γfact_PT` at first). `γfact` overrides all.
"""
function refresh_coefficients!(hier::MGHierarchy{T}, par::Parameters, op::StokesOp;
                               γfact = nothing) where {T}
    nlev = length(hier.levels)
    γ_coarse = hier.levels[nlev].cache.γfact
    γf(l) = γfact !== nothing ? T(γfact) :
            l == nlev ? (γ_coarse > zero(T) ? γ_coarse : T(par.γfact_PT)) : T(par.mg_gfact)
    for (l, lvl) in enumerate(hier.levels)
        if l > 1
            f = lvl.fluid;  ff = hier.levels[l - 1].fluid
            restrict_cell!(f.eta, ff.eta)
            restrict_cell!(f.rho, ff.rho)
            # derived from the coarse cell fields, so the coarse operator stays a
            # discretization of the fine one
            derived_coefficients!(f, op)
        end
        penalty!(lvl.cache, lvl.fluid, lvl.grid, op; γfact = γf(l))
        check_preconditioner(lvl.cache)
        _set_smoother_damping!(lvl.cache, par.mg_cfact)
        update_dτ_α_β!(lvl.cache, par.CFL_PT)
    end
    return nothing
end

# compact (ghost-free) view of velocity component d
_compact(vd::AbstractArray{<:Any, D}, d) where {D} =
    view(vd, ntuple(k -> k == d ? Colon() : (2:size(vd, k) - 1), Val(D))...)
