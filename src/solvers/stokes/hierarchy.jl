# Multigrid hierarchy: per level a grid, a fluid state holding the level's
# (v, P) and coarsened coefficients, and a `StokesCache` (whose `P0` is the
# pseudo-transient reference pressure). Level 1 wraps the caller's fluid state.

"""
    coarsen(grid) -> grid

Halve every cell count and double `h`. Errors on a dimension that cannot be halved.
"""
function coarsen(grid::Grid{T}) where {T}
    N = cells(grid)
    (iseven(N[1]) && all(n -> iseven(n) || n == 1, Base.tail(N))) ||
        error("multigrid: cannot coarsen grid $(join(N, "×")) (need even dims)")
    # A single-cell axis stays one cell wide, so its length doubles with h.
    return Grid(map(n -> n == 1 ? 1 : n ÷ 2, N), map((n, l) -> n == 1 ? 2 * grid.h : l, N, grid.li))
end

"""
    n_levels(N; minlvl) -> Int

Number of levels from `N` cells down to the coarsest grid whose dimensions stay
≥ `minlvl` (a horizontal extent of 1 is kept).
"""
function n_levels(N::NTuple{D, Int}; minlvl::Int = 4) where {D}
    halvable(n, k) = k == 1 ? (iseven(n) && n ÷ 2 ≥ minlvl) :
                              ((iseven(n) || n == 1) && (n == 1 || n ÷ 2 ≥ minlvl))
    lvls = 1;  n = N
    while all(map(halvable, n, ntuple(identity, Val(D))))
        n = map(m -> m == 1 ? 1 : m ÷ 2, n);  lvls += 1
    end
    return lvls
end

struct MGLevel{G, F, C}
    grid::G
    fluid::F
    cache::C
end

struct MGHierarchy{T, L, W}
    levels::Vector{L}     # levels[1] = finest
    cc::Vector{W}         # per-level (p, f, r, h) of the Cahouet–Chabard Poisson V-cycle
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
                    inner, γ_mg) -> MGHierarchy

Level 1 wraps `fluid` with a fresh cache; coarse levels own zero-initialized
states down to the coarsest grid with every dimension ≥ `minlvl`. Coefficients
are filled per solve by `refresh_coefficients!`. Defaults from `par.mg_*`.
Every dimension must be a multiple of 32 (clean coarsening, GPU warps).
"""
function build_hierarchy(fluid, grid, par::Parameters{T};
                         minlvl::Int = par.mg_minlvl, npre::Int = par.mg_npre,
                         npost::Int = par.mg_npost, ncoarse::Int = par.mg_ncoarse,
                         coarse_rtol::Real = par.mg_coarse_rtol, inner::Int = par.mg_inner,
                         γ_mg::Real = par.mg_gamma) where {T}
    N = cells(grid)
    all(n -> n % 32 == 0, N) || throw(ArgumentError(
        "multigrid solvers need every grid dimension to be a multiple of 32 for clean " *
        "coarsening and GPU-friendly sizes (got $(join(N, "×"))); e.g. N = 128/192/256 " *
        "(with L = 1.5·D choose N a multiple of 64 so that Nx = 1.5N is a multiple of 32)"))
    backend = KernelAbstractions.get_backend(fluid.P)
    nlev = n_levels(N; minlvl)
    # Level 1's cache.P0 is set per solve. Coarse levels keep cache.P0 = 0:
    # their P is the pressure correction (started at 0 each cycle), so
    # (P − 0)/(η_b·dt) is the linearized pseudo-transient term the fine operator
    # applies to its own per-cycle increment.
    levels = [MGLevel(grid, fluid, StokesCache(T, backend, grid))]
    g = grid
    for _ in 2:nlev
        g = coarsen(g)
        push!(levels, MGLevel(g, similar_fluid(fluid, backend, cells(g)), StokesCache(T, backend, g)))
    end
    cell(l) = xcore_zeros(backend, T, cells(l.grid)...)
    cc = [(p = cell(l), f = cell(l), r = cell(l), h = T(l.grid.h)) for l in levels]
    return MGHierarchy{T, eltype(levels), eltype(cc)}(levels, cc, minlvl, npre, npost, ncoarse,
                                                      T(coarse_rtol), inner, T(γ_mg))
end

"""
    derived_coefficients!(fluid, op) -> nothing

Rebuild the face densities and edge viscosities of `fluid` from its cell
`rho` and `eta` (arithmetic face means, geometric edge means).
"""
function derived_coefficients!(fluid, op::StokesOp{D}) where {D}
    ρf = face_density(fluid)
    for k in 1:D
        face_mean!(ρf[k], fluid.rho, k, op.periodic)
    end
    edge_geomean!(edge_viscosity(fluid), fluid.eta, op.periodic)
    return nothing
end

# Fixed per-level smoother damping: a smoother only damps the upper half of its
# level's preconditioned spectrum (λ ≳ λ̄max/4; coarser levels handle the rest),
# whose critical damping is c = 2·√(λ̄max/4) = √λ̄max. The coarsest level's
# adaptive solve overrides c through its own Rayleigh retunes.
function set_smoother_damping!(cache::StokesCache{D, T}, mg_cfact::Real) where {D, T}
    λ̄ = foldl(+, ntuple(d -> sum(cache.λmax[d]), Val(D))) /
         foldl(+, ntuple(d -> length(cache.λmax[d]), Val(D)))
    cache.c = T(mg_cfact) * sqrt(λ̄)
    return nothing
end

"""
    refresh_coefficients!(hier, par, op; γfact = nothing) -> nothing

Restrict η and ρ down the hierarchy (cell means), rebuild each coarse level's
face densities and edge viscosities from them, and set every level's penalty,
diagonals and smoother damping. The finest level's coefficients are
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
        set_smoother_damping!(lvl.cache, par.mg_cfact)
    end
    return nothing
end
