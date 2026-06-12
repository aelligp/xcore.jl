# Grid hierarchy for geometric multigrid. Each MGLevel carries a coarse Grid,
# restricted coefficient fields, a coarse-sized FluidState/PhaseState (the
# smoother operates on these), and a DyrelCache (per-level autotune/residual
# scratch). Built once per solve; coefficients refreshed each Picard sweep.
#
# (Coefficient coarsening + per-level Gershgorin/γ recompute is filled in
# alongside the V-cycle; this file currently provides the hierarchy scaffolding
# and the coarse-grid constructor.)

"""
    coarsen_grid(grid) -> Grid

Halve `Nz`,`Nx` and double `h` (h_c = 2h). Errors if a dimension is odd (the
hierarchy builder enforces power-of-two-friendly sizes down to `minlvl`).
"""
function coarsen_grid(grid::Grid{T}) where {T}
    Nz, Nx = grid.Nz, grid.Nx
    (iseven(Nz) && (iseven(Nx) || Nx == 1)) ||
        error("GMG: cannot coarsen grid $(Nz)×$(Nx) (need even dims)")
    Nzc = Nz ÷ 2
    Nxc = Nx == 1 ? 1 : Nx ÷ 2
    # Build a coarse Grid via the same Parameters-free path: reuse Grid fields,
    # scaling h. We reconstruct coordinate/index vectors at the coarse size.
    hc = grid.h * 2
    return _grid_at(T, Nzc, Nxc, hc, grid.D, grid.L)
end

# Construct a Grid at an explicit (Nz,Nx,h) without going through Parameters.
# Mirrors `Grid(par)` coordinate/index construction.
function _grid_at(::Type{T}, Nz::Int, Nx::Int, h::T, D::T, L::T) where {T}
    Xc = collect(((1:Nx) .- T(0.5)) .* h)
    Zc = collect(((1:Nz) .- T(0.5)) .* h)
    Xf = collect((0:Nx) .* h)
    Zf = collect((0:Nz) .* h)
    XX = repeat(reshape(Xc, 1, Nx), Nz, 1)
    ZZ = repeat(reshape(Zc, Nz, 1), 1, Nx)
    icx = [Nx; collect(1:Nx); 1]
    icz = [1;  collect(1:Nz); Nz]
    ifx = [Nx; collect(1:Nx + 1); 2]
    ifz = [2;  collect(1:Nz + 1); Nz]
    return Grid{T}(Nx, Nz, h, D, L, Xc, Zc, Xf, Zf, XX, ZZ, icx, icz, ifx, ifz)
end

"""
    n_levels(grid; minlvl) -> Int

Number of grid levels from `grid` down to a coarsest grid with min dimension
≥ `minlvl` (coarsest still even-coarsenable).
"""
function n_levels(grid::Grid; minlvl::Int = 4)
    Nz, Nx = grid.Nz, grid.Nx
    lvls = 1
    while iseven(Nz) && (iseven(Nx) || Nx == 1) &&
          (Nz ÷ 2) ≥ minlvl && ((Nx == 1) || (Nx ÷ 2) ≥ minlvl)
        Nz ÷= 2;  Nx = Nx == 1 ? 1 : Nx ÷ 2;  lvls += 1
    end
    return lvls
end

# ----------------------------------------------------------------------------
# Per-level state + hierarchy
# ----------------------------------------------------------------------------

"""
    MGLevel

One multigrid level: a (coarse) `Grid`, a `FluidState`/`PhaseState` pair holding
the level's `(W,U,P)` solution + restricted coefficients + stress scratch, and a
`DyrelCache` (per-level Gershgorin diagonals, dτ/α/β, residual + injected-RHS
scratch). Level 1 (finest) wraps the caller's own fluid/phase; coarse levels own
freshly-allocated coarse-sized state.
"""
struct MGLevel{T, FS, PS, CA}
    grid::Grid{T}
    fluid::FS
    phase::PS
    cache::CA
end

struct MGHierarchy{T, L}
    levels::Vector{L}     # levels[1] = finest
    minlvl::Int
    npre::Int
    npost::Int
    ncoarse::Int          # max coarsest-level PH steps (solve-to-tolerance budget)
    coarse_rtol::T        # coarsest-level relative residual drop
    inner::Int            # inner DR-iteration cap per smoothing PH step (coarsest uses maxit_PT)
    γ_mg::T
end

"""
    build_hierarchy(fluid, phase, grid, par; minlvl, npre, npost, ncoarse,
                    coarse_rtol, inner, γ_mg) -> MGHierarchy

Allocate the level-1 wrapper (caller's fluid/phase + a fresh fine DyrelCache) and
the coarse levels (own coarse FluidState/PhaseState/DyrelCache), down to a
coarsest grid with min dim ≥ `minlvl`. Coefficients are filled by
`refresh_coefficients!` (called per Picard sweep). `inner` bounds the DR sub-solve
inside each *smoothing* PH step (the coarsest level is solved to `coarse_rtol`).
Defaults are taken from the `par.mg_*` fields.
"""
function build_hierarchy(fluid::FluidState{T}, phase::PhaseState{T}, grid::Grid{T},
                         par::Parameters{T};
                         minlvl::Int = par.mg_minlvl, npre::Int = par.mg_npre,
                         npost::Int = par.mg_npost, ncoarse::Int = par.mg_ncoarse,
                         coarse_rtol::Real = par.mg_coarse_rtol, inner::Int = par.mg_inner,
                         γ_mg::Real = par.mg_gamma) where {T}
    # GPU-friendliness + clean coarsening: both grid dims must be multiples of 32
    # (warp size; and 32 = 2·16 halves cleanly down to the 16² coarsest floor that
    # still resolves the nonlinear rheology). Awkward sizes (e.g. 200×300) coarsen
    # to a large odd-blocked coarsest and make the V-cycle slow — reject them here.
    @assert grid.Nz % 32 == 0 && grid.Nx % 32 == 0 (
        "GMG (:gmg) requires grid dimensions to be multiples of 32 for GPU-" *
        "friendliness and clean multigrid coarsening (got $(grid.Nz)×$(grid.Nx)). " *
        "Pick a 32-multiple resolution, e.g. N=128/192/256 (with L=1.5·D ⇒ choose N " *
        "a multiple of 64 so Nx=1.5N is also a multiple of 32).")
    backend = KernelAbstractions.get_backend(fluid.W)
    nlev = n_levels(grid; minlvl = minlvl)

    L1 = MGLevel(grid, fluid, phase, DyrelCache(T, backend, grid.Nz, grid.Nx))
    levels = Any[L1]
    g = grid
    for _ in 2:nlev
        g = coarsen_grid(g)
        cf = FluidState(T, backend, g.Nz, g.Nx)
        cp = PhaseState(T, backend, g.Nz, g.Nx)
        cc = DyrelCache(T, backend, g.Nz, g.Nx)
        push!(levels, MGLevel(g, cf, cp, cc))
    end
    levels = [l for l in levels]                      # narrow eltype
    return MGHierarchy{T, eltype(levels)}(levels, minlvl, npre, npost, ncoarse,
                                          T(coarse_rtol), inner, T(γ_mg))
end

"""
    refresh_coefficients!(hier, par; a1, gamma, dt)

Restrict the fine-grid coefficients (η, ρ, ρw, ρu) down the hierarchy, recompute
`etaco` from coarse η, and (re)build the per-level Gershgorin diagonals + γ_eff +
dτ/α/β. Call once per `fluidmech_gmg!` (coefficients change each Picard sweep).
The finest level already has live coefficients (from `update!`); only its cache
autotune is refreshed.
"""
function refresh_coefficients!(hier::MGHierarchy{T}, par::Parameters{T};
                               a1::Real, gamma::Real, dt::Real) where {T}
    h1 = hier.levels[1]
    nlev = length(hier.levels)
    # Per-level AL penalty: SMOOTHING levels (all but the coarsest, incl. the
    # finest) use the O(1) `mg_gfact` — over-penalization (γfact_PT=50) makes
    # the penalty dominate the momentum diagonal, leaving div-free HF modes
    # un-smoothable by point-DR and the V-cycle divergent (see parameters.jl
    # mg_gfact). The hierarchy replaces the multiplier acceleration the big
    # penalty buys single-grid. The COARSEST level is a true solve with no
    # coarser correction ⇒ it keeps the strong single-grid γfact_PT.
    γf(l) = l == nlev ? T(par.γfact_PT) : T(par.mg_gfact)
    # finest level: coefficients are live; build the operator-derived penalty
    # (γ_eff = γfact/s_P_phys, leaves D_W/D_U = D_full) exactly as the
    # single-grid solver, then refresh s_P from D_full.
    compute_AL_penalty!(h1.cache, h1.fluid, h1.grid.h; γfact = γf(1), a1, gamma, dt)
    compute_schur_diag!(h1.cache, h1.fluid, h1.grid.h)
    _set_smoother_damping!(h1.cache, par.mg_cfact)
    update_dτ_α_β!(h1.cache, par.CFL_PT)

    for l in 2:length(hier.levels)
        f = hier.levels[l].fluid;  ff = hier.levels[l - 1].fluid
        restrict_p!(f.eta,  ff.eta)
        restrict_p!(f.rho,  ff.rho)
        restrict_w!(f.rhow, ff.rhow)
        restrict_u!(f.rhou, ff.rhou)
        # etaco recomputed from coarse η (consistent with the fine construction)
        compute_corner_eta!(f.etaco, f.eta; xBC = :periodic, zBC = :closed)
        compute_AL_penalty!(hier.levels[l].cache, f, hier.levels[l].grid.h;
                            γfact = γf(l), a1, gamma, dt)
        compute_schur_diag!(hier.levels[l].cache, f, hier.levels[l].grid.h)
        _set_smoother_damping!(hier.levels[l].cache, par.mg_cfact)
        update_dτ_α_β!(hier.levels[l].cache, par.CFL_PT)
    end
    return nothing
end

# Fixed per-level smoother damping: a smoother only has to damp the upper half
# of its level's preconditioned spectrum (λ ≳ λ̄max/4; coarser levels handle the
# rest), whose critical damping is c = 2·√(λ̄max/4) = √λ̄max. The single-grid
# Rayleigh damping targets the GLOBAL λmin instead — far too light for
# smoothing, and never even computed within a short fixed-count sweep (c stays
# 0 ⇒ undamped ⇒ the old empirical mg_inner=32 stability floor). The coarsest
# level's adaptive solve overrides c via its own Rayleigh retunes.
function _set_smoother_damping!(cache::DyrelCache{T}, mg_cfact::Real) where {T}
    λ̄ = (sum(cache.λmax_W) + sum(cache.λmax_U)) /
        (length(cache.λmax_W) + length(cache.λmax_U))
    cache.c[] = T(mg_cfact) * sqrt(λ̄)
    return nothing
end
