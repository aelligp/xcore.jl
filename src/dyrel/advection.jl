# Momentum advection for xcore-DYREL (compressible Navier-Stokes, Re>1).
#
# Replicates the momentum-advection terms from `src/fluidmech.jl` (the
# `advn_mz` / `advn_mx` blocks) using the same `advect_centered!` machinery and
# the same staggered interpolations, so the DR residual carries an identical
# advection term to the direct solver.
#
# CRITICAL — frozen-coefficient treatment: like `fluidmech!`, advection is
# computed ONCE from the velocity at entry to `fluidmech_dyrel!` and held fixed
# throughout the PH/DR iteration. It is a constant RHS contribution for the
# linear solve that one `fluidmech_dyrel!` call performs; the outer Picard loop
# (in `run!`) updates it by calling the solver again with the new velocity.
# This matches the direct solver's semantics exactly and keeps the DR inner
# loop cheap (no per-iteration WENO5 advection).

"""
    dyrel_momentum_advection!(cache, fluid, grid, par; xBC, zBC) -> nothing

Fill `cache.advn_mz` (interior z-faces, (Nz-1, Nx)) and `cache.advn_mx`
(all x-faces, (Nz, Nx+1)) with the divergence of advected momentum
`div(v · ρv)`, using the current `fluid.W`, `fluid.U`, face densities
`fluid.rhow`, `fluid.rhou`, and the advection scheme `par.ADVN`.

Call once at the start of `fluidmech_dyrel!`. Allocates interpolation
temporaries per call (acceptable: once per Picard iteration, same as
`fluidmech!`).
"""
function dyrel_momentum_advection!(cache::DyrelCache{T}, fluid::FluidState{T},
                                   grid::Grid{T}, par::Parameters{T};
                                   xBC::Symbol = :periodic,
                                   zBC::Symbol = :closed) where {T<:AbstractFloat}
    Nz = grid.Nz;  Nx = grid.Nx;  h = grid.h
    W = fluid.W;  U = fluid.U
    rhow = fluid.rhow;  rhou = fluid.rhou

    # --- z-momentum advection (mirrors fluidmech.jl W-block) ---
    # f_mz = ρW·W on interior z-faces; advecting velocities interpolated onto them
    f_mz = @views rhow[2:end-1, :] .* W[2:end-1, 2:end-1]            # (Nz-1, Nx)
    u_mz = @views (U[2:end-2, :] .+ U[3:end-1, :]) ./ T(2)           # (Nz-1, Nx+1)
    w_mz = @views (W[1:end-1, 2:end-1] .+ W[2:end, 2:end-1]) ./ T(2) # (Nz, Nx)
    advect_centered!(cache.advn_mz, f_mz, u_mz, w_mz, h, par.ADVN; xBC, zBC)

    # --- x-momentum advection (mirrors fluidmech.jl U-block) ---
    ifx = [Nx; collect(1:(Nx + 1)); 2]                              # length Nx+3
    u_mx = @views (U[2:end-1, ifx[1:end-1]] .+ U[2:end-1, ifx[2:end]]) ./ T(2)  # (Nz, Nx+2)
    w_mx = @views (W[:, 1:end-1] .+ W[:, 2:end]) ./ T(2)            # (Nz+1, Nx+1)
    f_mx = @views rhou .* U[2:end-1, :]                            # (Nz, Nx+1)
    advect_centered!(cache.advn_mx, f_mx, u_mx, w_mx, h, par.ADVN; xBC, zBC)
    # average the periodic-equivalent boundary columns (fluidmech.jl:271-274)
    @views begin
        col_avg = (cache.advn_mx[:, 1] .+ cache.advn_mx[:, end]) ./ T(2)
        cache.advn_mx[:, 1]   .= col_avg
        cache.advn_mx[:, end] .= col_avg
    end
    return nothing
end
