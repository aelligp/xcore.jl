# Per-step "store the previous solution" — mirrors src/store.m.
# In MATLAB this is `[Xoo, Xo] = deal(Xo, X);` for each pair; the equivalent
# in Julia is a two-step copy that walks `Xo → Xoo` first so we don't clobber
# the older buffer before snapshotting the newer one.

"""
    store_previous!(fluid::FluidState, phase::PhaseState) -> nothing

Snapshot the current state into the `*o` and `*oo` history buffers used by
the BD2 time integrator. Mirrors `src/store.m`. Call once per outer time
step, before the nonlinear iteration loop.
"""
function store_previous!(fluid::FluidState{T}, phase::PhaseState{T}) where {T<:AbstractFloat}
    # phase densities + rates
    phase.Xoo  .= phase.Xo;     phase.Xo  .= phase.X
    phase.Moo  .= phase.Mo;     phase.Mo  .= phase.M
    phase.dXdtoo .= phase.dXdto; phase.dXdto .= phase.dXdt

    # bulk density + rates
    fluid.rhooo .= fluid.rhoo;  fluid.rhoo .= fluid.rho
    fluid.drhodtoo .= fluid.drhodto;  fluid.drhodto .= fluid.drhodt

    # momentum fluxes on the staggered grids
    fluid.rhoWoo .= fluid.rhoWo
    @views fluid.rhoWo .= fluid.rhow .* fluid.W[:, 2:end-1]
    fluid.rhoUoo .= fluid.rhoUo
    @views fluid.rhoUo .= fluid.rhou .* fluid.U[2:end-1, :]

    return nothing
end

"""
    update_segregation_speed!(phase, grid, par; xBC=:periodic) -> nothing

Update the terminal segregation speed `wx` (crystal) and corresponding melt
speed `wm` on the staggered z-face grid. Mirrors lines 348–359 of
`src/fluidmech.m`:

    wx[:, interior]  =  d0² / η_sw · Δρ_x · g0
    wx              .= wx .* bndtaper       # zero at closed boundaries
    wx[:, ghosts]   .= wx[:, opposite side] # periodic x-BC
    wm              = -x_w / m_w · wx

`Drhox`, `etasw`, `x_w`, `m_w`, and `bndtaperw` are precomputed by `update!`
and `initialize!` respectively.
"""
function update_segregation_speed!(phase::PhaseState{T}, grid::Grid{T},
                                   par::Parameters{T};
                                   xBC::Symbol = :periodic) where {T<:AbstractFloat}
    Nx = grid.Nx
    icx = xBC === :periodic ? [Nx; collect(1:Nx); 1] : [1; collect(1:Nx); Nx]

    # interior wx: Stokes terminal velocity scaled by Δρ
    @views phase.wx[:, 2:end-1] .= par.d0^2 ./ phase.etasw .* phase.Drhox .* par.g0
    # apply closed-boundary taper (zeros wx at z=0, z=D when open_sgr=false)
    phase.wx .*= phase.bndtaperw
    # x-direction ghost columns: periodic wrap-around to match advect's BC
    @views phase.wx[:, 1]   .= phase.wx[:, end - 1]
    @views phase.wx[:, end] .= phase.wx[:, 2]

    # melt segregation speed: wm = -x_w/m_w * wx on the extended (Nz+1, Nx+2) grid
    @views begin
        x_w_ext = phase.x_w[:, icx]      # (Nz+1, Nx+2)
        m_w_ext = phase.m_w[:, icx]
        phase.wm .= -x_w_ext ./ max.(m_w_ext, eps(T)) .* phase.wx
    end
    return nothing
end

"""
    update_phase_velocities!(phase, fluid, grid, par; xBC=:periodic) -> nothing

Set the phase velocities `Wx`, `Wm`, `Ux`, `Um` from the bulk `W`, `U` plus
segregation contributions. Computes `wx`, `wm` first, then:

    Wx = W + wx              # crystals settle down
    Wm = W + wm               # melt rises compensatorily
    Ux = U;  Um = U          # no lateral segregation (noise pending)
"""
function update_phase_velocities!(phase::PhaseState{T}, fluid::FluidState{T},
                                  ns::NoiseState{T},
                                  grid::Grid{T}, par::Parameters{T};
                                  xBC::Symbol = :periodic,
                                  zBC::Symbol = :closed) where {T<:AbstractFloat}
    update_segregation_speed!(phase, grid, par; xBC)

    Nx = grid.Nx
    Nz = grid.Nz
    icx = xBC === :periodic ? [Nx; collect(1:Nx); 1] : [1; collect(1:Nx); Nx]
    icz = zBC === :periodic ? [Nz; collect(1:Nz); 1] : [1; collect(1:Nz); Nz]

    @views begin
        x_w_ext = phase.x_w[:, icx]
        m_w_ext = phase.m_w[:, icx]
        x_u_ext = phase.x_u[icz, :]
        m_u_ext = phase.m_u[icz, :]
    end

    @inbounds @. phase.Wx = fluid.W + phase.wx + (ns.xisw + ns.xixw) + ns.xiew
    @inbounds @. phase.Ux = fluid.U +            (ns.xisu + ns.xixu) + ns.xieu
    @inbounds @. phase.Wm = fluid.W + phase.wm - x_w_ext / max(m_w_ext, eps(T)) * (ns.xisw + ns.xixw) + ns.xiew
    @inbounds @. phase.Um = fluid.U +            - x_u_ext / max(m_u_ext, eps(T)) * (ns.xisu + ns.xixu) + ns.xieu

    return nothing
end
