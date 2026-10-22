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

    # momentum fluxes ρ_d v_d on the compact faces of each axis
    for (d, (lag1, lag2, ρd, vd)) in enumerate(zip(momentum_lag1(fluid), momentum_lag2(fluid),
                                                   face_density(fluid), velocity(fluid)))
        lag2 .= lag1
        lag1 .= ρd .* face_interior(vd, d)
    end
    return nothing
end

"""
    update_segregation_speed!(phase, grid, par; xBC=:periodic, yBC=:periodic) -> nothing

Update the terminal segregation speed `wx` (crystal) and corresponding melt
speed `wm` on the ghosted z-faces. Mirrors lines 348–359 of `src/fluidmech.m`:

    wx[interior]  =  d0² / η_sw · Δρ_x · g0
    wx           .= wx .* bndtaper       # zero at closed boundaries
    wm[interior]  = -x_w / m_w · wx
    ghosts        : ghost rule of each horizontal axis (periodic wrap, else repeat)

`Drhox`, `etasw`, `x_w`, `m_w`, and `bndtaperw` are precomputed by `update!`
and `initialize!` respectively. `yBC` is read only in 3-D.
"""
function update_segregation_speed!(phase::PhaseState{T, D}, grid::Grid{T, D},
                                   par::Parameters{T};
                                   xBC::Symbol = :periodic,
                                   yBC::Symbol = :periodic) where {T<:AbstractFloat, D}
    periodic = periodic_axes("update_segregation_speed!", Val(D), xBC, yBC, :closed)
    wx = face_interior(phase.wx, 1);  wm = face_interior(phase.wm, 1)
    # Stokes terminal velocity scaled by Δρ, zero at closed boundaries (open_sgr = false)
    wx .= par.d0^2 ./ phase.etasw .* phase.Drhox .* par.g0
    phase.wx .*= phase.bndtaperw
    # melt segregation speed balancing the crystal mass flux
    wm .= .-phase.x_w ./ phase.m_w .* wx
    fill_face_ghosts!(phase.wx, 1, periodic)
    fill_face_ghosts!(phase.wm, 1, periodic)
    return nothing
end

# crystal and melt velocities on the ghosted faces of `axis`: bulk velocity, plus
# the segregation speeds `segx`, `segm` (z-faces only, `nothing` otherwise), plus
# the noise; the melt carries the crystal noise flux weighted by −x/m (x, m on
# compact faces, gathered by the ghost rule)
@kernel function phase_velocity_kernel!(vx, vm, @Const(v), segx, segm,
                                        @Const(xis), @Const(xix), @Const(xie),
                                        @Const(xf), @Const(mf), axis, periodic)
    J = @index(Global, Cartesian)
    d = unval(axis)
    Jc = CartesianIndex(ntuple(k -> k == d ? J[k] : ghost_index(J[k] - 1, size(xf, k), periodic[k]),
                               Val(length(J))))
    s = xis[J] + xix[J]
    bx = segx === nothing ? v[J] : v[J] + segx[J]
    bm = segm === nothing ? v[J] : v[J] + segm[J]
    vx[J] = bx + s + xie[J]
    vm[J] = bm - xf[Jc] / mf[Jc] * s + xie[J]
end

"""
    update_phase_velocities!(phase, fluid, ns, grid, par; xBC=:periodic, yBC=:periodic,
                             zBC=:closed) -> nothing

Set the phase velocities from the bulk velocity plus segregation and noise.
Computes `wx`, `wm` first, then on the ghosted faces of each axis d
(`Wx`/`Wm`, `Ux`/`Um`[, `Vx`/`Vm`]):

    crystal = v_d + seg_d + (xis_d + xix_d) + xie_d
    melt    = v_d + segm_d − x_d/m_d · (xis_d + xix_d) + xie_d

with the segregation speeds `wx`, `wm` on z only. `yBC` is read only in 3-D.
"""
function update_phase_velocities!(phase::PhaseState{T, D}, fluid::FluidState{T, D},
                                  ns::NoiseState{T, D},
                                  grid::Grid{T, D}, par::Parameters{T};
                                  xBC::Symbol = :periodic, yBC::Symbol = :periodic,
                                  zBC::Symbol = :closed) where {T<:AbstractFloat, D}
    update_segregation_speed!(phase, grid, par; xBC, yBC)
    periodic = periodic_axes("update_phase_velocities!", Val(D), xBC, yBC, zBC)
    backend = KernelAbstractions.get_backend(phase.Wx)
    args = zip(per_axis(Val(D), phase.Wx, phase.Ux, phase.Vx), per_axis(Val(D), phase.Wm, phase.Um, phase.Vm),
               velocity(fluid), per_axis(Val(D), phase.wx, nothing, nothing),
               per_axis(Val(D), phase.wm, nothing, nothing),
               per_axis(Val(D), ns.xisw, ns.xisu, ns.xisv), per_axis(Val(D), ns.xixw, ns.xixu, ns.xixv),
               per_axis(Val(D), ns.xiew, ns.xieu, ns.xiev),
               per_axis(Val(D), phase.x_w, phase.x_u, phase.x_v), per_axis(Val(D), phase.m_w, phase.m_u, phase.m_v))
    for (d, (vx, vm, v, segx, segm, xis, xix, xie, xf, mf)) in enumerate(args)
        phase_velocity_kernel!(backend)(vx, vm, v, segx, segm, xis, xix, xie, xf, mf, Val(d), periodic;
                                        ndrange = size(vx))
    end
    KernelAbstractions.synchronize(backend)
    return nothing
end
