using LinearAlgebra

# Port of src/phsevo.m. Updates the phase mass density `X` (and derived
# `M`, `x`, `m`) via a semi-implicit residual:
#
#   res_X = (a1 X - a2 Xo - a3 Xoo) - (b1 ∂_t X + b2 ∂_t Xo + b3 ∂_t Xoo) dt
#   X  ← X - α res_X / a1
#
# where ∂_t X = -∇·(v_x X) + ∇·(X k_x ∇χ) + G_x, v_x is the crystal-phase
# velocity (bulk + segregation + noise, `update_phase_velocities!`) and G_x the
# boundary crystallisation.

"""
    phsevo!(phase, fluid, grid, par, scales;
            ADVN=:weno5, xBC=:periodic, zBC=:closed,
            a1, a2, a3, b1, b2, b3, dt, alpha = par.alpha) -> norm_upd_X

Advance the phase mass density `X` by one Picard step. Writes `advn_X`,
`advn_M`, `advn_rho`, `dffn_X`, `dXdt` into `phase`; updates `phase.X`,
`phase.M`, `phase.x`, `phase.m`. Mirrors `src/phsevo.m`.

Returns the 2-norm of the update `upd_X = -α res_X / a1` before `X` is clamped
to `[ρε, ρ(1-ε)]`; pass it to [`compute_resnorm`](@ref).

The boundary crystallisation term `Gx = G0 · (1 - x) · bndshape` activates
when `Da > 0` (which sets `G0` non-zero via `scales`). `bndshape` is
populated by `compute_bndshape!` during `initialize!`.
"""
function phsevo!(phase::PhaseState{T}, fluid::FluidState{T}, grid::Grid{T},
                 par::Parameters{T}, scales::Scales{T};
                 ADVN::Symbol = :weno5,
                 xBC::Symbol = :periodic,
                 zBC::Symbol = :closed,
                 a1::Real, a2::Real, a3::Real,
                 b1::Real, b2::Real, b3::Real,
                 dt::Real,
                 alpha::Real = par.alpha) where {T<:AbstractFloat}
    h = grid.h

    # crop ghost cols/rows off the staggered phase velocities to get the
    # face-velocity sizing that advect! wants
    Wx_face = @view phase.Wx[:, 2:end-1]            # (Nz+1, Nx)
    Wm_face = @view phase.Wm[:, 2:end-1]
    Ux_face = @view phase.Ux[2:end-1, :]            # (Nz, Nx+1)
    Um_face = @view phase.Um[2:end-1, :]

    # phase advection — populate cell-centred rates AND face-centred fluxes
    # (face fluxes are needed by record_history! to do exact boundary-flux
    # accounting; matches MATLAB phsevo.m:8-9 returning [advn, qz, qx])
    advect_with_flux!(phase.advn_X, phase.qx_advn_X, phase.qz_advn_X,
                      phase.X, Ux_face, Wx_face, h, ADVN; xBC, zBC)
    advect_with_flux!(phase.advn_M, phase.qx_advn_M, phase.qz_advn_M,
                      phase.M, Um_face, Wm_face, h, ADVN; xBC, zBC)
    phase.advn_rho .= phase.advn_X .+ phase.advn_M

    # diffusion: f = χ, k = X·k_x. Emit fluxes too (matches phsevo.m:13).
    Xkx = Broadcast.instantiate(Broadcast.broadcasted(*, phase.X, phase.kx))
    diffus_with_flux!(phase.dffn_X, phase.qx_dffn_X, phase.qz_dffn_X,
                      phase.chi, Xkx, h; xBC, zBC)

    # boundary crystallisation reaction (G0 = 0 when par.Da = 0)
    @. phase.Gx = T(scales.G0) * (one(T) - phase.x) * phase.bndshape

    # rate of change and Picard residual update
    phase.dXdt .= .-phase.advn_X .+ phase.dffn_X .+ phase.Gx
    A1, A2, A3 = T(a1), T(a2), T(a3);  B1, B2, B3 = T(b1), T(b2), T(b3)
    α = T(alpha);  dtT = T(dt)
    upd(X, Xo, Xoo, r, ro, roo) = -α * ((A1 * X - A2 * Xo - A3 * Xoo) - (B1 * r + B2 * ro + B3 * roo) * dtT) / A1
    args = (phase.X, phase.Xo, phase.Xoo, phase.dXdt, phase.dXdto, phase.dXdtoo)
    # ‖upd_X‖ is taken before the clamp: it is the mass term of the Picard residual (report.m:2)
    nupd = sqrt(sum(abs2, Broadcast.instantiate(Broadcast.broadcasted(upd, args...))))
    phase.X .+= upd.(args...)

    # clamp to [ρ·ε, ρ·(1-ε)] then derive M, x, m
    eps_t = eps(T)
    @. phase.X = clamp(phase.X, fluid.rho * eps_t, fluid.rho * (one(T) - eps_t))
    phase.M .= fluid.rho .- phase.X
    phase.x .= phase.X ./ fluid.rho
    phase.m .= phase.M ./ fluid.rho

    return nupd
end
