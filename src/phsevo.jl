using LinearAlgebra

# Port of src/phsevo.m. Updates the phase mass density `X` (and derived
# `M`, `x`, `m`) via a semi-implicit residual:
#
#   res_X = (a1 X - a2 Xo - a3 Xoo) - (b1 ∂_t X + b2 ∂_t Xo + b3 ∂_t Xoo) dt
#   X  ← X - α res_X / a1
#
# where ∂_t X = -∇·(v X) + ∇·(X k_x ∇χ) + G_x. Segregation and noise are not
# in the phase velocity yet (we use the bulk W, U); the boundary reaction
# term G_x is zero until we add the boundary-shape function (Da = 0 path).

"""
    advect_centered!(adv, f, u, w, h, scheme; xBC=:periodic, zBC=:closed) -> adv

Convenience wrapper around `advect!` for non-halo'd cell-centred inputs.
Allocates a halo buffer, embeds `f`, fills ghosts, and dispatches to the
appropriate KA kernel. Allocates per call — fine for the per-iteration use
inside `phsevo!`; can be pre-allocated as workspace if it becomes a hot spot.
"""
function advect_centered!(adv::AbstractMatrix{T}, f::AbstractMatrix{T},
                          u::AbstractMatrix{T}, w::AbstractMatrix{T},
                          h::Real, scheme::Symbol;
                          xBC::Symbol = :periodic,
                          zBC::Symbol = :closed) where {T<:AbstractFloat}
    halo = scheme_halo(scheme)
    f_halo = similar(f, size(f, 1) + 2halo, size(f, 2) + 2halo)
    fill!(f_halo, zero(T))
    embed_interior!(f_halo, f, halo)
    fill_ghosts!(f_halo, halo; xBC, zBC)

    # TVD needs one extra face on each side of u and w. Build padded velocity
    # buffers with periodic-x / repeat-z BCs (matching the f_halo fill above).
    if scheme === :tvdim
        Nz, Nx = size(adv)
        u_pad = similar(u, Nz, Nx + 3)
        w_pad = similar(w, Nz + 3, Nx)
        @views u_pad[:, 2:Nx+2] .= u
        @views w_pad[2:Nz+2, :] .= w
        if xBC === :periodic
            @views u_pad[:, 1]    .= u[:, Nx]     # face 0 ↔ face Nx (periodic wrap)
            @views u_pad[:, Nx+3] .= u[:, 2]      # face Nx+2 ↔ face 2
        else
            @views u_pad[:, 1]    .= u[:, 1]
            @views u_pad[:, Nx+3] .= u[:, Nx+1]
        end
        if zBC === :periodic
            @views w_pad[1,    :] .= w[Nz, :]
            @views w_pad[Nz+3, :] .= w[2,  :]
        else
            @views w_pad[1,    :] .= w[1,    :]
            @views w_pad[Nz+3, :] .= w[Nz+1, :]
        end
        advect!(adv, f_halo, u_pad, w_pad, h, scheme)
    else
        advect!(adv, f_halo, u, w, h, scheme)
    end
    return adv
end

"""
    diffus_centered!(dff, f, k, h; xBC=:periodic, zBC=:closed) -> dff

Convenience wrapper around `diffus!` for non-halo'd cell-centred `f` and
diffusivity `k`. Mirrors `advect_centered!` — embed, ghost-fill, dispatch.
"""
function diffus_centered!(dff::AbstractMatrix{T}, f::AbstractMatrix{T},
                          k::AbstractMatrix{T}, h::Real;
                          xBC::Symbol = :periodic,
                          zBC::Symbol = :closed) where {T<:AbstractFloat}
    halo = 1
    f_halo = similar(f, size(f, 1) + 2halo, size(f, 2) + 2halo)
    k_halo = similar(k, size(k, 1) + 2halo, size(k, 2) + 2halo)
    fill!(f_halo, zero(T));  fill!(k_halo, zero(T))
    embed_interior!(f_halo, f, halo);  embed_interior!(k_halo, k, halo)
    fill_ghosts!(f_halo, halo; xBC, zBC);  fill_ghosts!(k_halo, halo; xBC, zBC)
    diffus!(dff, f_halo, k_halo, h; halo)
    return dff
end

"""
    phsevo!(phase, fluid, grid, par, scales;
            ADVN=:weno5, xBC=:periodic, zBC=:closed,
            a1, a2, a3, b1, b2, b3, dt, alpha = par.alpha) -> nothing

Advance the phase mass density `X` by one Picard step. Writes `advn_X`,
`advn_M`, `advn_rho`, `dffn_X`, `dXdt` into `phase`; updates `phase.X`,
`phase.M`, `phase.x`, `phase.m`. Mirrors `src/phsevo.m`.

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

    advect_centered!(phase.advn_X, phase.X, Ux_face, Wx_face, h, ADVN; xBC, zBC)
    advect_centered!(phase.advn_M, phase.M, Um_face, Wm_face, h, ADVN; xBC, zBC)
    phase.advn_rho .= phase.advn_X .+ phase.advn_M

    # diffusion: f = χ, k = X·k_x
    Xkx = phase.X .* phase.kx
    diffus_centered!(phase.dffn_X, phase.chi, Xkx, h; xBC, zBC)

    # boundary crystallisation reaction (G0 = 0 when par.Da = 0)
    @. phase.Gx = T(scales.G0) * (one(T) - phase.x) * phase.bndshape

    # rate of change and Picard residual update
    phase.dXdt .= .-phase.advn_X .+ phase.dffn_X .+ phase.Gx
    res_X = (T(a1) .* phase.X .- T(a2) .* phase.Xo .- T(a3) .* phase.Xoo) .-
            (T(b1) .* phase.dXdt .+ T(b2) .* phase.dXdto .+ T(b3) .* phase.dXdtoo) .* T(dt)
    upd_X = -T(alpha) .* res_X ./ T(a1)
    phase.X .= phase.X .+ upd_X

    # clamp to [ρ·ε, ρ·(1-ε)] then derive M, x, m
    eps_t = eps(T)
    @. phase.X = clamp(phase.X, fluid.rho * eps_t, fluid.rho * (one(T) - eps_t))
    phase.M .= fluid.rho .- phase.X
    phase.x .= phase.X ./ fluid.rho
    phase.m .= phase.M ./ fluid.rho

    return nothing
end
