using KernelAbstractions

# Boundary-condition application for xcore-DYREL.
#
# The velocity BCs run once per inner DR iteration, so they are FUSED into two KA
# kernels (a rows pass, then a cols pass) instead of ~7 separate slice broadcasts:
# on GPU each broadcast was its own kernel dispatch, and the inner loop is
# launch-latency-bound. Two passes are needed because the periodic-column fill
# reads interior columns that the rows pass writes (corner dependency), and KA
# orders same-backend launches so rows-then-cols is correct. The pressure BCs
# (below) run once per Powell-Hestenes step, not per DR iter, so they stay as
# broadcasts.
#
# xcore conventions (matching the matrix-assembly BCs in fluidmech.jl):
#   * x : periodic (left ≡ right)
#   * z : W Dirichlet (= 0) top & bottom; U free/no-slip via top_cnv/bot_cnv
#   * P : periodic-x ghosts, Neumann-z ghosts (repeat boundary)
#
# Velocity field shapes:
#   W (Nz+1, Nx+2)  — z-faces, x-ghost cols 1 & Nx+2
#   U (Nz+2, Nx+1)  — x-faces, z-ghost rows 1 & Nz+2
#   P (Nz+2, Nx+2)  — cells + ghost ring

# Rows pass: W Dirichlet top + bottom (interior x-cols only; ghost cols are
# filled by the cols pass), and U z-ghost rows (all x-faces). `mode` selects the
# bottom-W BC: 0 = closed (W_bot=0), 1 = drain (W_bot = mfs_mean·Ddom/ρw),
# 2 = open (bottom left to the host broadcast in the wrapper). Launched over the
# Nx+1 x-faces (W uses ix ≤ Nx ⇒ fluid col ix+1).
@kernel function _dyrel_vbc_rows_kernel!(W, U, @Const(rhow),
                                         top_cnv, bot_cnv, mfs_mean, Ddom, mode)
    ix = @index(Global)
    @inbounds begin
        Nzf  = size(W, 1)            # Nz+1 (z-faces)
        Nx   = size(W, 2) - 2        # interior x-cols
        Uend = size(U, 1)            # Nz+2
        # U free/no-slip z-ghost rows (all x-faces)
        U[1, ix]    = -top_cnv * U[2, ix]
        U[Uend, ix] = -bot_cnv * U[Uend - 1, ix]
        # W top (=0) and bottom faces — interior x-cols only (fluid col ix+1)
        if ix <= Nx
            W[1, ix + 1] = zero(eltype(W))
            if mode == 0
                W[Nzf, ix + 1] = zero(eltype(W))
            elseif mode == 1
                W[Nzf, ix + 1] = mfs_mean * Ddom / rhow[Nzf, ix]
            end
            # mode == 2 (open): bottom set by the wrapper's broadcast post-launch
        end
    end
end

# Cols pass: periodic-x ghost columns of W and the periodic-equivalent last
# x-face of U. Reads interior columns written by the rows pass (run after it).
@kernel function _dyrel_vbc_cols_kernel!(W, U)
    iz = @index(Global)
    @inbounds begin
        Wrows = size(W, 1)           # Nz+1
        Wend  = size(W, 2)           # Nx+2
        Uend  = size(U, 2)           # Nx+1
        if iz <= Wrows
            W[iz, 1]    = W[iz, Wend - 1]   # periodic wrap
            W[iz, Wend] = W[iz, 2]
        end
        U[iz, Uend] = U[iz, 1]
    end
end

"""
    dyrel_apply_velocity_bcs!(fluid; top_cnv, bot_cnv) -> nothing

Enforce velocity boundary conditions after each DR velocity update:

* `W` = 0 on the top and bottom z-boundary faces (Dirichlet closed).
* `W` x-ghost columns filled by periodic wrap.
* `U` top/bottom z-ghost rows set by reflection: `U[ghost] = −cnv·U[interior]`
  (`cnv = +1` ⇒ no-slip, `cnv = −1` ⇒ free-slip), matching fluidmech.jl.
* `U` periodic x-faces kept in sync (`U[:,end] = U[:,1]`).

Fused into a rows pass + cols pass (see file header). Bit-identical to the old
slice-broadcast version: the cols pass overwrites the ghost columns of rows
1/end, so the rows pass only needs interior columns.
"""
function dyrel_apply_velocity_bcs!(fluid::FluidState{T};
                                   top_cnv::Integer = 1,
                                   bot_cnv::Integer = 1,
                                   open_cnv::Bool = false,
                                   h::Real = 0,
                                   mfs_mean::Real = 0,
                                   Ddom::Real = 0) where {T<:AbstractFloat}
    W = fluid.W;  U = fluid.U
    backend = KernelAbstractions.get_backend(W)
    mode = open_cnv ? 2 : (iszero(mfs_mean) ? 0 : 1)

    _dyrel_vbc_rows_kernel!(backend)(W, U, fluid.rhow,
                                     T(top_cnv), T(bot_cnv), T(mfs_mean), T(Ddom), mode;
                                     ndrange = size(U, 2))
    if open_cnv
        # Open bottom: discrete continuity (mass-flux) balance solved for W_bot
        # (mirrors fluidmech.jl:170-191 with oc=1). Rare path → kept as a host
        # broadcast; runs between the rows and cols passes (KA-ordered).
        ρw = fluid.rhow;  ρu = fluid.rhou;  MFS = fluid.MFS
        @views W[end, 2:end-1] .= (
            MFS[end, :] .* T(h)
            .+ ρw[end - 1, :] .* W[end - 1, 2:end-1]
            .- ρu[end, 2:end] .* U[end - 1, 2:end]
            .+ ρu[end, 1:end-1] .* U[end - 1, 1:end-1]
        ) ./ ρw[end, :]
    end
    _dyrel_vbc_cols_kernel!(backend)(W, U; ndrange = size(U, 1))
    return nothing
end

"""
    dyrel_apply_pressure_bcs!(fluid) -> nothing

Fill the ghost ring of `fluid.P`: periodic in x, Neumann (repeat) in z.
Required before each momentum-residual evaluation since the pressure gradient
reads `P` ghost cells across boundaries.
"""
function dyrel_apply_pressure_bcs!(fluid::FluidState{T}) where {T<:AbstractFloat}
    P = fluid.P
    # x first so the subsequent z fill picks up correct corner columns
    @views P[:, 1]   .= P[:, end - 1]     # periodic wrap
    @views P[:, end] .= P[:, 2]
    @views P[1, :]   .= P[2, :]           # Neumann repeat (z)
    @views P[end, :] .= P[end - 1, :]
    return nothing
end

"""
    dyrel_apply_pressure_gauge!(fluid; target = 0) -> nothing

Fix the pressure gauge by shifting the interior pressure so its mean equals
`target`. With periodic-x and Neumann-z boundaries (and the closed/drain
bottom) the pressure is determined only up to an additive constant, so the
Powell-Hestenes multiplier iteration has a free constant mode. The
artificial-compressibility term `(P−P0)/(η_b·dt)` anchors it only weakly
(η_b = γ_eff is large), letting the mode drift and stalling the continuity
residual. Pinning the mean each PH update removes the null space without
changing gradients (∇P, hence velocity, is untouched) — the discrete analogue
of fluidmech.jl pinning one reference cell.

Pass `target = mean(P0_interior)` to keep P referenced to the previous step
(so the compressibility term's mean stays ~0 and the two don't fight); the
default `target = 0` gives a plain zero-mean gauge. The open-bottom outflow
does NOT fix the pressure level (the P null space remains), so this gauge is
required there too — fluidmech.jl pins the bottom-row P for the same reason.
"""
function dyrel_apply_pressure_gauge!(fluid::FluidState{T}; target::Real = 0) where {T<:AbstractFloat}
    @views Pint = fluid.P[2:end-1, 2:end-1]
    shift = sum(Pint) / length(Pint) - T(target)
    Pint .-= shift
    return nothing
end

"""
    dyrel_apply_pressure_gauge!(fluid, P0, γ_eff) -> nothing

γ_eff-weighted gauge: shift the interior pressure so `Σ (P − P0)/γ_eff = 0`,
i.e. so the artificial-compressibility term `comp = (P−P0)/(γ_eff·dt)`
integrates to ZERO over the domain. Global mass balance requires exactly this
(the bottom drain already carries Σ MFS, so any nonzero Σ comp is an
unremovable CONSTANT continuity residual). The plain unweighted-mean gauge
pins `Σ (P − P0) = 0` instead, which differs whenever γ_eff varies per cell —
observed as a bit-exact P-residual floor with `corr(R_P, 1/γ_eff) ≈ −1` at
N≥256. Weighting by 1/γ_eff makes the gauge and the solvability constraint
coincide; the shift is a constant, so ∇P (hence velocity) is untouched.
"""
function dyrel_apply_pressure_gauge!(fluid::FluidState{T}, P0::AbstractMatrix{T},
                                     γ_eff::AbstractMatrix{T}) where {T<:AbstractFloat}
    @views Pint  = fluid.P[2:end-1, 2:end-1]
    @views P0int = P0[2:end-1, 2:end-1]
    num = sum(_bc((p, p0, γ) -> (p - p0) / γ, Pint, P0int, γ_eff))
    den = sum(_bc(γ -> inv(γ), γ_eff))
    Pint .-= num / den
    return nothing
end
