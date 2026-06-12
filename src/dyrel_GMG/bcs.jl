using KernelAbstractions

# Boundary-condition application for xcore-DYREL.
#
# Velocity BCs run once per inner DR iteration → FUSED into a rows pass + a cols
# pass (two KA kernels) instead of ~7 slice broadcasts, to cut per-iteration
# kernel dispatches on GPU. Two passes because the periodic-column fill reads
# interior columns the rows pass writes (KA orders same-backend launches).
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

# Rows pass: W top (=0) + bottom (mode 0 closed / 1 drain), interior x-cols only;
# U z-ghost rows (all x-faces). Ghost cols of W are filled by the cols pass.
@kernel function _dyrel_vbc_rows_kernel!(W, U, @Const(rhow),
                                         top_cnv, bot_cnv, mfs_mean, Ddom, mode)
    ix = @index(Global)
    @inbounds begin
        Nzf  = size(W, 1)            # Nz+1
        Nx   = size(W, 2) - 2        # interior x-cols
        Uend = size(U, 1)            # Nz+2
        U[1, ix]    = -top_cnv * U[2, ix]
        U[Uend, ix] = -bot_cnv * U[Uend - 1, ix]
        if ix <= Nx
            W[1, ix + 1] = zero(eltype(W))
            if mode == 0
                W[Nzf, ix + 1] = zero(eltype(W))
            else
                W[Nzf, ix + 1] = mfs_mean * Ddom / rhow[Nzf, ix]
            end
        end
    end
end

# Cols pass: periodic-x ghost columns of W + periodic-equivalent last x-face of U.
@kernel function _dyrel_vbc_cols_kernel!(W, U)
    iz = @index(Global)
    @inbounds begin
        Wrows = size(W, 1)           # Nz+1
        Wend  = size(W, 2)           # Nx+2
        Uend  = size(U, 2)           # Nx+1
        if iz <= Wrows
            W[iz, 1]    = W[iz, Wend - 1]
            W[iz, Wend] = W[iz, 2]
        end
        U[iz, Uend] = U[iz, 1]
    end
end

"""
    dyrel_apply_velocity_bcs!(fluid; top_cnv, bot_cnv) -> nothing

Enforce velocity boundary conditions after each DR velocity update:

* `W` = 0 on the top and bottom z-boundary faces (Dirichlet closed); bottom is a
  uniform drain `W_bot = MFSmean·D/ρw` when `mfs_mean ≠ 0` (fluidmech.jl:197-206).
* `W` x-ghost columns filled by periodic wrap.
* `U` top/bottom z-ghost rows reflected (`U[ghost] = −cnv·U[interior]`).
* `U` periodic x-faces kept in sync (`U[:,end] = U[:,1]`).

Fused into a rows pass + cols pass; bit-identical to the old slice broadcasts.
"""
function dyrel_apply_velocity_bcs!(fluid::FluidState{T};
                                   top_cnv::Integer = 1,
                                   bot_cnv::Integer = 1,
                                   mfs_mean::Real = 0,
                                   Ddom::Real = 0) where {T<:AbstractFloat}
    W = fluid.W;  U = fluid.U
    backend = KernelAbstractions.get_backend(W)
    mode = iszero(mfs_mean) ? 0 : 1
    _dyrel_vbc_rows_kernel!(backend)(W, U, fluid.rhow,
                                     T(top_cnv), T(bot_cnv), T(mfs_mean), T(Ddom), mode;
                                     ndrange = size(U, 2))
    _dyrel_vbc_cols_kernel!(backend)(W, U; ndrange = size(U, 1))
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
residual — same rationale as src/dyrel/bcs.jl. Applied on the FINE level only
(the prolonged coarse corrections are already mean-removed in the V-cycle).

Pass `target = mean(P0_interior)` so the gauge and the compressibility anchor
don't fight.
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
making the artificial-compressibility term integrate to zero (the global mass
solvability constraint — the bottom drain already carries Σ MFS). The
unweighted gauge pins `Σ (P − P0) = 0` instead, which differs when γ_eff
varies per cell and leaves a bit-exact constant P-residual floor (observed at
N≥256: `corr(R_P, 1/γ_eff) ≈ −1`). Same fix as src/dyrel/bcs.jl.
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
