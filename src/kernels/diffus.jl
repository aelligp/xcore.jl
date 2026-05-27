using KernelAbstractions

# Two-point centred diffusion operator, faithful to src/diffus.m:
#   q  = - (k_face) ⋅ grad(f)              (k_face = arithmetic average to face)
#   dff = - div(q)                         (returned positive for use as a source term)
#
# Requires a 1-cell halo on both `f_halo` and `k_halo`; call `fill_ghosts!` first.

@kernel function _diffus_kernel!(dff, @Const(f), @Const(k), invh2, halo, half)
    iz, ix = @index(Global, NTuple)
    j = iz + halo
    i = ix + halo
    @inbounds begin
        fc  = f[j, i]
        fmx = f[j, i - 1];   fpx = f[j, i + 1]
        fmz = f[j - 1, i];   fpz = f[j + 1, i]
        kc  = k[j, i]
        kmx = k[j, i - 1];   kpx = k[j, i + 1]
        kmz = k[j - 1, i];   kpz = k[j + 1, i]

        # face-averaged k times the face gradient; the factor 1/2 from the
        # face average is folded into `half * invh2` below.
        flux_xr = (kpx + kc) * (fpx - fc)
        flux_xl = (kc + kmx) * (fc - fmx)
        flux_zb = (kpz + kc) * (fpz - fc)
        flux_zt = (kc + kmz) * (fc - fmz)

        dff[iz, ix] = half * invh2 *
                      (flux_xr - flux_xl + flux_zb - flux_zt)
    end
end

"""
    diffus!(dff, f_halo, k_halo, h; halo=1) -> dff

In-place diffusion operator: writes `-div(-k grad f)` to `dff` (size `(Nz, Nx)`),
reading from halo'd `f_halo` and `k_halo` (size `(Nz + 2halo, Nx + 2halo)`).
Returns positive values where `f` is locally concave — i.e. the standard
"diffusion as a positive source" sign convention used in `src/diffus.m` and
throughout the MATLAB code (`dffn_X` enters `dXdt` with a `+` sign).

The caller is responsible for filling ghost cells with the correct BC values
beforehand (typically via `fill_ghosts!`).
"""
function diffus!(dff::AbstractMatrix, f_halo::AbstractMatrix, k_halo::AbstractMatrix,
                 h::Real; halo::Integer = 1)
    @assert size(f_halo) == size(k_halo)                "diffus!: f and k must have matching halo sizes"
    @assert size(dff, 1) == size(f_halo, 1) - 2halo     "diffus!: output row count must match f interior"
    @assert size(dff, 2) == size(f_halo, 2) - 2halo     "diffus!: output col count must match f interior"
    backend = KernelAbstractions.get_backend(dff)
    T = eltype(dff)
    _diffus_kernel!(backend, (16, 16))(dff, f_halo, k_halo,
                                       T(inv(h)^2), halo, T(0.5);
                                       ndrange = size(dff))
    KernelAbstractions.synchronize(backend)
    return dff
end

# ============================================================================
# Per-face diffusive-flux kernels — write face-centred q = -k·∂f signed to
# match MATLAB's `q_x = -(k_face)·(∂f/∂x)` convention from `src/diffus.m`.
#
# Interior sizing:
#   qx_int  (Nz,   Nx+1)   — x-face fluxes at face ix ∈ 1..Nx+1
#   qz_int  (Nz+1, Nx  )   — z-face fluxes at face iz ∈ 1..Nz+1
# ============================================================================

@kernel function _diffus_flux_x_kernel!(qx_int, @Const(f), @Const(k), invh, halo, half)
    iz, ix = @index(Global, NTuple)
    j = iz + halo
    @inbounds begin
        fL = f[j, ix + halo - 1]
        fR = f[j, ix + halo    ]
        kL = k[j, ix + halo - 1]
        kR = k[j, ix + halo    ]
        # q_x = -(k_face) (f_R - f_L) / h; k_face = (k_L + k_R)/2
        qx_int[iz, ix] = -(kL + kR) * half * (fR - fL) * invh
    end
end

@kernel function _diffus_flux_z_kernel!(qz_int, @Const(f), @Const(k), invh, halo, half)
    iz, ix = @index(Global, NTuple)
    i = ix + halo
    @inbounds begin
        fT = f[iz + halo - 1, i]
        fB = f[iz + halo,     i]
        kT = k[iz + halo - 1, i]
        kB = k[iz + halo,     i]
        qz_int[iz, ix] = -(kT + kB) * half * (fB - fT) * invh
    end
end

"""
    diffus_with_flux!(dff, qx, qz, f_halo, k_halo, h; halo=1, xBC, zBC) -> dff

Same as `diffus!` plus emits MATLAB-sized face fluxes:
* `qx` size `(Nz+2, Nx+1)` — `q_x = -(k_face)·∂_x f`
* `qz` size `(Nz+1, Nx+2)` — `q_z = -(k_face)·∂_z f`

Boundary rows/columns are filled per MATLAB `diffus.m:70-81`: periodic ⇒
wrap, otherwise ⇒ repeat. Sign convention matches `src/diffus.m`: the stored
flux carries the minus sign so that `dff = -div(q)` is positive where `f` is
locally concave.
"""
function diffus_with_flux!(dff::AbstractMatrix, qx::AbstractMatrix, qz::AbstractMatrix,
                           f_halo::AbstractMatrix, k_halo::AbstractMatrix,
                           h::Real; halo::Integer = 1,
                           xBC::Symbol = :periodic,
                           zBC::Symbol = :closed)
    Nz = size(dff, 1);  Nx = size(dff, 2)
    @assert size(qx) == (Nz + 2, Nx + 1)  "diffus_with_flux!: qx must be (Nz+2, Nx+1)"
    @assert size(qz) == (Nz + 1, Nx + 2)  "diffus_with_flux!: qz must be (Nz+1, Nx+2)"

    diffus!(dff, f_halo, k_halo, h; halo)

    backend = KernelAbstractions.get_backend(dff)
    T = eltype(dff)
    invh = T(inv(h))
    half = T(0.5)

    qx_int = view(qx, 2:Nz+1, 1:Nx+1)
    qz_int = view(qz, 1:Nz+1, 2:Nx+1)

    _diffus_flux_x_kernel!(backend, (16, 16))(qx_int, f_halo, k_halo, invh, halo, half;
                                               ndrange = size(qx_int))
    _diffus_flux_z_kernel!(backend, (16, 16))(qz_int, f_halo, k_halo, invh, halo, half;
                                               ndrange = size(qz_int))
    KernelAbstractions.synchronize(backend)

    # Boundary fills per src/diffus.m:70-81
    if xBC === :periodic && Nx > 1
        @views qz[:, 1]     .= qz[:, Nx + 1]
        @views qz[:, Nx + 2] .= qz[:, 2]
    else
        @views qz[:, 1]     .= qz[:, 2]
        @views qz[:, Nx + 2] .= qz[:, Nx + 1]
    end
    if zBC === :periodic && Nz > 1
        @views qx[1, :]     .= qx[Nz + 1, :]
        @views qx[Nz + 2, :] .= qx[2, :]
    else
        @views qx[1, :]     .= qx[2, :]
        @views qx[Nz + 2, :] .= qx[Nz + 1, :]
    end

    return dff
end
