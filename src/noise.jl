using FFTW
using Random
using Statistics

# Port of src/noise.m + the filter precomputation from src/init.m.
#
# Ornstein–Uhlenbeck noise on the stream function / potential field for three
# noise types:
#   psie  — eddy (mixture) stream function
#   psix  — particle-eddy potential
#   psis  — particle-settling potential
#
# Each potential is updated with an OU step, filtered spatially via a
# Gaussian kernel in Fourier space, then differentiated to produce
# staggered-grid noise flux components (xiew, xieu, xixw, xixu, xisw, xisu).
# Those components are consumed by update_phase_velocities! to add noise to
# the crystal/melt phase velocities.

"""
    NoiseState{T}

Mutable workspace for the OU + Gaussian-FFT noise algorithm. CPU-only
because FFTW cannot run on GPU without additional packages.

Fields (all cell-centred (Nz,Nx) unless noted):
* `psie/psix/psis`   — current OU stream-function / potential
* `psieo/psixo/psiso` — previous step (history for OU integrator)
* `re/rs`            — white noise drawn once per timestep
* `Gkpe/Gkps`        — precomputed Gaussian FFT kernels in padded spaces
* `padL0/padl0`      — row-padding widths for eddy / settling filters
* `fL/fl`            — scalar variance-normalisation factors (precomputed)
* `xiew/xieu`        — eddy noise flux on z-faces / x-faces (ghost-extended)
* `xixw/xixu`        — particle-eddy noise flux
* `xisw/xisu`        — particle-settling noise flux
* `xie/xix/xis`      — noise speed magnitudes (cell-centred; for diagnostics)
* `rng`              — seeded Mersenne-Twister used for all white noise
"""
struct NoiseState{T<:AbstractFloat}
    psie::Matrix{T};   psix::Matrix{T};   psis::Matrix{T}
    psieo::Matrix{T};  psixo::Matrix{T};  psiso::Matrix{T}
    re::Matrix{T};     rs::Matrix{T}
    Gkpe::Matrix{Complex{T}}
    Gkps::Matrix{Complex{T}}
    padL0::Int
    padl0::Int
    fL::T
    fl::T
    xiew::Matrix{T};   xieu::Matrix{T}
    xixw::Matrix{T};   xixu::Matrix{T}
    xisw::Matrix{T};   xisu::Matrix{T}
    xie::Matrix{T};    xix::Matrix{T};    xis::Matrix{T}
    rng::MersenneTwister
end

"""
    NoiseState(::Type{T}, grid, scales; seed=0) -> NoiseState{T}

Allocate and precompute all static fields. Mirrors the filter kernel
construction in `src/init.m` (lines 83–101).
"""
function NoiseState(::Type{T}, grid::Grid{T}, scales::Scales{T};
                    seed::Integer = 0) where {T<:AbstractFloat}
    Nz, Nx = grid.Nz, grid.Nx
    h   = Float64(grid.h)
    L0h = Float64(scales.L0h)
    l0h = Float64(scales.l0h)

    # padding widths: 4 × ceil(length_scale / h) rows on each side
    padL0 = 4 * ceil(Int, L0h / h)
    padl0 = 4 * ceil(Int, l0h / h)

    # precompute Gaussian filter kernels for padded domains
    Gkpe = _make_gaussian_filter(T, Nz + padL0, Nx, h, L0h)
    Gkps = _make_gaussian_filter(T, Nz + padl0, Nx, h, l0h)

    # stream-function to flux-component variance normalisation (init.m line 28-29)
    fL = T(2 / sqrt(1 - exp(-h^2 / (2 * L0h^2))))
    fl = T(2 / sqrt(1 - exp(-h^2 / (2 * l0h^2))))

    z = (args...) -> zeros(T, args...)

    return NoiseState{T}(
        z(Nz, Nx), z(Nz, Nx), z(Nz, Nx),   # psie, psix, psis
        z(Nz, Nx), z(Nz, Nx), z(Nz, Nx),   # psieo, psixo, psiso
        z(Nz, Nx), z(Nz, Nx),               # re, rs
        Gkpe, Gkps, padL0, padl0, fL, fl,
        z(Nz + 1, Nx + 2), z(Nz + 2, Nx + 1),   # xiew, xieu
        z(Nz + 1, Nx + 2), z(Nz + 2, Nx + 1),   # xixw, xixu
        z(Nz + 1, Nx + 2), z(Nz + 2, Nx + 1),   # xisw, xisu
        z(Nz, Nx), z(Nz, Nx), z(Nz, Nx),         # xie, xix, xis
        MersenneTwister(seed),
    )
end

# Build a Gaussian filter kernel exp(-σ² |k|²) for an (Nz_pad × Nx) FFT domain.
# σ here is the *full* scale (L0h or l0h); the padded filter omits the 1/2
# factor present in the unpadded kernel — matches init.m lines 101-102.
function _make_gaussian_filter(::Type{T}, Nz_pad::Int, Nx::Int,
                                h::Real, sigma::Real) where {T<:AbstractFloat}
    kz = 2π * fftfreq(Nz_pad, 1 / h)
    kx = 2π * fftfreq(Nx,     1 / h)
    Gk = Matrix{Complex{T}}(undef, Nz_pad, Nx)
    @inbounds for i in 1:Nx, j in 1:Nz_pad
        Gk[j, i] = exp(-sigma^2 * (kz[j]^2 + kx[i]^2))
    end
    return Gk
end

"""
    store_noise!(ns::NoiseState) -> nothing

Snapshot `psie/psix/psis` into the `*o` history buffers. Call once per
outer time step, before the Picard iteration (alongside `store_previous!`).
"""
function store_noise!(ns::NoiseState)
    ns.psieo .= ns.psie
    ns.psixo .= ns.psix
    ns.psiso .= ns.psis
    return nothing
end

"""
    noise!(ns, phase, grid, par, scales, dt; first_iter=true) -> nothing

One OU + spatial-filter noise update. Mirrors `src/noise.m`.

When `first_iter = true` (first Picard sweep of each outer step), fresh
Gaussian white noise is drawn into `ns.re` and `ns.rs`. Subsequent sweeps
reuse the same noise so the Picard iteration does not diverge due to
changing forcing.

Writes:
  `ns.xiew`, `ns.xieu`  — eddy noise flux on z/x staggered faces
  `ns.xixw`, `ns.xixu`  — particle-eddy noise flux
  `ns.xisw`, `ns.xisu`  — particle-settling noise flux
  `ns.xie`,  `ns.xix`, `ns.xis` — noise speed magnitudes (diagnostics)
"""
function noise!(ns::NoiseState{T}, phase::PhaseState{T}, grid::Grid{T},
                par::Parameters{T}, scales::Scales{T}, dt::Real;
                first_iter::Bool = true) where {T<:AbstractFloat}

    Nz, Nx = grid.Nz, grid.Nx
    icz = grid.icz   # [1; 1:Nz; Nz]  length Nz+2
    icx = grid.icx   # [Nx; 1:Nx; 1]  length Nx+2

    Xi   = T(par.Xi)
    L0h  = T(scales.L0h)
    l0h  = T(scales.l0h)
    h    = T(grid.h)
    txi0 = T(scales.txi0)
    eps_T = eps(T)

    # draw white noise once per outer step
    if first_iter
        randn!(ns.rng, ns.re)
        randn!(ns.rng, ns.rs)
    end

    # --- per-cell decorrelation times ---
    taue = @. L0h / (T(2) * (phase.V  + eps_T))   # eddy (Nz,Nx)
    taus = @. l0h / (T(2) * (phase.vx + eps_T))   # settling

    St   = @. txi0 / (taue + eps_T)

    # --- noise flux amplitudes ---
    sge = @. Xi * sqrt(max(zero(T),             phase.fReL * phase.ke / taue))                          # eddy mixture noise speed
    sgx = @. Xi * sqrt(max(zero(T), phase.chi * phase.fReL * phase.ke / taue * St / (one(T) + St^2)))   # eddy crystal noise speed
    sgs = @. Xi * sqrt(max(zero(T), phase.chi *              phase.ks / taus))                          # settling noise speed

    # Ornstein–Uhlenbeck time update for evolving noise
    @. taue = inv(inv(taue) + inv(T(1e2) * T(dt))) + T(dt)
    @. taus = inv(inv(taus) + inv(T(1e2) * T(dt))) + T(dt)

    Fte = @. exp(-T(dt) / taue)     # eddy noise time evolution factor
    Fts = @. exp(-T(dt) / taus)     # settling noise time evolution factor

    fL   = T(2)/sqrt(T(1) - exp(-h^2/(2*L0h^2))); # scaling factor for potential field to noise component variance
    fl   = T(2)/sqrt(T(1) - exp(-h^2/(2*l0h^2))); # scaling factor for potential field to noise component variance

    # --- OU time update ---
    @. ns.psie = Fte * ns.psieo + sqrt(max(zero(T), one(T) - Fte^2)) * sge * fL * ns.re
    @. ns.psix = Fte * ns.psixo + sqrt(max(zero(T), one(T) - Fte^2)) * sgx * fL * ns.re
    @. ns.psis = Fts * ns.psiso + sqrt(max(zero(T), one(T) - Fts^2)) * sgs * fl * ns.rs

    # --- spatial Gaussian filtering ---
    psie_flt = compute_fft_filter(ns.psie, ns.Gkpe, ns.padL0)
    psix_flt = compute_fft_filter(ns.psix, ns.Gkpe, ns.padL0)
    psis_flt = compute_fft_filter(ns.psis, ns.Gkps, ns.padl0)

    # --- eddy noise fluxes from stream function (noise.m lines 61-64) ---
    # corner average of psie_flt to (Nz+1, Nx+1) corner points.
    # psie_ext is already ghost-extended via icz/icx, so use sequential
    # 1:end-1 / 2:end slices — equivalent to MATLAB's
    # psie_flt(icz(1:end-1), icx(1:end-1)) etc.
    psie_ext = psie_flt[icz, icx]             # (Nz+2, Nx+2)
    psiec = (psie_ext[1:end-1, 1:end-1] .+ psie_ext[1:end-1, 2:end] .+
             psie_ext[2:end,   1:end-1] .+ psie_ext[2:end,   2:end]) ./ T(4)  # (Nz+1, Nx+1)

    # raw differences (h=1 convention — matches MATLAB `ddz(f,1)`)
    xieu_int =   psiec[2:end, :] .- psiec[1:end-1, :]   # ddz  → (Nz, Nx+1)
    xiew_int = -(psiec[:, 2:end] .- psiec[:, 1:end-1])  # -ddx → (Nz+1, Nx)

    # ghost-extend to staggered grid sizes (index into interior, then copy whole field)
    ns.xieu .= xieu_int[icz, :]   # (Nz+2, Nx+1) — extend z-ghosts via icz
    ns.xiew .= xiew_int[:, icx]   # (Nz+1, Nx+2) — extend x-ghosts via icx

    # --- noise taper: suppress near chi=0 and where mu<0.4 ---
    xtaperw = @. (one(T) - exp(-phase.chiw / T(1e-4))) *
                 (one(T) - exp(-max(zero(T), phase.muw - T(0.4)) / T(0.05)))
    xtaperu = @. (one(T) - exp(-phase.chiu / T(1e-4))) *
                 (one(T) - exp(-max(zero(T), phase.muu - T(0.4)) / T(0.05)))

    # --- particle-eddy and settling noise (noise.m lines 71-76) ---
    psix_ext = psix_flt[icz, icx]   # (Nz+2, Nx+2)
    psis_ext = psis_flt[icz, icx]

    ns.xixu .= (-(psix_ext[:, 2:end] .- psix_ext[:, 1:end-1])) .* xtaperu   # (Nz+2, Nx+1)
    ns.xixw .= (-(psix_ext[2:end, :] .- psix_ext[1:end-1, :])) .* xtaperw   # (Nz+1, Nx+2)
    ns.xisu .= (-(psis_ext[:, 2:end] .- psis_ext[:, 1:end-1])) .* xtaperu
    ns.xisw .= (-(psis_ext[2:end, :] .- psis_ext[1:end-1, :])) .* xtaperw

    # --- noise speed magnitudes (cell-centred, for diagnostics) ---
    @inbounds for i in 1:Nx, j in 1:Nz
        # interior z-face index in the ghost-extended arrays: col i+1
        wz_s = (ns.xisw[j, i+1] + ns.xisw[j+1, i+1]) / T(2)
        wx_s = (ns.xisu[j+1, i] + ns.xisu[j+1, i+1]) / T(2)
        ns.xis[j, i] = sqrt(wz_s^2 + wx_s^2)

        wz_x = (ns.xixw[j, i+1] + ns.xixw[j+1, i+1]) / T(2)
        wx_x = (ns.xixu[j+1, i] + ns.xixu[j+1, i+1]) / T(2)
        ns.xix[j, i] = sqrt(wz_x^2 + wx_x^2)

        wz_e = (ns.xiew[j, i+1] + ns.xiew[j+1, i+1]) / T(2)
        wx_e = (ns.xieu[j+1, i] + ns.xieu[j+1, i+1]) / T(2)
        ns.xie[j, i] = sqrt(wz_e^2 + wx_e^2)
    end

    return nothing
end

# Pad `psi` (Nz×Nx) with `pad` zero rows (half on each side), apply the
# Gaussian FFT filter `Gk`, unpad, then rescale the result to match the
# original field's mean and standard deviation.
function compute_fft_filter(psi::Matrix{T}, Gk::Matrix{Complex{T}}, pad::Int) where {T<:AbstractFloat}
    Nz, Nx  = size(psi)
    halfpad = pad ÷ 2
    Nz_pad  = Nz + pad

    # embed in zero-padded array
    psi_pad = zeros(Complex{T}, Nz_pad, Nx)
    psi_pad[halfpad + 1 : halfpad + Nz, :] .= psi

    # FFT → multiply filter → IFFT
    psi_pad .= fft(psi_pad, (1, 2))
    psi_pad .*= Gk
    psi_pad .= ifft(psi_pad, (1, 2))

    psi_flt = real.(psi_pad[halfpad + 1 : halfpad + Nz, :])

    # rescale to original mean and std (noise.m lines 54-56)
    raw_std  = std(psi)
    raw_mean = mean(psi)
    flt_std  = std(psi_flt)
    flt_mean = mean(psi_flt)
    if flt_std > eps(T)
        @. psi_flt = (psi_flt - flt_mean) * raw_std / flt_std + raw_mean
    end

    return psi_flt
end
