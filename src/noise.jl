using FFTW
using Random
using Statistics

# White-noise draw, dispatched by array backend. KernelAbstractions does not
# abstract RNG, so this is the one place that needs an explicit backend split:
#   - host `Array`  → seeded MersenneTwister (reproducible across runs)
#   - device array  → the array package's own device RNG (GPUArrays /
#     CUDA.CURAND / AMDGPU.rocRAND), since a host RNG cannot fill device memory.
_noise_randn!(rng::Random.AbstractRNG, A::Array) = Random.randn!(rng, A)
_noise_randn!(_,                       A::AbstractArray) = Random.randn!(A)

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

State and workspace of the OU + Gaussian-FFT noise algorithm, on the backend
of the model state (`NoiseState(T, backend, grid, scales)`).

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
# Array-type-generic so the noise fields live on whatever backend the rest of
# the state uses (CPU `Array`, CUDA `CuArray`, AMD `ROCArray`, …). `A` is the
# real field type, `AC` the complex FFT-filter type, `R` the RNG (host
# `MersenneTwister` on CPU; on a device the per-element `randn!(A)` device RNG
# is used and `rng` is ignored — see `_noise_randn!`).
struct NoiseState{T<:AbstractFloat, A<:AbstractMatrix{T},
                  AC<:AbstractMatrix{Complex{T}}, R}
    psie::A;   psix::A;   psis::A
    psieo::A;  psixo::A;  psiso::A
    re::A;     rs::A
    Gkpe::AC
    Gkps::AC
    padL0::Int
    padl0::Int
    fL::T
    fl::T
    xiew::A;   xieu::A
    xixw::A;   xixu::A
    xisw::A;   xisu::A
    xie::A;    xix::A;    xis::A
    rng::R
    # scratch, reused for psie, psix and psis in turn: the filtered potential
    # (Nz,Nx), its ghost extension (Nz+2,Nx+2), and the padded FFT buffers of the
    # eddy (Gkpe/padL0) and settling (Gkps/padl0) filters
    psi_flt::A
    psi_ext::A
    eddy_pad::AC; settle_pad::AC
    # eddy stream function at corners (Nz+1,Nx+1) and its unextended derivatives
    psiec::A
    xieu_int::A; xiew_int::A
end

"""
    NoiseState(::Type{T}, grid, scales; seed=0) -> NoiseState{T}

Allocate and precompute all static fields. Mirrors the filter kernel
construction in `src/init.m` (lines 83–101).
"""
function NoiseState(::Type{T}, backend, grid::Grid{T}, scales::Scales{T};
                    seed::Integer = 0) where {T<:AbstractFloat}
    Nz, Nx = grid.Nz, grid.Nx
    h   = Float64(grid.h)
    L0h = Float64(scales.L0h)
    l0h = Float64(scales.l0h)

    # padding widths: 4 × ceil(length_scale / h) rows on each side
    padL0 = 4 * ceil(Int, L0h / h)
    padl0 = 4 * ceil(Int, l0h / h)

    # Gaussian filters: built on the host (small, setup-only) then moved to the
    # target backend so the per-step FFT filtering runs on-device.
    Gkpe = adapt_backend(backend, _make_gaussian_filter(T, Nz + padL0, Nx, h, L0h))
    Gkps = adapt_backend(backend, _make_gaussian_filter(T, Nz + padl0, Nx, h, l0h))

    # stream-function to flux-component variance normalisation (init.m line 28-29)
    fL = T(2 / sqrt(1 - exp(-h^2 / (2 * L0h^2))))
    fl = T(2 / sqrt(1 - exp(-h^2 / (2 * l0h^2))))

    z  = (args...) -> xcore_zeros(backend, T, args...)
    zc = (args...) -> xcore_zeros(backend, Complex{T}, args...)

    return NoiseState(
        z(Nz, Nx), z(Nz, Nx), z(Nz, Nx),   # psie, psix, psis
        z(Nz, Nx), z(Nz, Nx), z(Nz, Nx),   # psieo, psixo, psiso
        z(Nz, Nx), z(Nz, Nx),               # re, rs
        Gkpe, Gkps, padL0, padl0, fL, fl,
        z(Nz + 1, Nx + 2), z(Nz + 2, Nx + 1),   # xiew, xieu
        z(Nz + 1, Nx + 2), z(Nz + 2, Nx + 1),   # xixw, xixu
        z(Nz + 1, Nx + 2), z(Nz + 2, Nx + 1),   # xisw, xisu
        z(Nz, Nx), z(Nz, Nx), z(Nz, Nx),         # xie, xix, xis
        MersenneTwister(seed),
        z(Nz, Nx),                                # psi_flt
        z(Nz + 2, Nx + 2),                        # psi_ext
        zc(Nz + padL0, Nx), zc(Nz + padl0, Nx),  # eddy_pad, settle_pad
        z(Nz + 1, Nx + 1),                        # psiec
        z(Nz, Nx + 1), z(Nz + 1, Nx),             # xieu_int, xiew_int
    )
end

# Back-compat: default to the CPU backend so existing `NoiseState(T, grid, scales)`
# call sites keep working unchanged.
NoiseState(::Type{T}, grid::Grid{T}, scales::Scales{T}; seed::Integer = 0) where {T} =
    NoiseState(T, KernelAbstractions.CPU(), grid, scales; seed = seed)

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

# Pointwise pieces of the OU noise update (noise.m): decorrelation time of a
# length scale l at speed v, noise-flux amplitudes, the OU step with the
# decorrelation time relaxed towards the time step, and the noise taper.
@inline _decorrelation(l, v, ε) = l / (2 * (v + ε))
@inline _eddy_amplitude(Xi, fReL, ke, τ) = Xi * sqrt(max(zero(τ), fReL * ke / τ))
@inline function _xtal_amplitude(Xi, chi, fReL, ke, τ, txi0)
    St = txi0 / τ
    return Xi * sqrt(max(zero(τ), chi * fReL * ke / τ * St / (one(τ) + St^2)))
end
@inline _settling_amplitude(Xi, chi, ks, τ) = Xi * sqrt(max(zero(τ), chi * ks / τ))
@inline function _ou_step(ψo, r, τ, sg, f, dt, τcap)
    F = exp(-dt / (inv(inv(τ) + inv(τcap)) + dt))
    return F * ψo + sqrt(max(zero(F), one(F) - F^2)) * sg * f * r
end
@inline _noise_taper(chi, mu, χ0, μ0, dμ) =
    (one(chi) - exp(-chi / χ0)) * (one(chi) - exp(-max(zero(chi), mu - μ0) / dμ))

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
    L0   = T(scales.L0)
    l0   = T(scales.l0)
    txi0 = T(scales.txi0)
    eps_T = eps(T)
    # scalars stay outside `@.`: a literal converted inside a broadcast is
    # evaluated per element, in Float64, which Float32-only GPUs reject
    dtT  = T(dt)
    τcap = T(1e2) * dtT
    χ0 = T(1e-4);  μ0 = T(0.4);  dμ = T(0.05)

    # draw white noise once per outer step (host MersenneTwister on CPU;
    # the array package's device RNG on GPU — see `_noise_randn!`)
    if first_iter
        _noise_randn!(ns.rng, ns.re)
        _noise_randn!(ns.rng, ns.rs)
    end

    fL = ns.fL   # precomputed scaling factor for potential field to noise component variance
    fl = ns.fl

    # --- OU time update of the eddy, particle-eddy and settling potentials ---
    @. ns.psie = _ou_step(ns.psieo, ns.re, _decorrelation(L0, phase.V, eps_T),
                          _eddy_amplitude(Xi, phase.fReL, phase.ke, _decorrelation(L0, phase.V, eps_T)),
                          fL, dtT, τcap)
    @. ns.psix = _ou_step(ns.psixo, ns.re, _decorrelation(L0, phase.V, eps_T),
                          _xtal_amplitude(Xi, phase.chi, phase.fReL, phase.ke,
                                          _decorrelation(L0, phase.V, eps_T), txi0),
                          fL, dtT, τcap)
    @. ns.psis = _ou_step(ns.psiso, ns.rs, _decorrelation(l0, phase.vx, eps_T),
                          _settling_amplitude(Xi, phase.chi, phase.ks, _decorrelation(l0, phase.vx, eps_T)),
                          fl, dtT, τcap)

    # --- eddy noise fluxes from the filtered stream function (noise.m lines 61-64) ---
    # corner average of the ghost-extended psie_flt to (Nz+1, Nx+1) corner points,
    # with sequential 1:end-1 / 2:end slices — equivalent to MATLAB's
    # psie_flt(icz(1:end-1), icx(1:end-1)) etc.
    compute_fft_filter!(ns.psi_flt, ns.eddy_pad, ns.psie, ns.Gkpe, ns.padL0)
    ns.psi_ext .= @view ns.psi_flt[icz, icx]             # (Nz+2, Nx+2)
    @views @. ns.psiec = (ns.psi_ext[1:end-1, 1:end-1] + ns.psi_ext[1:end-1, 2:end] +
                          ns.psi_ext[2:end,   1:end-1] + ns.psi_ext[2:end,   2:end]) / T(4)  # (Nz+1, Nx+1)

    # raw differences (h=1 convention — matches MATLAB `ddz(f,1)`)
    @views @. ns.xieu_int =   ns.psiec[2:end, :] - ns.psiec[1:end-1, :]   # ddz  → (Nz, Nx+1)
    @views @. ns.xiew_int = -(ns.psiec[:, 2:end] - ns.psiec[:, 1:end-1])  # -ddx → (Nz+1, Nx)

    # ghost-extend to staggered grid sizes (index into interior, then copy whole field)
    ns.xieu .= @view ns.xieu_int[icz, :]   # (Nz+2, Nx+1) — extend z-ghosts via icz
    ns.xiew .= @view ns.xiew_int[:, icx]   # (Nz+1, Nx+2) — extend x-ghosts via icx

    # --- particle-eddy and settling noise (noise.m lines 71-76), tapered to
    # vanish near chi = 0 and where mu < 0.4 ---
    compute_fft_filter!(ns.psi_flt, ns.eddy_pad, ns.psix, ns.Gkpe, ns.padL0)
    ns.psi_ext .= @view ns.psi_flt[icz, icx]   # (Nz+2, Nx+2)
    @views @. ns.xixu = -(ns.psi_ext[:, 2:end] - ns.psi_ext[:, 1:end-1]) * _noise_taper(phase.chiu, phase.muu, χ0, μ0, dμ)   # (Nz+2, Nx+1)
    @views @. ns.xixw = -(ns.psi_ext[2:end, :] - ns.psi_ext[1:end-1, :]) * _noise_taper(phase.chiw, phase.muw, χ0, μ0, dμ)   # (Nz+1, Nx+2)

    compute_fft_filter!(ns.psi_flt, ns.settle_pad, ns.psis, ns.Gkps, ns.padl0)
    ns.psi_ext .= @view ns.psi_flt[icz, icx]
    @views @. ns.xisu = -(ns.psi_ext[:, 2:end] - ns.psi_ext[:, 1:end-1]) * _noise_taper(phase.chiu, phase.muu, χ0, μ0, dμ)
    @views @. ns.xisw = -(ns.psi_ext[2:end, :] - ns.psi_ext[1:end-1, :]) * _noise_taper(phase.chiw, phase.muw, χ0, μ0, dμ)

    # --- noise speed magnitudes (cell-centred, for diagnostics) ---
    backend = KernelAbstractions.get_backend(ns.xie)
    _noise_magnitudes_kernel!(backend)(ns.xis, ns.xix, ns.xie,
                                       ns.xisw, ns.xisu, ns.xixw, ns.xixu,
                                       ns.xiew, ns.xieu; ndrange = (Nz, Nx))
    KernelAbstractions.synchronize(backend)

    return nothing
end

# Cell-centred noise speed magnitudes from the staggered, ghost-extended flux
# components (interior z-face column is i+1 in the ghost-extended arrays).
@kernel function _noise_magnitudes_kernel!(xis, xix, xie,
                                           @Const(xisw), @Const(xisu),
                                           @Const(xixw), @Const(xixu),
                                           @Const(xiew), @Const(xieu))
    j, i = @index(Global, NTuple)
    @inbounds begin
        T = eltype(xis)
        wz_s = (xisw[j, i+1] + xisw[j+1, i+1]) / T(2)
        wx_s = (xisu[j+1, i] + xisu[j+1, i+1]) / T(2)
        xis[j, i] = sqrt(wz_s^2 + wx_s^2)

        wz_x = (xixw[j, i+1] + xixw[j+1, i+1]) / T(2)
        wx_x = (xixu[j+1, i] + xixu[j+1, i+1]) / T(2)
        xix[j, i] = sqrt(wz_x^2 + wx_x^2)

        wz_e = (xiew[j, i+1] + xiew[j+1, i+1]) / T(2)
        wx_e = (xieu[j+1, i] + xieu[j+1, i+1]) / T(2)
        xie[j, i] = sqrt(wz_e^2 + wx_e^2)
    end
end

# Pad `psi` (Nz×Nx) with `pad` zero rows (half on each side) into caller-owned
# scratch `psi_pad`, apply the Gaussian FFT filter `Gk`, unpad, then rescale to
# match the original field's mean and standard deviation, writing into
# caller-owned `psi_flt`. `psi_pad`/`psi_flt` are reused across calls by the
# caller (`noise!`) instead of being allocated here.
# Array-type-generic: `fft!`/`ifft!` are the AbstractFFTs generics that dispatch
# automatically to FFTW (CPU), cuFFT (CUDA), or rocFFT (AMD) when the
# corresponding package is loaded — no explicit per-backend code needed. `Gk`
# must already be on the same backend (NoiseState moves it there at construction).
function compute_fft_filter!(psi_flt::AbstractMatrix{T}, psi_pad::AbstractMatrix{Complex{T}},
                             psi::AbstractMatrix{T}, Gk::AbstractMatrix{Complex{T}},
                             pad::Int) where {T<:AbstractFloat}
    Nz      = size(psi, 1)
    halfpad = pad ÷ 2

    fill!(psi_pad, zero(Complex{T}))
    @views psi_pad[halfpad + 1 : halfpad + Nz, :] .= psi

    # FFT → multiply filter → IFFT, all IN-PLACE (`fft!`/`ifft!`) so the
    # transforms reuse `psi_pad` instead of allocating fresh output arrays each
    # call. AbstractFFTs in-place generics dispatch to FFTW / cuFFT / rocFFT.
    fft!(psi_pad, (1, 2))
    psi_pad .*= Gk
    ifft!(psi_pad, (1, 2))

    @views @. psi_flt = real(psi_pad[halfpad + 1 : halfpad + Nz, :])

    # rescale to original mean and std (noise.m lines 54-56). std/mean reduce to
    # host scalars; the final broadcast stays on-device.
    raw_std  = std(psi)
    raw_mean = mean(psi)
    flt_std  = std(psi_flt)
    flt_mean = mean(psi_flt)
    if flt_std > eps(T)
        @. psi_flt = (psi_flt - flt_mean) * raw_std / flt_std + raw_mean
    end

    return nothing
end
