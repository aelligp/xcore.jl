using FFTW
using Random
using Statistics

# White-noise draw, dispatched by array backend. KernelAbstractions does not
# abstract RNG, so this is the one place that needs an explicit backend split:
#   - host `Array`  → seeded MersenneTwister (reproducible across runs)
#   - device array  → the array package's own device RNG (GPUArrays /
#     CUDA.CURAND / AMDGPU.rocRAND), since a host RNG cannot fill device memory.
noise_randn!(rng::Random.AbstractRNG, A::Array) = Random.randn!(rng, A)
noise_randn!(_,                       A::AbstractArray) = Random.randn!(A)

# Port of src/noise.m + the filter precomputation from src/init.m.
#
# Ornstein–Uhlenbeck noise on potential fields for three noise types:
#   psie  — eddy (mixture) stream function / vector potential
#   psix  — particle-eddy potential
#   psis  — particle-settling potential
#
# Each potential is updated with an OU step, filtered spatially via a Gaussian
# kernel in Fourier space, then differentiated to produce noise fluxes on the
# ghosted faces of every axis (xie*, xix*, xis*), which update_phase_velocities!
# adds to the crystal/melt phase velocities. The eddy noise is ξ = −∇×A: in 2-D
# A has the single (y-) component `psie`, the stream function; in 3-D A has the
# components `(psie_z, psie_x, psie)`, each on the edges parallel to its axis.
# The particle noises are gradients of the scalar potentials psix, psis; psix is
# driven by the eddy white noise (in 3-D its normalized sum over the components).

# Array-type-generic so the noise fields live on whatever backend the rest of
# the state uses (CPU `Array`, CUDA `CuArray`, AMD `ROCArray`, …). `A` is the
# real field type, `AC` the complex FFT-filter type, `R` the RNG (host
# `MersenneTwister` on CPU; on a device the per-element `randn!(A)` device RNG
# is used and `rng` is ignored — see `noise_randn!`).

"""
    NoiseState{T,D}

State and workspace of the OU + Gaussian-FFT noise algorithm, on the backend
of the model state (`NoiseState(T, backend, grid, scales)`).

Fields (all cell-centered, size N = `cells(grid)`, unless noted):
* `psie/psix/psis`   — current OU potentials (`psie`: y-component of the eddy
  vector potential, the stream function in 2-D)
* `psie_z/psie_x`    — z- and x-components of the eddy vector potential (3-D)
* `psieo/psixo/psiso`, `psieo_z/psieo_x` — previous step (history for OU integrator)
* `re/rs`, `re_z/re_x` — white noise drawn once per timestep
* `Gkpe/Gkps`        — precomputed Gaussian FFT kernels in z-padded spaces
* `padL0/padl0`      — z-padding widths for eddy / settling filters
* `fL/fl`            — scalar variance-normalisation factors (precomputed)
* `xiew/xieu/xiev`   — eddy noise flux on ghosted z-, x-, y-faces
* `xixw/xixu/xixv`   — particle-eddy noise flux
* `xisw/xisu/xisv`   — particle-settling noise flux
* `xie/xix/xis`      — noise speed magnitudes (cell-centered; for diagnostics)
* `rng`              — seeded Mersenne-Twister used for all white noise

The 3-D-only fields (`psie_z`, `psie_x`, their history and white noise, the
y-fluxes) are `nothing` in 2-D.
"""
struct NoiseState{T<:AbstractFloat, D, A<:AbstractArray{T, D},
                  AC<:AbstractArray{Complex{T}, D}, R, B<:Union{A, Nothing}}
    psie::A;   psix::A;   psis::A
    psieo::A;  psixo::A;  psiso::A
    re::A;     rs::A
    psie_z::B;  psie_x::B
    psieo_z::B; psieo_x::B
    re_z::B;    re_x::B
    Gkpe::AC
    Gkps::AC
    padL0::Int
    padl0::Int
    fL::T
    fl::T
    xiew::A;   xieu::A;   xiev::B
    xixw::A;   xixu::A;   xixv::B
    xisw::A;   xisu::A;   xisv::B
    xie::A;    xix::A;    xis::A
    rng::R
    # scratch, reused for every potential in turn: the filtered potential (N),
    # its ghost extension (N .+ 2), and the padded FFT buffers of the eddy
    # (Gkpe/padL0) and settling (Gkps/padl0) filters
    psi_flt::A
    psi_ext::A
    eddy_pad::AC; settle_pad::AC
    # eddy vector potential on its edges: y-component on the z–x edges (the 2-D
    # corners), z-component on the x–y edges, x-component on the z–y edges
    psiec::A;  psiec_z::B;  psiec_x::B
end

# the eddy vector-potential components (z, x, y) present in D dimensions
eddy_components(::Val{2}, z, x, y) = (y,)
eddy_components(::Val{3}, z, x, y) = (z, x, y)
# edge axis pair of each component: z ↔ (x, y), x ↔ (z, y), y ↔ (z, x)
eddy_edge_pairs(::Val{2}) = ((1, 2),)
eddy_edge_pairs(::Val{3}) = ((2, 3), (1, 3), (1, 2))

eddy_potential(ns::NoiseState{<:Any, D}) where {D} = eddy_components(Val(D), ns.psie_z, ns.psie_x, ns.psie)
eddy_previous(ns::NoiseState{<:Any, D}) where {D}  = eddy_components(Val(D), ns.psieo_z, ns.psieo_x, ns.psieo)
eddy_white_noise(ns::NoiseState{<:Any, D}) where {D} = eddy_components(Val(D), ns.re_z, ns.re_x, ns.re)
eddy_edges(ns::NoiseState{<:Any, D}) where {D} = eddy_components(Val(D), ns.psiec_z, ns.psiec_x, ns.psiec)

# white noise of the particle-eddy potential: the eddy white noise `re` (noise.m);
# in 3-D the normalized sum over the vector-potential components, which is again
# standard normal, so psix keeps the 2-D variance and couples to all components alike
xtal_white_noise(ns::NoiseState{<:Any, 2}) = ns.re
function xtal_white_noise(ns::NoiseState{T, 3}) where {T}
    s = inv(sqrt(T(3)))
    return Broadcast.broadcasted((a, b, c) -> (a + b + c) * s, ns.re_z, ns.re_x, ns.re)
end

"""
    NoiseState(::Type{T}, [backend,] grid, scales; seed=0) -> NoiseState{T,D}

Allocate and precompute all static fields for a 2-D or 3-D `grid`. Mirrors the
filter kernel construction in `src/init.m` (lines 83–101).
"""
function NoiseState(::Type{T}, backend, grid::Grid{T, D}, scales::Scales{T};
                    seed::Integer = 0) where {T<:AbstractFloat, D}
    N = cells(grid);  Nz = N[1];  Nh = Base.tail(N)
    h   = Float64(grid.h)
    L0h = Float64(scales.L0h)
    l0h = Float64(scales.l0h)

    # padding widths: 4 × ceil(length_scale / h) rows on each side
    padL0 = 4 * ceil(Int, L0h / h)
    padl0 = 4 * ceil(Int, l0h / h)

    # Gaussian filters: built on the host (small, setup-only) then moved to the
    # target backend so the per-step FFT filtering runs on-device.
    Gkpe = adapt_backend(backend, gaussian_filter(T, (Nz + padL0, Nh...), h, L0h))
    Gkps = adapt_backend(backend, gaussian_filter(T, (Nz + padl0, Nh...), h, l0h))

    # stream-function to flux-component variance normalisation (init.m line 28-29)
    fL = T(2 / sqrt(1 - exp(-h^2 / (2 * L0h^2))))
    fl = T(2 / sqrt(1 - exp(-h^2 / (2 * l0h^2))))

    z(dims)  = xcore_zeros(backend, T, dims...)
    zc(dims) = xcore_zeros(backend, Complex{T}, dims...)
    y(dims)  = D == 3 ? z(dims) : nothing
    face(d) = (z(ghost_face_shape(N, 1)), z(ghost_face_shape(N, 2)), y(ghost_face_shape(N, 3)))
    A = typeof(z(N))

    return NoiseState{T, D, A, typeof(Gkpe), MersenneTwister, D == 3 ? A : Nothing}(
        z(N), z(N), z(N),                                  # psie, psix, psis
        z(N), z(N), z(N),                                  # psieo, psixo, psiso
        z(N), z(N),                                        # re, rs
        y(N), y(N), y(N), y(N), y(N), y(N),                # psie_z/x, psieo_z/x, re_z/x
        Gkpe, Gkps, padL0, padl0, fL, fl,
        face(1)..., face(2)..., face(3)...,                # xie*, xix*, xis* on faces z, x, y
        z(N), z(N), z(N),                                  # xie, xix, xis
        MersenneTwister(seed),
        z(N),                                              # psi_flt
        z(N .+ 2),                                         # psi_ext
        zc((Nz + padL0, Nh...)), zc((Nz + padl0, Nh...)),  # eddy_pad, settle_pad
        z(edge_shape(N, 1, 2)),                            # psiec
        D == 3 ? z(edge_shape(N, 2, 3)) : nothing,         # psiec_z
        D == 3 ? z(edge_shape(N, 1, 3)) : nothing,         # psiec_x
    )
end

NoiseState(::Type{T}, grid::Grid{T}, scales::Scales{T}; seed::Integer = 0) where {T} =
    NoiseState(T, KernelAbstractions.CPU(), grid, scales; seed)

# Gaussian filter kernel exp(-σ² |k|²) on an FFT domain of size `dims` (axis
# order z, x[, y]). σ here is the *full* scale (L0h or l0h); the padded filter
# omits the 1/2 factor present in the unpadded kernel — matches init.m lines 101-102.
function gaussian_filter(::Type{T}, dims::NTuple{D, Int}, h::Real, sigma::Real) where {T<:AbstractFloat, D}
    k = map(n -> 2π * fftfreq(n, 1 / h), dims)
    return Complex{T}[exp(-sigma^2 * foldl(+, (k[d][I[d]]^2 for d in 1:D)))
                      for I in CartesianIndices(dims)]
end

"""
    store_noise!(ns::NoiseState) -> nothing

Snapshot the potentials into the `*o` history buffers. Call once per outer
time step, before the Picard iteration (alongside `store_previous!`).
"""
function store_noise!(ns::NoiseState)
    foreach((o, p) -> o .= p, eddy_previous(ns), eddy_potential(ns))
    ns.psixo .= ns.psix
    ns.psiso .= ns.psis
    return nothing
end

# Pointwise pieces of the OU noise update (noise.m): decorrelation time of a
# length scale l at speed v, noise-flux amplitudes, the OU step with the
# decorrelation time relaxed towards the time step, and the noise taper.
@inline decorrelation_time(l, v, ε) = l / (2 * (v + ε))
@inline eddy_amplitude(Xi, fReL, ke, τ) = Xi * sqrt(max(zero(τ), fReL * ke / τ))
@inline function xtal_amplitude(Xi, chi, fReL, ke, τ, txi0)
    St = txi0 / τ
    return Xi * sqrt(max(zero(τ), chi * fReL * ke / τ * St / (one(τ) + St^2)))
end
@inline settling_amplitude(Xi, chi, ks, τ) = Xi * sqrt(max(zero(τ), chi * ks / τ))
@inline function ou_step(ψo, r, τ, sg, f, dt, τcap)
    F = exp(-dt / (inv(inv(τ) + inv(τcap)) + dt))
    return F * ψo + sqrt(max(zero(F), one(F) - F^2)) * sg * f * r
end
@inline noise_taper(chi, mu, χ0, μ0, dμ) =
    (one(chi) - exp(-chi / χ0)) * (one(chi) - exp(-max(zero(chi), mu - μ0) / dμ))

# forward difference of `A` along axis k (one entry shorter along k)
forward_diff(A, k) = Broadcast.broadcasted(-, selectdim(A, k, 2:size(A, k)), selectdim(A, k, 1:size(A, k) - 1))

# ξ = −∇×A on the compact faces of each axis, from the edge components of A
# (h = 1 convention, as MATLAB `ddz(f, 1)`; the grid spacing is in the amplitude)
function eddy_curl!(ξ::NTuple{2}, (Ay,)::Tuple)
    ξ[1] .= .-forward_diff(Ay, 2)
    ξ[2] .= forward_diff(Ay, 1)
    return ξ
end
function eddy_curl!(ξ::NTuple{3}, (Az, Ax, Ay)::Tuple)
    ξ[1] .= .-(forward_diff(Ay, 2) .- forward_diff(Ax, 3))
    ξ[2] .= .-(forward_diff(Az, 3) .- forward_diff(Ay, 1))
    ξ[3] .= .-(forward_diff(Ax, 1) .- forward_diff(Az, 2))
    return ξ
end

# mean of the ghosted cell field `ext` over the four cells around each edge of
# the axis pair (a, b)
function edge_average!(Ae::AbstractArray{T, D}, ext::AbstractArray{T, D}, (a, b)) where {T, D}
    part(sa, sb) = view(ext, ntuple(k -> k == a ? (1 + sa:size(ext, k) - 1 + sa) :
                                         k == b ? (1 + sb:size(ext, k) - 1 + sb) : (2:size(ext, k) - 1),
                                    Val(D))...)
    Ae .= (part(0, 0) .+ part(0, 1) .+ part(1, 0) .+ part(1, 1)) ./ T(4)
    return Ae
end

"""
    noise!(ns, phase, grid, par, scales, dt; first_iter=true, xBC=:periodic, yBC=:periodic) -> nothing

One OU + spatial-filter noise update. Mirrors `src/noise.m`; in 3-D the eddy
stream function becomes a vector potential (see the file header) whose flux
amplitude is scaled by 1/√2, so that each flux component keeps the 2-D variance.

When `first_iter = true` (first Picard sweep of each outer step), fresh
Gaussian white noise is drawn into `ns.re` (and `re_z`, `re_x` in 3-D) and
`ns.rs`. Subsequent sweeps reuse the same noise so the Picard iteration does
not diverge due to changing forcing. The particle-eddy potential is driven by
the eddy white noise: `re` in 2-D, as in `noise.m`, and `(re_z + re_x + re)/√3`
in 3-D. `xBC`/`yBC` set the ghost rule of the noise fluxes (z is closed); `yBC`
is read only in 3-D.

Writes the noise fluxes `ns.xie*`, `ns.xix*`, `ns.xis*` on the ghosted faces
and the noise speed magnitudes `ns.xie`, `ns.xix`, `ns.xis` (diagnostics).
"""
function noise!(ns::NoiseState{T, D}, phase::PhaseState{T, D}, grid::Grid{T, D},
                par::Parameters{T}, scales::Scales{T}, dt::Real;
                first_iter::Bool = true, xBC::Symbol = :periodic,
                yBC::Symbol = :periodic) where {T<:AbstractFloat, D}
    periodic = periodic_axes("noise!", Val(D), xBC, yBC, :closed)

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
    # potential to flux-component variance normalisation; the 3-D curl sums two
    # independent potential derivatives per component
    fL = ns.fL;  fl = ns.fl
    fLe = D == 3 ? fL / sqrt(T(2)) : fL

    # draw white noise once per outer step (host MersenneTwister on CPU;
    # the array package's device RNG on GPU — see `noise_randn!`)
    if first_iter
        foreach(r -> noise_randn!(ns.rng, r), eddy_white_noise(ns))
        noise_randn!(ns.rng, ns.rs)
    end

    # --- OU time update of the eddy, particle-eddy and settling potentials ---
    foreach(eddy_potential(ns), eddy_previous(ns), eddy_white_noise(ns)) do psi, psio, r
        @. psi = ou_step(psio, r, decorrelation_time(L0, phase.V, eps_T),
                         eddy_amplitude(Xi, phase.fReL, phase.ke, decorrelation_time(L0, phase.V, eps_T)),
                         fLe, dtT, τcap)
    end
    rx = xtal_white_noise(ns)
    @. ns.psix = ou_step(ns.psixo, rx, decorrelation_time(L0, phase.V, eps_T),
                         xtal_amplitude(Xi, phase.chi, phase.fReL, phase.ke,
                                        decorrelation_time(L0, phase.V, eps_T), txi0),
                         fL, dtT, τcap)
    @. ns.psis = ou_step(ns.psiso, ns.rs, decorrelation_time(l0, phase.vx, eps_T),
                         settling_amplitude(Xi, phase.chi, phase.ks, decorrelation_time(l0, phase.vx, eps_T)),
                         fl, dtT, τcap)

    # --- eddy noise fluxes from the filtered potential (noise.m lines 61-64):
    # ghost-extend each filtered component, average it onto its edges, take −curl
    for (psi, Ae, pair) in zip(eddy_potential(ns), eddy_edges(ns), eddy_edge_pairs(Val(D)))
        filtered_extension!(ns, psi, ns.eddy_pad, ns.Gkpe, ns.padL0, periodic)
        edge_average!(Ae, ns.psi_ext, pair)
    end
    xie = per_axis(Val(D), ns.xiew, ns.xieu, ns.xiev)
    eddy_curl!(map(face_interior, xie, ntuple(identity, Val(D))), eddy_edges(ns))
    fill_face_ghosts!(xie, periodic)

    # --- particle-eddy and settling noise (noise.m lines 71-76): −∇ψ on the
    # ghosted faces, tapered to vanish near chi = 0 and where mu < 0.4 ---
    chif = per_axis(Val(D), phase.chiw, phase.chiu, phase.chiv)
    muf  = per_axis(Val(D), phase.muw,  phase.muu,  phase.muv)
    for (psi, pad, Gk, npad, ξ) in ((ns.psix, ns.eddy_pad, ns.Gkpe, ns.padL0, per_axis(Val(D), ns.xixw, ns.xixu, ns.xixv)),
                                    (ns.psis, ns.settle_pad, ns.Gkps, ns.padl0, per_axis(Val(D), ns.xisw, ns.xisu, ns.xisv)))
        filtered_extension!(ns, psi, pad, Gk, npad, periodic)
        for d in 1:D
            ξ[d] .= .-forward_diff(ns.psi_ext, d) .* noise_taper.(chif[d], muf[d], χ0, μ0, dμ)
        end
    end

    # --- noise speed magnitudes (cell-centered, for diagnostics) ---
    backend = KernelAbstractions.get_backend(ns.xie)
    for (mag, ξ) in ((ns.xis, per_axis(Val(D), ns.xisw, ns.xisu, ns.xisv)),
                     (ns.xix, per_axis(Val(D), ns.xixw, ns.xixu, ns.xixv)),
                     (ns.xie, xie))
        noise_magnitude_kernel!(backend)(mag, ξ; ndrange = size(mag))
    end
    KernelAbstractions.synchronize(backend)

    return nothing
end

# filter `psi` into `ns.psi_flt` and ghost-extend it into `ns.psi_ext`
function filtered_extension!(ns::NoiseState, psi, pad, Gk, npad, periodic)
    compute_fft_filter!(ns.psi_flt, pad, psi, Gk, npad)
    interior(ns.psi_ext) .= ns.psi_flt
    fill_cell_ghosts!(ns.psi_ext, periodic)
    return ns.psi_ext
end

# cell-centered speed magnitude from the per-axis noise fluxes on ghosted faces
@kernel function noise_magnitude_kernel!(mag, @Const(ξ))
    c = @index(Global, Cartesian)
    o = oneunit(c)
    comp(d) = (ξ[d][c + o - unit_index(c, d)] + ξ[d][c + o]) / 2
    sq = ntuple(d -> comp(d)^2, Val(length(c)))
    mag[c] = sqrt(tuple_foldl(+, first(sq), Base.tail(sq)))
end

# Pad `psi` with `pad` zero layers along z (half on each side) into caller-owned
# scratch `psi_pad`, apply the Gaussian FFT filter `Gk`, unpad, then rescale to
# match the original field's mean and standard deviation, writing into
# caller-owned `psi_flt`. The horizontal axes are periodic in the FFT.
# Array-type-generic: `fft!`/`ifft!` are the AbstractFFTs generics that dispatch
# automatically to FFTW (CPU), cuFFT (CUDA), or rocFFT (AMD) when the
# corresponding package is loaded — no explicit per-backend code needed. `Gk`
# must already be on the same backend (NoiseState moves it there at construction).
function compute_fft_filter!(psi_flt::AbstractArray{T}, psi_pad::AbstractArray{Complex{T}},
                             psi::AbstractArray{T}, Gk::AbstractArray{Complex{T}},
                             pad::Int) where {T<:AbstractFloat}
    Nz      = size(psi, 1)
    halfpad = pad ÷ 2
    inner   = selectdim(psi_pad, 1, halfpad + 1 : halfpad + Nz)

    fill!(psi_pad, zero(Complex{T}))
    inner .= psi

    # FFT → multiply filter → IFFT, all IN-PLACE (`fft!`/`ifft!`) so the
    # transforms reuse `psi_pad` instead of allocating fresh output arrays each
    # call. AbstractFFTs in-place generics dispatch to FFTW / cuFFT / rocFFT.
    fft!(psi_pad, 1:ndims(psi_pad))
    psi_pad .*= Gk
    ifft!(psi_pad, 1:ndims(psi_pad))

    psi_flt .= real.(inner)

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
