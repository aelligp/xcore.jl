using KernelAbstractions
using Statistics

# Port of src/update.m, split into one function per block of the MATLAB script;
# `update!` runs them in the order of update.m. Per-cell work is done by
# KernelAbstractions kernels, index shuffling by broadcasts and `face_mean!`.
#
# Every function works for 2-D and 3-D states (`PhaseState{T,D}`), axes ordered
# (z, x[, y]). Per-axis and per-edge-pair fields are reached as tuples
# (`per_axis`, `per_pair`, `axis_pairs`); sums over the diagonal components run
# horizontal axes first, then z.

# ----------------------------------------------------------------------------
# volume fractions, bulk density, mass densities, phase indicators
# ----------------------------------------------------------------------------

@kernel function compute_volfrac_kernel!(rho, chi, mu, hasx, hasm,
                                         @Const(x), @Const(m), rhom0, rhox0, eps_t, one_meps)
    I = @index(Global, Cartesian)
    x_c = x[I];  m_c = m[I]
    # bulk density: harmonic mean of the phase densities
    rho_c = one(eps_t) / (m_c / rhom0 + x_c / rhox0)
    rho[I] = rho_c
    # volume fractions, kept inside [eps, 1-eps] so powers and ratios stay finite
    chi[I] = clamp(x_c * rho_c / rhox0, eps_t, one_meps)
    mu[I]  = clamp(m_c * rho_c / rhom0, eps_t, one_meps)
    # phase presence indicators
    hasx[I] = x_c >= sqrt(eps_t)
    hasm[I] = m_c >= sqrt(eps_t)
end

"""
    update_volume_fractions!(phase, fluid, par) -> nothing

Compute `chi`, `mu`, `rho`, `hasx`, `hasm` from the primary fields `x`, `m`.
Mirrors lines 5–35 of `src/update.m`. Does not touch `X` or `M`: those are the
advected phase densities, owned by `phsevo!`; overwriting them with the new
`rho` would feed spot-wise undershoots at sharp fronts into the next WENO5
reconstruction.
"""
function update_volume_fractions!(phase::PhaseState{T}, fluid::FluidState{T},
                                  par::Parameters{T}) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(phase.x)
    compute_volfrac_kernel!(backend)(fluid.rho, phase.chi, phase.mu,
                                     phase.hasx, phase.hasm,
                                     phase.x, phase.m, par.rhom0, par.rhox0,
                                     eps(T), one(T) - eps(T);
                                     ndrange = size(phase.x))
    return nothing
end

# ----------------------------------------------------------------------------
# face interpolations of densities and fractions (update.m lines 11-48)
# ----------------------------------------------------------------------------

"""
    update_interpolations!(phase, fluid, par; xBC=:periodic, yBC=:periodic, zBC=:closed) -> nothing

Compute face-averaged densities (`rhow`, `rhou`[, `rhov`]) and the density
anomaly `Drho` (`rhow` minus its horizontal mean), density contrasts (`Drhom`,
`Drhox`) on z-faces, and face-averaged volume fractions (ghosted faces) and mass
fractions (compact faces). Mirrors lines 11–48 of `src/update.m`. Default BCs
match `init.m`; `yBC` is read only in 3-D.
"""
function update_interpolations!(phase::PhaseState{T, D}, fluid::FluidState{T, D},
                                par::Parameters{T};
                                xBC::Symbol = :periodic, yBC::Symbol = :periodic,
                                zBC::Symbol = :closed) where {T<:AbstractFloat, D}
    periodic = periodic_axes("update_interpolations!", Val(D), xBC, yBC, zBC)
    rhof = face_density(fluid)
    chif = per_axis(Val(D), phase.chiw, phase.chiu, phase.chiv)
    muf  = per_axis(Val(D), phase.muw,  phase.muu,  phase.muv)
    xf   = per_axis(Val(D), phase.x_w,  phase.x_u,  phase.x_v)
    mf   = per_axis(Val(D), phase.m_w,  phase.m_u,  phase.m_v)
    for d in 1:D
        face_mean!(rhof[d], fluid.rho, d, periodic)
        face_mean!(chif[d], phase.chi, d, periodic; ghosted = true)
        face_mean!(muf[d],  phase.mu,  d, periodic; ghosted = true)
        face_mean!(xf[d],   phase.x,   d, periodic)
        face_mean!(mf[d],   phase.m,   d, periodic)
    end

    # density anomaly: rhow minus its mean over the horizontal axes
    fluid.Drho .= fluid.rhow .- mean(fluid.rhow, dims = horizontal_dims(Val(D)))

    # density contrasts (phase density minus face-interpolated bulk) drive the
    # segregation speed
    @. phase.Drhom = par.rhom0 - fluid.rhow
    @. phase.Drhox = par.rhox0 - fluid.rhow
    return nothing
end

# the horizontal axes 2..D, as a `dims` argument
horizontal_dims(::Val{D}) where {D} = ntuple(k -> k + 1, Val(D - 1))

# ----------------------------------------------------------------------------
# lithostatic + total pressure (update.m lines 50–53)
# ----------------------------------------------------------------------------

"""
    update_pressure!(phase, fluid, grid, par) -> nothing

Set `Pl` from the cumulative weight of the horizontal mean of `rhow`, then
`Pt = max(Ptop/100, Pl + P_dyn)` with `P_dyn` the interior of `fluid.P`.
"""
function update_pressure!(phase::PhaseState{T, D}, fluid::FluidState{T, D},
                          grid::Grid{T, D}, par::Parameters{T}) where {T<:AbstractFloat, D}
    Nz = grid.ni[1];  h = grid.h
    # Pl[1] = Ptop + rhoref[1]·g·h/2, then Pl[j] = Pl[j-1] + rhoref[j]·g·h. The
    # horizontal mean of rhow is reduced on the device; the short column is
    # summed sequentially on the host, because not every KernelAbstractions
    # backend implements a device scan (`cumsum`).
    rhoref = Array(vec(mean(fluid.rhow, dims = horizontal_dims(Val(D)))))   # length Nz+1
    gh = par.g0 * h
    Pl_h = Vector{T}(undef, Nz)
    Pl_h[1] = rhoref[1] * gh / T(2) + par.Ptop
    for j in 2:Nz
        Pl_h[j] = Pl_h[j - 1] + rhoref[j] * gh
    end
    Pl_d = copyto!(similar(fluid.rhow, Nz), Pl_h)
    phase.Pl .= reshape(Pl_d, Nz, ntuple(_ -> 1, Val(D - 1))...)
    P_floor = par.Ptop / T(100)
    phase.Pt .= max.(P_floor, phase.Pl .+ interior(fluid.P))
    return nothing
end

# ----------------------------------------------------------------------------
# permission weights + mixture viscosity (update.m lines 55–71)
# The 2x2 rheology matrices AA, BB, CC are passed as scalars.
# ----------------------------------------------------------------------------

@kernel function compute_rheology_kernel!(etamix, @Const(chi), @Const(mu),
                                   AA11, AA12, AA21, AA22,
                                   BB11, BB12, BB21, BB22,
                                   CC11, CC12, CC21, CC22,
                                   etax0, etam0)
    I = @index(Global, Cartesian)
    f1 = chi[I];  f2 = mu[I]

    # Sf[i, j] = (f_j / BB[i,j])^(1/CC[i,j]), normalized so Σ_j Sf[i,j] = 1
    s11 = (f1 / BB11)^(one(etax0) / CC11);  s12 = (f2 / BB12)^(one(etax0) / CC12)
    s21 = (f1 / BB21)^(one(etax0) / CC21);  s22 = (f2 / BB22)^(one(etax0) / CC22)
    n1 = s11 + s12
    n2 = s21 + s22
    Sf11 = s11 / n1;  Sf12 = s12 / n1
    Sf21 = s21 / n2;  Sf22 = s22 / n2

    # Xf[i, j] = α_i * f_j + (1 - α_i) * Sf[i, j], with α_i = Σ_k AA[i,k] Sf[i,k]
    a1 = AA11 * Sf11 + AA12 * Sf12
    a2 = AA21 * Sf21 + AA22 * Sf22
    Xf11 = a1 * f1 + (one(etax0) - a1) * Sf11
    Xf12 = a1 * f2 + (one(etax0) - a1) * Sf12
    Xf21 = a2 * f1 + (one(etax0) - a2) * Sf21
    Xf22 = a2 * f2 + (one(etax0) - a2) * Sf22

    # thtv[i] = Π_j (η_j / η_i)^Xf[i,j]; etaf[i] = η_i · thtv[i] (= geometric mean of η_j^Xf)
    thtv1 = (etax0 / etax0)^Xf11 * (etam0 / etax0)^Xf12
    thtv2 = (etax0 / etam0)^Xf21 * (etam0 / etam0)^Xf22
    etamix[I] = f1 * (etax0 * thtv1) + f2 * (etam0 * thtv2)
end

"""
    update_rheology!(phase, par) -> nothing

Compute `etamix` from `chi`, `mu` using the permission-weight rheology
(AA/BB/CC) from `Parameters`. Mirrors lines 55–71 of `src/update.m`.
"""
function update_rheology!(phase::PhaseState{T}, par::Parameters{T}) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(phase.etamix)
    AA = par.AA;  BB = par.BB;  CC = par.CC
    compute_rheology_kernel!(backend)(phase.etamix, phase.chi, phase.mu,
                                     AA[1,1], AA[1,2], AA[2,1], AA[2,2],
                                     BB[1,1], BB[1,2], BB[2,1], BB[2,2],
                                     CC[1,1], CC[1,2], CC[2,1], CC[2,2],
                                     par.etax0, par.etam0;
                                     ndrange = size(phase.etamix))
    return nothing
end

# ----------------------------------------------------------------------------
# kinematics: Div_V, strain rates, eII, V  (update.m lines 73–104)
# ----------------------------------------------------------------------------

# diagonal deviatoric strain rates e_dd = ∂_d v_d − div/3 and the divergence, at
# cell centers; v[d] is ghosted, so cell c spans faces c + 1 − e_d and c + 1
@kernel function compute_strain_kernel!(ediag, Div_V, @Const(v), invh)
    c = @index(Global, Cartesian)
    o = oneunit(c)
    grad(a) = (d = unval(a); (v[d][c + o] - v[d][c + o - unit_index(c, d)]) * invh)
    ax = hz_vaxes(Val(length(c)))
    div = tuple_foldl((s, a) -> s + grad(a), grad(first(ax)), Base.tail(ax))
    Div_V[c] = div
    foreach_axis(Val(length(c))) do a
        ediag[unval(a)][c] = grad(a) - div / 3
    end
end

# shear strain rate e_ab = (∂_a v_b + ∂_b v_a)/2 on the edges of the axis pair
# (a, b); the velocities carry one ghost layer along the remaining axis
@kernel function compute_shear_kernel!(eab, @Const(va), @Const(vb), pa, pb, invh)
    E = @index(Global, Cartesian)
    ea = unit_index(E, unval(pa));  eb = unit_index(E, unval(pb))
    K = E + oneunit(E) - ea - eb
    dvb_da = (vb[K + ea] - vb[K]) * invh
    dva_db = (va[K + eb] - va[K]) * invh
    eab[E] = (dvb_da + dva_db) / 2
end

# mean of A² over the four edges of the axis pair (a, b) around cell c
@inline function edge_square_mean(A, (pa, pb), c)
    ea = unit_index(c, unval(pa));  eb = unit_index(c, unval(pb))
    e1 = A[c];  e2 = A[c + ea];  e3 = A[c + eb];  e4 = A[c + ea + eb]
    return (e1 * e1 + e2 * e2 + e3 * e3 + e4 * e4) / 4
end

# second invariant √(½ Σ_d A_dd² + Σ_(a,b) ⟨A_ab²⟩) at cell c of a tensor with
# cell-centered diagonal and edge-centered off-diagonal components
@inline function staggered_invariant(diag::Tuple, shear::Tuple, c)
    ax = hz_vaxes(Val(length(c)))
    sq(a) = diag[unval(a)][c]^2
    dsum = tuple_foldl((s, a) -> s + sq(a), sq(first(ax)), Base.tail(ax))
    ssum = map((A, p) -> edge_square_mean(A, p, c), shear, axis_pairs(Val(length(c))))
    return sqrt(tuple_foldl(+, dsum / 2, ssum))
end

@kernel function compute_eII_V_kernel!(eII, V, @Const(ediag), @Const(eshear), @Const(v), eps_t)
    c = @index(Global, Cartesian)
    o = oneunit(c)
    eII[c] = staggered_invariant(ediag, eshear, c) + eps_t
    # cell-center velocity magnitude from the two faces of each component
    vc(d) = (v[d][c + o - unit_index(c, d)] + v[d][c + o]) / 2
    vsq = ntuple(d -> (u = vc(d); u * u), Val(length(c)))
    V[c] = sqrt(tuple_foldl(+, first(vsq), Base.tail(vsq)))
end

# cell-centered magnitude of the segregation speeds `wx`, `wm` from the two
# flanking (ghosted) z-faces, offset by eps to stay strictly positive
@kernel function compute_segspeed_kernel!(vx, vm, @Const(wx), @Const(wm), eps_t)
    c = @index(Global, Cartesian)
    lo = c + oneunit(c) - unit_index(c, 1);  hi = c + oneunit(c)
    vx[c] = abs((wx[lo] + wx[hi]) / 2) + eps_t
    vm[c] = abs((wm[lo] + wm[hi]) / 2) + eps_t
end

# diagonal and off-diagonal strain-rate and stress components as tuples
strain_diag(p::PhaseState{<:Any, D}) where {D}  = per_axis(Val(D), p.ezz, p.exx, p.eyy)
strain_shear(p::PhaseState{<:Any, D}) where {D} = per_pair(Val(D), p.exz, p.eyz, p.exy)
stress_diag(p::PhaseState{<:Any, D}) where {D}  = per_axis(Val(D), p.tzz, p.txx, p.tyy)
stress_shear(p::PhaseState{<:Any, D}) where {D} = per_pair(Val(D), p.txz, p.tyz, p.txy)

"""
    update_kinematics!(phase, fluid, grid) -> nothing

Compute `Div_V`, the deviatoric strain rates (diagonal `ezz`, `exx`[, `eyy`] at
cells; shear `exz`[, `eyz`, `exy`] on edges), their second invariant `eII`, the
convection-speed magnitude `V`, and the segregation-speed magnitudes `vx`, `vm`.
Mirrors lines 73–104 of `src/update.m`.
"""
function update_kinematics!(phase::PhaseState{T, D}, fluid::FluidState{T, D},
                            grid::Grid{T, D}) where {T<:AbstractFloat, D}
    backend = KernelAbstractions.get_backend(phase.eII)
    invh = inv(grid.h)
    v = velocity(fluid)
    ediag = strain_diag(phase);  eshear = strain_shear(phase)
    compute_strain_kernel!(backend)(ediag, phase.Div_V, v, invh; ndrange = size(phase.Div_V))
    for (eab, (pa, pb)) in zip(eshear, axis_pairs(Val(D)))
        compute_shear_kernel!(backend)(eab, v[unval(pa)], v[unval(pb)], pa, pb, invh; ndrange = size(eab))
    end
    compute_eII_V_kernel!(backend)(phase.eII, phase.V, ediag, eshear, v, eps(T);
                                   ndrange = size(phase.eII))
    compute_segspeed_kernel!(backend)(phase.vx, phase.vm, phase.wx, phase.wm, eps(T);
                                      ndrange = size(phase.vx))
    return nothing
end

# ----------------------------------------------------------------------------
# diffusivities + viscosity blend  (update.m lines 106–134)
# ----------------------------------------------------------------------------

# fReL is the ramp factor left by the previous update (update.m:108-111 use it
# before update.m:147 refreshes it); `update!` refreshes it last.
@kernel function compute_viscblend_kernel!(etae, ke, @Const(fReL), @Const(eII), @Const(rho), L0)
    I = @index(Global, Cartesian)
    ke_c = eII[I] * L0 * L0
    etae[I] = fReL[I] * ke_c * rho[I]
    ke[I]   = ke_c
end

# Relax `eta` toward `etamix + extra`, whose contrast is capped at
# `min(etamix + extra) * etacntr` by a harmonic blend (update.m:118-127):
# eta = w·cap(etamix + extra) + (1 - w)·eta.
function relax_capped!(eta, etamix, extra, etacntr, w)
    T = eltype(eta)
    total = Broadcast.instantiate(Broadcast.broadcasted(+, etamix, extra))
    emax = minimum(total) * T(etacntr)
    w = T(w)
    @. eta = w * (T(1) / (T(1) / emax + T(1) / (etamix + extra))) + (T(1) - w) * eta
    return eta
end

"""
    update_viscosity!(phase, fluid, par, scales; xBC=:periodic, yBC=:periodic,
                      zBC=:closed, relaxation = 1//2) -> nothing

Blend `etamix + etae` into `eta` by the linear continuation
`eta = relaxation*etai + (1 - relaxation)*eta_old`, with `etae = fReL·ke·rho`
from the ramp factor of the previous update. Computes `ke` and `ReD` along the
way.

`relaxation` must match how often the caller applies it: once per Picard sweep
wants a large step (`1//2`, the `update.m` blend), whereas a refresh inside the
DYREL iteration is applied many times per solve and needs a proportionally
smaller one.
"""
function update_viscosity!(phase::PhaseState{T}, fluid::FluidState{T},
                           par::Parameters{T},
                           scales::Scales{T};
                           xBC::Symbol = :periodic, yBC::Symbol = :periodic,
                           zBC::Symbol = :closed,
                           relaxation::Real = T(1//2)) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(phase.etamix)
    compute_viscblend_kernel!(backend)(phase.etae, phase.ke, phase.fReL, phase.eII, fluid.rho,
                                       scales.L0; ndrange = size(fluid.eta))

    relax_capped!(fluid.eta, phase.etamix, phase.etae, par.etacntr, relaxation)
    # ReD uses the capped and blended eta
    @. phase.ReD = phase.V * scales.D0 / (fluid.eta / fluid.rho)

    # Edge viscosities (update.m:130). The ghost extension must match the one
    # `update_interpolations!` uses for the face densities: wrapping the x-ghost
    # of a walled domain would average in the viscosity of the opposite wall.
    update_edge_viscosity!(fluid; xBC, yBC, zBC)
    return nothing
end

# fReL and fRel are the ramp factors of the previous update, as for etae.
@kernel function compute_segvisc_kernel!(ks, etat, kx, Rc,
                                          @Const(vx), @Const(V), @Const(rho), @Const(ke),
                                          @Const(fReL), @Const(fRel), l0)
    I = @index(Global, Cartesian)
    vx_c = vx[I]
    # particle diffusivity from segregation speed; combined diffusivity
    ks_c = vx_c * l0
    ks[I] = ks_c
    # turbulent drag viscosity contribution
    etat[I] = fRel[I] * ks_c * rho[I]
    kx[I] = ks_c + fReL[I] * ke[I]
    Rc[I] = V[I] / vx_c
end

"""
    update_segregation_viscosity!(phase, fluid, par, scales; xBC=:periodic, yBC=:periodic,
                                  zBC=:closed) -> nothing

Compute the segregation-side rheology from the ramp factors of the previous
update: particle diffusivity `ks` and combined `kx = ks + fReL·ke`, turbulent
drag viscosity `etat = fRel·ks·rho`, blended segregation viscosity `etas`
(Picard 50/50), the z-face geometric mean `etasw` used by the segregation
speed, `Red` and `Rc`.
"""
function update_segregation_viscosity!(phase::PhaseState{T, D}, fluid::FluidState{T, D},
                                       par::Parameters{T}, scales::Scales{T};
                                       xBC::Symbol = :periodic, yBC::Symbol = :periodic,
                                       zBC::Symbol = :closed) where {T<:AbstractFloat, D}
    backend = KernelAbstractions.get_backend(phase.etas)
    compute_segvisc_kernel!(backend)(phase.ks, phase.etat, phase.kx, phase.Rc,
                                     phase.vx, phase.V, fluid.rho, phase.ke,
                                     phase.fReL, phase.fRel, scales.l0;
                                     ndrange = size(phase.etas))

    # fixed 50/50 Picard relaxation (update.m:122-123, 127)
    relax_capped!(phase.etas, phase.etamix, phase.etat, par.etacntr, 1//2)
    # Red uses the capped and blended etas
    @. phase.Red = phase.vx * par.d0 / (phase.etas / fluid.rho)

    # etasw: geometric mean of etas across the compact z-faces
    periodic = periodic_axes("update_segregation_viscosity!", Val(D), xBC, yBC, zBC)
    face_mean!(phase.etasw, phase.etas, 1, periodic; mean2 = geometric_mean)
    return nothing
end

"""
    update_edge_viscosity!(fluid; xBC=:periodic, yBC=:periodic, zBC=:closed) -> nothing

Set the edge viscosities (`etaco`[, `etayz`, `etaxy`]) to the geometric mean of
`fluid.eta` over the four cells around each edge; `yBC` is read only in 3-D.
"""
function update_edge_viscosity!(fluid::FluidState{T, D}; xBC::Symbol = :periodic,
                                yBC::Symbol = :periodic, zBC::Symbol = :closed) where {T, D}
    periodic = periodic_axes("update_edge_viscosity!", Val(D), xBC, yBC, zBC)
    edge_geomean!(edge_viscosity(fluid), fluid.eta, periodic)
    return nothing
end

# ----------------------------------------------------------------------------
# stresses + Rayleigh (update.m lines 136–158)
# ----------------------------------------------------------------------------

@kernel function compute_stress_kernel!(tdiag, tII, Ra, @Const(eta), @Const(ediag),
                                        @Const(tshear), @Const(V), @Const(kx), D0, eps_t)
    c = @index(Global, Cartesian)
    foreach_axis(Val(length(c))) do a
        d = unval(a)
        tdiag[d][c] = eta[c] * ediag[d][c]
    end
    tII[c] = staggered_invariant(tdiag, tshear, c) + eps_t
    Ra[c]  = V[c] * D0 / kx[c]
end

"""
    update_stresses!(phase, fluid, scales) -> nothing

Compute the deviatoric stresses (diagonal at cells with `eta`, shear on edges
with the edge viscosities), the cell-center stress invariant `tII`, and the
Rayleigh number `Ra = V D0 / kx`.
"""
function update_stresses!(phase::PhaseState{T, D}, fluid::FluidState{T, D},
                          scales::Scales{T}) where {T<:AbstractFloat, D}
    backend = KernelAbstractions.get_backend(phase.txx)
    tshear = stress_shear(phase)
    for (t, η, e) in zip(tshear, edge_viscosity(fluid), strain_shear(phase))
        t .= η .* e
    end
    compute_stress_kernel!(backend)(stress_diag(phase), phase.tII, phase.Ra, fluid.eta,
                                    strain_diag(phase), tshear, phase.V, phase.kx,
                                    scales.D0, eps(T); ndrange = size(phase.tII))
    return nothing
end

# ----------------------------------------------------------------------------
# time-step update (diffusive limit and CFL on the phase velocities)
# ----------------------------------------------------------------------------

"""
    update_dt(phase, grid, par, dt_prev) -> T

CFL-limited time step from the current `kx` and the **phase** velocities
(`Wx`, `Ux`[, `Vx`] and `Wm`, `Um`[, `Vm`]). Matches `update.m` line 162 exactly:
`dta = h/2 / max(abs([Um(:); Wm(:); Ux(:); Wx(:)]))`.

The phase velocities, not the bulk `(W, U)`, are what advect `X`/`M`; they
include segregation (`wx`, `wm`) and noise (`ξ`).
"""
function update_dt(phase::PhaseState{T, D}, grid::Grid{T, D},
                   par::Parameters{T}, dt_prev::Real) where {T<:AbstractFloat, D}
    h = grid.h
    dtk = (h / T(2))^2 / maximum(phase.kx)
    vabs(v) = abs(v + eps(T))
    vel = (per_axis(Val(D), phase.Wx, phase.Ux, phase.Vx)...,
           per_axis(Val(D), phase.Wm, phase.Um, phase.Vm)...)
    vmax = maximum(v -> maximum(vabs, v), vel)
    dta = (h / T(2)) / vmax
    return min(T(1.5) * T(dt_prev), min(dtk, par.CFL * dta), par.dtmax)
end

# Reynolds numbers on the eddy (L0) and particle (l0) scales and the ramp
# factors 1 - exp(-Re) that switch on the eddy and particle contributions in the
# next update (update.m:136-137, 147-148).
function update_ramp_factors!(phase::PhaseState, fluid::FluidState, scales::Scales)
    L0 = scales.L0;  l0 = scales.l0
    @. phase.ReL = phase.V  * L0 / (phase.etamix / fluid.rho)
    @. phase.Rel = phase.vx * l0 / (phase.etamix / fluid.rho)
    @. phase.fReL = 1 - exp(-phase.ReL)
    @. phase.fRel = 1 - exp(-phase.Rel)
    return nothing
end

# ----------------------------------------------------------------------------
# top-level orchestrator
# ----------------------------------------------------------------------------

"""
    update!(phase, fluid, grid, par, scales; xBC=:periodic, yBC=:periodic, zBC=:closed) -> nothing

Run the full constitutive pipeline:

  1. `update_volume_fractions!`       — `rho`, `chi`, `mu`, `hasx`, `hasm`
  2. `update_interpolations!`         — face averages of densities and fractions
  3. `update_pressure!`               — `Pl`, `Pt`
  4. `update_rheology!`               — `etamix` from permission weights
  5. `update_kinematics!`             — `Div_V`, strain rates, `eII`, `V`, `vx`, `vm`
  6. `update_viscosity!`              — eddy regularization, blended `eta`, edge viscosities, `ReD`
  7. `update_segregation_viscosity!`  — `ks`, `kx`, `etat`, blended `etas`, `etasw`, `Red`, `Rc`
  8. `update_stresses!`               — deviatoric stresses, `tII`, `Ra`
  9. `update_ramp_factors!`           — `ReL`, `Rel`, `fReL`, `fRel` for the next update

`update_dt` is called separately by `run!`, after `update!` in every sweep.
"""
function update!(phase::PhaseState{T}, fluid::FluidState{T}, grid::Grid{T},
                 par::Parameters{T}, scales::Scales{T};
                 xBC::Symbol = :periodic, yBC::Symbol = :periodic,
                 zBC::Symbol = :closed) where {T<:AbstractFloat}
    update_volume_fractions!(phase, fluid, par)
    update_interpolations!(phase, fluid, par; xBC, yBC, zBC)
    update_pressure!(phase, fluid, grid, par)
    update_rheology!(phase, par)
    update_kinematics!(phase, fluid, grid)
    update_viscosity!(phase, fluid, par, scales; xBC, yBC, zBC)
    update_segregation_viscosity!(phase, fluid, par, scales; xBC, yBC, zBC)
    update_stresses!(phase, fluid, scales)
    update_ramp_factors!(phase, fluid, scales)
    return nothing
end
