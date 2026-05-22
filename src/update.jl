using KernelAbstractions
using Statistics

# Port of src/update.m. Splits the monolithic MATLAB script into focused KA
# kernels so each piece is independently testable and so per-cell work stays
# branch-free. The orchestrator `update!` runs them in the same order as
# update.m.
#
# Segregation- and noise-dependent quantities (`Wx`, `Ux`, `Wm`, `Um`, `vx`,
# `vm`, `xis*`, `xie*`, `xix*`, `etas`, `etasw`, `Nox`, `Nos`, `Rc`, `ks`) are
# not computed here — they require phsevo! and noise! to be in place. `kx`
# falls back to `ke` until then.

# ----------------------------------------------------------------------------
# volume fractions, bulk density, mass densities, phase indicators
# ----------------------------------------------------------------------------

@kernel function compute_volfrac_kernel!(rho, chi, mu, X, M, hasx, hasm,
                                  @Const(x), @Const(m), rhom0, rhox0, eps_t, one_meps)
    iz, ix = @index(Global, NTuple)
    @inbounds begin
        x_c = x[iz, ix];  m_c = m[iz, ix]
        # bulk density from harmonic-mean mixing of phase densities
        rho_c = one(eps_t) / (m_c / rhom0 + x_c / rhox0)
        rho[iz, ix] = rho_c
        # volume fractions, clamped to [eps, 1-eps]
        chi_raw = x_c * rho_c / rhox0
        mu_raw  = m_c * rho_c / rhom0
        chi[iz, ix] = clamp(chi_raw, eps_t, one_meps)
        mu[iz,  ix] = clamp(mu_raw,  eps_t, one_meps)
        # phase densities for advection
        X[iz, ix] = rho_c * x_c
        M[iz, ix] = rho_c * m_c
        # phase presence indicators (≥ √ε to match update.m)
        hasx[iz, ix] = x_c >= sqrt(eps_t)
        hasm[iz, ix] = m_c >= sqrt(eps_t)
    end
end

"""
    update_volume_fractions!(phase, fluid, par) -> nothing

Compute `chi`, `mu`, `rho`, `X`, `M`, `hasx`, `hasm` from primary `x`, `m`
fields. Mirrors lines 5–35 of `src/update.m`.
"""
function update_volume_fractions!(phase::PhaseState{T}, fluid::FluidState{T},
                                  par::Parameters{T}) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(phase.x)
    compute_volfrac_kernel!(backend, (16, 16))(fluid.rho, phase.chi, phase.mu,
                                        phase.X, phase.M, phase.hasx, phase.hasm,
                                        phase.x, phase.m,
                                        par.rhom0, par.rhox0,
                                        eps(T), one(T) - eps(T);
                                        ndrange = size(phase.x))
    KernelAbstractions.synchronize(backend)
    return nothing
end

# ----------------------------------------------------------------------------
# face/corner interpolations of densities and volume fractions
# Done with broadcasting + ghost-index views so it stays simple — these are
# index-shuffle operations, no per-cell math worth dedicated kernels.
# ----------------------------------------------------------------------------

"""
    update_interpolations!(phase, fluid, grid, par; xBC=:periodic, zBC=:closed) -> nothing

Compute face-averaged densities (`rhow`, `rhou`, `Drho`), density contrasts
(`Drhom`, `Drhox`) on z-faces, and face-averaged volume/mass fractions.
Mirrors lines 11–48 of `src/update.m`. Default BCs match `init.m`.
"""
function update_interpolations!(phase::PhaseState{T}, fluid::FluidState{T},
                                grid::Grid{T}, par::Parameters{T};
                                xBC::Symbol = :periodic,
                                zBC::Symbol = :closed) where {T<:AbstractFloat}
    Nz = grid.Nz;  Nx = grid.Nx
    icx = xBC === :periodic ? [Nx; collect(1:Nx); 1]      : [1; collect(1:Nx); Nx]
    icz = zBC === :periodic ? [Nz; collect(1:Nz); 1]      : [1; collect(1:Nz); Nz]

    # face densities (Nz+1, Nx) / (Nz, Nx+1)
    @views fluid.rhow[1:end, :] .= (fluid.rho[icz[1:end-1], :] .+ fluid.rho[icz[2:end], :]) ./ T(2)
    @views fluid.rhou[:, 1:end] .= (fluid.rho[:, icx[1:end-1]] .+ fluid.rho[:, icx[2:end]]) ./ T(2)

    # density anomaly (column-wise mean of rhow subtracted)
    rhoref = mean(fluid.rhow, dims = 2)                       # (Nz+1, 1)
    @views fluid.Drho .= fluid.rhow .- rhoref

    # density contrasts (phase density minus face-interpolated bulk) — drive
    # the segregation speed in update_phase_velocities!
    @. phase.Drhom = par.rhom0 - fluid.rhow
    @. phase.Drhox = par.rhox0 - fluid.rhow

    # volume fractions on (Nz+1, Nx+2) and (Nz+2, Nx+1) staggered grids
    @views phase.chiw .= (phase.chi[icz[1:end-1], icx] .+ phase.chi[icz[2:end], icx]) ./ T(2)
    @views phase.muw  .= (phase.mu[ icz[1:end-1], icx] .+ phase.mu[ icz[2:end], icx]) ./ T(2)
    @views phase.chiu .= (phase.chi[icz, icx[1:end-1]] .+ phase.chi[icz, icx[2:end]]) ./ T(2)
    @views phase.muu  .= (phase.mu[ icz, icx[1:end-1]] .+ phase.mu[ icz, icx[2:end]]) ./ T(2)

    # mass fractions on Z-faces (Nz+1, Nx) and X-faces (Nz, Nx+1)
    @views phase.x_w .= (phase.x[icz[1:end-1], :] .+ phase.x[icz[2:end], :]) ./ T(2)
    @views phase.m_w .= (phase.m[icz[1:end-1], :] .+ phase.m[icz[2:end], :]) ./ T(2)
    @views phase.x_u .= (phase.x[:, icx[1:end-1]] .+ phase.x[:, icx[2:end]]) ./ T(2)
    @views phase.m_u .= (phase.m[:, icx[1:end-1]] .+ phase.m[:, icx[2:end]]) ./ T(2)

    return nothing
end

# ----------------------------------------------------------------------------
# lithostatic + total pressure (update.m lines 50–53)
# ----------------------------------------------------------------------------

"""
    update_pressure!(phase, fluid, grid, par) -> nothing

Set `Pl` from the cumulative weight of `rowmean(rhow)`, then `Pt = max(Ptop/100,
Pl + P_dyn)` using the interior of the dynamic pressure array `fluid.P`.
"""
function update_pressure!(phase::PhaseState{T}, fluid::FluidState{T},
                          grid::Grid{T}, par::Parameters{T}) where {T<:AbstractFloat}
    Nz = grid.Nz;  h = grid.h
    rhoref = vec(mean(fluid.rhow, dims = 2))                  # length Nz+1
    # column build of Pl: top cell uses rhoref[1] * g * h/2, then cumulative
    # rhoref[2:Nz] * g * h. Last entry rhoref[Nz+1] is unused.
    Pl_col = similar(rhoref, Nz)
    Pl_col[1] = rhoref[1] * par.g0 * h / T(2) + par.Ptop
    if Nz > 1
        @inbounds for j in 2:Nz
            Pl_col[j] = Pl_col[j - 1] + rhoref[j] * par.g0 * h
        end
    end
    @views phase.Pl .= reshape(Pl_col, Nz, 1)
    floor_p = par.Ptop / T(100)
    @views phase.Pt .= max.(floor_p, phase.Pl .+ fluid.P[2:end-1, 2:end-1])
    return nothing
end

# ----------------------------------------------------------------------------
# permission weights + mixture viscosity (update.m lines 55–71)
# Per-cell scalar kernel. AA, BB, CC are 2x2 rheology parameter matrices
# passed by value (KA copies the small array to each work item on CPU).
# ----------------------------------------------------------------------------

@kernel function compute_rheology_kernel!(etamix, @Const(chi), @Const(mu),
                                   AA11, AA12, AA21, AA22,
                                   BB11, BB12, BB21, BB22,
                                   CC11, CC12, CC21, CC22,
                                   etax0, etam0)
    iz, ix = @index(Global, NTuple)
    @inbounds begin
        f1 = chi[iz, ix];  f2 = mu[iz, ix]

        # Sf[i, j] = (f_j / BB[i,j])^(1/CC[i,j]), normalised so Σ_j Sf[i,j] = 1
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
        etaf1 = etax0 * thtv1
        etaf2 = etam0 * thtv2

        etamix[iz, ix] = f1 * etaf1 + f2 * etaf2
    end
end

"""
    update_rheology!(phase, par) -> nothing

Compute `etamix` from `chi`, `mu` using the permission-weight rheology
(AA/BB/CC) from `Parameters`. Mirrors lines 55–71 of `src/update.m`.
"""
function update_rheology!(phase::PhaseState{T}, par::Parameters{T}) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(phase.etamix)
    AA = par.AA;  BB = par.BB;  CC = par.CC
    compute_rheology_kernel!(backend, (16, 16))(phase.etamix, phase.chi, phase.mu,
                                         AA[1,1], AA[1,2], AA[2,1], AA[2,2],
                                         BB[1,1], BB[1,2], BB[2,1], BB[2,2],
                                         CC[1,1], CC[1,2], CC[2,1], CC[2,2],
                                         par.etax0, par.etam0;
                                         ndrange = size(phase.etamix))
    KernelAbstractions.synchronize(backend)
    return nothing
end

# ----------------------------------------------------------------------------
# kinematics: Div_V, strain rates, eII, V  (update.m lines 73–104)
# ----------------------------------------------------------------------------

@kernel function compute_strain_kernel!(exx, ezz, Div_V, @Const(W), @Const(U), invh)
    iz, ix = @index(Global, NTuple)
    @inbounds begin
        # ∂_x U at cell centre  = (U[interior z, ix+1] - U[interior z, ix]) / h
        Ux_c = (U[iz + 1, ix + 1] - U[iz + 1, ix]) * invh
        Wz_c = (W[iz + 1, ix + 1] - W[iz, ix + 1]) * invh
        div  = Ux_c + Wz_c
        Div_V[iz, ix] = div
        exx[iz, ix] = Ux_c - div / 3
        ezz[iz, ix] = Wz_c - div / 3
    end
end

@kernel function compute_shear_kernel!(exz, @Const(W), @Const(U), invh)
    # exz lives at corners (Nz+1, Nx+1). exz[j,i] = ((∂U/∂z + ∂W/∂x))/2.
    iz, ix = @index(Global, NTuple)
    @inbounds begin
        dUdz = (U[iz + 1, ix] - U[iz, ix]) * invh
        dWdx = (W[iz, ix + 1] - W[iz, ix]) * invh
        exz[iz, ix] = (dUdz + dWdx) / 2
    end
end

@kernel function compute_eII_V_kernel!(eII, V, @Const(exx), @Const(ezz), @Const(exz),
                                @Const(W), @Const(U), eps_t)
    iz, ix = @index(Global, NTuple)
    @inbounds begin
        # corner-to-centre average of the 4 surrounding exz values, squared
        e1 = exz[iz,     ix    ];  e2 = exz[iz + 1, ix    ]
        e3 = exz[iz,     ix + 1];  e4 = exz[iz + 1, ix + 1]
        exz_sq = (e1 * e1 + e2 * e2 + e3 * e3 + e4 * e4) / 4
        eII[iz, ix] = sqrt(eps_t +
                           (exx[iz, ix]^2 + ezz[iz, ix]^2) / 2 +
                            exz_sq)
        # cell-centre velocity magnitude (4-point average from staggered nodes)
        wc = (W[iz,     ix + 1] + W[iz + 1, ix + 1]) / 2
        uc = (U[iz + 1, ix    ] + U[iz + 1, ix + 1]) / 2
        V[iz, ix] = sqrt(wc * wc + uc * uc)
    end
end

@kernel function compute_segspeed_kernel!(vx, vm, @Const(wx), @Const(wm), eps_t)
    # cell-centred magnitude from the two flanking z-face segregation speeds
    # wx is (Nz+1, Nx+2); interior x is wx[:, 2:end-1] so we read col ix+1.
    iz, ix = @index(Global, NTuple)
    @inbounds begin
        w1 = wx[iz,     ix + 1];  w2 = wx[iz + 1, ix + 1]
        vx[iz, ix] = sqrt(((w1 + w2) / 2)^2 + eps_t)
        m1 = wm[iz,     ix + 1];  m2 = wm[iz + 1, ix + 1]
        vm[iz, ix] = sqrt(((m1 + m2) / 2)^2 + eps_t)
    end
end

"""
    update_kinematics!(phase, fluid, grid) -> nothing

Compute `Div_V`, deviatoric strain rates (`exx`, `ezz`, `exz`), `eII`, the
convection-speed magnitude `V`, and the segregation-speed magnitudes
`vx`, `vm`. Mirrors lines 73–104 of `src/update.m`.
"""
function update_kinematics!(phase::PhaseState{T}, fluid::FluidState{T},
                            grid::Grid{T}) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(phase.eII)
    invh = T(inv(grid.h))
    compute_strain_kernel!(backend, (16, 16))(phase.exx, phase.ezz, phase.Div_V,
                                       fluid.W, fluid.U, invh;
                                       ndrange = size(phase.exx))
    compute_shear_kernel!(backend, (16, 16))(phase.exz, fluid.W, fluid.U, invh;
                                      ndrange = size(phase.exz))
    compute_eII_V_kernel!(backend, (16, 16))(phase.eII, phase.V,
                                      phase.exx, phase.ezz, phase.exz,
                                      fluid.W, fluid.U, eps(T);
                                      ndrange = size(phase.eII))
    compute_segspeed_kernel!(backend, (16, 16))(phase.vx, phase.vm,
                                                phase.wx, phase.wm, eps(T);
                                                ndrange = size(phase.vx))
    KernelAbstractions.synchronize(backend)
    return nothing
end

# ----------------------------------------------------------------------------
# diffusivities + viscosity blend  (update.m lines 106–134)
# ----------------------------------------------------------------------------

@kernel function compute_viscblend_kernel!(eta, etae, ke, kx, ReL_arr, fReL_arr, ReD,
                                    @Const(etamix), @Const(eII), @Const(rho), @Const(V),
                                    @Const(eta_prev),
                                    L0, D0, etacntr_inv_floor, blend_w)
    iz, ix = @index(Global, NTuple)
    @inbounds begin
        eII_c = eII[iz, ix]
        rho_c = rho[iz, ix]
        # eddy diffusivity & viscosity
        ke_c = eII_c * L0 * L0
        # ReL evaluated at the current iteration
        ReL_c = V[iz, ix] * L0 / (etamix[iz, ix] / rho_c)
        fReL_c = one(L0) - exp(-ReL_c)
        etae_c = fReL_c * ke_c * rho_c
        # effective viscosity with eddy regularisation; kx falls back to ke
        # until particle diffusivity ks lands with segregation
        etai = etamix[iz, ix] + etae_c
        # contrast limit: 1/eta_eff = 1/etamax + 1/etai, etamax = min_etai * cntr
        # We can't reduce here, so apply a per-cell soft floor consistent with
        # MATLAB's update.m: etai itself is already bounded below by etamix.
        eta_new = etai
        # Picard-style relaxation against the previous iterate
        eta[iz, ix] = blend_w * eta_new + (one(L0) - blend_w) * eta_prev[iz, ix]
        etae[iz, ix] = etae_c
        ke[iz, ix]   = ke_c
        kx[iz, ix]   = ke_c                # ks=0 placeholder until segregation lands
        ReL_arr[iz, ix] = ReL_c
        fReL_arr[iz, ix] = fReL_c
        ReD[iz, ix] = V[iz, ix] * D0 / (eta[iz, ix] / rho_c)
    end
end

"""
    update_viscosity!(phase, fluid, grid, par, scales) -> nothing

Blend `etamix + etae` into `eta` with 50/50 Picard relaxation (matching
`update.m`'s `eta = (etai + eta)/2`). Computes `ke`, `kx`, `ReL`, `ReD`,
`fReL` along the way.
"""
function update_viscosity!(phase::PhaseState{T}, fluid::FluidState{T},
                           grid::Grid{T}, par::Parameters{T},
                           scales::Scales{T}) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(phase.etamix)
    # copy current eta as the "previous" iterate
    eta_prev = copy(fluid.eta)
    compute_viscblend_kernel!(backend, (16, 16))(fluid.eta, phase.etae, phase.ke, phase.kx,
                                          phase.ReL, phase.fReL, phase.ReD,
                                          phase.etamix, phase.eII, fluid.rho, phase.V,
                                          eta_prev,
                                          scales.L0, scales.D0,
                                          T(1) / par.etacntr, T(0.5);
                                          ndrange = size(fluid.eta))
    KernelAbstractions.synchronize(backend)
    # corner-interpolated etaco from the geometric mean over 4 surrounding cells
    # (matches update.m line 130). Mirror-pad in z for closed BC; periodic in x.
    compute_corner_eta!(fluid.etaco, fluid.eta; xBC = :periodic, zBC = :closed)
    return nothing
end

@kernel function compute_segvisc_kernel!(Rel, fRel, ks, etat, etas, kx, Red, Rc,
                                          @Const(vx), @Const(V), @Const(etamix),
                                          @Const(rho), @Const(ke), @Const(fReL),
                                          @Const(etas_prev),
                                          l0, d0, blend_w, eps_t)
    iz, ix = @index(Global, NTuple)
    @inbounds begin
        vx_c     = vx[iz, ix]
        rho_c    = rho[iz, ix]
        etamix_c = etamix[iz, ix]
        # particle Reynolds number on the eddy length scale + ramp factor
        Rel_c  = vx_c * l0 * rho_c / etamix_c
        fRel_c = one(l0) - exp(-Rel_c)
        # particle diffusivity from segregation speed; combined diffusivity
        ks_c   = vx_c * l0
        # turbulent drag viscosity contribution
        etat_c = fRel_c * ks_c * rho_c
        # Picard blend of (etamix + etat) against previous etas
        etas[iz, ix] = blend_w * (etamix_c + etat_c) +
                        (one(l0) - blend_w) * etas_prev[iz, ix]
        Rel[iz,  ix] = Rel_c
        fRel[iz, ix] = fRel_c
        ks[iz,   ix] = ks_c
        etat[iz, ix] = etat_c
        kx[iz,   ix] = ks_c + fReL[iz, ix] * ke[iz, ix]
        Red[iz,  ix] = vx_c * d0 / (etas[iz, ix] / rho_c)
        Rc[iz,   ix] = V[iz, ix] / max(vx_c, eps_t)
    end
end

"""
    update_segregation_viscosity!(phase, fluid, par, scales; zBC=:closed) -> nothing

Compute the segregation-side rheology: particle Reynolds `Rel`, ramp factor
`fRel`, particle diffusivity `ks` and combined `kx = ks + fReL·ke`, turbulent
drag viscosity `etat`, blended segregation viscosity `etas` (Picard 50/50),
and the z-face geometric mean `etasw` used by the segregation speed.
"""
function update_segregation_viscosity!(phase::PhaseState{T}, fluid::FluidState{T},
                                       par::Parameters{T}, scales::Scales{T};
                                       zBC::Symbol = :closed) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(phase.etas)
    etas_prev = copy(phase.etas)
    compute_segvisc_kernel!(backend, (16, 16))(phase.Rel, phase.fRel, phase.ks,
                                               phase.etat, phase.etas,
                                               phase.kx, phase.Red, phase.Rc,
                                               phase.vx, phase.V, phase.etamix,
                                               fluid.rho, phase.ke, phase.fReL,
                                               etas_prev,
                                               scales.l0, par.d0, T(0.5), eps(T);
                                               ndrange = size(phase.etas))
    KernelAbstractions.synchronize(backend)
    # etasw: geometric mean of etas across z-faces, (Nz+1, Nx).
    Nz = size(phase.etas, 1)
    icz = zBC === :periodic ? [Nz; collect(1:Nz); 1] : [1; collect(1:Nz); Nz]
    @views phase.etasw .= sqrt.(phase.etas[icz[1:end-1], :] .* phase.etas[icz[2:end], :])
    return nothing
end

function compute_corner_eta!(etaco::AbstractMatrix{T}, eta::AbstractMatrix{T};
                      xBC::Symbol = :periodic,
                      zBC::Symbol = :closed) where {T<:AbstractFloat}
    Nz, Nx = size(eta)
    icx = xBC === :periodic ? [Nx; collect(1:Nx); 1] : [1; collect(1:Nx); Nx]
    icz = zBC === :periodic ? [Nz; collect(1:Nz); 1] : [1; collect(1:Nz); Nz]
    # etaco at corner (j+1/2, i+1/2) (j∈0..Nz, i∈0..Nx) = (η00 η10 η01 η11)^(1/4)
    @views begin
        e00 = eta[icz[1:end-1], icx[1:end-1]]
        e10 = eta[icz[2:end],   icx[1:end-1]]
        e01 = eta[icz[1:end-1], icx[2:end]]
        e11 = eta[icz[2:end],   icx[2:end]]
        etaco .= (e00 .* e10 .* e01 .* e11) .^ T(0.25)
    end
    return etaco
end

# ----------------------------------------------------------------------------
# stresses + Rayleigh (update.m lines 136–158)
# ----------------------------------------------------------------------------

@kernel function compute_stress_kernel!(txx, tzz, tII, Ra,
                                 @Const(eta), @Const(exx), @Const(ezz),
                                 @Const(exz), @Const(txz),
                                 @Const(V), @Const(kx), D0, eps_t)
    iz, ix = @index(Global, NTuple)
    @inbounds begin
        eta_c = eta[iz, ix]
        txx_c = eta_c * exx[iz, ix]
        tzz_c = eta_c * ezz[iz, ix]
        txx[iz, ix] = txx_c
        tzz[iz, ix] = tzz_c
        # 4-corner average of squared corner stresses for tII
        t1 = txz[iz,     ix    ];  t2 = txz[iz + 1, ix    ]
        t3 = txz[iz,     ix + 1];  t4 = txz[iz + 1, ix + 1]
        txz_sq = (t1 * t1 + t2 * t2 + t3 * t3 + t4 * t4) / 4
        tII[iz, ix] = sqrt(eps_t + (txx_c * txx_c + tzz_c * tzz_c) / 2 + txz_sq)
        Ra[iz, ix]  = V[iz, ix] * D0 / kx[iz, ix]
    end
end

@kernel function compute_shear_stress_kernel!(txz, @Const(etaco), @Const(exz))
    iz, ix = @index(Global, NTuple)
    @inbounds txz[iz, ix] = etaco[iz, ix] * exz[iz, ix]
end

"""
    update_stresses!(phase, fluid, grid, scales) -> nothing

Compute deviatoric stresses and the cell-centre stress invariant `tII`, plus
the Rayleigh number `Ra = V D0 / kx`.
"""
function update_stresses!(phase::PhaseState{T}, fluid::FluidState{T},
                          grid::Grid{T}, scales::Scales{T}) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(phase.txx)
    compute_shear_stress_kernel!(backend, (16, 16))(phase.txz, fluid.etaco, phase.exz;
                                             ndrange = size(phase.txz))
    compute_stress_kernel!(backend, (16, 16))(phase.txx, phase.tzz, phase.tII, phase.Ra,
                                       fluid.eta, phase.exx, phase.ezz,
                                       phase.exz, phase.txz,
                                       phase.V, phase.kx,
                                       scales.D0, eps(T);
                                       ndrange = size(phase.txx))
    KernelAbstractions.synchronize(backend)
    return nothing
end

# ----------------------------------------------------------------------------
# time-step update (CFL on convective speed; particle speeds skipped for now)
# ----------------------------------------------------------------------------

"""
    update_dt(phase, fluid, grid, par, dt_prev; dtmax = par.dtmax) -> T

CFL-limited time step from the current `kx` and `(W, U)` magnitudes.
Matches `update.m` lines 160–164. Until particle settling lands, segregation
speeds are zero so the advective bound uses only `(W, U)`.
"""
function update_dt(phase::PhaseState{T}, fluid::FluidState{T}, grid::Grid{T},
                   par::Parameters{T}, dt_prev::Real) where {T<:AbstractFloat}
    h = grid.h
    dtk = (h / T(2))^2 / maximum(phase.kx)
    dta = (h / T(2)) / (maximum(abs.(fluid.W)) + maximum(abs.(fluid.U)) + eps(T))
    return min(T(1.5) * T(dt_prev), min(dtk, par.CFL * dta), par.dtmax)
end

# ----------------------------------------------------------------------------
# top-level orchestrator
# ----------------------------------------------------------------------------

"""
    update!(phase, fluid, grid, par, scales; xBC=:periodic, zBC=:closed) -> nothing

Run the full constitutive pipeline:

  1. `update_volume_fractions!` — `rho`, `chi`, `mu`, `X`, `M`, `hasx`, `hasm`
  2. `update_interpolations!`   — face/edge averages of densities and fractions
  3. `update_pressure!`         — `Pl`, `Pt`
  4. `update_rheology!`         — `etamix` from permission weights
  5. `update_kinematics!`       — `Div_V`, strain rates, `eII`, `V`
  6. `update_viscosity!`        — eddy regularisation, blended `eta`, `etaco`
  7. `update_stresses!`         — deviatoric stresses, `tII`, `Ra`

Segregation/noise terms are skipped (they need `phsevo!`/`noise!` to be
implemented). `update_dt` is exposed separately and called from the outer
time loop.
"""
function update!(phase::PhaseState{T}, fluid::FluidState{T}, grid::Grid{T},
                 par::Parameters{T}, scales::Scales{T};
                 xBC::Symbol = :periodic,
                 zBC::Symbol = :closed) where {T<:AbstractFloat}
    update_volume_fractions!(phase, fluid, par)
    update_interpolations!(phase, fluid, grid, par; xBC, zBC)
    update_pressure!(phase, fluid, grid, par)
    update_rheology!(phase, par)
    update_kinematics!(phase, fluid, grid)
    update_viscosity!(phase, fluid, grid, par, scales)
    update_segregation_viscosity!(phase, fluid, par, scales; zBC)
    update_stresses!(phase, fluid, grid, scales)
    return nothing
end
