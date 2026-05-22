"""
    Scales{T}

Characteristic scales and dimensionless numbers derived from `Parameters` + `Grid`.
Direct port of `src/scales.m` — same fixed-point iteration on `(W0, w0)` for the
Re-dependent ramp factors, then derived diffusivities, viscosities, noise amplitudes
and dimensionless numbers.

Stored as a flat struct of `T`-typed scalars so it can travel with state on either
CPU or GPU backends. Use `print_scales(io, s, par)` to emit the same human-readable
report that `scales.m` writes to stdout.
"""
struct Scales{T<:AbstractFloat}
    # length scales
    h0::T; D0::T; d0::T
    L0::T; L0h::T
    l0::T; l0h::T
    bnd_w::T

    # material scales
    rho0::T; Drho0::T
    chi0::T; Dchi0::T
    eta0::T

    # velocity scales
    W0::T; w0::T
    W0l::T; w0l::T
    W0t::T; w0t::T
    W0i::T
    Ri0::T

    # Re-dependent ramp factors at characteristic state
    ReL0::T; Rel0::T
    fReL0::T; fRel0::T

    # diffusivities
    eII0::T; ke0::T; ks0::T; kx0::T

    # times
    tW0::T; tw0::T; tk0::T; ti0::T; txi0::T
    t0::T; dt0::T

    # noise amplitudes
    taue0::T; taus0::T; St0::T
    xie0::T; xix0::T; xis0::T

    # reaction
    G0::T

    # viscosities / stress
    etae0::T; etat0::T; p0::T

    # dimensionless numbers
    Noe0::T; Nox0::T; Nos0::T
    Rc0::T; Ra0::T
    ReD0::T; Red0::T
end

"""
    compute_scales(par::Parameters{T}, grid::Grid{T}; tol=1e-9, maxiter=200) -> Scales{T}

Solve the coupled `(W0, w0)` fixed point with Re-dependent drag, then derive all
secondary scales. The iteration uses the closed-form quadratic solution from
`scales.m` (lines 47–50) at every step. Converges in a handful of iterations for
all parameter ranges we care about.

Float32 note: the MATLAB code uses `vpa` (24-digit variable precision) inside the
loop to avoid cancellation near `eta0`-dominated limits. We don't have vpa in Julia
stdlib, so for `T == Float32` we do the iteration in Float64 internally and narrow
at the end. This costs nothing and avoids a class of stagnation bugs.
"""
function compute_scales(par::Parameters{T}, grid::Grid{T};
                        tol::Real = 1e-9, maxiter::Int = 200) where {T<:AbstractFloat}
    # promote iteration to Float64 for numerical safety, narrow result at end
    s = _compute_scales_inner(Float64(par.D), Float64(par.d0), Float64(par.L0),
                              Float64(par.l0), Float64(par.x0), Float64(par.Da),
                              Float64(par.rhom0), Float64(par.rhox0),
                              Float64(par.etam0), Float64(par.g0),
                              Float64(par.Xi), Float64(grid.h),
                              Float64(tol), maxiter)
    return _narrow_scales(s, T)
end

# returns a NamedTuple of Float64 scales; narrowing happens in compute_scales
function _compute_scales_inner(D, d0, L0_in, l0_in, x0, Da, rhom0, rhox0,
                               etam0, g0, Xi, h, tol, maxiter)
    # length scales
    h0    = h
    D0    = D / 10
    L0    = L0_in;  L0h = (L0 + h0) / 2
    l0    = l0_in;  l0h = (l0 + h0) / 2
    bnd_w = l0/2 + D/100

    # material scales
    rho0  = rhom0
    Drho0 = rhox0 - rhom0
    if Da > 0
        chi0  = Da
        Dchi0 = Da / 10
    else
        chi0  = x0
        Dchi0 = x0 / 10
    end
    eta0 = etam0

    # initial guesses: laminar speeds
    W0l = Dchi0 * Drho0 * g0 * D0^2 / eta0
    w0l =          Drho0 * g0 * d0^2 / eta0

    w0t = sqrt(       Drho0 * g0 * d0^2 / (l0  * rho0))
    W0t = sqrt(Dchi0 * Drho0 * g0 * D0^3 / (L0^2 * rho0))
    W0i = sqrt(Dchi0 * Drho0 * g0 * D    /          rho0)
    Ri0 = W0i / W0t

    W0 = W0l
    w0 = w0l
    ReL0 = W0 * L0 / (eta0 / rho0)
    Rel0 = w0 * l0 / (eta0 / rho0)
    fReL0 = 1 - exp(-ReL0)
    fRel0 = 1 - exp(-Rel0)

    for _ in 1:maxiter
        W0prv = W0; w0prv = w0
        ReL0  = W0 * L0 / (eta0 / rho0)
        Rel0  = w0 * l0 / (eta0 / rho0)
        fReL0 = 1 - exp(-ReL0)
        fRel0 = 1 - exp(-Rel0)
        # closed-form quadratic solution (mirrors scales.m lines 47–50)
        W0 = (sqrt(4/Ri0^2 * Dchi0 * Drho0 * g0 * rho0 * fReL0 * L0^2 * D0 + eta0^2) - eta0) *
             D0 / (2 * fReL0 * L0^2 * rho0 / Ri0^2)
        w0 = (sqrt(4         * Drho0 * g0 * rho0 * fRel0 * l0   * d0^2 + eta0^2) - eta0) /
             (2 * fRel0 * l0 * rho0)
        res = abs(W0 - W0prv)/W0 + abs(w0 - w0prv)/w0
        res <= tol && break
    end

    # diffusivities
    eII0 = W0 / D0
    ke0  = eII0 * L0^2
    ks0  = w0 * l0
    kx0  = ks0 + fReL0 * ke0

    # times
    tW0  = D / W0
    tw0  = D / w0
    tk0  = D^2 / kx0
    ti0  = D / W0i
    txi0 = rhox0 * d0^2 / 18 / eta0
    t0   = inv(1/(ti0 + tW0) + 1/tw0 + 1/tk0)
    dt0  = min((h0/2)^2 / kx0, (h0/2) / (W0 + w0))

    # noise amplitudes
    taue0 = L0 / 2 / W0
    taus0 = l0 / 2 / w0
    St0   = txi0 / taue0
    xie0  = Xi * sqrt(        fReL0 * ke0 / taue0)
    xix0  = Xi * sqrt(chi0  * fReL0 * ke0 / taue0 * St0 / (1 + St0^2))
    xis0  = Xi * sqrt(chi0  *         ks0 / taus0)

    # reaction
    G0 = Da * rho0 / t0 * D / D0

    # viscosities and stress
    etae0 = fReL0 * ke0 * rho0
    etat0 = fRel0 * ks0 * rho0
    p0    = (eta0 + etae0) * eII0

    # dimensionless numbers
    Noe0 = xie0 / W0
    Nox0 = xix0 / w0
    Nos0 = xis0 / w0
    Rc0  = W0 / w0
    Ra0  = W0 * D0 / kx0
    ReD0 = W0 * D0 / ((eta0 + etae0) / rho0)
    Red0 = w0 * d0 / ((eta0 + etat0) / rho0)

    return (; h0, D0, d0, L0, L0h, l0, l0h, bnd_w, rho0, Drho0, chi0, Dchi0, eta0,
            W0, w0, W0l, w0l, W0t, w0t, W0i, Ri0,
            ReL0, Rel0, fReL0, fRel0,
            eII0, ke0, ks0, kx0,
            tW0, tw0, tk0, ti0, txi0, t0, dt0,
            taue0, taus0, St0, xie0, xix0, xis0,
            G0, etae0, etat0, p0,
            Noe0, Nox0, Nos0, Rc0, Ra0, ReD0, Red0)
end

# narrow Float64 NamedTuple into Scales{T}
function _narrow_scales(s::NamedTuple, ::Type{T}) where {T<:AbstractFloat}
    return Scales{T}(
        T(s.h0), T(s.D0), T(s.d0),
        T(s.L0), T(s.L0h),
        T(s.l0), T(s.l0h),
        T(s.bnd_w),
        T(s.rho0), T(s.Drho0),
        T(s.chi0), T(s.Dchi0),
        T(s.eta0),
        T(s.W0), T(s.w0),
        T(s.W0l), T(s.w0l),
        T(s.W0t), T(s.w0t),
        T(s.W0i),
        T(s.Ri0),
        T(s.ReL0), T(s.Rel0),
        T(s.fReL0), T(s.fRel0),
        T(s.eII0), T(s.ke0), T(s.ks0), T(s.kx0),
        T(s.tW0), T(s.tw0), T(s.tk0), T(s.ti0), T(s.txi0),
        T(s.t0), T(s.dt0),
        T(s.taue0), T(s.taus0), T(s.St0),
        T(s.xie0), T(s.xix0), T(s.xis0),
        T(s.G0),
        T(s.etae0), T(s.etat0), T(s.p0),
        T(s.Noe0), T(s.Nox0), T(s.Nos0),
        T(s.Rc0), T(s.Ra0),
        T(s.ReD0), T(s.Red0),
    )
end

"""
    VisScales{T}

Visualization scale factors and unit strings for output plots. Computed by
`compute_vis_scales` from `Parameters`, `Scales`, and `Grid`. Mirrors the
`if ndm_op ... else ...` block in `src/scales.m`.

In dimensionless mode (`par.ndm_op = true`) every field is divided by its
characteristic scale so colorbars/axes read O(1). In dimensional mode the
scale factors are chosen so that velocities appear in m/yr, m/hr, or m/s;
depth in m/km/Mm; pressure in Pa/kPa/MPa/GPa; viscosity in Pa·s; etc.
"""
struct VisScales{T<:AbstractFloat}
    # spatial axis
    ssc::T;  sun::String
    # convection velocity (W, U, ξe, ξx)
    Wsc::T;  Wun::String
    # settling velocity (wx heatmap)
    wxsc::T
    # melt settling velocity (wm heatmap; = wxsc in dim, = w0·χ₀/(1-χ₀) in ndm)
    wmsc::T;  wun::String
    # history settling scale (= Wsc in dimensional, = w0 in ndm — mirrors whsc in MATLAB)
    whsc::T
    # profile speed scale (max(W0,w0) range)
    wmpsc::T; wpun::String
    # pressure
    psc::T;  pun::String
    # diffusivities
    kssc::T; kesc::T; kxsc::T; kun::String
    # noise speeds
    xiesc::T; xieun::String
    xissc::T; xisun::String
    xixsc::T; xixun::String
    # viscosity: eta / (esc + eesc), etas / (esc + etsc)
    esc::T;  eun::String
    eesc::T; etsc::T
    # density
    rsc::T;  dun::String
    # mass-flux source
    MFSsc::T; MFSun::String
    # crystallinity
    xsc::T;  xun::String
    # dimensionless numbers
    Rasc::T; ReDsc::T; Redsc::T; Rcsc::T
    Noesc::T; Noxsc::T; Nossc::T
end

"""
    compute_vis_scales(par, scales, grid) -> VisScales{T}

Port of the `if ndm_op ... else ...` block in `src/scales.m` (lines 137–249).
"""
function compute_vis_scales(par::Parameters{T}, scales::Scales{T},
                            grid::Grid{T}) where {T<:AbstractFloat}
    yr = 365.25 * 24 * 3600.0
    hr = 3600.0

    if par.ndm_op
        # ---- dimensionless output: divide by characteristic scales -----------
        chi0 = Float64(scales.chi0)
        wmsc_val = T(scales.w0 * chi0 / (1 - chi0))
        return VisScales{T}(
            T(grid.D), "1",
            T(scales.W0), "1",
            T(scales.w0), wmsc_val, "1",
            T(scales.w0),            # whsc = w0 in ndm (mirrors MATLAB whsc = w0)
            T(scales.W0), "1",
            T(scales.p0), "1",
            T(scales.ks0), T(scales.ke0), T(scales.kx0), "1",
            T(scales.xie0), "1",
            T(scales.xis0), "1",
            T(scales.xix0), "1",
            T(scales.eta0), "1", T(scales.etae0), T(scales.etat0),
            T(scales.rho0), "1",
            T(scales.rho0 / scales.t0), "1",
            T(scales.chi0), "1",
            T(scales.Ra0), T(scales.ReD0), T(scales.Red0), T(scales.Rc0),
            T(scales.Noe0), T(scales.Nox0), T(scales.Nos0),
        )
    else
        # ---- dimensional output: human-friendly units by magnitude ----------
        D  = Float64(grid.D)
        W0 = Float64(scales.W0);  w0 = Float64(scales.w0)
        p0 = Float64(scales.p0)

        ssc, sun = D  < 1e3 ? (1.0, "m") : D < 1e6 ? (1e3, "km") : (1e6, "Mm")

        Wsc, Wun = W0 < 1000/yr ? (1/yr, "m/yr") :
                   W0 < 1000/hr ? (1/hr, "m/hr") : (1.0, "m/s")

        wxsc, wun = w0 < 1000/yr ? (1/yr, "m/yr") :
                    w0 < 1000/hr ? (1/hr, "m/hr") : (1.0, "m/s")

        Wpsc_val  = max(W0, w0)
        wmpsc, wpun = Wpsc_val < 1000/yr ? (1/yr, "m/yr") :
                      Wpsc_val < 1000/hr ? (1/hr, "m/hr") : (1.0, "m/s")

        psc, pun = p0 < 1e2 ? (1.0, "Pa") : p0 < 1e6 ? (1e3, "kPa") :
                   p0 < 1e9 ? (1e6, "MPa") : (1e9, "GPa")

        return VisScales{T}(
            T(ssc), sun,
            T(Wsc), Wun,
            T(wxsc), T(wxsc), wun,      # wmsc = wxsc in dimensional mode
            T(Wsc),                     # whsc = Wsc in dimensional (mirrors MATLAB line 244)
            T(wmpsc), wpun,
            T(psc), pun,
            T(1), T(1), T(1), "m²/s",
            T(Wsc), Wun,                 # xiesc = Wsc (mirrors MATLAB line 241)
            T(wxsc), wun,                # xissc = wxsc
            T(Wsc), Wun,                 # xixsc = Wsc
            T(1), "Pa·s", T(0), T(0),
            T(1), "kg/m³",
            T(1), "kg/m³/s",
            T(0.01), "wt%",              # xsc = 1/100 so x/xsc gives percent
            T(1), T(1), T(1), T(1),
            T(1), T(1), T(1),
        )
    end
end

"""
    print_scales([io,] s::Scales, par::Parameters)

Emit the same human-readable scaling report as `src/scales.m`.
"""
print_scales(s::Scales, par::Parameters) = print_scales(stdout, s, par)

function print_scales(io::IO, s::Scales, par::Parameters)
    @printf(io, "\n  Scaled domain depth D0    = %1.0e [m]",   s.D0)
    @printf(io, "\n  Crystal size        d0    = %1.0e [m]",   s.d0)
    @printf(io, "\n  Eddy  corrl. length L0    = %1.0e [m]",   s.L0)
    @printf(io, "\n  Segr. corrl. length l0    = %1.0e [m]\n", s.l0)

    @printf(io, "\n  Density             rho0  = %1.0f  [kg/m3]",  s.rho0)
    @printf(io, "\n  Density contrast    Drho0 = %1.0f   [kg/m3]", s.Drho0)
    @printf(io, "\n  Cristal. contrast   Dchi0 = %1.3f [wt]",      s.Dchi0)
    @printf(io, "\n  Viscosity           eta0  = %1.0e [Pas]\n",   s.eta0)

    @printf(io, "\n  Convection  speed   W0    = %1.2e [m/s]",   s.W0)
    @printf(io, "\n  Segregation speed   w0    = %1.2e [m/s]\n", s.w0)

    @printf(io, "\n  Eddy  diffusivity   ke0   = %1.1e [m2/s]", s.ke0)
    @printf(io, "\n  Segr. diffusivity   ks0   = %1.1e [m2/s]", s.ks0)
    @printf(io, "\n  Eddy  viscosity     etae  = %1.1e [Pas]",  s.etae0)
    @printf(io, "\n  Segr. viscosity     etas0 = %1.1e [Pas]\n",s.etat0)

    @printf(io, "\n  Mixture-Eddy noise  xie0  = %1.2e [m/s]",   s.xie0)
    @printf(io, "\n  Particle-Eddy noise xix0  = %1.2e [m/s]",   s.xix0)
    @printf(io, "\n  Settling noise      xis0  = %1.2e [m/s]\n", s.xis0)

    @printf(io, "\n  Reaction rate       G0    = %1.2e [kg/m3/s]\n", s.G0)

    @printf(io, "\n  Inertial    time    ti0   = %1.2e [s]",   s.ti0)
    @printf(io, "\n  Convection  time    tW0   = %1.2e [s]",   s.tW0)
    @printf(io, "\n  Segregation time    tw0   = %1.2e [s]",   s.tw0)
    @printf(io, "\n  Diffusion   time    tk0   = %1.2e [s]\n", s.tk0)

    @printf(io, "\n  Dahmköhler No       Da0   = %1.2e [1]",   par.Da)
    @printf(io, "\n  Mixt.-Eddy Noise No Noe0  = %1.2e [1]",   s.Noe0)
    @printf(io, "\n  Part.-Eddy Noise No Nox0  = %1.2e [1]",   s.Nox0)
    @printf(io, "\n  Settling Noise No   Nos0  = %1.2e [1]\n", s.Nos0)

    @printf(io, "\n  Convection No       Rc0   = %1.2e [1]",   s.Rc0)
    @printf(io, "\n  Rayleigh No         Ra0   = %1.2e [1]",   s.Ra0)
    @printf(io, "\n  Domain  Reynolds No ReD0  = %1.2e [1]",   s.ReD0)
    @printf(io, "\n  Crystal Reynolds No Red0  = %1.2e [1]\n\n\n", s.Red0)
    return nothing
end
