using Statistics

# Port of src/history.m.  Accumulates per-step scalar diagnostics into
# growing vectors via push!.  The three-column convention from MATLAB
# (column 1 = min, 2 = mean/rms/geomean, 3 = max, 4 = std where present)
# is preserved as named triples for clarity.

_rms(A)    = sqrt(mean(x -> x^2, A))
_geomean(A) = exp(mean(log, max.(A, eps(eltype(A)))))

"""
    History{T}

Growing record of per-step diagnostics.  All vectors are appended via
`push!` inside `record_history!`; they are empty until the first call.

Conservation errors (`EB`, `EM`, `EX`) track fractional drift relative to
the initial total mass, melt-mass, and crystal-mass. For closed z-boundaries
with Da = 0 these should stay near machine precision.
"""
mutable struct History{T<:AbstractFloat}
    time::Vector{T};  dt::Vector{T}

    # total mass, melt mass, crystal mass (per unit depth, ×h²)
    sumB::Vector{T};  sumM::Vector{T};  sumX::Vector{T}
    EB::Vector{T};    EM::Vector{T};    EX::Vector{T}

    # BD2-integrated expected mass change (drift from boundary fluxes + reaction)
    DB::Vector{T};    DM::Vector{T};    DX::Vector{T}

    # cached rates of change for BD2 (current, 1-step lagged, 2-step lagged).
    # Shifted at the top of every call to record_history! exactly like the MATLAB
    # globals in src/history.m:3-5.
    dsumBdt::T;   dsumBdto::T;   dsumBdtoo::T
    dsumMdt::T;   dsumMdto::T;   dsumMdtoo::T
    dsumXdt::T;   dsumXdto::T;   dsumXdtoo::T

    # crystallinity x: min, mean, max, std
    x_min::Vector{T};  x_mean::Vector{T};  x_max::Vector{T};  x_std::Vector{T}

    # convection speed: min, rms, max
    V_min::Vector{T};  V_rms::Vector{T};  V_max::Vector{T}

    # segregation speeds: min, rms, max
    vx_min::Vector{T}; vx_rms::Vector{T}; vx_max::Vector{T}
    vm_min::Vector{T}; vm_rms::Vector{T}; vm_max::Vector{T}

    # noise speeds: min, rms, max
    xie_min::Vector{T}; xie_rms::Vector{T}; xie_max::Vector{T}
    xix_min::Vector{T}; xix_rms::Vector{T}; xix_max::Vector{T}
    xis_min::Vector{T}; xis_rms::Vector{T}; xis_max::Vector{T}

    # dimensionless numbers: min, geomean, max
    Ra_min::Vector{T};  Ra_gm::Vector{T};  Ra_max::Vector{T}
    Rc_min::Vector{T};  Rc_gm::Vector{T};  Rc_max::Vector{T}
    ReD_min::Vector{T}; ReD_gm::Vector{T}; ReD_max::Vector{T}
    Red_min::Vector{T}; Red_gm::Vector{T}; Red_max::Vector{T}

    # diffusivities: min, geomean, max
    ks_min::Vector{T};  ks_gm::Vector{T};  ks_max::Vector{T}
    ke_min::Vector{T};  ke_gm::Vector{T};  ke_max::Vector{T}
    kx_min::Vector{T};  kx_gm::Vector{T};  kx_max::Vector{T}

    # viscosities: min, geomean, max
    eta_min::Vector{T};    eta_gm::Vector{T};    eta_max::Vector{T}
    etamix_min::Vector{T}; etamix_gm::Vector{T}; etamix_max::Vector{T}
    etae_min::Vector{T};   etae_gm::Vector{T};   etae_max::Vector{T}
    etas_min::Vector{T};   etas_gm::Vector{T};   etas_max::Vector{T}
    etat_min::Vector{T};   etat_gm::Vector{T};   etat_max::Vector{T}
end

"""
    History(::Type{T}) -> History{T}

Construct an empty History accumulator.
"""
function History(::Type{T}) where {T<:AbstractFloat}
    v() = T[]
    z = zero(T)
    return History{T}(
        v(), v(),                          # time, dt
        v(), v(), v(), v(), v(), v(),      # sumB/M/X, EB/EM/EX
        v(), v(), v(),                     # DB/DM/DX
        z, z, z, z, z, z, z, z, z,         # dsumBdt/dto/dtoo for B, M, X
        v(), v(), v(), v(),                # x stats
        v(), v(), v(),                     # V stats
        v(), v(), v(), v(), v(), v(),      # vx, vm stats
        v(), v(), v(), v(), v(), v(), v(), v(), v(),   # xie, xix, xis
        v(), v(), v(), v(), v(), v(), v(), v(), v(), v(), v(), v(),  # Ra Rc ReD Red
        v(), v(), v(), v(), v(), v(), v(), v(), v(),   # ks ke kx
        v(), v(), v(), v(), v(), v(), v(), v(), v(), v(), v(), v(), v(), v(), v(),  # eta etamix etae etas etat
    )
end

"""
    record_history!(hst, step, time, dt, phase, fluid, ns, grid)

Append one record to `hst`. Called every `nrh` steps. The `ns` argument
may be `nothing`; in that case noise speeds are recorded as zero.
"""
function record_history!(hst::History{T},
                         time::Real, dt::Real,
                         phase::PhaseState{T}, fluid::FluidState{T},
                         ns::NoiseState{T}, grid::Grid{T},
                         a1::Real = T(1), a2::Real = T(1), a3::Real = T(0),
                         b1::Real = T(1), b2::Real = T(0), b3::Real = T(0)) where {T<:AbstractFloat}
    h  = grid.h
    h2 = h^2

    # Shift cached rates BEFORE computing the new ones — exactly matching
    # src/history.m:3-5. After this, dsumBdto holds the rate from the
    # previous call, dsumBdtoo the call before that.
    hst.dsumBdtoo = hst.dsumBdto;  hst.dsumBdto = hst.dsumBdt
    hst.dsumMdtoo = hst.dsumMdto;  hst.dsumMdto = hst.dsumMdt
    hst.dsumXdtoo = hst.dsumXdto;  hst.dsumXdto = hst.dsumXdt

    push!(hst.time, T(time))
    push!(hst.dt,   T(dt))

    # --- total mass [kg per unit y-depth] (src/history.m:14-16) ---
    # MATLAB adds a +eps to avoid div-by-zero in EB; preserve that exactly.
    sumB = sum(fluid.rho) * h2 + eps(T)
    sumM = sum(phase.M)   * h2 + eps(T)
    sumX = sum(phase.X)   * h2 + eps(T)
    push!(hst.sumB, sumB);  push!(hst.sumM, sumM);  push!(hst.sumX, sumX)

    # --- boundary flux integrals (src/history.m:19-28) ---
    # qz_* are sized (Nz+1, Nx+2); row 1 = top face, row end = bottom face;
    # interior x-columns are 2:end-1. Each `sum(qz[...,2:end-1]*h)` is the
    # line integral of the z-flux across the top/bottom boundary, in [kg/s].
    qzaX = phase.qz_advn_X;  qzaM = phase.qz_advn_M
    qzdX = phase.qz_dffn_X;  qzdM = phase.qz_dffn_M
    Nz1, Nxp2 = size(qzaX)
    interior = 2:(Nxp2 - 1)

    top_aX  = sum(@view qzaX[1,    interior]) * h
    bot_aX  = sum(@view qzaX[Nz1,  interior]) * h
    top_aM  = sum(@view qzaM[1,    interior]) * h
    bot_aM  = sum(@view qzaM[Nz1,  interior]) * h
    top_dX  = sum(@view qzdX[1,    interior]) * h
    bot_dX  = sum(@view qzdX[Nz1,  interior]) * h
    top_dM  = sum(@view qzdM[1,    interior]) * h
    bot_dM  = sum(@view qzdM[Nz1,  interior]) * h

    sumGx = sum(phase.Gx) * h2

    # Net expected rates of change driven by boundary fluxes + reaction Gx.
    # Sign convention from MATLAB: stored advection flux is signed v*f,
    # diffusion flux is -k*∂f. The MATLAB formula sums (top - bottom) which
    # corresponds to inflow at top minus outflow at bottom.
    dsumBdt_new = (top_aX - bot_aX) + (top_aM - bot_aM) +
                  (top_dX - bot_dX) + (top_dM - bot_dM)
    dsumMdt_new = -sumGx + (top_aM - bot_aM) + (top_dM - bot_dM)
    dsumXdt_new = +sumGx + (top_aX - bot_aX) + (top_dX - bot_dX)

    hst.dsumBdt = dsumBdt_new
    hst.dsumMdt = dsumMdt_new
    hst.dsumXdt = dsumXdt_new

    # --- BD2-integrated drift D{B,M,X} (src/history.m:30-32) ---
    stp = length(hst.time)
    if stp >= 2
        DBo  = hst.DB[stp - 1];  DBoo  = hst.DB[max(1, stp - 2)]
        DMo  = hst.DM[stp - 1];  DMoo  = hst.DM[max(1, stp - 2)]
        DXo  = hst.DX[stp - 1];  DXoo  = hst.DX[max(1, stp - 2)]
        DB_new = (T(a2)*DBo + T(a3)*DBoo +
                  (T(b1)*dsumBdt_new + T(b2)*hst.dsumBdto + T(b3)*hst.dsumBdtoo) * T(dt)) / T(a1)
        DM_new = (T(a2)*DMo + T(a3)*DMoo +
                  (T(b1)*dsumMdt_new + T(b2)*hst.dsumMdto + T(b3)*hst.dsumMdtoo) * T(dt)) / T(a1)
        DX_new = (T(a2)*DXo + T(a3)*DXoo +
                  (T(b1)*dsumXdt_new + T(b2)*hst.dsumXdto + T(b3)*hst.dsumXdtoo) * T(dt)) / T(a1)
        push!(hst.DB, DB_new); push!(hst.DM, DM_new); push!(hst.DX, DX_new)
    else
        push!(hst.DB, zero(T)); push!(hst.DM, zero(T)); push!(hst.DX, zero(T))
    end

    # --- fractional conservation error (src/history.m:35-37) ---
    sumB0 = hst.sumB[1]
    push!(hst.EB, (sumB - hst.DB[stp] - sumB0) / sumB0)
    push!(hst.EM, (sumM - hst.DM[stp] - hst.sumM[1]) / sumB0)
    push!(hst.EX, (sumX - hst.DX[stp] - hst.sumX[1]) / sumB0)

    # --- crystallinity ---
    push!(hst.x_min,  minimum(phase.x))
    push!(hst.x_mean, mean(phase.x))
    push!(hst.x_max,  maximum(phase.x))
    push!(hst.x_std,  std(phase.x))

    # --- convection speed ---
    push!(hst.V_min, minimum(phase.V))
    push!(hst.V_rms, _rms(phase.V))
    push!(hst.V_max, maximum(phase.V))

    # --- segregation speeds ---
    push!(hst.vx_min, minimum(phase.vx));  push!(hst.vx_rms, _rms(phase.vx));  push!(hst.vx_max, maximum(phase.vx))
    push!(hst.vm_min, minimum(phase.vm));  push!(hst.vm_rms, _rms(phase.vm));  push!(hst.vm_max, maximum(phase.vm))

    # --- noise speeds ---
    push!(hst.xie_min, minimum(ns.xie));  push!(hst.xie_rms, _rms(ns.xie));  push!(hst.xie_max, maximum(ns.xie))
    push!(hst.xix_min, minimum(ns.xix));  push!(hst.xix_rms, _rms(ns.xix));  push!(hst.xix_max, maximum(ns.xix))
    push!(hst.xis_min, minimum(ns.xis));  push!(hst.xis_rms, _rms(ns.xis));  push!(hst.xis_max, maximum(ns.xis))

    # --- dimensionless numbers ---
    push!(hst.Ra_min, minimum(phase.Ra));   push!(hst.Ra_gm, _geomean(phase.Ra));  push!(hst.Ra_max, maximum(phase.Ra))
    push!(hst.Rc_min, minimum(phase.Rc));   push!(hst.Rc_gm, _geomean(phase.Rc));  push!(hst.Rc_max, maximum(phase.Rc))
    push!(hst.ReD_min, minimum(phase.ReD)); push!(hst.ReD_gm, _geomean(phase.ReD)); push!(hst.ReD_max, maximum(phase.ReD))
    push!(hst.Red_min, minimum(phase.Red)); push!(hst.Red_gm, _geomean(phase.Red)); push!(hst.Red_max, maximum(phase.Red))

    # --- diffusivities ---
    push!(hst.ks_min, minimum(phase.ks));  push!(hst.ks_gm, _geomean(phase.ks));  push!(hst.ks_max, maximum(phase.ks))
    push!(hst.ke_min, minimum(phase.ke));  push!(hst.ke_gm, _geomean(phase.ke));  push!(hst.ke_max, maximum(phase.ke))
    push!(hst.kx_min, minimum(phase.kx));  push!(hst.kx_gm, _geomean(phase.kx));  push!(hst.kx_max, maximum(phase.kx))

    # --- viscosities ---
    push!(hst.eta_min, minimum(fluid.eta));       push!(hst.eta_gm, _geomean(fluid.eta));       push!(hst.eta_max, maximum(fluid.eta))
    push!(hst.etamix_min, minimum(phase.etamix)); push!(hst.etamix_gm, _geomean(phase.etamix)); push!(hst.etamix_max, maximum(phase.etamix))
    push!(hst.etae_min, minimum(phase.etae));     push!(hst.etae_gm, _geomean(phase.etae));     push!(hst.etae_max, maximum(phase.etae))
    push!(hst.etas_min, minimum(phase.etas));     push!(hst.etas_gm, _geomean(phase.etas));     push!(hst.etas_max, maximum(phase.etas))
    push!(hst.etat_min, minimum(phase.etat));     push!(hst.etat_gm, _geomean(phase.etat));     push!(hst.etat_max, maximum(phase.etat))

    return nothing
end
