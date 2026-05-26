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

    # conservation error
    DB::Vector{T};    DM::Vector{T};    DX::Vector{T}

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
    return History{T}(
        v(), v(),                          # time, dt
        v(), v(), v(), v(), v(), v(),      # sumB/M/X, EB/EM/EX
        v(), v(), v(),                     # DB/DM/DX
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
    h2 = grid.h^2

    push!(hst.time, T(time))
    push!(hst.dt,   T(dt))

    # --- conservation ---
    sumB = sum(fluid.rho) * h2 * eps(T)
    sumM = sum(phase.M)   * h2 * eps(T)
    sumX = sum(phase.X)   * h2 * eps(T)
    push!(hst.sumB, sumB);  push!(hst.sumM, sumM);  push!(hst.sumX, sumX)

    sB0 = hst.sumB[1];  sM0 = hst.sumM[1];  sX0 = hst.sumX[1]
    push!(hst.EB, (sB - sB0) / (sB0 + eps(T)))
    push!(hst.EM, (sM - sM0) / (sB0 + eps(T)))
    push!(hst.EX, (sX - sX0) / (sB0 + eps(T)))

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
