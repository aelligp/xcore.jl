using Statistics

# Port of src/history.m. Per-step scalar diagnostics are appended to growing
# vectors. MATLAB's column convention (1 = min, 2 = mean/rms/geomean, 3 = max,
# 4 = std where present) is kept as `_min`, `_mean`/`_rms`/`_gm`, `_max` fields.

rms(A)     = sqrt(mean(x -> x^2, A))
geomean(A) = exp(mean(log, A))

# Append min, `center(A)` and max of `A` to the three history vectors.
function push_range!(vmin, vcenter, vmax, A, center)
    push!(vmin, minimum(A));  push!(vcenter, center(A));  push!(vmax, maximum(A))
    return nothing
end

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
    record_history!(hst, time, dt, phase, fluid, ns, grid, a1=1, a2=1, a3=0, b1=1, b2=0, b3=0)

Append one record to `hst`. Called every `nrh` steps. `a1..a3`, `b1..b3` are
the BD2 coefficients of the time integrator used for the mass-drift
bookkeeping `DB`, `DM`, `DX`.
"""
function record_history!(hst::History{T},
                         time::Real, dt::Real,
                         phase::PhaseState{T}, fluid::FluidState{T},
                         ns::NoiseState{T}, grid::Grid{T, D},
                         a1::Real = T(1), a2::Real = T(1), a3::Real = T(0),
                         b1::Real = T(1), b2::Real = T(0), b3::Real = T(0)) where {T<:AbstractFloat, D}
    h  = grid.h
    vol  = prod(ntuple(_ -> h, Val(D)))        # cell volume h^D
    area = prod(ntuple(_ -> h, Val(D - 1)))    # z-face area h^(D-1)

    # Shift cached rates before computing the new ones (src/history.m:3-5). After this, dsumBdto holds the rate from the
    # previous call, dsumBdtoo the call before that.
    hst.dsumBdtoo = hst.dsumBdto;  hst.dsumBdto = hst.dsumBdt
    hst.dsumMdtoo = hst.dsumMdto;  hst.dsumMdto = hst.dsumMdt
    hst.dsumXdtoo = hst.dsumXdto;  hst.dsumXdto = hst.dsumXdt

    push!(hst.time, T(time))
    push!(hst.dt,   T(dt))

    # --- total mass [kg per unit y-depth in 2-D, kg in 3-D] (src/history.m:14-16) ---
    # MATLAB adds a +eps to avoid div-by-zero in EB; preserve that exactly.
    sumB = sum(fluid.rho) * vol + eps(T)
    sumM = sum(phase.M)   * vol + eps(T)
    sumX = sum(phase.X)   * vol + eps(T)
    push!(hst.sumB, sumB);  push!(hst.sumM, sumM);  push!(hst.sumX, sumX)

    # --- boundary flux integrals (src/history.m:19-28) ---
    # qz_* have the ghosted z-face shape (Nz+1, Nx+2[, Ny+2]); plane 1 is the top
    # face, plane Nz+1 the bottom one, and the horizontal interior is 2:end-1 along
    # every other axis. Each sum is the surface integral of the z-flux across the
    # top/bottom boundary, in [kg/s] (per unit y-depth in 2-D).
    qzaX = phase.qz_advn_X;  qzaM = phase.qz_advn_M
    # `qz_dffn_M` is identically zero: only X diffuses, so `phsevo!` emits no melt
    # diffusive flux. It stays in the budget because adding melt diffusion (or an
    # open diffusive boundary) must show up here rather than be silently dropped.
    qzdX = phase.qz_dffn_X;  qzdM = phase.qz_dffn_M
    plane(q, k) = sum(view(q, k, ntuple(d -> 2:size(q, d + 1) - 1, Val(D - 1))...)) * area
    Nz1 = size(qzaX, 1)

    top_aX  = plane(qzaX, 1);  bot_aX  = plane(qzaX, Nz1)
    top_aM  = plane(qzaM, 1);  bot_aM  = plane(qzaM, Nz1)
    top_dX  = plane(qzdX, 1);  bot_dX  = plane(qzdX, Nz1)
    top_dM  = plane(qzdM, 1);  bot_dM  = plane(qzdM, Nz1)

    sumGx = sum(phase.Gx) * vol

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

    # --- field statistics: (min, centre, max) triples, std for x ---
    push!(hst.x_mean, mean(phase.x))
    push!(hst.x_std,  std(phase.x))
    push!(hst.x_min,  minimum(phase.x))
    push!(hst.x_max,  maximum(phase.x))

    push_range!(hst.V_min,  hst.V_rms,  hst.V_max,  phase.V,  rms)
    push_range!(hst.vx_min, hst.vx_rms, hst.vx_max, phase.vx, rms)
    push_range!(hst.vm_min, hst.vm_rms, hst.vm_max, phase.vm, rms)

    push_range!(hst.xie_min, hst.xie_rms, hst.xie_max, ns.xie, rms)
    push_range!(hst.xix_min, hst.xix_rms, hst.xix_max, ns.xix, rms)
    push_range!(hst.xis_min, hst.xis_rms, hst.xis_max, ns.xis, rms)

    push_range!(hst.Ra_min,  hst.Ra_gm,  hst.Ra_max,  phase.Ra,  geomean)
    push_range!(hst.Rc_min,  hst.Rc_gm,  hst.Rc_max,  phase.Rc,  geomean)
    push_range!(hst.ReD_min, hst.ReD_gm, hst.ReD_max, phase.ReD, geomean)
    push_range!(hst.Red_min, hst.Red_gm, hst.Red_max, phase.Red, geomean)

    push_range!(hst.ks_min, hst.ks_gm, hst.ks_max, phase.ks, geomean)
    push_range!(hst.ke_min, hst.ke_gm, hst.ke_max, phase.ke, geomean)
    push_range!(hst.kx_min, hst.kx_gm, hst.kx_max, phase.kx, geomean)

    push_range!(hst.eta_min,    hst.eta_gm,    hst.eta_max,    fluid.eta,     geomean)
    push_range!(hst.etamix_min, hst.etamix_gm, hst.etamix_max, phase.etamix, geomean)
    push_range!(hst.etae_min,   hst.etae_gm,   hst.etae_max,   phase.etae,   geomean)
    push_range!(hst.etas_min,   hst.etas_gm,   hst.etas_max,   phase.etas,   geomean)
    push_range!(hst.etat_min,   hst.etat_gm,   hst.etat_max,   phase.etat,   geomean)

    return nothing
end

"""
    time_average(v::AbstractVector) -> Vector

Running time average of a history series `v` (e.g. `hst.x_mean`), as recorded
in `HST.*_tavg` by `src/history.m`: entry `n` is the mean of `v[n0:n]` with
`n0 = round((1/n^10 + 1/(30 + 0.75(n - 30))^10)^(-1/10))`, a smooth minimum of
`n` and `7.5 + 0.75n`, i.e. the trailing quarter of the record once `n ≳ 30`.
"""
function time_average(v::AbstractVector)
    Base.require_one_based_indexing(v)
    return map(eachindex(v)) do n
        n0 = round(Int, (1 / float(n)^10 + 1 / (30 + (n - 30) * 0.75)^10)^(-1 / 10), RoundNearestTiesAway)
        mean(@view v[n0:n])
    end
end
