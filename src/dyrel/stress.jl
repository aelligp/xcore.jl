using KernelAbstractions

# Minimal strain-rate → deviatoric-stress evaluation for the DYREL inner loop.
# Computes only what the momentum residual needs (txx, tzz cell-centred; txz
# corner-staggered), skipping the eII / V / Ra / segregation work that the
# full `update_stresses!` does. Writes into `phase.txx/tzz/txz` (reusing that
# storage — it IS the stress, and nothing else touches it mid-solve).
#
# Indexing matches xcore's `compute_strain_kernel!` / `compute_shear_kernel!`
# exactly, so DYREL stresses are bit-identical to the constitutive path's for
# the same (W, U, η, ηco).

@kernel function compute_dyrel_stress_cell_kernel!(txx, tzz, @Const(W), @Const(U),
                                                   @Const(eta), invh, third)
    iz, ix = @index(Global, NTuple)
    @inbounds begin
        Ux_c = (U[iz + 1, ix + 1] - U[iz + 1, ix]) * invh
        Wz_c = (W[iz + 1, ix + 1] - W[iz, ix + 1]) * invh
        div  = Ux_c + Wz_c
        e    = eta[iz, ix]
        txx[iz, ix] = e * (Ux_c - div * third)
        tzz[iz, ix] = e * (Wz_c - div * third)
    end
end

@kernel function compute_dyrel_stress_corner_kernel!(txz, @Const(W), @Const(U),
                                                     @Const(etaco), invh, half)
    iz, ix = @index(Global, NTuple)
    @inbounds begin
        dUdz = (U[iz + 1, ix] - U[iz, ix]) * invh
        dWdx = (W[iz, ix + 1] - W[iz, ix]) * invh
        txz[iz, ix] = etaco[iz, ix] * (dUdz + dWdx) * half
    end
end

"""
    dyrel_update_stress!(phase, fluid, grid) -> nothing

Refresh `phase.txx`, `phase.tzz`, `phase.txz` from the current velocity
(`fluid.W`, `fluid.U`) and viscosity (`fluid.eta`, `fluid.etaco`). Called once
per PH outer iteration and once per DR inner iteration.
"""
function dyrel_update_stress!(phase::PhaseState{T}, fluid::FluidState{T},
                              grid::Grid{T}) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(fluid.W)
    invh = T(inv(grid.h))
    compute_dyrel_stress_cell_kernel!(backend)(phase.txx, phase.tzz,
                                                         fluid.W, fluid.U, fluid.eta,
                                                         invh, T(1//3);
                                                         ndrange = size(phase.txx))
    compute_dyrel_stress_corner_kernel!(backend)(phase.txz, fluid.W, fluid.U,
                                                           fluid.etaco, invh, T(0.5);
                                                           ndrange = size(phase.txz))
    # No synchronize: KA keeps same-backend launches ordered; sync happens at
    # the next host read (the reductions / convergence checks in solve.jl).
    return nothing
end
