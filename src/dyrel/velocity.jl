using KernelAbstractions

# Velocity update kernels for xcore-DYREL. Each combines the JustRelax
# `update_V_damping!` (damping-pong) and `update_DR_V!` (velocity step) into a
# single pass per direction:
#
#   dVdτ   ← α · dVdτ + R          (damped pseudo-rate; R is preconditioned)
#   ΔV     = β · dτ · dVdτ          (this iteration's increment)
#   V     += ΔV                     (advance the velocity)
#   dV_step = ΔV                    (recorded for the Rayleigh-quotient λ_min)
#
# Index mapping: cache.dWdτ etc. are (Nz+1, Nx); they update the interior
# x-columns of state.W (= W[:, 2:end-1]). cache.dUdτ etc. are (Nz, Nx+1);
# they update the interior z-rows of state.U (= U[2:end-1, :]).

@kernel function compute_dyrel_update_W_kernel!(W, dWdτ, dW_step, @Const(R_W),
                                                @Const(α_W), @Const(β_W), @Const(dτ_W))
    iz_f, ix = @index(Global, NTuple)
    @inbounds begin
        dwdt = α_W[iz_f, ix] * dWdτ[iz_f, ix] + R_W[iz_f, ix]
        dWdτ[iz_f, ix] = dwdt
        step = β_W[iz_f, ix] * dτ_W[iz_f, ix] * dwdt
        dW_step[iz_f, ix] = step
        W[iz_f, ix + 1] += step          # state.W interior x-col = ix + 1
    end
end

@kernel function compute_dyrel_update_U_kernel!(U, dUdτ, dU_step, @Const(R_U),
                                                @Const(α_U), @Const(β_U), @Const(dτ_U))
    iz, ix = @index(Global, NTuple)
    @inbounds begin
        dudt = α_U[iz, ix] * dUdτ[iz, ix] + R_U[iz, ix]
        dUdτ[iz, ix] = dudt
        step = β_U[iz, ix] * dτ_U[iz, ix] * dudt
        dU_step[iz, ix] = step
        U[iz + 1, ix] += step            # state.U interior z-row = iz + 1
    end
end

"""
    dyrel_update_velocity!(cache, fluid) -> nothing

Advance `fluid.W`, `fluid.U` by one DR pseudo-time step using the current
preconditioned residuals (`cache.R_W`, `cache.R_U`) and per-face damping /
step coefficients. Records the applied increments into `cache.dW_step` /
`cache.dU_step` for the subsequent Rayleigh-quotient λ_min estimate.
"""
function dyrel_update_velocity!(cache::DyrelCache{T}, fluid::FluidState{T}) where {T<:AbstractFloat}
    backend = KernelAbstractions.get_backend(fluid.W)
    compute_dyrel_update_W_kernel!(backend)(fluid.W, cache.dWdτ, cache.dW_step,
                                                      cache.R_W, cache.α_W, cache.β_W, cache.dτ_W;
                                                      ndrange = size(cache.R_W))
    compute_dyrel_update_U_kernel!(backend)(fluid.U, cache.dUdτ, cache.dU_step,
                                                      cache.R_U, cache.α_U, cache.β_U, cache.dτ_U;
                                                      ndrange = size(cache.R_U))
    # No synchronize: KA keeps same-backend launches ordered; sync happens at
    # the next host read (the reductions / convergence checks in solve.jl).
    return nothing
end
