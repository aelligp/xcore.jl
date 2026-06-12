# Grid-transfer operators for geometric multigrid on xcore's staggered MAC grid.
# Restriction (fine→coarse) and prolongation (coarse→fine) for the three field
# locations, in xcore's (iz, ix) = (z=dim1, x=dim2) convention. Coarsening halves
# each dimension: Nz_c = Nz÷2, Nx_c = Nx÷2, h_c = 2h.
#
# Field shapes (cache/interior sizing):
#   p (cell)   : (Nz,   Nx  ) → (Nz_c,   Nx_c  )
#   w (z-face) : (Nz+1, Nx  ) → (Nz_c+1, Nx_c  )
#   u (x-face) : (Nz,   Nx+1) → (Nz_c,   Nx_c+1)
#
# BCs in the prolongation ghosts: periodic in x, "closed" (linear extrapolation)
# in z — matching `dyrel_apply_*_bcs!`. Ported from the MATLAB restrict_*/prolong_*.

# ============================================================================
# RESTRICTION  (fine → coarse)
# ============================================================================

@kernel function restrict_p_kernel!(coarse, @Const(fine))
    J, I = @index(Global, NTuple)
    @inbounds begin
        jf = 2J - 1;  iff = 2I - 1
        coarse[J, I] = (fine[jf, iff] + fine[jf + 1, iff] +
                        fine[jf, iff + 1] + fine[jf + 1, iff + 1]) * oftype(fine[1, 1], 0.25)
    end
end

# z-face: coarse face J coincides with fine face (2J-1); average the two fine
# x-cells. Coarse (Nz_c+1, Nx_c).
@kernel function restrict_w_kernel!(coarse, @Const(fine))
    J, I = @index(Global, NTuple)
    @inbounds begin
        jf = 2J - 1;  iff = 2I - 1
        coarse[J, I] = (fine[jf, iff] + fine[jf, iff + 1]) * oftype(fine[1, 1], 0.5)
    end
end

# x-face: coarse face I coincides with fine face (2I-1); average the two fine
# z-rows. Coarse (Nz_c, Nx_c+1).
@kernel function restrict_u_kernel!(coarse, @Const(fine))
    J, I = @index(Global, NTuple)
    @inbounds begin
        jf = 2J - 1;  iff = 2I - 1
        coarse[J, I] = (fine[jf, iff] + fine[jf + 1, iff]) * oftype(fine[1, 1], 0.5)
    end
end

# No synchronize in the wrappers: KA keeps same-backend launches ordered; sync
# happens at the next host read (reductions / convergence checks), matching the
# rest of the solver. Per-kernel syncs here serialized the V-cycle on GPU.
function restrict_p!(coarse, fine)
    backend = KernelAbstractions.get_backend(coarse)
    restrict_p_kernel!(backend)(coarse, fine; ndrange = size(coarse))
    return coarse
end
function restrict_w!(coarse, fine)
    backend = KernelAbstractions.get_backend(coarse)
    restrict_w_kernel!(backend)(coarse, fine; ndrange = size(coarse))
    return coarse
end
function restrict_u!(coarse, fine)
    backend = KernelAbstractions.get_backend(coarse)
    restrict_u_kernel!(backend)(coarse, fine; ndrange = size(coarse))
    return coarse
end

# ============================================================================
# PROLONGATION  (coarse → fine), bilinear with periodic-x / closed-z ghosts.
# Each fine point reads coarse values via index arithmetic; BC-aware neighbour
# lookups are done inline (periodic wrap in x, clamp+extrapolate in z).
# ============================================================================

# periodic-x coarse column index
@inline _wrapx(I, Nxc) = I < 1 ? Nxc : (I > Nxc ? 1 : I)
# closed-z: clamp the coarse row (boundary uses nearest; extrapolation handled by
# the 3/4–1/4 weighting falling back to the edge cell)
@inline _clampz(J, Nzc) = J < 1 ? 1 : (J > Nzc ? Nzc : J)

# --- p (cell): tensor-product {1/4,3/4} bilinear. Fine cell (jf,iff) maps to a
# coarse cell plus the neighbour toward its half of that cell. ---
@kernel function prolong_p_kernel!(fine, @Const(coarse))
    jf, iff = @index(Global, NTuple)
    @inbounds begin
        Nzc, Nxc = size(coarse)
        T = eltype(fine)
        # coarse cell containing this fine cell, and the neighbour direction
        Jc = (jf + 1) ÷ 2
        Ic = (iff + 1) ÷ 2
        # within-cell side: odd fine index → "low" side (weight 3/4 self, 1/4 prev)
        zlow = isodd(jf)
        xlow = isodd(iff)
        Jn = zlow ? Jc - 1 : Jc + 1
        In = xlow ? Ic - 1 : Ic + 1
        Jn = _clampz(Jn, Nzc);  In = _wrapx(In, Nxc)
        wz_self = T(0.75);  wz_nb = T(0.25)
        wx_self = T(0.75);  wx_nb = T(0.25)
        fine[jf, iff] =
            (wz_self * wx_self) * coarse[Jc, Ic] +
            (wz_nb   * wx_self) * coarse[Jn, Ic] +
            (wz_self * wx_nb  ) * coarse[Jc, In] +
            (wz_nb   * wx_nb  ) * coarse[Jn, In]
    end
end

# --- w (z-face): coarse z-faces coincide with odd fine z-faces. Even fine
# z-faces interpolate between two coarse z-faces. x uses {1/4,3/4}. ---
@kernel function prolong_w_kernel!(fine, @Const(coarse))
    jf, iff = @index(Global, NTuple)
    @inbounds begin
        Nzc1, Nxc = size(coarse)        # coarse is (Nz_c+1, Nx_c)
        T = eltype(fine)
        Ic = (iff + 1) ÷ 2
        xlow = isodd(iff)
        In = xlow ? Ic - 1 : Ic + 1
        In = _wrapx(In, Nxc)
        wx_self = T(0.75);  wx_nb = T(0.25)
        if isodd(jf)                    # coincides with coarse face Jc=(jf+1)/2
            Jc = (jf + 1) ÷ 2
            fine[jf, iff] = wx_self * coarse[Jc, Ic] + wx_nb * coarse[Jc, In]
        else                            # between coarse faces jf/2 and jf/2+1
            Jc = jf ÷ 2
            a = T(0.5) * (coarse[Jc, Ic] + coarse[Jc + 1, Ic])
            b = T(0.5) * (coarse[Jc, In] + coarse[Jc + 1, In])
            fine[jf, iff] = wx_self * a + wx_nb * b
        end
    end
end

# --- u (x-face): mirror of w with z↔x. Coarse x-faces coincide with odd fine
# x-faces; even fine x-faces interpolate between two coarse x-faces; z {1/4,3/4}. ---
@kernel function prolong_u_kernel!(fine, @Const(coarse))
    jf, iff = @index(Global, NTuple)
    @inbounds begin
        Nzc, Nxc1 = size(coarse)        # coarse is (Nz_c, Nx_c+1)
        T = eltype(fine)
        Jc = (jf + 1) ÷ 2
        zlow = isodd(jf)
        Jn = zlow ? Jc - 1 : Jc + 1
        Jn = _clampz(Jn, Nzc)
        wz_self = T(0.75);  wz_nb = T(0.25)
        if isodd(iff)
            Ic = (iff + 1) ÷ 2
            fine[jf, iff] = wz_self * coarse[Jc, Ic] + wz_nb * coarse[Jn, Ic]
        else
            Ic = iff ÷ 2
            a = T(0.5) * (coarse[Jc, Ic] + coarse[Jc, Ic + 1])
            b = T(0.5) * (coarse[Jn, Ic] + coarse[Jn, Ic + 1])
            fine[jf, iff] = wz_self * a + wz_nb * b
        end
    end
end

function prolong_p!(fine, coarse)
    backend = KernelAbstractions.get_backend(fine)
    prolong_p_kernel!(backend)(fine, coarse; ndrange = size(fine))
    return fine
end
function prolong_w!(fine, coarse)
    backend = KernelAbstractions.get_backend(fine)
    prolong_w_kernel!(backend)(fine, coarse; ndrange = size(fine))
    return fine
end
function prolong_u!(fine, coarse)
    backend = KernelAbstractions.get_backend(fine)
    prolong_u_kernel!(backend)(fine, coarse; ndrange = size(fine))
    return fine
end
