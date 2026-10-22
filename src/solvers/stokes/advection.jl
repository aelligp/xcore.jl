# Momentum advection div(v·ρv_d), frozen at the velocity a solve starts from:
# the outer Picard loop updates it by calling the solver again, which is the
# treatment `fluidmech!` applies, and it keeps per-iteration work free of the
# wide-stencil advection.
#
# Each component's momentum ρ_d·v_d is advected as a cell field on the control
# volumes (CVs) centered on its faces, with the flux-form schemes of `advect!` (`face_fluxes`). Along
# its own axis d:
#   * bounded (always z): the CVs are the interior faces 2..N_d, advected by v_d
#     interpolated to cell centers; boundary faces get zero.
#   * periodic: all N_d + 1 faces (the two boundary faces are the same physical
#     face, so the CV row has period N_d + 1), advected by v_d at the midpoints
#     between faces with periodic wrap; the two boundary values are averaged
#     afterwards so they agree.
# Along every other axis k the CVs follow the cell grid, advected by v_k
# averaged along d onto the CV faces.
#
# One kernel per component evaluates the momentum and the CV-face velocities
# directly from the state, applying the ghost rule of each axis on the fly
# (periodic wrap, otherwise repeat of the edge value); the TVD scheme reads one
# face further out, with the same rule.

# momentum ρ_d·v_d at CV index c (ghost rule applied); `own` = periodic own axis
Base.@propagate_inbounds @inline function cv_momentum(v, ρd, c::CartesianIndex{D}, d, own, n,
                                                      periodic) where {D}
    J = CartesianIndex(ntuple(k -> (i = ghost_index(c[k], n[k], k == d ? own : periodic[k]);
                                    k == d && !own ? i + 1 : i), Val(D)))
    return @inbounds ρd[J] * v[d][J + oneunit(J) - unit_index(J, d)]
end

# velocity on CV face i along axis k at the transverse position I
Base.@propagate_inbounds @inline function cv_velocity(v, I::CartesianIndex{D}, d, k, i, own, N) where {D}
    if k == d
        Nd = N[d]
        if own
            # midpoints between faces a(j), a(j + 1) with a = [N_d; 1:N_d+1; 2]
            a(j) = j == 1 ? Nd : (j == Nd + 3 ? 2 : j - 1)
            lo = CartesianIndex(ntuple(m -> m == d ? a(i) : I[m] + 1, Val(D)))
            hi = replace_index(lo, d, a(i + 1))
        else
            lo = CartesianIndex(ntuple(m -> m == d ? i : I[m] + 1, Val(D)))
            hi = lo + unit_index(lo, d)
        end
        return @inbounds (v[d][lo] + v[d][hi]) / 2
    end
    # v_k averaged along d over the two cells adjacent to the CV
    lo = CartesianIndex(ntuple(m -> m == d ? (own ? I[m] : I[m] + 1) : (m == k ? i : I[m] + 1), Val(D)))
    return @inbounds (v[k][lo] + v[k][lo + unit_index(lo, d)]) / 2
end

@kernel function momentum_advection_kernel!(advn, comp, @Const(v), @Const(ρd), scheme,
                                             periodic, N, n, g_tvd, invh, half)
    I = @index(Global, Cartesian)
    @inbounds begin
        d = unval(comp);  D = length(I)
        own = periodic[d]
        pcv(k) = k == d ? own : periodic[k]
        # flux balance along axis k: horizontal axes first, then z
        q(k) = (ek = unit_index(I, k);  nk = n[k];  p = pcv(k);
                vel(i) = cv_velocity(v, I, d, k, ghost_face_index(i, nk, p), own, N);
                face_fluxes(scheme, m -> cv_momentum(v, ρd, I + m * ek, d, own, n, periodic),
                        vel(I[k]), vel(I[k] + 1),
                        g_tvd ? vel(I[k] - 1) : zero(half), g_tvd ? vel(I[k] + 2) : zero(half), half))
        ax = hz_axes(Val(D))
        q1 = q(ax[1])
        acc = tuple_foldl((s, k) -> (qk = q(k); s + qk[1] - qk[2]), q1[1] - q1[2], Base.tail(ax))
        advn[own ? I : I + unit_index(I, d)] = acc * invh
    end
end

# boundary faces of component d along its own axis: periodic ⇒ the two copies
# of the boundary face take their mean; bounded ⇒ zero
@kernel function advection_boundary_kernel!(advn, comp, own::Bool)
    I = @index(Global, Cartesian)          # plane, I[d] = 1
    @inbounds begin
        d = unval(comp);  n = size(advn, d)
        i1 = replace_index(I, d, 1);  in_ = replace_index(I, d, n)
        if own
            avg = (advn[i1] + advn[in_]) / 2
            advn[i1] = avg;  advn[in_] = avg
        else
            advn[i1] = zero(eltype(advn));  advn[in_] = zero(eltype(advn))
        end
    end
end

"""
    momentum_advection!(cache, fluid, grid, par, op) -> nothing

Fill `cache.advn[d]` with `div(v·ρ_d v_d)` for every component from the current
velocity and face densities, scheme `par.ADVN` ∈ (`:centr`, `:upwd1`, `:quick`,
`:fromm`, `:weno3`, `:weno5`, `:tvdim`); see the file header.
"""
function momentum_advection!(cache::StokesCache{D, T}, fluid, grid, par, op::StokesOp{D}) where {D, T}
    scheme = par.ADVN
    scheme in ADVN_SCHEMES ||
        throw(ArgumentError("momentum_advection!: unsupported scheme = :$scheme"))
    return momentum_advection!(cache, fluid, grid, op, Val(scheme))
end

function momentum_advection!(cache::StokesCache{D, T}, fluid, grid, op::StokesOp{D},
                              scheme::Val{S}) where {D, T, S}
    backend = KernelAbstractions.get_backend(cache.R_P)
    v = velocity(fluid);  ρf = face_density(fluid)
    N = cells(grid);  invh = T(inv(grid.h))
    foreach_axis(Val(D)) do comp
        d = unval(comp);  own = op.periodic[d]
        n = ntuple(k -> k == d ? (own ? N[k] + 1 : N[k] - 1) : N[k], Val(D))
        momentum_advection_kernel!(backend)(cache.advn[d], comp, v, ρf[d], scheme, op.periodic,
                                             N, n, S === :tvdim, invh, T(0.5); ndrange = n)
        advection_boundary_kernel!(backend)(cache.advn[d], comp, own;
                                             ndrange = ntuple(k -> k == d ? 1 : size(cache.advn[d], k), Val(D)))
    end
    return nothing
end
