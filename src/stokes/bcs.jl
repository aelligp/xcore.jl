# Boundary conditions. Velocity BCs run once per DR iteration, as one kernel per
# axis over that axis's boundary plane (z first, then the horizontal axes in
# order). Each pass reads only values the earlier passes have finalized, so the
# edges and corners of the ghost layer come out consistent.
#
#   z      W = 0 on the top face; bottom closed (W = 0), draining
#          (W = mfs_mean·Ddom/ρw: the net volume source leaves through the
#          bottom), or open (W from the discrete continuity of the bottom cell
#          row). Tangential components reflect: v[ghost] = −cnv·v[interior]
#          (cnv = +1 no slip, −1 free slip).
#   x, y   periodic: the normal component's two boundary faces are the same
#          face, tangential ghosts wrap. Solid walls: normal component = 0,
#          tangential ghosts reflect v[ghost] = −sds·v[interior]
#          (sds = −1 free slip, +1 no slip).
#   P      Neumann (repeat) in z and at walls, periodic wrap otherwise.

@inline _inside(I, A, a) = all(ntuple(k -> k == a || I[k] <= size(A, k), Val(length(I))))

# z pass over the horizontal plane (I[1] = 1). `mode`: 0 closed, 1 drain, 2 open.
@kernel function _velocity_bc_z_kernel!(v, ρf, @Const(MFS), top_cnv, bot_cnv,
                                        mfs_mean, Ddom, mode, h)
    I = @index(Global, Cartesian)
    @inbounds begin
        D = length(I)
        for d in 2:D
            A = v[d]
            if _inside(I, A, 1)
                n = size(A, 1)
                A[_setidx(I, 1, 1)] = -top_cnv * A[_setidx(I, 1, 2)]
                A[_setidx(I, 1, n)] = -bot_cnv * A[_setidx(I, 1, n - 1)]
            end
        end
        W = v[1]
        if all(ntuple(k -> k == 1 || 2 <= I[k] <= size(W, k) - 1, Val(D)))
            nb = size(W, 1)
            W[I] = zero(eltype(W))
            o  = _ones(I)
            Ib = _setidx(I, 1, nb)                 # bottom W face
            cb = _setidx(I - o, 1, nb - 1)         # bottom cell row, compact
            if mode == 0
                W[Ib] = zero(eltype(W))
            elseif mode == 1
                W[Ib] = mfs_mean * Ddom / ρf[1][cb + _unit(cb, 1)]
            elseif mode == 2
                s = _tfoldl(MFS[cb] * h + ρf[1][cb] * W[cb + o - _unit(cb, 1)],
                            ntuple(k -> k + 1, Val(D - 1))) do acc, k
                    ek = _unit(cb, k)
                    @inbounds acc - ρf[k][cb + ek] * v[k][cb + o] + ρf[k][cb] * v[k][cb + o - ek]
                end
                W[Ib] = s / ρf[1][cb + _unit(cb, 1)]
            end
        end
    end
end

# horizontal pass along axis a over its boundary plane (I[a] = 1)
@kernel function _velocity_bc_h_kernel!(v, axis, periodic::Bool, sds)
    I = @index(Global, Cartesian)
    @inbounds begin
        a = _val(axis)
        for d in 1:length(I)
            A = v[d]
            if _inside(I, A, a)
                n = size(A, a)
                i1 = _setidx(I, a, 1);  i2 = _setidx(I, a, 2)
                im = _setidx(I, a, n - 1);  in_ = _setidx(I, a, n)
                if d == a
                    if periodic
                        A[in_] = A[i1]
                    else
                        A[i1]  = zero(eltype(A))
                        A[in_] = zero(eltype(A))
                    end
                elseif periodic
                    A[i1]  = A[im]
                    A[in_] = A[i2]
                else
                    A[i1]  = -sds * A[i2]
                    A[in_] = -sds * A[im]
                end
            end
        end
    end
end

# boundary plane of axis a: extent 1 along a, the largest ghosted extent elsewhere
_plane(N::NTuple{D, Int}, a) where {D} = ntuple(k -> k == a ? 1 : N[k] + 2, Val(D))

"""
    apply_velocity_bcs!(fluid, grid, op; mfs_mean = 0, Ddom = 0, open_cnv = false)

Enforce the velocity boundary conditions (see the file header). `mfs_mean ≠ 0`
drains the net mass source through the bottom; `open_cnv = true` sets the
bottom W from the continuity of the bottom cell row instead.
"""
function apply_velocity_bcs!(fluid, grid, op::StokesOp{D, T};
                             mfs_mean::Real = 0, Ddom::Real = 0,
                             open_cnv::Bool = false) where {D, T}
    v = velocity(fluid);  N = cells(grid)
    backend = KernelAbstractions.get_backend(fluid.P)
    mode = open_cnv ? 2 : (iszero(mfs_mean) ? 0 : 1)
    _velocity_bc_z_kernel!(backend)(v, face_density(fluid), fluid.MFS, op.top_cnv, op.bot_cnv,
                                    T(mfs_mean), T(Ddom), mode, T(grid.h); ndrange = _plane(N, 1))
    _each_axis(Val(D)) do ax
        a = _val(ax)
        a == 1 || _velocity_bc_h_kernel!(backend)(v, ax, op.periodic[a], op.sds; ndrange = _plane(N, a))
    end
    return nothing
end

@kernel function _pressure_bc_kernel!(P, axis, periodic::Bool)
    I = @index(Global, Cartesian)
    @inbounds begin
        a = _val(axis);  n = size(P, a)
        if periodic
            P[_setidx(I, a, 1)] = P[_setidx(I, a, n - 1)]
            P[_setidx(I, a, n)] = P[_setidx(I, a, 2)]
        else
            P[_setidx(I, a, 1)] = P[_setidx(I, a, 2)]
            P[_setidx(I, a, n)] = P[_setidx(I, a, n - 1)]
        end
    end
end

"""
    apply_pressure_bcs!(fluid, op) -> nothing

Fill the pressure ghost ring: periodic or Neumann along each horizontal axis,
then Neumann in z.
"""
function apply_pressure_bcs!(fluid, op::StokesOp{D}) where {D}
    P = fluid.P
    backend = KernelAbstractions.get_backend(P)
    plane(a) = ntuple(k -> k == a ? 1 : size(P, k), Val(D))
    _each_axis(Val(D)) do ax
        a = _val(ax)
        a == 1 || _pressure_bc_kernel!(backend)(P, ax, op.periodic[a]; ndrange = plane(a))
    end
    _pressure_bc_kernel!(backend)(P, Val(1), false; ndrange = plane(1))
    return nothing
end

"""
    pressure_gauge!(fluid, P0) -> nothing

Remove the free constant pressure mode by shifting the interior so that
`Σ (P − P0) = 0`: the pressure keeps the mean level of `P0`, which
`update_pressure!` reads back as `Pt = Pl + P`. ∇P (hence velocity) is untouched.
"""
function pressure_gauge!(fluid, P0)
    Pint = _interior(fluid.P);  P0int = _interior(P0)
    target = sum(P0int) / length(P0int)
    shift = sum(Pint) / length(Pint) - target
    Pint .-= shift
    return nothing
end

"""
    pressure_gauge!(fluid, P0, η_b) -> nothing

η_b-weighted gauge `Σ (P − P0)/η_b = 0` of the multigrid smoother, whose
continuity residual carries `(P − P0)/(η_b·dt)`: the weighted gauge makes that
term integrate to zero, the global mass-solvability constraint.
"""
function pressure_gauge!(fluid, P0, η_b)
    Pint = _interior(fluid.P);  P0int = _interior(P0)
    num = sum(_bc((p, p0, η) -> (p - p0) / η, Pint, P0int, η_b))
    den = sum(_bc(inv, η_b))
    Pint .-= num / den
    return nothing
end
