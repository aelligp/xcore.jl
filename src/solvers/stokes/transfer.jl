# Multigrid grid transfers on the staggered grid. Coarsening halves every axis
# (h_c = 2h); coarse face J of component d coincides with fine face 2J − 1.
#
#   cell field   restrict: mean of the 2^D fine cells (arithmetic: geometric
#                and harmonic means contract more slowly under strong
#                viscosity contrast). prolong: tensor-product {3/4, 1/4}.
#   face field   restrict: mean of the 2^(D−1) fine faces on the coinciding
#                coarse face. prolong: along d, coinciding faces copy and
#                in-between faces average; across d, {3/4, 1/4} with the
#                neighbor from the ghost rule of that axis.
#
# Ghost rules of the prolongation neighbor: periodic wrap; walls and the z
# boundary of a cell field repeat the edge (Neumann); a face field across a
# wall reflects with −sds, across the z boundary with −cnv (top/bottom), the
# same values the velocity BCs write into the ghosts.

@kernel function restrict_cell_kernel!(coarse, @Const(fine))
    J = @index(Global, Cartesian)
    D = length(J);  T = eltype(coarse)
    base = 2J - oneunit(J)
    offs = CartesianIndices(ntuple(_ -> 0:1, Val(D)))
    s = fine[base]
    for k in 2:length(offs)
        s += fine[base + offs[k]]
    end
    coarse[J] = s * T(1 / 2^D)
end

@kernel function restrict_face_kernel!(coarse, @Const(fine), comp)
    J = @index(Global, Cartesian)
    d = unval(comp);  D = length(J);  T = eltype(coarse)
    base = 2J - oneunit(J)
    offs = CartesianIndices(ntuple(k -> k == d ? (0:0) : (0:1), Val(D)))
    s = fine[base]
    for k in 2:length(offs)
        s += fine[base + offs[k]]
    end
    coarse[J] = s * T(1 / 2^(D - 1))
end

"""
    restrict_cell!(coarse, fine) -> coarse

Mean of the 2^D fine cells of each coarse cell.
"""
function restrict_cell!(coarse, fine)
    restrict_cell_kernel!(KernelAbstractions.get_backend(coarse))(coarse, fine; ndrange = size(coarse))
    return coarse
end

"""
    restrict_face!(coarse, fine, d) -> coarse

Mean of the 2^(D−1) fine faces of component `d` (an `Int` or `Val(d)`)
covering each coarse face.
"""
function restrict_face!(coarse, fine, comp::Val)
    restrict_face_kernel!(KernelAbstractions.get_backend(coarse))(coarse, fine, comp;
                                                                  ndrange = size(coarse))
    return coarse
end
restrict_face!(coarse, fine, d::Int) = restrict_face!(coarse, fine, Val(d))

# coarse neighbor index along axis k of a cell-centered field
@inline neighbor_cell(i, n, periodic) = i < 1 ? (periodic ? n : 1) : (i > n ? (periodic ? 1 : n) : i)

@kernel function prolong_cell_kernel!(fine, @Const(coarse), periodic)
    I = @index(Global, Cartesian)
    D = length(I);  T = eltype(fine)
    Jc  = CartesianIndex(ntuple(k -> (I[k] + 1) ÷ 2, Val(D)))
    Jn  = CartesianIndex(ntuple(k -> neighbor_cell(isodd(I[k]) ? Jc[k] - 1 : Jc[k] + 1,
                                              size(coarse, k), periodic[k]), Val(D)))
    offs = CartesianIndices(ntuple(_ -> 0:1, Val(D)))
    acc = zero(T)
    for o in offs
        w = one(T);  idx = Jc
        for k in 1:D
            if o[k] == 1
                w *= T(0.25);  idx = replace_index(idx, k, Jn[k])
            else
                w *= T(0.75)
            end
        end
        acc += w * coarse[idx]
    end
    fine[I] = acc
end

"""
    prolong_cell!(fine, coarse, periodic) -> fine

Tensor-product {3/4, 1/4} interpolation of a cell field; `periodic[k]` selects
wrap or edge repeat along axis k.
"""
function prolong_cell!(fine, coarse, periodic::NTuple)
    prolong_cell_kernel!(KernelAbstractions.get_backend(fine))(fine, coarse, periodic;
                                                               ndrange = size(fine))
    return fine
end

# transverse coarse neighbor along axis k of a face field, and its weight sign
@inline function neighbor_face(k, d, I, Jc, n, periodic, sds, top_cnv, bot_cnv)
    T = typeof(sds)
    k == d && return (Jc[k], one(T))
    r = isodd(I[k]) ? Jc[k] - 1 : Jc[k] + 1
    1 <= r <= n && return (r, one(T))
    k == 1 && return (r < 1 ? 1 : n, r < 1 ? -top_cnv : -bot_cnv)
    periodic[k] && return (r < 1 ? n : 1, one(T))
    return (r < 1 ? 1 : n, -sds)
end

@kernel function prolong_face_kernel!(fine, @Const(coarse), comp, periodic, sds, top_cnv, bot_cnv)
    I = @index(Global, Cartesian)
    d = unval(comp);  D = length(I);  T = eltype(fine)
    ed = unit_index(I, d)
    odd = isodd(I[d])
    # coarse index of the cell-shaped (transverse) position, face along d
    Jc = CartesianIndex(ntuple(k -> k == d ? (odd ? (I[k] + 1) ÷ 2 : I[k] ÷ 2) : (I[k] + 1) ÷ 2, Val(D)))
    nb = ntuple(k -> neighbor_face(k, d, I, Jc, size(coarse, k), periodic, sds, top_cnv, bot_cnv),
                Val(D))
    along(J) = odd ? coarse[J] : T(0.5) * (coarse[J] + coarse[J + ed])
    offs = CartesianIndices(ntuple(k -> k == d ? (0:0) : (0:1), Val(D)))
    acc = zero(T)
    for o in offs
        w = one(T);  idx = Jc
        for k in 1:D
            k == d && continue
            if o[k] == 1
                w *= T(0.25) * nb[k][2];  idx = replace_index(idx, k, nb[k][1])
            else
                w *= T(0.75)
            end
        end
        acc += w * along(idx)
    end
    fine[I] = acc
end

"""
    prolong_face!(fine, coarse, d, op) -> fine

Interpolate component `d` (an `Int` or `Val(d)`; compact face shape) from the
coarse to the fine grid with the BC-consistent ghost rule of each transverse
axis (file header).
"""
function prolong_face!(fine, coarse, comp::Val, op::StokesOp)
    prolong_face_kernel!(KernelAbstractions.get_backend(fine))(
        fine, coarse, comp, op.periodic, op.sds, op.top_cnv, op.bot_cnv; ndrange = size(fine))
    return fine
end
prolong_face!(fine, coarse, d::Int, op::StokesOp) = prolong_face!(fine, coarse, Val(d), op)
