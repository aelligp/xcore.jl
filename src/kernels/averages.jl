using KernelAbstractions

# Face and edge averages of cell fields. Neighbors outside the domain follow the
# ghost rule of their axis (`ghost_index`: periodic wrap, otherwise edge repeat).

arithmetic_mean(a, b) = (a + b) / 2
geometric_mean(a, b) = sqrt(a * b)

# out[J]: `mean2` of the two cells adjacent to face J normal to `axis`; with
# `ghosted`, `out` carries one ghost layer along every other axis
@kernel function face_mean_kernel!(out, @Const(f), axis, ghosted::Bool, periodic, mean2)
    J = @index(Global, Cartesian)
    d = unval(axis);  N = size(f)
    cell(m) = CartesianIndex(ntuple(k -> k == d ? ghost_index(J[k] - 1 + m, N[k], periodic[k]) :
                                         ghosted ? ghost_index(J[k] - 1, N[k], periodic[k]) : J[k],
                                    Val(ndims(f))))
    out[J] = mean2(f[cell(0)], f[cell(1)])
end

"""
    face_mean!(out, f, d, periodic; ghosted = false, mean2 = arithmetic_mean) -> out

Average the cell field `f` onto the faces normal to axis `d`: `out[J]` is
`mean2` of the two cells adjacent to face `J`. `out` has the compact face shape
`face_shape(size(f), d)`, or with `ghosted` the ghosted face shape (one ghost
layer along every other axis). `periodic[k]` selects the ghost rule of axis k.
"""
function face_mean!(out::AbstractArray, f::AbstractArray, d::Integer, periodic::Tuple;
                    ghosted::Bool = false, mean2 = arithmetic_mean)
    N = size(f)
    expected = ghosted ? ghost_face_shape(N, d) : face_shape(N, d)
    size(out) == expected ||
        throw(DimensionMismatch("face_mean!: out has size $(size(out)), expected $expected"))
    Base.require_one_based_indexing(out, f)
    face_mean_kernel!(KernelAbstractions.get_backend(out))(out, f, Val(d), ghosted, periodic, mean2;
                                                           ndrange = size(out))
    return out
end

# geometric mean of the four cells around edge E of the axis pair (a, b)
@kernel function edge_geomean_kernel!(out, @Const(f), pa, pb, periodic)
    E = @index(Global, Cartesian)
    a = unval(pa);  b = unval(pb);  N = size(f)
    cell(ma, mb) = CartesianIndex(ntuple(k -> k == a ? ghost_index(E[k] - 1 + ma, N[k], periodic[k]) :
                                              k == b ? ghost_index(E[k] - 1 + mb, N[k], periodic[k]) : E[k],
                                         Val(ndims(f))))
    out[E] = (f[cell(0, 0)] * f[cell(1, 0)] * f[cell(0, 1)] * f[cell(1, 1)])^(one(eltype(f)) / 4)
end

"""
    edge_geomean!(edges, f, periodic) -> edges

Geometric mean of the cell field `f` over the four cells around every edge, for
each axis pair in the order (1,2), (1,3), (2,3); `edges[i]` has the shape
`edge_shape(size(f), a, b)` of its pair.
"""
function edge_geomean!(edges::Tuple, f::AbstractArray{<:Any, D}, periodic::Tuple) where {D}
    Base.require_one_based_indexing(f, edges...)
    backend = KernelAbstractions.get_backend(f)
    for (e, (pa, pb)) in zip(edges, axis_pairs(Val(D)))
        expected = edge_shape(size(f), unval(pa), unval(pb))
        size(e) == expected ||
            throw(DimensionMismatch("edge_geomean!: edge array has size $(size(e)), expected $expected"))
        edge_geomean_kernel!(backend)(e, f, pa, pb, periodic; ndrange = size(e))
    end
    return edges
end
