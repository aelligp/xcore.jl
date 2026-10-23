using KernelAbstractions

# Dimension-generic index arithmetic on CartesianIndex, shared by the stencil
# kernels and the Stokes solvers.
@inline unval(::Val{d}) where {d} = d
@inline unit_index(::CartesianIndex{D}, d::Int) where {D} =
    CartesianIndex(ntuple(k -> ifelse(k == d, 1, 0), Val(D)))
@inline replace_index(I::CartesianIndex{D}, d::Int, i::Int) where {D} =
    CartesianIndex(ntuple(k -> ifelse(k == d, i, I[k]), Val(D)))

# horizontal axes first, then z; `hz_vaxes` as `Val`s, so that indexing a
# heterogeneous tuple by axis stays static inside a GPU kernel
@inline hz_axes(::Val{D}) where {D} = (ntuple(k -> k + 1, Val(D - 1))..., 1)
@inline hz_vaxes(::Val{D}) where {D} = (ntuple(k -> Val(k + 1), Val(D - 1))..., Val(1))

# left fold over a tuple, unrolled at compile time; `f` is inlined at every step
# (kernel bodies exceed the inliner's default budget)
@inline tuple_foldl(f, acc, ::Tuple{}) = acc
@inline tuple_foldl(f, acc, t::Tuple) = tuple_foldl(f, (@inline f(acc, first(t))), Base.tail(t))

# compile-time loop over the axes 1..D, passing each as a `Val`
@inline foreach_axis(f, ::Val{D}) where {D} = (ntuple(d -> (f(Val(d)); nothing), Val(D)); nothing)

# Per-axis fields in axis order (z, x[, y]) and per-pair fields in the pair order
# (1,2), (1,3), (2,3), i.e. (z–x[, z–y, x–y]); the y-entries are dropped in 2-D.
per_axis(::Val{2}, z, x, y) = (z, x)
per_axis(::Val{3}, z, x, y) = (z, x, y)
per_pair(::Val{2}, zx, zy, xy) = (zx,)
per_pair(::Val{3}, zx, zy, xy) = (zx, zy, xy)
axis_pairs(::Val{2}) = ((Val(1), Val(2)),)
axis_pairs(::Val{3}) = ((Val(1), Val(2)), (Val(1), Val(3)), (Val(2), Val(3)))

# Array shapes from the interior cell counts N, with e_d the unit vector along
# axis d: compact faces N .+ e_d, ghosted faces N .+ 2 .- e_d (one ghost layer
# along every other axis), edges of the axis pair (a, b) N .+ e_a .+ e_b.
unit_offset(N::NTuple{D, Int}, d) where {D} = ntuple(k -> Int(k == d), Val(D))
face_shape(N, d)       = N .+ unit_offset(N, d)
ghost_face_shape(N, d) = N .+ 2 .- unit_offset(N, d)
edge_shape(N, a, b)    = N .+ unit_offset(N, a) .+ unit_offset(N, b)

# periodicity of the axes (z, x[, y]) from the boundary conditions; yBC is read
# only for D = 3
function periodic_axes(name, ::Val{D}, xBC::Symbol, yBC::Symbol, zBC::Symbol) where {D}
    D in (2, 3) || throw(ArgumentError("$name: need 2-D or 3-D fields, got D = $D"))
    bcs = D == 2 ? (zBC, xBC) : (zBC, xBC, yBC)
    for bc in bcs
        bc in (:periodic, :closed) || throw(ArgumentError("$name: unsupported BC = :$bc"))
    end
    return ntuple(k -> bcs[k] === :periodic, Val(D))
end

# Ghosted face arrays (ghosted-face shape) carry one ghost layer along every axis
# k ≠ d; `face_interior` drops it, `fill_face_ghosts!` fills it as
# advect.m:144-161 and diffus.m:70-81 do: periodic ⇒ wrap, otherwise ⇒ edge repeat.
face_interior(qd::AbstractArray{<:Any,D}, d) where {D} =
    view(qd, ntuple(k -> k == d ? (1:size(qd, k)) : (2:size(qd, k) - 1), Val(D))...)

# fill the ghost layer of `a` along axis k: periodic ⇒ wrap, otherwise ⇒ edge repeat
function fill_ghost_layer!(a::AbstractArray, k::Integer, periodic::Bool)
    n = size(a, k)
    if periodic && n > 3
        selectdim(a, k, 1) .= selectdim(a, k, n - 1)
        selectdim(a, k, n) .= selectdim(a, k, 2)
    else
        selectdim(a, k, 1) .= selectdim(a, k, 2)
        selectdim(a, k, n) .= selectdim(a, k, n - 1)
    end
    return a
end

# ghosted face array of axis d: fill its ghost layers along every other axis
function fill_face_ghosts!(qd::AbstractArray, d::Integer, periodic)
    foreach(k -> k == d || fill_ghost_layer!(qd, k, periodic[k]), eachindex(periodic))
    return qd
end

# ghosted cell array (ghost ring along every axis): fill all its ghost layers
function fill_cell_ghosts!(a::AbstractArray, periodic)
    foreach(k -> fill_ghost_layer!(a, k, periodic[k]), eachindex(periodic))
    return a
end

fill_face_ghosts!(q::Tuple, periodic) = (foreach(d -> fill_face_ghosts!(q[d], d, periodic), eachindex(q)); q)

function check_padded_fluxes(name, q::Tuple, N)
    length(q) == length(N) && all(d -> size(q[d]) == ghost_face_shape(N, d), eachindex(q)) ||
        throw(DimensionMismatch("$name: need flux q[d] of size N .+ 2 .- e_d for N = $N; " *
                                "got $(map(size, q))"))
    return nothing
end


# ----------------------------------------------------------------------------
# Bare first-difference operators (no halo required — just shrink by one in
# the differentiated direction). Faithful to MATLAB's ddx.m/ddz.m, which use
# `diff(a,1,dim)./h` for constant grid spacing.
# ----------------------------------------------------------------------------

@kernel function ddx_kernel!(out, @Const(f), invh)
    iz, ix = @index(Global, NTuple)
    out[iz, ix] = (f[iz, ix + 1] - f[iz, ix]) * invh
end

@kernel function ddz_kernel!(out, @Const(f), invh)
    iz, ix = @index(Global, NTuple)
    out[iz, ix] = (f[iz + 1, ix] - f[iz, ix]) * invh
end

"""
    ddx!(out, f, h) -> out

In-place x-direction finite difference: `out[j,i] = (f[j,i+1] - f[j,i]) / h`.
`size(out) == (size(f,1), size(f,2)-1)`. Backend-agnostic via
`KernelAbstractions.get_backend(out)`.
"""
function ddx!(out::AbstractMatrix, f::AbstractMatrix, h::Real)
    @assert size(out, 1) == size(f, 1)            "ddx!: row count mismatch"
    @assert size(out, 2) == size(f, 2) - 1        "ddx!: output must be one col narrower than input"
    backend = KernelAbstractions.get_backend(out)
    T = eltype(out)
    ddx_kernel!(backend)(out, f, T(inv(h)); ndrange = size(out))
    return out
end

"""
    ddz!(out, f, h) -> out

In-place z-direction finite difference: `out[j,i] = (f[j+1,i] - f[j,i]) / h`.
`size(out) == (size(f,1)-1, size(f,2))`.
"""
function ddz!(out::AbstractMatrix, f::AbstractMatrix, h::Real)
    @assert size(out, 1) == size(f, 1) - 1        "ddz!: output must be one row shorter than input"
    @assert size(out, 2) == size(f, 2)            "ddz!: column count mismatch"
    backend = KernelAbstractions.get_backend(out)
    T = eltype(out)
    ddz_kernel!(backend)(out, f, T(inv(h)); ndrange = size(out))
    return out
end
