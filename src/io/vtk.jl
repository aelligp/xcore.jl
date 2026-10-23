# ParaView output, following JustRelax.DataIO: one rectilinear `.vtr` file per
# frame on the vertex grid, cell fields as cell data, vertex fields and the
# velocity as point data, optionally collected in a `.pvd` time series.
#
# xcore arrays are ordered (z, x[, y]) with z = depth, increasing downward.
# VTK arrays are ordered (x[, y], elevation) with elevation = -z, so the top
# boundary sits at 0 and up is up in ParaView.

# Host array holding the values of `x`; a host `Array` is returned as is.
host(x) = x isa Array ? x : Array(x)

"""
    pack_velocity(velocity::Tuple, precision)

Pack the velocity components into the `(3, size...)` array expected for a VTK
vector attribute. Readers copy three values per tuple regardless of the
declared `NumberOfComponents`, so components beyond `length(velocity)` are
written as zeros.
"""
function pack_velocity(velocity::Tuple, precision)
    length(velocity) ≤ 3 ||
        throw(ArgumentError("velocity must have at most 3 components, got $(length(velocity))"))
    for v in velocity
        axes(v) == axes(first(velocity)) ||
            throw(DimensionMismatch("velocity components must share their axes: $(axes(v)) vs $(axes(first(velocity)))"))
    end
    velocity_field = zeros(precision, 3, size(first(velocity))...)
    for (i, v) in enumerate(velocity)
        selectdim(velocity_field, 1, i) .= host(v)
    end
    return velocity_field
end

"""
    vtk_field(name, array, npoints, ncells, precision) -> (name, data, location)

Host copy of `array` with the VTK location, point data or cell data, whose
size it matches.
A vertex grid holds one cell fewer than it has nodes per dimension, so the two
sizes are always distinguishable.
"""
function vtk_field(name, array, npoints, ncells, precision)
    data = convert(Array{precision}, host(array))
    location = if size(data) == npoints
        VTKPointData()
    elseif size(data) == ncells
        VTKCellData()
    else
        throw(DimensionMismatch(
            "$name has size $(size(data)), which matches neither the $npoints vertices nor the $ncells cells of the grid"))
    end
    return string(name), data, location
end

"""
    save_vtk(fname, xvi, xci, data_v::NamedTuple, data_c::NamedTuple, velocity::Tuple;
             precision = Float32, t = 0, pvd = nothing)

Write the rectilinear VTK file `fname` (`.vtr` is appended) on the grid
spanned by the vertex coordinates `xvi`.

- `xci`: cell-center coordinates, one entry fewer per dimension than `xvi`.
- `data_v`, `data_c`: fields written as point or cell data according to their
  size, so a cell field passed in `data_v` still lands on the cells.
- `velocity`: components on the vertices, written as the point vector `Velocity`.
- `t`: written as the field data `TimeValue`.
- `pvd`: if given, the file is added at time `t` to the ParaView collection
  `pvd.pvd`, which is created or appended to.

Device arrays are copied to the host.
"""
function save_vtk(fname::AbstractString, xvi, xci, data_v::NamedTuple, data_c::NamedTuple,
                  velocity::Tuple; precision = Float32, t::Number = 0,
                  pvd::Union{Nothing, AbstractString} = nothing)
    length.(xvi) == length.(xci) .+ 1 || throw(DimensionMismatch(
        "the vertex grid must have one node more per dimension than the center grid: $(length.(xvi)) vs $(length.(xci))"))
    size(first(velocity)) == length.(xvi) || throw(DimensionMismatch(
        "velocity must be given on the vertices: $(size(first(velocity))) vs $(length.(xvi))"))

    # Validate every field before the file is opened, so a bad field leaves no file.
    fields = [vtk_field(name, array, length.(xvi), length.(xci), precision)
              for (name, array) in Iterators.flatten((pairs(data_v), pairs(data_c)))]
    velocity_field = pack_velocity(velocity, precision)
    vtk_grid(fname, xvi...) do vtk
        for (name, data, location) in fields
            vtk[name, location] = data
        end
        vtk["Velocity", VTKPointData()] = velocity_field
        vtk["TimeValue"] = t
        if pvd !== nothing
            paraview_collection(pvd; append = true) do collection
                collection_add_timestep(collection, vtk, t)
            end
        end
    end
    return nothing
end

"""
    vtk_order(A) -> Array

Host copy of the (z, x[, y]) array `A`, reordered to (x[, y], elevation).
"""
vtk_order(A::AbstractArray{<:Any, D}) where {D} =
    reverse!(permutedims(host(A), (2:D..., 1)); dims = D)

"""
    vtk_coordinates(grid) -> (xvi, xci)

Vertex and cell-center coordinates of `grid` along the VTK axes
(x[, y], elevation).
"""
function vtk_coordinates(grid::Grid)
    to_vtk(x) = (Base.tail(x)..., -reverse(first(x)))
    return to_vtk(grid.xvi), to_vtk(grid.xci)
end

"""
    face_to_vertex(q, d, N) -> AbstractArray

Average the ghosted `d`-face array `q` of a grid with `N` cells onto the cell
vertices by taking the mean of neighboring faces along every other axis; the
result has size `N .+ 1`. Ghost values follow the boundary conditions, so the
boundary vertices see periodic or closed neighbors as the solver does.
"""
function face_to_vertex(q::AbstractArray{<:Any, D}, d, N::NTuple{D, Int}) where {D}
    Base.require_one_based_indexing(q)
    size(q) == ghost_face_shape(N, d) ||
        throw(DimensionMismatch("face array along axis $d has size $(size(q)), expected $(ghost_face_shape(N, d))"))
    return foldl(filter(!=(d), ntuple(identity, D)); init = q) do v, e
        n = size(v, e)
        (selectdim(v, e, 1:n-1) .+ selectdim(v, e, 2:n)) ./ 2
    end
end

"""
    save_vtk(fname, phase, fluid, ns, grid; t = 0, pvd = nothing)

Write one ParaView frame of the model state on `grid` (2-D or 3-D) with the
fields of the `save_output` figures: the mixture velocity as `Velocity`, the
segregation speeds `wx`, `wm` on the vertices, and as cell data `P`, `rho`,
`eta`, `MFS`, crystallinity `x`, `etas`, the diffusivities `ks`, `kx`, `ke`,
the noise speeds `xie`, `xix`, `xis` and the dimensionless numbers `ReD`, `Ra`,
`Rc`. Vertical components are positive upward (`-W`, `-wx`, `-wm`).
"""
function save_vtk(fname::AbstractString, phase::PhaseState, fluid::FluidState,
                  ns::NoiseState, grid::Grid; t::Number = 0,
                  pvd::Union{Nothing, AbstractString} = nothing)
    N = cells(grid)
    vertex(q, d) = vtk_order(face_to_vertex(q, d, N))
    w, horizontal... = map(vertex, velocity(fluid), ntuple(identity, length(N)))
    data_v = (; wx = -vertex(phase.wx, 1), wm = -vertex(phase.wm, 1))
    data_c = map(vtk_order, (; P = fluid.P[ntuple(d -> 2:N[d]+1, length(N))...],
        fluid.rho, fluid.eta, fluid.MFS, phase.x, phase.etas, phase.ks, phase.kx, phase.ke,
        ns.xie, ns.xix, ns.xis, phase.ReD, phase.Ra, phase.Rc))
    xvi, xci = vtk_coordinates(grid)
    return save_vtk(fname, xvi, xci, data_v, data_c, (horizontal..., -w); t, pvd)
end
