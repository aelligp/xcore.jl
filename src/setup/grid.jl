"""
    Grid{T,D}

Staggered finite-difference grid in `D = 2` or `3` dimensions with square
cells of size `h`, stored per axis like JustRelax's `Geometry`. Mirrors the
coordinate setup at the top of `init.m`.

Every per-axis field is a `D`-tuple ordered `(z, x[, y])`: z points down
(gravity), x and y are horizontal.

- `ni`: interior cell counts `(Nz, Nx[, Ny])`.
- `li`: domain lengths `(depth, width[, width])`.
- `xci`: cell-center coordinates, `length(xci[d]) == ni[d]`.
- `xvi`: cell-face (vertex) coordinates, `length(xvi[d]) == ni[d] + 1`.

Layout (cell-centered + face-staggered), in the z–x plane:

```
xvi[1][j+1] |---- W ----|----  ----|
            |           |          |
xci[1][j]   U   P,x,m   U          |   <- P at cell center, U on x-faces, W on z-faces
            |           |          |
xvi[1][j]   |---- W ----|----  ----|
         xvi[2][i] xci[2][i] xvi[2][i+1]
```
"""
struct Grid{T<:AbstractFloat, D}
    ni::NTuple{D, Int}
    li::NTuple{D, T}
    h::T
    xci::NTuple{D, Vector{T}}
    xvi::NTuple{D, Vector{T}}
end

"""
    Grid(ni::NTuple{D,Integer}, li::NTuple{D,Real}) -> Grid{T,D}

Grid with `ni` cells spanning the lengths `li`, both ordered `(z, x[, y])`.
The dimension is `length(ni)`. The cells must be square: `li[d] / ni[d]` is
the same spacing `h` along every axis.
"""
function Grid(ni::NTuple{D, Integer}, li::NTuple{D, Real}) where {D}
    D in (2, 3) || throw(ArgumentError("Grid: need 2 or 3 dimensions, got $D"))
    all(>(0), ni) || throw(ArgumentError("Grid: cell counts must be positive, got $ni"))
    T  = float(promote_type(map(typeof, li)...))
    li = map(T, li)
    h  = li[1] / ni[1]
    all(d -> li[d] / ni[d] ≈ h, 2:D) ||
        throw(ArgumentError("Grid: cells must be square, but li ./ ni = $(li ./ ni)"))
    xci = map(n -> collect(((1:n) .- T(0.5)) .* h), Tuple(ni))
    xvi = map(n -> collect((0:n) .* h), Tuple(ni))
    return Grid{T, D}(Tuple(ni), li, h, xci, xvi)
end

"""
    Grid(par::Parameters, Val(D) = Val(2)) -> Grid{T,D}

Grid of a `Parameters` instance: `h = par.D / par.N`, `par.N` cells in depth
and `round(Int, par.L / h)` cells across the width `par.L` of each horizontal
axis. For `par.L = h` this collapses to a single column (1-D mode).
"""
function Grid(par::Parameters{T}, ::Val{D} = Val(2)) where {T<:AbstractFloat, D}
    Nh = max(1, round(Int, par.L / (par.D / par.N)))
    return Grid((par.N, ntuple(_ -> Nh, D - 1)...), (par.D, ntuple(_ -> par.L, D - 1)...))
end

"""
    cells(grid) -> NTuple{D,Int}

Interior cell counts in axis order: `(Nz, Nx)` or `(Nz, Nx, Ny)`.
"""
cells(g::Grid) = g.ni
