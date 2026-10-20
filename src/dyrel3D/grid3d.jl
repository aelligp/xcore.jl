# 3-D staggered grid — the dimension lift of `src/grid.jl::Grid`. Axes follow the
# convention used by the Stokes solvers (see `src/stokes/layout.jl`):
#
#   (iz, ix, iy) = (z, x, y);  z = dim1 (gravity), x = dim2, y = dim3.
#
# z is closed (Dirichlet W faces); the horizontal boundary conditions are chosen
# per solve. Cells are isotropic (single `h`), as in 2-D. Coordinate vectors
# mirror the 2-D `Grid` construction (cell centers at (i-½)h, faces at i·h).
# There are no ghost-index vectors: the solver kernels apply the boundary
# conditions directly.

"""
    Grid3D{T}

Staggered 3-D finite-difference grid: `Nz×Nx×Ny` interior cells, spacing `h`,
domain depth `D` (z-extent) and width `L` (x- and y-extent). Coordinate vectors:
`Xc/Yc/Zc` cell centres, `Xf/Yf/Zf` faces.
"""
struct Grid3D{T<:AbstractFloat}
    Nx::Int
    Ny::Int
    Nz::Int
    h::T
    D::T
    L::T
    Xc::Vector{T}    # length Nx
    Yc::Vector{T}    # length Ny
    Zc::Vector{T}    # length Nz
    Xf::Vector{T}    # length Nx+1
    Yf::Vector{T}    # length Ny+1
    Zf::Vector{T}    # length Nz+1
end

"""
    Grid3D(par::Parameters{T}) -> Grid3D{T}

Build a cubic-cell 3-D grid from `Parameters`: `h = D/N`, `Nz = N`, and
`Nx = Ny = round(L/h)` (square horizontal footprint). Mirrors `Grid(par)`.
"""
function Grid3D(par::Parameters{T}) where {T<:AbstractFloat}
    h  = par.D / par.N
    Nz = par.N
    Nx = max(1, round(Int, par.L / h))
    Ny = Nx
    return _grid3d_at(T, Nz, Nx, Ny, h, par.D, par.L)
end

# Construct a Grid3D at explicit sizes/spacing (used by Grid3D(par) and by the
# multigrid coarsening, which halves dims and doubles h).
function _grid3d_at(::Type{T}, Nz::Int, Nx::Int, Ny::Int, h::T, D::T, L::T) where {T}
    Xc = collect(((1:Nx) .- T(0.5)) .* h)
    Yc = collect(((1:Ny) .- T(0.5)) .* h)
    Zc = collect(((1:Nz) .- T(0.5)) .* h)
    Xf = collect((0:Nx) .* h)
    Yf = collect((0:Ny) .* h)
    Zf = collect((0:Nz) .* h)
    return Grid3D{T}(Nx, Ny, Nz, h, D, L, Xc, Yc, Zc, Xf, Yf, Zf)
end

"""
    coarsen_grid3d(grid) -> Grid3D

Halve `Nz`,`Nx`,`Ny` and double `h`. Errors on an odd coarsenable dimension;
the multigrid hierarchy builder requires multiples of 32.
"""
function coarsen_grid3d(grid::Grid3D{T}) where {T}
    Nz, Nx, Ny = grid.Nz, grid.Nx, grid.Ny
    (iseven(Nz) && (iseven(Nx) || Nx == 1) && (iseven(Ny) || Ny == 1)) ||
        error("multigrid: cannot coarsen grid $(Nz)×$(Nx)×$(Ny) (need even dims)")
    half(n) = n == 1 ? 1 : n ÷ 2
    return _grid3d_at(T, half(Nz), half(Nx), half(Ny), grid.h * 2, grid.D, grid.L)
end
