"""
    Grid{T}

Staggered finite-difference grid. Mirrors the coordinate setup at the top of `src/init.m`.

Layout (cell-centred + face-staggered):

```
Zf[j+1] |---- W ----|----  ----|
        |           |          |
Zc[j]   U   P,x,m   U          |   <- P at cell centre, U on x-faces, W on z-faces
        |           |          |
Zf[j]   |---- W ----|----  ----|
        Xf[i]  Xc[i]  Xf[i+1]
```

`Xc`, `Zc` are cell centres (length `Nx`, `Nz`); `Xf`, `Zf` are faces
(length `Nx+1`, `Nz+1`). `XX`/`ZZ` are the 2D meshgrids of cell-centred coordinates,
matching MATLAB convention `[XX,ZZ] = meshgrid(Xc,Zc)` so the leading axis is z.

Ghost-index vectors `icx`, `icz`, `ifx`, `ifz` reproduce MATLAB's wrap-around indexing
(`icx = [Nx, 1:Nx, 1]` etc.) used throughout `update.m` for periodic-x / closed-z BCs.
"""
struct Grid{T<:AbstractFloat}
    Nx::Int
    Nz::Int
    h::T
    D::T
    L::T

    Xc::Vector{T}    # cell-centre x coordinates, length Nx
    Zc::Vector{T}    # cell-centre z coordinates, length Nz
    Xf::Vector{T}    # face x coordinates, length Nx+1
    Zf::Vector{T}    # face z coordinates, length Nz+1
    XX::Matrix{T}    # Nz x Nx meshgrid of Xc
    ZZ::Matrix{T}    # Nz x Nx meshgrid of Zc

    # ghosted index vectors (periodic-x, closed-z by default; reproduce MATLAB indexing)
    icx::Vector{Int}
    icz::Vector{Int}
    ifx::Vector{Int}
    ifz::Vector{Int}
end

"""
    Grid(par::Parameters{T}) -> Grid{T}

Build a grid from a `Parameters` instance. `h = D / N` (square cells), with `L`
controlling the number of x-cells via `Nx = round(Int, L / h)`. For `L = h` this
collapses to a single column (1-D mode).
"""
function Grid(par::Parameters{T}) where {T<:AbstractFloat}
    h  = par.D / par.N
    Nz = par.N
    Nx = max(1, round(Int, par.L / h))

    # MATLAB: Xc = -h/2:h:L+h/2; then Xc(2:end-1).
    # Net effect: Nx interior centres at h/2, 3h/2, ..., (Nx-1/2)h.
    Xc = collect(((1:Nx) .- T(0.5)) .* h)
    Zc = collect(((1:Nz) .- T(0.5)) .* h)
    Xf = collect((0:Nx) .* h)
    Zf = collect((0:Nz) .* h)

    # meshgrid: leading axis z, trailing axis x (matches MATLAB [XX,ZZ]=meshgrid(Xc,Zc))
    XX = repeat(reshape(Xc, 1, Nx), Nz, 1)
    ZZ = repeat(reshape(Zc, Nz, 1), 1, Nx)

    icx = [Nx; collect(1:Nx); 1]
    icz = [1;  collect(1:Nz); Nz]
    ifx = [Nx; collect(1:Nx+1); 2]
    ifz = [2;  collect(1:Nz+1); Nz]

    return Grid{T}(Nx, Nz, h, par.D, par.L, Xc, Zc, Xf, Zf, XX, ZZ,
                   icx, icz, ifx, ifz)
end
