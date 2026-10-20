# 3-D fluid + phase state — the dimension lift of `src/state.jl`. Field NAMES are
# kept identical to the 2-D structs (and to the `f.`/`c.` accesses in
# `src/dyrel3D/schur.jl`) so the residual/Schur kernel wrappers operate on these
# real structs unchanged. Array shapes follow the schur.jl header table:
#
#   field   | shape                  | location
#   --------|------------------------|--------------------
#   W       | (Nz+1, Nx+2, Ny+2)     | z-faces (x/y ghosts)
#   U       | (Nz+2, Nx+1, Ny+2)     | x-faces (z/y ghosts)
#   V       | (Nz+2, Nx+2, Ny+1)     | y-faces (z/x ghosts)
#   P       | (Nz+2, Nx+2, Ny+2)     | cells (ghost ring)
#   eta,rho | (Nz, Nx, Ny)           | cells
#   etaxz   | (Nz+1, Nx+1, Ny)       | z-x edges
#   etayz   | (Nz+1, Nx,   Ny+1)     | z-y edges
#   etaxy   | (Nz,   Nx+1, Ny+1)     | x-y edges
#   rhow    | (Nz+1, Nx, Ny)         | z-faces
#   rhou    | (Nz, Nx+1, Ny)         | x-faces
#   rhov    | (Nz, Nx, Ny+1)         | y-faces
#   Drho    | (Nz+1, Nx, Ny)         | z-faces (buoyancy anomaly)
#
# No sparse-LU solver cache (there is no `:direct` 3-D path — GMG only). The
# DYREL scratch lives on a separate `DyrelCache3D` (see cache3d.jl), as in 2-D.

"""
    FluidState3D{T,A}

Fields touched by the 3-D Navier-Stokes / GMG solver. `A<:AbstractArray{T,3}`
so GPU backends drop in unchanged. Shapes per the header table above.
"""
struct FluidState3D{T<:AbstractFloat, A<:AbstractArray{T,3}}
    # primary unknowns
    W::A;  U::A;  V::A;  P::A
    # viscosities (cell + three edge sets)
    eta::A;    etaxz::A;  etayz::A;  etaxy::A
    # densities (current + BD2 history) and face interpolations
    rho::A;    rhoo::A;   rhooo::A
    rhow::A;   rhou::A;   rhov::A;   Drho::A
    # density-evolution rates (current + lagged, for the MFS source)
    drhodt::A;  drhodto::A;  drhodtoo::A
    # source term
    MFS::A
    # momentum-flux history for the inertial RHS (BD2)
    rhoWo::A;   rhoWoo::A
    rhoUo::A;   rhoUoo::A
    rhoVo::A;   rhoVoo::A
end

"""
    FluidState3D(::Type{T}, backend, Nz, Nx, Ny) -> FluidState3D

Allocate a zero-initialised `FluidState3D` on the given KA backend.
"""
function FluidState3D(::Type{T}, backend, Nz::Integer, Nx::Integer, Ny::Integer) where {T<:AbstractFloat}
    z(dims...) = xcore_zeros(backend, T, dims...)
    return FluidState3D{T, typeof(z(1, 1, 1))}(
        z(Nz + 1, Nx + 2, Ny + 2),                         # W
        z(Nz + 2, Nx + 1, Ny + 2),                         # U
        z(Nz + 2, Nx + 2, Ny + 1),                         # V
        z(Nz + 2, Nx + 2, Ny + 2),                         # P
        z(Nz, Nx, Ny),                                     # eta
        z(Nz + 1, Nx + 1, Ny),                             # etaxz
        z(Nz + 1, Nx, Ny + 1),                             # etayz
        z(Nz, Nx + 1, Ny + 1),                             # etaxy
        z(Nz, Nx, Ny), z(Nz, Nx, Ny), z(Nz, Nx, Ny),      # rho, rhoo, rhooo
        z(Nz + 1, Nx, Ny),                                 # rhow
        z(Nz, Nx + 1, Ny),                                 # rhou
        z(Nz, Nx, Ny + 1),                                 # rhov
        z(Nz + 1, Nx, Ny),                                 # Drho
        z(Nz, Nx, Ny), z(Nz, Nx, Ny), z(Nz, Nx, Ny),      # drhodt, drhodto, drhodtoo
        z(Nz, Nx, Ny),                                     # MFS
        z(Nz + 1, Nx, Ny), z(Nz + 1, Nx, Ny),             # rhoWo, rhoWoo
        z(Nz, Nx + 1, Ny), z(Nz, Nx + 1, Ny),             # rhoUo, rhoUoo
        z(Nz, Nx, Ny + 1), z(Nz, Nx, Ny + 1),             # rhoVo, rhoVoo
    )
end

FluidState3D(::Type{T}, backend, grid::Grid3D) where {T<:AbstractFloat} =
    FluidState3D(T, backend, grid.Nz, grid.Nx, grid.Ny)

"""
    PhaseState3D{T,A}

Constitutive workspace consumed/produced by the 3-D `update!` path. Phase-1
scaffold: carries only the fields the GMG driver currently reads (`advn_rho`,
the density-advection rate that feeds the MFS source). It will grow field-for-
field as the 3-D `update!`/`phsevo!` path is ported (Phase 4) — mirror
`PhaseState` in `src/state.jl` for the full set.
"""
struct PhaseState3D{T<:AbstractFloat, A<:AbstractArray{T,3}}
    advn_rho::A
end

function PhaseState3D(::Type{T}, backend, Nz::Integer, Nx::Integer, Ny::Integer) where {T<:AbstractFloat}
    z(dims...) = xcore_zeros(backend, T, dims...)
    return PhaseState3D{T, typeof(z(1, 1, 1))}(z(Nz, Nx, Ny))
end

PhaseState3D(::Type{T}, backend, grid::Grid3D) where {T<:AbstractFloat} =
    PhaseState3D(T, backend, grid.Nz, grid.Nx, grid.Ny)
