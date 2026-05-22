"""
    fill_ghosts!(f, halo; zBC=:closed, xBC=:periodic) -> f

Fill the outer `halo` rings of a 2D field according to per-axis boundary
conditions. After this call, stencil kernels can read `f[j±halo, i±halo]`
without any bounds checks or branches.

`f` has size `(Nz + 2halo, Nx + 2halo)`; the interior lives at
`f[halo+1:Nz+halo, halo+1:Nx+halo]`. Supports:

* `:periodic` — wrap-around copy from the opposite interior side.
* `:closed`   — repeat-boundary (zero-gradient Neumann).

Corners are filled correctly by applying x first, then z. Uses
`@views` + broadcasting so KA backends (incl. GPU) dispatch through their
standard kernels — no separate GPU implementation needed for the BC fill.
"""
function fill_ghosts!(f::AbstractMatrix, halo::Integer;
                      zBC::Symbol = :closed, xBC::Symbol = :periodic)
    Nz_t, Nx_t = size(f)
    Nz = Nz_t - 2halo
    Nx = Nx_t - 2halo
    @assert Nz ≥ 1 && Nx ≥ 1   "fill_ghosts!: interior must be non-empty (halo too large?)"

    # ----- x first so the subsequent z fill picks up the correct corner columns
    if xBC === :periodic
        @views f[:, 1:halo]                  .= f[:, (Nx + 1):(Nx + halo)]
        @views f[:, (Nx + halo + 1):Nx_t]    .= f[:, (halo + 1):(2halo)]
    elseif xBC === :closed
        for k in 1:halo
            @views f[:, k]              .= f[:, halo + 1]
            @views f[:, Nx + halo + k]  .= f[:, Nx + halo]
        end
    else
        throw(ArgumentError("fill_ghosts!: unsupported xBC = $(xBC)"))
    end

    if zBC === :periodic
        @views f[1:halo, :]                  .= f[(Nz + 1):(Nz + halo), :]
        @views f[(Nz + halo + 1):Nz_t, :]    .= f[(halo + 1):(2halo), :]
    elseif zBC === :closed
        for k in 1:halo
            @views f[k, :]              .= f[halo + 1, :]
            @views f[Nz + halo + k, :]  .= f[Nz + halo, :]
        end
    else
        throw(ArgumentError("fill_ghosts!: unsupported zBC = $(zBC)"))
    end

    return f
end

"""
    embed_interior!(f_halo, f, halo) -> f_halo

Copy `f` (size `(Nz, Nx)`) into the interior of a halo'd buffer `f_halo`
(size `(Nz + 2halo, Nx + 2halo)`). Convenience for tests and for callers that
keep MATLAB-sized fields as their primary storage and only embed-into-halo
when they need a wide-stencil operator.
"""
function embed_interior!(f_halo::AbstractMatrix, f::AbstractMatrix, halo::Integer)
    @views f_halo[(halo + 1):(end - halo), (halo + 1):(end - halo)] .= f
    return f_halo
end
