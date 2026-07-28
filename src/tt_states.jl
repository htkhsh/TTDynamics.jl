using TTSolver

"""
    localized_tt_state(dims; state_indices=1, sites=1) → TTTensor

Create a localized (product) TT state with specified state indices at given sites.
All other sites are initialized to ground state (index 1).

# Arguments
- `dims`: Vector of dimensions for each site
- `state_indices`: State index (Int) or indices (Vector{Int}) for the specified sites
- `sites`: Site index (Int) or indices (Vector{Int}) where to place the states (1-indexed)

# Examples
```julia
# Single site: |↑, 0, 0, ...⟩
psi = localized_tt_state(dims; state_indices=1, sites=1)

# Multiple sites: |↑, 2, 0, ...⟩ (spin up, boson in state 2)
psi = localized_tt_state(dims; state_indices=[1, 2], sites=[1, 2])
```
"""
function localized_tt_state(dims::Vector{Int}; 
                            state_indices::Union{Int, Vector{Int}}=1, 
                            sites::Union{Int, Vector{Int}}=1)
    d = length(dims)
    j = ones(Int, d)  # Default: all ground states (index 1)
    
    # Convert to vectors if single values
    site_vec = sites isa Int ? [sites] : collect(sites)
    idx_vec = state_indices isa Int ? [state_indices] : collect(state_indices)
    
    @assert length(site_vec) == length(idx_vec) "sites and state_indices must have same length"
    
    for (site, idx) in zip(site_vec, idx_vec)
        @assert 1 <= site <= d "site $site out of range [1, $d]"
        @assert 1 <= idx <= dims[site] "state_index $idx out of range [1, $(dims[site])] for site $site"
        j[site] = idx
    end
    
    return tt_unit(dims, j)
end

"""
    localized_tt_state(spaces; state_indices=1, sites=1) → TTTensor

Create a localized TT state using PhysicalSpace objects.
"""
function localized_tt_state(spaces::Vector; 
                            state_indices::Union{Int, Vector{Int}}=1, 
                            sites::Union{Int, Vector{Int}}=1,
                            # Backward compatibility
                            state_index::Union{Int, Nothing}=nothing,
                            site::Union{Int, Nothing}=nothing)
    # Handle backward compatibility
    if !isnothing(state_index)
        state_indices = state_index
    end
    if !isnothing(site)
        sites = site
    end
    dims = [dim(s) for s in spaces]
    return localized_tt_state(dims; state_indices=state_indices, sites=sites)
end

# Backward compatibility alias
const initial_tt_state = localized_tt_state

"""    zero_tt_state(dims; rk=1) → TTTensor"""
function zero_tt_state(dims::Vector{Int}; rk::Union{Int, Vector{Int}}=1)
    d = length(dims)
    if rk isa Int
        ranks = vcat([1], fill(rk, d-1), [1])
    else
        ranks = rk
    end
    
    cores = Vector{Array{ComplexF64, 3}}(undef, d)
    for k in 1:d
        cores[k] = zeros(ComplexF64, ranks[k], dims[k], ranks[k+1])
    end
    return TTTensor(cores)
end

"""    product_tt_state(states) → TTTensor"""
function product_tt_state(states::Vector{Vector{ComplexF64}})
    d = length(states)
    cores = Vector{Array{ComplexF64, 3}}(undef, d)
    for k in 1:d
        n = length(states[k])
        cores[k] = reshape(states[k], (1, n, 1))
    end
    return TTTensor(cores)
end

"""    product_tt_state(states::Vector{<:AbstractVector}) → TTTensor"""
function product_tt_state(states::Vector{<:AbstractVector})
    complex_states = [Vector{ComplexF64}(s) for s in states]
    return product_tt_state(complex_states)
end

# =============================================
# Density Matrix States
# =============================================

"""
    localized_dm_state(dim::Int, state::Int) → Vector{ComplexF64}

Create a vectorized pure-state density matrix |state⟩⟨state| for a single subsystem.
Returns vec(ρ) where ρ[state, state] = 1, others = 0.
"""
function localized_dm_state(dim::Int, state::Int)
    @assert 1 <= state <= dim "state $state out of range [1, $dim]"
    rho = zeros(ComplexF64, dim, dim)
    rho[state, state] = 1.0
    return vec(rho)
end

"""
    dm_state_from_matrix(rho::AbstractMatrix) → Vector{ComplexF64}

Vectorize a density matrix.
"""
function dm_state_from_matrix(rho::AbstractMatrix)
    return vec(ComplexF64.(rho))
end

"""
    tt_svd(v::AbstractVector, dims::Vector{Int}; tol=1e-12, rmax=nothing) → TTTensor

Convert a full vector to TT format using SVD-based decomposition.

# Arguments
- `v`: Full vector of length prod(dims)
- `dims`: Dimensions of each TT core
- `tol`: SVD truncation tolerance (relative to largest singular value)
- `rmax`: Maximum rank (nothing for no limit)

# Returns
- TTTensor with specified dimensions
"""
function tt_svd(v::AbstractVector{T}, dims::Vector{Int}; 
                tol::Float64=1e-12, rmax::Union{Nothing, Int}=nothing) where T
    @assert length(v) == prod(dims) "Vector length must equal prod(dims)"
    
    d = length(dims)
    if d == 1
        # Single core case
        cores = [reshape(ComplexF64.(v), (1, dims[1], 1))]
        return TTTensor(cores)
    end
    
    cores = Vector{Array{ComplexF64, 3}}(undef, d)
    
    # Start with the full vector as a matrix (1 × prod(dims))
    C = reshape(ComplexF64.(v), (1, prod(dims)))
    
    r_prev = 1
    for k in 1:d-1
        n_k = dims[k]
        n_rest = prod(dims[k+1:end])
        
        # Reshape C to matrix: (r_prev * n_k) × n_rest
        C = reshape(C, (r_prev * n_k, n_rest))
        
        # SVD: Julia's svd returns F where F.U, F.S, F.V (not Vt!)
        # Use F.Vt to get the transposed V matrix
        F = svd(C)
        U, S, Vt = F.U, F.S, F.Vt
        
        # Truncate based on tolerance
        s_max = S[1]  # S is sorted in descending order
        if s_max > 0
            r_k = count(s -> s / s_max > tol, S)
            r_k = max(r_k, 1)  # At least rank 1
        else
            r_k = 1
        end
        
        # Apply rmax if specified
        if !isnothing(rmax)
            r_k = min(r_k, rmax)
        end
        
        # Core k: reshape U[:, 1:r_k] to (r_prev, n_k, r_k)
        cores[k] = reshape(U[:, 1:r_k], (r_prev, n_k, r_k))
        
        # Prepare for next iteration: C = diag(S[1:r_k]) * Vt[1:r_k, :]
        C = Diagonal(S[1:r_k]) * Vt[1:r_k, :]
        
        r_prev = r_k
    end
    
    # Last core: C is (r_prev × dims[d])
    cores[d] = reshape(C, (r_prev, dims[d], 1))
    
    return TTTensor(cores)
end

"""
    dm_to_tt(rho::AbstractMatrix, sys_dims::Vector{Int}; tol=1e-12, rmax=nothing) → TTTensor

Convert a (possibly mixed) density matrix to TT format using SVD decomposition.

The density matrix ρ for a multi-partite system with subsystem dimensions `sys_dims`
is vectorized and then decomposed into TT format. Each core has dimension n² where n
is the subsystem dimension.

# Arguments
- `rho`: Density matrix of dimension (prod(sys_dims), prod(sys_dims))
- `sys_dims`: Vector of subsystem dimensions
- `tol`: SVD truncation tolerance
- `rmax`: Maximum TT rank

# Example
```julia
# Mixed state of two qubits
rho = 0.5 * [1 0 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 1]  # (|00⟩⟨00| + |11⟩⟨11|)/2
psi_tt = dm_to_tt(rho, [2, 2])
```
"""
function dm_to_tt(rho::AbstractMatrix, sys_dims::Vector{Int}; 
                  tol::Float64=1e-12, rmax::Union{Nothing, Int}=nothing)
    n_total = prod(sys_dims)
    @assert size(rho) == (n_total, n_total) "Density matrix size must match prod(sys_dims)"
    
    # Vectorize density matrix
    rho_vec = vec(ComplexF64.(rho))
    
    # TT core dimensions: each n → n²
    tt_dims = sys_dims .^ 2
    
    return tt_svd(rho_vec, tt_dims; tol=tol, rmax=rmax)
end

"""
    product_dm_tt_state(rhos::Vector{<:AbstractMatrix}; tol=1e-12) → TTTensor

Create a TT state from a tensor product of density matrices.
Each density matrix ρᵢ (nᵢ × nᵢ) is vectorized to a vector of length nᵢ².

For HEOM-type problems where the system density matrix spans multiple TT cores,
this function creates |ρ₁⟩ ⊗ |ρ₂⟩ ⊗ ... where |ρᵢ⟩ = vec(ρᵢ).

# Arguments
- `rhos`: Vector of density matrices

# Example
```julia
# Two-spin system: ρ = |↑⟩⟨↑| ⊗ |↓⟩⟨↓|
rho1 = [1.0 0; 0 0]   # |↑⟩⟨↑|
rho2 = [0 0; 0 1.0]   # |↓⟩⟨↓|
psi_tt = product_dm_tt_state([rho1, rho2])
```
"""
function product_dm_tt_state(rhos::Vector{<:AbstractMatrix}; tol::Float64=1e-12)
    d = length(rhos)
    cores = Vector{Array{ComplexF64, 3}}(undef, d)
    for k in 1:d
        rho_vec = dm_state_from_matrix(rhos[k])
        n = length(rho_vec)
        cores[k] = reshape(rho_vec, (1, n, 1))
    end
    return TTTensor(cores)
end

"""
    localized_dm_tt_state(sys_dims::Vector{Int}; state_indices=1, sites=nothing, tol=1e-12) → TTTensor

Create a localized TT density matrix state |i⟩⟨i| for specified subsystems.
Each subsystem with dimension n contributes a TT core of dimension n².

# Arguments
- `sys_dims`: Vector of subsystem dimensions (each subsystem is n → n² in TT core)
- `state_indices`: State index (Int) or indices (Vector{Int}) for the specified sites
- `sites`: Site indices to set (default: all sites). If not specified, state_indices should have same length as sys_dims

# Examples
```julia
# Single 2-level system in state |1⟩⟨1|
psi = localized_dm_tt_state([2]; state_indices=1)

# Two 2-level systems: |↑⟩⟨↑| ⊗ |↓⟩⟨↓|
psi = localized_dm_tt_state([2, 2]; state_indices=[1, 2])

# Three systems, only set first two: |1⟩⟨1| ⊗ |2⟩⟨2| ⊗ |1⟩⟨1|
psi = localized_dm_tt_state([2, 3, 2]; state_indices=[1, 2], sites=[1, 2])
```
"""
function localized_dm_tt_state(sys_dims::Vector{Int}; 
                               state_indices::Union{Int, Vector{Int}}=1,
                               sites::Union{Nothing, Int, Vector{Int}}=nothing,
                               tol::Float64=1e-12)
    n_sys = length(sys_dims)
    
    # Convert to vectors
    idx_vec = state_indices isa Int ? [state_indices] : collect(state_indices)
    
    if isnothing(sites)
        # Default: state_indices applies to sites 1, 2, ..., length(idx_vec)
        site_vec = collect(1:length(idx_vec))
    else
        site_vec = sites isa Int ? [sites] : collect(sites)
    end
    
    @assert length(site_vec) == length(idx_vec) "sites and state_indices must have same length"
    
    # Build density matrices
    rhos = Vector{Matrix{ComplexF64}}(undef, n_sys)
    for k in 1:n_sys
        dim_k = sys_dims[k]
        rhos[k] = zeros(ComplexF64, dim_k, dim_k)
        rhos[k][1, 1] = 1.0  # Default: ground state
    end
    
    # Set specified states
    for (site, idx) in zip(site_vec, idx_vec)
        @assert 1 <= site <= n_sys "site $site out of range [1, $n_sys]"
        dim_s = sys_dims[site]
        @assert 1 <= idx <= dim_s "state_index $idx out of range [1, $dim_s] for site $site"
        rhos[site] = zeros(ComplexF64, dim_s, dim_s)
        rhos[site][idx, idx] = 1.0
    end
    
    return product_dm_tt_state(rhos; tol=tol)
end

"""
    dm_tt_state(rhos::Vector{<:AbstractMatrix}, aux_dims::Vector{Int}; tol=1e-12) → TTTensor

Create a TT state for HEOM-type problems: system density matrices ⊗ auxiliary (bath) modes.

The system part consists of vectorized density matrices, and the auxiliary modes
are initialized to ground state (index 1).

# Arguments
- `rhos`: Vector of system density matrices
- `aux_dims`: Dimensions of auxiliary (bath) modes, initialized to |0⟩

# Example
```julia
# Single spin coupled to 3 bath modes (each with Nh=10 levels)
rho_sys = [1.0 0; 0 0]  # |↑⟩⟨↑|
psi = dm_tt_state([rho_sys], [10, 10, 10])
```
"""
function dm_tt_state(rhos::Vector{<:AbstractMatrix}, aux_dims::Vector{Int}; tol::Float64=1e-12)
    # System cores from density matrices
    n_sys = length(rhos)
    n_aux = length(aux_dims)
    d = n_sys + n_aux
    
    cores = Vector{Array{ComplexF64, 3}}(undef, d)
    
    # System density matrix cores
    for k in 1:n_sys
        rho_vec = dm_state_from_matrix(rhos[k])
        n = length(rho_vec)
        cores[k] = reshape(rho_vec, (1, n, 1))
    end
    
    # Auxiliary (bath) cores - initialized to ground state |0⟩
    for k in 1:n_aux
        aux_vec = zeros(ComplexF64, aux_dims[k])
        aux_vec[1] = 1.0
        cores[n_sys + k] = reshape(aux_vec, (1, aux_dims[k], 1))
    end
    
    return TTTensor(cores)
end

"""
    dm_tt_state(sys_dims::Vector{Int}, state_indices::Union{Int, Vector{Int}}, 
                aux_dims::Vector{Int}; sites=nothing, tol=1e-12) → TTTensor

Create a localized TT state for HEOM-type problems with pure-state density matrices.

# Arguments
- `sys_dims`: Dimensions of system subsystems
- `state_indices`: State indices for localized states |i⟩⟨i|
- `aux_dims`: Dimensions of auxiliary (bath) modes
- `sites`: Which system sites to set (default: 1:length(state_indices))

# Example
```julia
# Two spins (both dim=2) coupled to 3 bath modes (Nh=10)
# Initial: |↑⟩⟨↑| ⊗ |↓⟩⟨↓| ⊗ |0,0,0⟩
psi = dm_tt_state([2, 2], [1, 2], [10, 10, 10])
```
"""
function dm_tt_state(sys_dims::Vector{Int}, 
                     state_indices::Union{Int, Vector{Int}},
                     aux_dims::Vector{Int};
                     sites::Union{Nothing, Int, Vector{Int}}=nothing,
                     tol::Float64=1e-12)
    n_sys = length(sys_dims)
    
    # Convert to vectors
    idx_vec = state_indices isa Int ? [state_indices] : collect(state_indices)
    
    if isnothing(sites)
        site_vec = collect(1:length(idx_vec))
    else
        site_vec = sites isa Int ? [sites] : collect(sites)
    end
    
    @assert length(site_vec) == length(idx_vec) "sites and state_indices must have same length"
    
    # Build density matrices
    rhos = Vector{Matrix{ComplexF64}}(undef, n_sys)
    for k in 1:n_sys
        dim_k = sys_dims[k]
        rhos[k] = zeros(ComplexF64, dim_k, dim_k)
        rhos[k][1, 1] = 1.0  # Default: ground state
    end
    
    # Set specified states
    for (site, idx) in zip(site_vec, idx_vec)
        @assert 1 <= site <= n_sys "site $site out of range [1, $n_sys]"
        dim_s = sys_dims[site]
        @assert 1 <= idx <= dim_s "state_index $idx out of range [1, $dim_s] for site $site"
        rhos[site] = zeros(ComplexF64, dim_s, dim_s)
        rhos[site][idx, idx] = 1.0
    end
    
    return dm_tt_state(rhos, aux_dims; tol=tol)
end

"""    lcbs_tt_state(dims, coeffs; site=1, tol=1e-12) → TTTensor"""
function lcbs_tt_state(dims::Vector{Int}, coeffs::Vector{<:Number}; site::Int=1, tol::Float64=1e-12)
    psi = ComplexF64(coeffs[1]) * localized_tt_state(dims; state_indices=1, sites=site)
    for k in 2:length(coeffs)
        psi = psi + ComplexF64(coeffs[k]) * localized_tt_state(dims; state_indices=k, sites=site)
        psi = tt_round(psi, tol)
    end
    
    # Normalize
    norm_psi = tt_norm(psi)
    if norm_psi > 1e-15
        psi = (1.0 / norm_psi) * psi
    end
    
    return psi
end

"""
    tt_partial_trace(psi::TTTensor, keep_sites, dims::Vector{Int}) → Matrix{ComplexF64}

Compute the reduced density matrix by tracing out all sites except `keep_sites`.

For a pure state |ψ⟩ in a tensor product space, computes ρ_sys = Tr_env(|ψ⟩⟨ψ|).

# Arguments
- `psi`: TT tensor representing the pure state
- `keep_sites`: Site index (Int) or indices (Vector{Int}) to keep (1-indexed)
- `dims`: Vector of dimensions for each site

# Returns
- Reduced density matrix of dimension (prod(dims[keep_sites]), prod(dims[keep_sites]))
"""
function tt_partial_trace(psi::TTTensor, keep_sites::Union{Int, Vector{Int}}, dims::Vector{Int})
    psi_full = vec(tt_full(psi))
    return tt_partial_trace(psi_full, keep_sites, dims)
end

"""
    tt_partial_trace(psi::Vector, keep_sites, dims::Vector{Int}) → Matrix{ComplexF64}

Compute the reduced density matrix from a full state vector.

# Arguments
- `psi`: Full state vector
- `keep_sites`: Site index (Int) or indices (Vector{Int}) to keep (1-indexed)
- `dims`: Vector of dimensions for each site
"""
function tt_partial_trace(psi::AbstractVector{<:Number}, keep_sites::Union{Int, Vector{Int}}, dims::Vector{Int})
    d = length(dims)
    
    # Convert single site to vector
    keep = keep_sites isa Int ? [keep_sites] : collect(keep_sites)
    trace_sites = setdiff(1:d, keep)
    
    n_sys = prod(dims[keep])
    n_env = prod(dims[trace_sites])
    
    # TTSolver uses reversed dimension order for linear indexing
    # reshape with reverse(dims), then permute back
    psi_tensor = reshape(psi, Tuple(reverse(dims)))
    
    # Permute to original dimension order: reverse the axes
    psi_tensor = permutedims(psi_tensor, reverse(1:d))
    
    # Now psi_tensor[i1, i2, ..., id] corresponds to |i1, i2, ..., id⟩
    # Permute to bring keep_sites first
    perm = vcat(keep, trace_sites)
    psi_perm = permutedims(psi_tensor, perm)
    
    # Reshape to (n_sys, n_env)
    psi_mat = reshape(psi_perm, (n_sys, n_env))
    
    # Compute reduced density matrix: ρ = ψ * ψ†
    ρ = psi_mat * psi_mat'
    
    return Matrix{ComplexF64}(ρ)
end

"""
    system_populations(psi::TTTensor, keep_sites, dims::Vector{Int}) → Vector{Float64}

Compute the diagonal elements (populations) of the reduced density matrix.

# Arguments
- `psi`: TT tensor representing the pure state
- `keep_sites`: Site index (Int) or indices (Vector{Int}) to keep (1-indexed)
- `dims`: Vector of dimensions for each site

# Returns
- Vector of populations for each system state
"""
function system_populations(psi::TTTensor, keep_sites::Union{Int, Vector{Int}}, dims::Vector{Int})
    ρ = tt_partial_trace(psi, keep_sites, dims)
    return real.(diag(ρ))
end

"""
    system_populations(psi::Vector, keep_sites, dims::Vector{Int}) → Vector{Float64}

Compute populations from a full state vector.
"""
function system_populations(psi::AbstractVector{<:Number}, keep_sites::Union{Int, Vector{Int}}, dims::Vector{Int})
    ρ = tt_partial_trace(psi, keep_sites, dims)
    return real.(diag(ρ))
end

"""
    system_populations(psi_vec::Vector{<:AbstractVector}, keep_sites, dims::Vector{Int}) → Matrix{Float64}

Compute populations for a time series of state vectors.

# Arguments
- `psi_vec`: Vector of state vectors at each time step
- `keep_sites`: Site index (Int) or indices (Vector{Int}) to keep (1-indexed)
- `dims`: Vector of dimensions for each site

# Returns
- Matrix of size (n_sys, n_times) where each column is populations at one time step
"""
function system_populations(psi_vec::Vector{<:AbstractVector}, keep_sites::Union{Int, Vector{Int}}, dims::Vector{Int})
    keep = keep_sites isa Int ? [keep_sites] : collect(keep_sites)
    n_times = length(psi_vec)
    n_sys = prod(dims[keep])
    pops = zeros(Float64, n_sys, n_times)
    for (i, psi) in enumerate(psi_vec)
        pops[:, i] = system_populations(psi, keep_sites, dims)
    end
    return pops
end

