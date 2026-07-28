using TTSolver
using LinearAlgebra

"""    local_tt_operator(op, spaces, site) → TTMatrix"""
function local_tt_operator(op::AbstractMatrix, 
                           spaces::AbstractVector{Int}, 
                           site::Int)
    n_sites = length(spaces)
    @assert 1 <= site <= n_sites "site must be in range [1, $n_sites]"
    @assert size(op, 1) == spaces[site] "operator size ($(size(op,1))) must match space dimension ($(spaces[site]))"
    
    ops = [i == site ? ComplexF64.(op) : Matrix{ComplexF64}(I, spaces[i], spaces[i]) 
           for i in 1:n_sites]
    
    return tt_mkron(ops...)
end

"""    local_tt_operator(op, spaces::Vector, site) → TTMatrix"""
function local_tt_operator(op::AbstractMatrix, 
                           spaces::Vector, 
                           site::Int)
    dims = [dim(s) for s in spaces]
    return local_tt_operator(op, dims, site)
end

"""    identity_tt(spaces) → TTMatrix"""
function identity_tt(spaces::AbstractVector{Int})
    ops = [Matrix{ComplexF64}(I, d, d) for d in spaces]
    return tt_mkron(ops...)
end

"""    identity_tt(spaces::Vector) → TTMatrix"""
function identity_tt(spaces::Vector)
    dims = [dim(s) for s in spaces]
    return identity_tt(dims)
end

"""    sum_local_operators(ops, spaces) → TTMatrix"""
function sum_local_operators(ops::Vector{<:AbstractMatrix}, 
                             spaces::AbstractVector{Int})
    n_sites = length(spaces)
    @assert length(ops) == n_sites "number of operators must match number of sites"
    
    result = local_tt_operator(ops[1], spaces, 1)
    for i in 2:n_sites
        result = result + local_tt_operator(ops[i], spaces, i)
    end
    return result
end
