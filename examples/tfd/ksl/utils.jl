using LinearAlgebra
using TTSolver

function _capped_product(values, cap::Int)
    result = 1
    for raw_value in values
        raw_value > 0 ||
            throw(ArgumentError("physical dimensions must be positive"))
        raw_value >= cap && return cap
        value = Int(raw_value)
        result > div(cap, value) && return cap
        result *= value
        result >= cap && return cap
    end
    return result
end

function admissible_tt_ranks(
    dims::AbstractVector{<:Integer},
    rmax::Integer=30,
)
    isempty(dims) && throw(ArgumentError("dims must not be empty"))
    rmax > 0 || throw(ArgumentError("rmax must be positive"))
    all(>(0), dims) || throw(ArgumentError("physical dimensions must be positive"))

    cap = Int(rmax)
    ranks = ones(Int, length(dims) + 1)
    for i in 1:(length(dims) - 1)
        left = _capped_product(@view(dims[1:i]), cap)
        right = _capped_product(@view(dims[(i + 1):end]), cap)
        ranks[i + 1] = min(cap, left, right)
    end
    return ranks
end

function pad_tt_ranks(
    x::TTTensor,
    ranks::AbstractVector{<:Integer},
)
    dims = tt_dims(x)
    old_ranks = collect(tt_ranks(x))
    requested = Int.(ranks)
    length(requested) == length(old_ranks) ||
        throw(ArgumentError("rank vector must have length $(length(old_ranks))"))
    requested[1] == 1 && requested[end] == 1 ||
        throw(ArgumentError("TT boundary ranks must be one"))
    all(requested .>= old_ranks) ||
        throw(ArgumentError("rank padding cannot shrink existing ranks"))

    limits = admissible_tt_ranks(dims, maximum(requested))
    all(requested .<= limits) ||
        throw(ArgumentError("requested ranks exceed Hilbert-space bounds"))

    T = promote_type(map(core -> eltype(core), x.cores)...)
    cores = Vector{Array{T,3}}(undef, length(x.cores))
    for i in eachindex(x.cores)
        old = x.cores[i]
        r1, n, r2 = size(old)
        core = zeros(T, requested[i], n, requested[i + 1])
        core[1:r1, :, 1:r2] .= old
        cores[i] = core
    end
    return TTTensor(cores)
end

function first_site_populations(psi::TTTensor)
    isempty(psi.cores) && throw(ArgumentError("TT state must contain a core"))

    T = promote_type(ComplexF64, map(core -> eltype(core), psi.cores)...)
    env = ones(T, 1, 1)
    for site in length(psi.cores):-1:2
        core = psi.cores[site]
        r1, n, r2 = size(core)
        next_env = zeros(T, r1, r1)
        workspace = zeros(T, r1, r2)
        for state in 1:n
            state_core = @view core[:, state, :]
            mul!(workspace, state_core, env)
            mul!(next_env, workspace, adjoint(state_core), one(T), one(T))
        end
        env = next_env
    end

    first_core = psi.cores[1]
    _, nsys, r2 = size(first_core)
    populations = zeros(Float64, nsys)
    workspace = zeros(T, 1, r2)
    population = zeros(T, 1, 1)
    conjugated = zeros(T, r2)
    for state in 1:nsys
        state_core = @view first_core[1, state, :]
        mul!(workspace, transpose(state_core), env)
        conjugated .= conj.(state_core)
        mul!(population, workspace, reshape(conjugated, r2, 1))
        populations[state] = real(population[1])
    end
    return populations
end
