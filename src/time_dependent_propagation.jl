function _schrodinger_operator_dimensions(operator::TTMatrix)
    isempty(operator.cores) && throw(ArgumentError("Hamiltonian must have at least one core"))
    rows, columns = tt_dims(operator)
    rows == columns || throw(DimensionMismatch("Hamiltonian must have square local dimensions"))
    all(>(0), rows) || throw(ArgumentError("Hamiltonian dimensions must be positive"))
    previous_rank = 1
    for core in operator.cores
        size(core, 1) == previous_rank || throw(DimensionMismatch("Hamiltonian TT ranks do not match"))
        all(isfinite, core) || throw(ArgumentError("Hamiltonian cores must be finite"))
        previous_rank = size(core, 4)
    end
    previous_rank == 1 || throw(ArgumentError("Hamiltonian boundary ranks must be one"))
    rows
end

"""
    TimeDependentHamiltonian(H0, terms)
    TimeDependentHamiltonian(terms)
    TimeDependentHamiltonian(H0::TTMatrix)

Represent `H(t) = H0 + Σ fₖ(t)Hₖ` using `coefficient => TTMatrix` pairs.
Coefficients are scalar-valued callables of absolute physical time. Omit
`H0`, or pass `0` or `nothing`, for a purely time-dependent Hamiltonian.
At least one operator is required to determine the local dimensions.
Calling `H(t)` evaluates the spatial TT operator without dense expansion.
"""
struct TimeDependentHamiltonian{Terms}
    terms::Terms
    dimensions::Vector{Int}

    function TimeDependentHamiltonian(H0, terms)
        static = if H0 isa TTMatrix
            ((t -> 1.0) => H0,)
        elseif H0 === nothing || (H0 isa Number && iszero(H0))
            ()
        else
            throw(ArgumentError("H0 must be a TTMatrix, zero, or nothing"))
        end
        raw_terms = (static..., Tuple(terms)...)
        isempty(raw_terms) && throw(ArgumentError("at least one Hamiltonian operator is required"))
        normalized = map(raw_terms) do term
            term isa Pair && last(term) isa TTMatrix ||
                throw(ArgumentError("terms must be coefficient => TTMatrix pairs"))
            coefficient, operator = first(term), last(term)
            applicable(coefficient, 0.0) ||
                throw(ArgumentError("each coefficient must be callable with a real time"))
            converted = TTMatrix([ComplexF64.(core) for core in operator.cores])
            _schrodinger_operator_dimensions(converted)
            coefficient => converted
        end
        dimensions = _schrodinger_operator_dimensions(last(first(normalized)))
        for term in normalized
            first(tt_dims(last(term))) == dimensions ||
                throw(DimensionMismatch("all Hamiltonian terms must have the same local dimensions"))
        end
        new{typeof(normalized)}(normalized, dimensions)
    end
end

TimeDependentHamiltonian(terms) = TimeDependentHamiltonian(nothing, terms)
TimeDependentHamiltonian(H0::TTMatrix) = TimeDependentHamiltonian(H0, ())

function _hamiltonian_coefficient(coefficient, time)
    value = coefficient(time)
    value isa Number && isfinite(value) ||
        throw(ArgumentError("Hamiltonian coefficients must return finite scalars"))
    result = ComplexF64(value)
    isfinite(result) || throw(ArgumentError("Hamiltonian coefficient is outside ComplexF64 range"))
    result
end

function (H::TimeDependentHamiltonian)(time::Real)
    isfinite(time) || throw(ArgumentError("time must be finite"))
    coefficient, operator = first(H.terms)
    result = _hamiltonian_coefficient(coefficient, time) * operator
    for term in Base.tail(H.terms)
        increment = _hamiltonian_coefficient(first(term), time) * last(term)
        # The native TTMatrix sum does not handle a single spatial core.
        result = length(H.dimensions) == 1 ?
            TTMatrix([result.cores[1] .+ increment.cores[1]]) : result + increment
    end
    result
end

function _schrodinger_times(H, state, times)
    tt_dims(state) == H.dimensions ||
        throw(DimensionMismatch("initial state dimensions do not match the Hamiltonian"))
    time_grid = collect(Float64, times)
    isempty(time_grid) && throw(ArgumentError("times must not be empty"))
    all(isfinite, time_grid) && all(diff(time_grid) .>= 0) ||
        throw(ArgumentError("times must be finite and nondecreasing"))
    all(isfinite, diff(time_grid)) || throw(ArgumentError("time intervals must be finite"))
    time_grid
end

function _schrodinger_trajectory(step, state, time_grid, callback)
    trajectory = TTTensor[state]
    current = state
    callback !== nothing && callback(0, time_grid[1], current)
    for index in 2:length(time_grid)
        start = time_grid[index - 1]
        Δt = time_grid[index] - start
        if Δt > 0
            current = step(current, start, Δt)
            callback !== nothing && callback(index - 1, time_grid[index], current)
        end
        push!(trajectory, current)
    end
    trajectory
end

"""
    propagate_ksl(H, state, times; tolerance=1e-12, rmax=typemax(Int),
                  symmetric=true, expmethod=:krylov, krylov_m=30, callback=nothing)

Solve `dψ/dt = -im * H(t) * ψ` with midpoint evaluation and KSL sweeps.
`H` is a `TimeDependentHamiltonian` or constant `TTMatrix`. Return a TT state
at each requested time. Symmetric sweeps give second-order time stepping
for smooth Hamiltonians, subject to rank and local solver accuracy.
The callback receives `(step, time, state)` initially and after nonzero steps.
"""
function propagate_ksl(H::Union{TimeDependentHamiltonian,TTMatrix}, state::TTTensor, times;
                       tolerance::Real=1e-12, rmax::Int=typemax(Int),
                       symmetric::Bool=true, expmethod::Symbol=:krylov,
                       krylov_m::Int=30, callback=nothing)
    isfinite(tolerance) && tolerance > 0 || throw(ArgumentError("tolerance must be finite and positive"))
    rmax > 0 || throw(ArgumentError("rmax must be positive"))
    krylov_m > 0 || throw(ArgumentError("krylov_m must be positive"))
    expmethod in (:krylov, :taylor) || throw(ArgumentError("expmethod must be :krylov or :taylor"))
    hamiltonian = H isa TTMatrix ? TimeDependentHamiltonian(H) : H
    time_grid = _schrodinger_times(hamiltonian, state, times)
    _schrodinger_trajectory(state, time_grid, callback) do current, start, Δt
        # tt_ksl solves dx/dt + A*x = 0.
        generator = 1im * hamiltonian(start + Δt / 2)
        tt_ksl(current, generator, Δt; tol=tolerance, rmax, symmetric, expmethod, krylov_m)
    end
end

function _hamiltonian_time_operator(H, times, Δt)
    spatial_modes = length(H.dimensions)
    nt = length(times)
    cells = Matrix{Any}(undef, spatial_modes + 1, length(H.terms))
    for (column, term) in enumerate(H.terms)
        coefficient, operator = first(term), last(term)
        for site in 1:spatial_modes
            cells[site, column] = operator.cores[site]
        end
        # tAMEn solves dx/dτ = A(τ)*x. Its adaptive step is applied internally.
        cells[1, column] = (-1im * Δt) * cells[1, column]
        values = [_hamiltonian_coefficient(coefficient, time) for time in times]
        cells[end, column] = reshape(Matrix(Diagonal(values)), 1, nt, nt, 1)
    end
    cells
end

"""
    propagate_tamen(H, state, times; temporal_basis_size=3, tolerance=1e-8,
                    state_tolerance=1e-12, options=Dict{Symbol,Any}(),
                    conserved_observables=nothing, workspace=nothing, callback=nothing)

Solve `dψ/dt = -im * H(t) * ψ` with separated spatial operators and sampled
time coefficients. `H` is a `TimeDependentHamiltonian` or constant `TTMatrix`.
Time coefficients are reevaluated at absolute physical times after adaptive
subdivision. Return one `TTTensor` per requested time. `options` accepts the
native tAMEn settings, including `time_scheme="cheb"` or `"cn"`, and is copied
before each solve. CN requires an odd temporal basis size of at least three.
"""
function propagate_tamen(H::Union{TimeDependentHamiltonian,TTMatrix}, state::TTTensor, times;
                         temporal_basis_size::Integer=3, tolerance::Real=1e-8,
                         state_tolerance::Real=1e-12, options=Dict{Symbol,Any}(),
                         conserved_observables=nothing,
                         workspace::Union{Nothing,TAMEnWorkspace}=nothing, callback=nothing)
    temporal_basis_size >= 2 || throw(ArgumentError("temporal_basis_size must be at least two"))
    isfinite(tolerance) && tolerance > 0 || throw(ArgumentError("tolerance must be finite and positive"))
    isfinite(state_tolerance) && state_tolerance >= 0 || throw(ArgumentError("state_tolerance must be finite and nonnegative"))
    hamiltonian = H isa TTMatrix ? TimeDependentHamiltonian(H) : H
    time_grid = _schrodinger_times(hamiltonian, state, times)
    local_options = Dict{Symbol,Any}(options)
    scheme = String(get(local_options, :time_scheme, "cheb"))
    if occursin("cheb", lowercase(scheme))
        # The solver constructs its own Chebyshev nodes.
    elseif occursin("cn", lowercase(scheme))
        isodd(temporal_basis_size) || throw(ArgumentError("CN requires an odd temporal_basis_size"))
    else
        throw(ArgumentError("time_scheme must be cheb or cn"))
    end
    work = workspace === nothing ? TAMEnWorkspace(ComplexF64) : workspace
    _schrodinger_trajectory(state, time_grid, callback) do current, start, Δt
        complex_state = TTTensor([ComplexF64.(core) for core in current.cores])
        space_time = tkron(complex_state, tt_ones(Int(temporal_basis_size); T=ComplexF64))
        generator = (_, τ) -> _hamiltonian_time_operator(hamiltonian, start .+ Δt .* τ, Δt)
        solution, local_times, _, _, _ = tamen(space_time, generator, tolerance,
            copy(local_options); conserved_observables, workspace=work)
        tt_round(extract_snapshot(solution, local_times, 1.0, scheme), Float64(state_tolerance))
    end
end
