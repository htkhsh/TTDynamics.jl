using LinearAlgebra
using Statistics
using TOML
using TTDynamics
using TTSolver

function step_count(duration_fs::Real, time_step_fs::Real; allow_zero::Bool=false)
    isfinite(duration_fs) && duration_fs >= 0 ||
        throw(ArgumentError("duration_fs must be finite and nonnegative"))
    isfinite(time_step_fs) && time_step_fs > 0 ||
        throw(ArgumentError("time_step_fs must be finite and positive"))
    !allow_zero && duration_fs == 0 &&
        throw(ArgumentError("duration_fs must include at least one time step"))
    ratio = duration_fs / time_step_fs
    steps = round(Int, ratio)
    isapprox(ratio, steps; rtol=0, atol=eps(Float64) * max(1, abs(ratio))) ||
        throw(ArgumentError("duration_fs must be an integral multiple of time_step_fs"))
    return steps
end

function gibbs_density(system::HEOMTTSystem, temperature_K::Real)
    isfinite(temperature_K) && temperature_K > 0 ||
        throw(ArgumentError("temperature_K must be finite and positive"))
    ishermitian(system.H_sys) ||
        throw(ArgumentError("the system Hamiltonian must be Hermitian"))
    eigensystem = eigen(Hermitian(system.H_sys))
    β = QFiND.ħ * 1e15 / (QFiND.kb * temperature_K)
    weights = exp.(-β .* (eigensystem.values .- minimum(eigensystem.values)))
    weights ./= sum(weights)
    density = eigensystem.vectors * Diagonal(weights) * eigensystem.vectors'
    return ComplexF64.((density + density') / 2)
end

function _heom_density_state(system::HEOMTTSystem, density::AbstractMatrix)
    n = size(system.H_sys, 1)
    size(density) == (n, n) || throw(DimensionMismatch("density has the wrong size"))
    isapprox(density, density'; atol=1e-10, rtol=0) ||
        throw(ArgumentError("density must be Hermitian"))
    isapprox(tr(density), 1; atol=1e-10, rtol=0) ||
        throw(ArgumentError("density must have unit trace"))
    ket_core = reshape(Matrix{ComplexF64}(I, n, n), 1, n, n)
    bra_core = reshape(ComplexF64.(density), n, n, 1)
    vacuum = [reshape(ComplexF64[1; zeros(nmode - 1)], 1, nmode, 1)
              for nmode in system.nb]
    return TTTensor([ket_core, bra_core, vacuum...])
end

function initial_heom_state(config, system::HEOMTTSystem)
    size(system.H_sys, 1) == config.site_count ||
        throw(DimensionMismatch("system dimension must equal config.site_count"))
    density = if config.initial_condition === :localized
        site = config.initial_site
        1 <= site <= config.site_count || throw(ArgumentError("initial_site is out of range"))
        result = zeros(ComplexF64, config.site_count, config.site_count)
        result[site, site] = 1
        result
    elseif config.initial_condition === :gibbs
        gibbs_density(system, config.temperature_K)
    else
        throw(ArgumentError("initial_condition must be :localized or :gibbs"))
    end
    return _heom_density_state(system, density)
end

function current_heom_operators(config, problem)
    n = config.site_count
    current = current_operator(config)
    identities = [Matrix{ComplexF64}(I, n, n),
                  [Matrix{ComplexF64}(I, size, size) for size in problem.system.nb]...]
    left_action = TTMatrix([
        reshape(ComplexF64.(matrix), 1, size(matrix, 1), size(matrix, 2), 1)
        for matrix in [current, identities...]
    ])
    left_action = tt_round(left_action, config.operator_tolerance)

    ket_core = zeros(ComplexF64, 1, n, n)
    for index in 1:n
        ket_core[1, index, index] = 1
    end
    bra_core = reshape(current, n, n, 1)
    vacuum = [reshape(ComplexF64[1; zeros(size - 1)], 1, size, 1)
              for size in problem.system.nb]
    observable = tt_round(TTTensor([ket_core, bra_core, vacuum...]),
                          config.operator_tolerance)
    return (; current, left_action, observable)
end

function tamen_step(state::TTTensor, liouvillian::TTMatrix, config,
                    trace_observable, workspace::TAMEnWorkspace)
    space_time_state = tt_kron(
        state,
        tt_ones(config.temporal_basis_size; T=ComplexF64),
    )
    space_time_state, time_grid, _, _, _ = tamen(
        space_time_state,
        config.time_step_fs * liouvillian,
        config.tamen_tolerance;
        verb=0,
        nswp=config.sweep_count,
        local_iters=config.local_iterations,
        kickrank=config.kick_rank,
        time_scheme="CN",
        time_error_damp=100.0,
        correct_2_norm=false,
        conserved_observables=[trace_observable],
        workspace=workspace,
    )
    return extract_snapshot(space_time_state, time_grid, 1.0, "CN")
end

function measure_density(state::TTTensor, problem)
    density = root_density_matrix(state, problem.system)
    pairs = [(row, column)
             for row in 1:(size(density, 1) - 1)
             for column in (row + 1):size(density, 2)]
    coherences = ComplexF64[density[row, column] for (row, column) in pairs]
    ranks = tt_ranks(state)
    return (; populations=real.(diag(density)),
            coherence_pairs=pairs,
            coherences,
            trace=real(tt_dot(problem.trace_observable, state)),
            maximum_rank=maximum(ranks),
            mean_rank=mean(ranks))
end

function _observations_to_result(observations, times)
    first_observation = first(observations)
    pairs = first_observation.coherence_pairs
    populations = hcat([observation.populations for observation in observations]...)
    coherences = hcat([observation.coherences for observation in observations]...)
    return (; times,
            populations,
            coherence_pairs=pairs,
            coherences,
            trace=[observation.trace for observation in observations],
            maximum_rank=[observation.maximum_rank for observation in observations],
            mean_rank=[observation.mean_rank for observation in observations])
end

function run_equilibration(config, problem;
                           duration_fs::Real,
                           initial_state=nothing)
    count = step_count(duration_fs, config.time_step_fs)
    state = isnothing(initial_state) ? initial_heom_state(config, problem.system) : initial_state
    workspace = TAMEnWorkspace(ComplexF64)
    observations = [measure_density(state, problem)]
    for _ in 1:count
        state = tamen_step(state, problem.liouvillian, config,
                           problem.trace_observable, workspace)
        push!(observations, measure_density(state, problem))
    end
    times = collect(0:count) .* config.time_step_fs
    return merge(_observations_to_result(observations, times), (; state))
end

function run_current_correlation(equilibrium_state::TTTensor, config, problem;
                                 duration_fs::Real)
    count = step_count(duration_fs, config.time_step_fs; allow_zero=true)
    operators = current_heom_operators(config, problem)
    exact_source = operators.left_action * equilibrium_state
    state = tt_round(exact_source, config.state_rounding_tolerance)
    workspace = TAMEnWorkspace(ComplexF64)
    correlations = ComplexF64[tt_dot(operators.observable, exact_source)]
    maximum_rank = [maximum(tt_ranks(state))]
    mean_rank = [mean(tt_ranks(state))]
    for _ in 1:count
        state = tamen_step(state, problem.liouvillian, config,
                           problem.trace_observable, workspace)
        push!(correlations, ComplexF64(tt_dot(operators.observable, state)))
        ranks = tt_ranks(state)
        push!(maximum_rank, maximum(ranks))
        push!(mean_rank, mean(ranks))
    end
    return (; times=collect(0:count) .* config.time_step_fs,
            correlation=correlations,
            maximum_rank,
            mean_rank,
            state)
end

function equilibrium_metadata(config, problem, state; equilibration_time_fs::Real)
    return Dict(
        "identifier" => "TTDynamics Holstein HEOM-TT equilibrium",
        "version" => 1,
        "site_count" => config.site_count,
        "initial_condition" => String(config.initial_condition),
        "equilibration_time_fs" => Float64(equilibration_time_fs),
        "hierarchy_sizes" => Int.(problem.system.nb),
        "tt_dimensions" => Int.(tt_dims(state)),
    )
end

function write_equilibrium_metadata(path::AbstractString, metadata; overwrite::Bool=true)
    !overwrite && ispath(path) && throw(ArgumentError("target already exists: $path"))
    open(String(path), "w") do io
        TOML.print(io, metadata)
    end
    return String(path)
end

read_equilibrium_metadata(path::AbstractString) = TOML.parsefile(String(path))

function validate_equilibrium_metadata(metadata, config, problem, state)
    required = ("identifier", "version", "site_count", "initial_condition",
                "equilibration_time_fs", "hierarchy_sizes", "tt_dimensions")
    all(key -> haskey(metadata, key), required) ||
        throw(ArgumentError("equilibrium metadata is missing a required key"))
    metadata["identifier"] == "TTDynamics Holstein HEOM-TT equilibrium" ||
        throw(ArgumentError("equilibrium metadata identifier does not match"))
    metadata["version"] == 1 || throw(ArgumentError("unsupported equilibrium metadata version"))
    metadata["site_count"] == config.site_count ||
        throw(DimensionMismatch("equilibrium site_count does not match config"))
    metadata["initial_condition"] == String(config.initial_condition) ||
        throw(ArgumentError("equilibrium initial_condition does not match config"))
    Int.(metadata["hierarchy_sizes"]) == Int.(problem.system.nb) ||
        throw(DimensionMismatch("equilibrium hierarchy sizes do not match problem"))
    Int.(metadata["tt_dimensions"]) == Int.(tt_dims(state)) ||
        throw(DimensionMismatch("equilibrium TT dimensions do not match state"))
    return nothing
end

function write_equilibration_csv(path::AbstractString, result; overwrite::Bool=true)
    !overwrite && ispath(path) && throw(ArgumentError("target already exists: $path"))
    open(String(path), "w") do io
        headers = ["time_fs", ["population_site_$site" for site in axes(result.populations, 1)]...]
        for (row, column) in result.coherence_pairs
            append!(headers, ["coherence_$(row)_$(column)_real",
                             "coherence_$(row)_$(column)_imag"])
        end
        append!(headers, ["trace", "max_rank", "mean_rank"])
        println(io, join(headers, ","))
        for index in eachindex(result.times)
            row = Any[result.times[index], result.populations[:, index]...]
            for value in result.coherences[:, index]
                append!(row, (real(value), imag(value)))
            end
            append!(row, (result.trace[index], result.maximum_rank[index], result.mean_rank[index]))
            println(io, join(row, ","))
        end
    end
    return String(path)
end

function write_current_correlation_csv(path::AbstractString, result; overwrite::Bool=true)
    !overwrite && ispath(path) && throw(ArgumentError("target already exists: $path"))
    open(String(path), "w") do io
        println(io, "time_fs,correlation_real_fs^-2,correlation_imag_fs^-2,max_rank,mean_rank")
        for index in eachindex(result.times)
            value = result.correlation[index]
            println(io, join((result.times[index], real(value), imag(value),
                             result.maximum_rank[index], result.mean_rank[index]), ","))
        end
    end
    return String(path)
end
