using LinearAlgebra
using Statistics
# Resolve TOML by UUID so including this example from the package test project
# does not make it an implicit package dependency.
if !isdefined(@__MODULE__, :TOML)
    @eval const TOML = Base.require(Base.PkgId(
        Base.UUID("fa267f1f-6049-4f14-aa54-33bafae1ed76"), "TOML",
    ))
end
using .TOML
using TTDynamics
using TTSolver
using HEOMKit: icm2ifs

"""
    lattice_frohlich_particle_current(config) -> Matrix{ComplexF64}

Return the hopping-generated Hermitian particle current for the periodic
lattice Fröhlich chain.
"""
function lattice_frohlich_particle_current(
    config::LatticeFrohlichCurrentCorrelationConfig,
)::Matrix{ComplexF64}
    validate_lattice_frohlich_current_correlation_config(config)
    N = config.site_count
    scale = config.hopping_cm * icm2ifs
    current = zeros(ComplexF64, N, N)
    for site in 1:(N - 1)
        current[site + 1, site] += im * scale
        current[site, site + 1] -= im * scale
    end
    if N > 2
        current[1, N] += im * scale
        current[N, 1] -= im * scale
    end
    return current
end

"""
    build_lattice_frohlich_current_operators(config, problem)

Embed the particle current as a ket-side HEOM-TT action and a root-space
observable in the canonical `[ket, bra, hierarchy...]` layout.
"""
function build_lattice_frohlich_current_operators(
    config::LatticeFrohlichCurrentCorrelationConfig,
    problem,
)
    current = lattice_frohlich_particle_current(config)
    N = config.site_count
    I_sys = Matrix{ComplexF64}(I, N, N)
    hierarchy_identities = [Matrix{ComplexF64}(I, n, n) for n in problem.system.nb]
    local_matrices = [current, I_sys, hierarchy_identities...]
    left_action = tt_round(
        TTMatrix([
            reshape(ComplexF64.(matrix), 1, size(matrix, 1), size(matrix, 2), 1)
            for matrix in local_matrices
        ]),
        config.operator_tolerance,
    )

    ket_core = zeros(ComplexF64, 1, N, N)
    for ket in 1:N
        ket_core[1, ket, ket] = 1
    end
    bra_core = reshape(current, N, N, 1)
    vacuum_cores = [reshape(ComplexF64[1; zeros(ComplexF64, n - 1)], 1, n, 1)
                    for n in problem.system.nb]
    observable = tt_round(
        TTTensor([ket_core, bra_core, vacuum_cores...]),
        config.operator_tolerance,
    )

    tt_dims(left_action) == tt_dims(problem.liouvillian) ||
        throw(ArgumentError("current left action dimensions must match the Liouvillian"))
    tt_dims(observable) == heom_tt_dimensions(problem.system) ||
        throw(ArgumentError("current observable dimensions must match the HEOM system"))
    return (; current, left_action, observable)
end

"""
    propagate_lattice_frohlich_cn_step(state, liouvillian, config;
                                       trace_observable=nothing) -> TTTensor

Advance one Crank-Nicolson tAMEn step using the configured solver controls.
"""
function propagate_lattice_frohlich_cn_step(
    state::TTTensor,
    liouvillian::TTMatrix,
    config::LatticeFrohlichCurrentCorrelationConfig;
    trace_observable=nothing,
)::TTTensor
    options = Dict(
        :verb => 0,
        :nswp => config.sweep_count,
        :local_iters => config.local_iterations,
        :kickrank => config.kick_rank,
        :time_scheme => "CN",
        :time_error_damp => 100.0,
    )
    !isnothing(trace_observable) && (options[:obs] = [trace_observable])
    space_time_state = tkron(
        state,
        tt_ones(config.temporal_basis_size; T=ComplexF64),
    )
    space_time_state, time_grid, _, _ = tamen(
        space_time_state,
        config.time_step_fs * liouvillian,
        config.tamen_tolerance,
        options,
    )
    snapshot = extract_snapshot(space_time_state, time_grid, 1.0, "CN")
    return tt_round(snapshot, config.state_rounding_tolerance)
end

"""
    propagate_lattice_frohlich_fixed_steps(state, problem, config, step_count;
                                            observe)

Run a fixed number of Crank-Nicolson steps, including an observation at zero.
"""
function propagate_lattice_frohlich_fixed_steps(
    state::TTTensor,
    problem,
    config::LatticeFrohlichCurrentCorrelationConfig,
    step_count::Integer;
    observe,
)
    step_count >= 0 || throw(ArgumentError("step_count must be nonnegative"))
    observations = [observe(0, 0.0, state)]
    trace_observable = hasproperty(problem, :trace_observable) ?
                       problem.trace_observable : nothing
    for step in 1:step_count
        state = propagate_lattice_frohlich_cn_step(
            state,
            problem.liouvillian,
            config;
            trace_observable,
        )
        push!(observations, observe(step, step * config.time_step_fs, state))
    end
    return (; state, observations)
end

"""
    normalize_lattice_frohlich_heom_state(state, trace_observable; atol=1e-14)

Scale every hierarchy component by the finite, nonzero complex root trace.
"""
function normalize_lattice_frohlich_heom_state(
    state::TTTensor,
    trace_observable::TTTensor;
    atol::Real=1e-14,
)::TTTensor
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    trace_value = tt_dot(trace_observable, state)
    isfinite(real(trace_value)) && isfinite(imag(trace_value)) ||
        throw(ArgumentError("HEOM trace must be finite"))
    abs(trace_value) > atol ||
        throw(ArgumentError("HEOM trace magnitude must exceed atol"))
    return inv(trace_value) * state
end

"""
    measure_lattice_frohlich_heom_state(state, trace_observable,
                                        population_observables)

Return root-space populations, real trace, and TT-rank diagnostics.
"""
function measure_lattice_frohlich_heom_state(
    state::TTTensor,
    trace_observable::TTTensor,
    population_observables,
)
    ranks = tt_ranks(state)
    return (;
        populations=real.([tt_dot(observable, state) for observable in population_observables]),
        trace=real(tt_dot(trace_observable, state)),
        maximum_rank=maximum(ranks),
        mean_rank=mean(ranks),
    )
end

function _lattice_frohlich_equilibration_step_count(
    config::LatticeFrohlichCurrentCorrelationConfig,
    equilibration_time_fs::Real,
)::Int
    isfinite(equilibration_time_fs) && equilibration_time_fs > 0 ||
        throw(ArgumentError("equilibration_time_fs must be finite and positive"))
    ratio = equilibration_time_fs / config.time_step_fs
    step_count = round(Int, ratio)
    isapprox(ratio, step_count; rtol=0, atol=eps(Float64) * max(1, abs(ratio))) ||
        throw(ArgumentError("equilibration_time_fs must be an integral multiple of time_step_fs"))
    step_count >= 1 ||
        throw(ArgumentError("equilibration_time_fs must include at least one time step"))
    return step_count
end

"""
    run_lattice_frohlich_equilibration(config, problem;
                                       initial_state=nothing,
                                       equilibration_time_fs=1000.0,
                                       progress_callback=nothing) -> NamedTuple

Propagate a localized lattice Fröhlich HEOM state and return its normalized
final state with root-space measurements at every step.
"""
function run_lattice_frohlich_equilibration(
    config::LatticeFrohlichCurrentCorrelationConfig,
    problem;
    initial_state=nothing,
    equilibration_time_fs::Real=1000.0,
    progress_callback=nothing,
)::NamedTuple
    validate_lattice_frohlich_current_correlation_config(config)
    step_count = _lattice_frohlich_equilibration_step_count(config, equilibration_time_fs)
    state = isnothing(initial_state) ? build_initial_state(
        problem.system,
        config.initial_site;
        tol=config.operator_tolerance,
    ) : initial_state
    state isa TTTensor || throw(ArgumentError("initial_state must be a TTTensor"))

    start_time_ns = time_ns()
    observe = function(step, time, rho)
        observation = measure_lattice_frohlich_heom_state(
            rho,
            problem.trace_observable,
            problem.population_observables,
        )
        should_report = step == 0 || step == step_count ||
                        step % config.progress_interval == 0
        if !isnothing(progress_callback) && should_report
            progress_callback((;
                step,
                step_count,
                time_fs=time,
                trace=observation.trace,
                maximum_rank=observation.maximum_rank,
                mean_rank=observation.mean_rank,
                elapsed_seconds=(time_ns() - start_time_ns) / 1.0e9,
            ))
        end
        return observation
    end
    propagated = propagate_lattice_frohlich_fixed_steps(
        state,
        problem,
        config,
        step_count;
        observe,
    )
    observations = propagated.observations
    times = collect(0:step_count) .* config.time_step_fs
    length(times) == length(observations) ||
        error("equilibration observation count is inconsistent")
    return (;
        times,
        populations=hcat([observation.populations for observation in observations]...),
        trace=[observation.trace for observation in observations],
        maximum_rank=[observation.maximum_rank for observation in observations],
        mean_rank=[observation.mean_rank for observation in observations],
        state=normalize_lattice_frohlich_heom_state(
            propagated.state,
            problem.trace_observable,
        ),
    )
end

function _print_lattice_frohlich_equilibration_progress(io::IO, p)::Nothing
    println(
        io,
        "Step $(p.step)/$(p.step_count) | time=$(p.time_fs) fs | " *
        "trace=$(p.trace) | max rank=$(p.maximum_rank) | " *
        "mean rank=$(p.mean_rank) | elapsed=$(round(p.elapsed_seconds; digits=2)) s",
    )
    return nothing
end

function _lattice_frohlich_correlation_step_count(
    config::LatticeFrohlichCurrentCorrelationConfig,
    correlation_time_fs::Real,
)::Int
    isfinite(correlation_time_fs) && correlation_time_fs >= 0 ||
        throw(ArgumentError("correlation_time_fs must be finite and nonnegative"))
    ratio = correlation_time_fs / config.time_step_fs
    step_count = round(Int, ratio)
    isapprox(ratio, step_count; rtol=0, atol=eps(Float64) * max(1, abs(ratio))) ||
        throw(ArgumentError("correlation_time_fs must be an integral multiple of time_step_fs"))
    return step_count
end

function _print_lattice_frohlich_current_progress(io::IO, p)::Nothing
    println(
        io,
        "Step $(p.step)/$(p.step_count) | time=$(p.time_fs) fs | " *
        "max rank=$(p.maximum_rank) | mean rank=$(p.mean_rank) | " *
        "elapsed=$(round(p.elapsed_seconds; digits=2)) s",
    )
    return nothing
end

"""
    run_lattice_frohlich_current_correlation(equilibrium_state, config, problem;
                                             correlation_time_fs=200.0,
                                             progress_callback=nothing) -> NamedTuple

Propagate the unnormalized `J * rho_eq` source and record the unsymmetrized
particle-current correlation at every time step.
"""
function run_lattice_frohlich_current_correlation(
    equilibrium_state::TTTensor,
    config::LatticeFrohlichCurrentCorrelationConfig,
    problem;
    correlation_time_fs::Real=200.0,
    progress_callback=nothing,
)::NamedTuple
    validate_lattice_frohlich_current_correlation_config(config)
    step_count = _lattice_frohlich_correlation_step_count(config, correlation_time_fs)
    operators = build_lattice_frohlich_current_operators(config, problem)
    exact_source = operators.left_action * equilibrium_state
    source = tt_round(exact_source, config.state_rounding_tolerance)
    start_time_ns = time_ns()
    observe = function(step, time, state)
        ranks = tt_ranks(state)
        observation = (
            correlation=ComplexF64(tt_dot(operators.observable, state)),
            maximum_rank=maximum(ranks),
            mean_rank=mean(ranks),
        )
        should_report = step == 0 || step == step_count ||
                        step % config.progress_interval == 0
        if !isnothing(progress_callback) && should_report
            progress_callback((;
                step,
                step_count,
                time_fs=time,
                maximum_rank=observation.maximum_rank,
                mean_rank=observation.mean_rank,
                elapsed_seconds=(time_ns() - start_time_ns) / 1.0e9,
            ))
        end
        return observation
    end
    propagated = propagate_lattice_frohlich_fixed_steps(
        source,
        problem,
        config,
        step_count;
        observe,
    )
    observations = propagated.observations
    times = collect(0:step_count) .* config.time_step_fs
    length(times) == length(observations) ||
        error("correlation observation count is inconsistent")
    correlation = ComplexF64[observation.correlation for observation in observations]
    correlation[1] = tt_dot(operators.observable, exact_source)
    return (;
        times,
        correlation,
        maximum_rank=[observation.maximum_rank for observation in observations],
        mean_rank=[observation.mean_rank for observation in observations],
        state=propagated.state,
    )
end

const _LATTICE_FROHLICH_METADATA_IDENTIFIER =
    "TTDynamics.LatticeFrohlichEquilibrium"
const _LATTICE_FROHLICH_METADATA_VERSION = 1
const _LATTICE_FROHLICH_HEOM_REPRESENTATION = "twin-space-v1"

const _LATTICE_FROHLICH_METADATA_INTEGER_FIELDS = (
    :site_count, :initial_site, :pade_order, :validation_sample_count,
    :hierarchy_local_size, :temporal_basis_size, :sweep_count,
    :local_iterations, :kick_rank, :progress_interval,
)
const _LATTICE_FROHLICH_METADATA_FLOAT_FIELDS = (
    :hopping_cm, :brownian_frequency_cm, :brownian_damping_cm,
    :reorganization_energy_cm, :temperature_K, :final_time_fs, :time_step_fs,
    :tpsd_tolerance, :validation_final_time_fs, :bcf_upper_bound_cm,
    :tamen_tolerance, :operator_tolerance, :state_rounding_tolerance,
)

_lattice_frohlich_metadata_name(field::Symbol) = String(field)

function _lattice_frohlich_metadata_value(metadata, field::AbstractString)
    haskey(metadata, field) || throw(ArgumentError("$field is missing from equilibrium metadata"))
    return metadata[field]
end
function _lattice_frohlich_metadata_string(metadata, field::AbstractString)
    value = _lattice_frohlich_metadata_value(metadata, field)
    value isa AbstractString || throw(ArgumentError("$field must be a string"))
    return String(value)
end
function _lattice_frohlich_metadata_integer(metadata, field::AbstractString)
    value = _lattice_frohlich_metadata_value(metadata, field)
    value isa Integer && !(value isa Bool) || throw(ArgumentError("$field must be an integer"))
    return Int(value)
end
function _lattice_frohlich_metadata_float(metadata, field::AbstractString)
    value = _lattice_frohlich_metadata_value(metadata, field)
    value isa Real && !(value isa Bool) && isfinite(value) ||
        throw(ArgumentError("$field must be a finite real number"))
    return Float64(value)
end
function _lattice_frohlich_metadata_float_vector(metadata, field::AbstractString)
    value = _lattice_frohlich_metadata_value(metadata, field)
    value isa AbstractVector || throw(ArgumentError("$field must be an array"))
    all(x -> x isa Real && !(x isa Bool) && isfinite(x), value) ||
        throw(ArgumentError("$field must contain finite real numbers"))
    return Float64.(value)
end
function _lattice_frohlich_metadata_integer_vector(metadata, field::AbstractString)
    value = _lattice_frohlich_metadata_value(metadata, field)
    value isa AbstractVector || throw(ArgumentError("$field must be an array"))
    all(x -> x isa Integer && !(x isa Bool), value) ||
        throw(ArgumentError("$field must contain integers"))
    return Int.(value)
end
function _validate_lattice_frohlich_toml_value(value, field::AbstractString)
    if value isa AbstractString || value isa Bool || (value isa Integer && !(value isa Bool))
        return nothing
    elseif value isa AbstractFloat
        isfinite(value) || throw(ArgumentError("$field must be finite"))
    elseif value isa AbstractVector
        for (index, element) in pairs(value)
            _validate_lattice_frohlich_toml_value(element, "$field[$index]")
        end
    elseif value isa AbstractDict
        for (key, element) in pairs(value)
            key isa AbstractString || throw(ArgumentError("$field dictionary keys must be strings"))
            _validate_lattice_frohlich_toml_value(element, "$field.$key")
        end
    else
        throw(ArgumentError("$field is not TOML-safe metadata"))
    end
    return nothing
end

function _validate_lattice_frohlich_equilibrium_metadata(metadata)
    metadata isa AbstractDict || throw(ArgumentError("metadata must be a dictionary"))
    for (field, value) in pairs(metadata)
        field isa AbstractString || throw(ArgumentError("metadata keys must be strings"))
        _validate_lattice_frohlich_toml_value(value, String(field))
    end
    _lattice_frohlich_metadata_string(metadata, "identifier") == _LATTICE_FROHLICH_METADATA_IDENTIFIER || throw(ArgumentError("identifier is unsupported"))
    _lattice_frohlich_metadata_integer(metadata, "version") == _LATTICE_FROHLICH_METADATA_VERSION || throw(ArgumentError("version is unsupported"))
    _lattice_frohlich_metadata_string(metadata, "heom_representation") == _LATTICE_FROHLICH_HEOM_REPRESENTATION || throw(ArgumentError("heom_representation is unsupported"))
    _lattice_frohlich_metadata_float(metadata, "equilibration_time_fs")
    _lattice_frohlich_metadata_float_vector(metadata, "site_energies_cm")
    for field in _LATTICE_FROHLICH_METADATA_FLOAT_FIELDS
        _lattice_frohlich_metadata_float(metadata, _lattice_frohlich_metadata_name(field))
    end
    for field in _LATTICE_FROHLICH_METADATA_INTEGER_FIELDS
        _lattice_frohlich_metadata_integer(metadata, _lattice_frohlich_metadata_name(field))
    end
    _lattice_frohlich_metadata_string(metadata, "pade_type")
    exponent_real = _lattice_frohlich_metadata_float_vector(metadata, "exponents_real")
    exponent_imag = _lattice_frohlich_metadata_float_vector(metadata, "exponents_imag")
    coefficient_real = _lattice_frohlich_metadata_float_vector(metadata, "coefficients_real")
    coefficient_imag = _lattice_frohlich_metadata_float_vector(metadata, "coefficients_imag")
    length(exponent_real) == length(exponent_imag) || throw(ArgumentError("exponents arrays must have equal length"))
    length(coefficient_real) == length(coefficient_imag) || throw(ArgumentError("coefficients arrays must have equal length"))
    length(exponent_real) == length(coefficient_real) || throw(ArgumentError("TPSD arrays must have equal length"))
    _lattice_frohlich_metadata_integer(metadata, "tpsd_term_count") == length(exponent_real) || throw(ArgumentError("tpsd_term_count must equal decomposition array length"))
    weights = _lattice_frohlich_metadata_value(metadata, "frohlich_weights")
    weights isa AbstractVector || throw(ArgumentError("frohlich_weights must be an array"))
    all(row -> row isa AbstractVector && all(x -> x isa Real && !(x isa Bool) && isfinite(x), row), weights) || throw(ArgumentError("frohlich_weights must be finite rows"))
    _lattice_frohlich_metadata_integer_vector(metadata, "hierarchy_sizes")
    _lattice_frohlich_metadata_integer_vector(metadata, "tt_dimensions")
    return nothing
end

function _lattice_frohlich_atomic_rename(oldpath::AbstractString, newpath::AbstractString)
    status = ccall(:jl_fs_rename, Int32, (Cstring, Cstring), String(oldpath), String(newpath))
    status < 0 && Base.uv_error("rename", status)
    return nothing
end

function lattice_frohlich_equilibrium_metadata(config::LatticeFrohlichCurrentCorrelationConfig,
                                                decomposition, problem, state::TTTensor;
                                                equilibration_time_fs)::Dict{String,Any}
    validate_lattice_frohlich_current_correlation_config(config)
    metadata = Dict{String,Any}(
        "identifier" => _LATTICE_FROHLICH_METADATA_IDENTIFIER,
        "version" => _LATTICE_FROHLICH_METADATA_VERSION,
        "heom_representation" => _LATTICE_FROHLICH_HEOM_REPRESENTATION,
        "equilibration_time_fs" => Float64(equilibration_time_fs),
        "site_count" => config.site_count,
        "site_energies_cm" => copy(config.site_energies_cm),
        "pade_type" => string(config.pade_type),
        "exponents_real" => real.(decomposition.exponents),
        "exponents_imag" => imag.(decomposition.exponents),
        "coefficients_real" => real.(decomposition.coefficients),
        "coefficients_imag" => imag.(decomposition.coefficients),
        "tpsd_term_count" => length(decomposition.exponents),
        "frohlich_weights" => [collect(row) for row in eachrow(problem.frohlich_weights)],
        "hierarchy_sizes" => copy(problem.system.nb),
        "tt_dimensions" => tt_dims(state),
    )
    for field in _LATTICE_FROHLICH_METADATA_FLOAT_FIELDS
        metadata[_lattice_frohlich_metadata_name(field)] = getfield(config, field)
    end
    for field in _LATTICE_FROHLICH_METADATA_INTEGER_FIELDS
        metadata[_lattice_frohlich_metadata_name(field)] = getfield(config, field)
    end
    _validate_lattice_frohlich_equilibrium_metadata(metadata)
    return metadata
end

function write_lattice_frohlich_equilibrium_metadata(path::AbstractString, metadata;
                                                     overwrite::Bool=false)::String
    target = String(path)
    _validate_lattice_frohlich_equilibrium_metadata(metadata)
    !overwrite && ispath(target) && throw(ArgumentError("target already exists: $target"))
    temporary_path = nothing
    try
        temporary_path, io = mktemp(dirname(abspath(target)))
        try TOML.print(io, metadata) finally close(io) end
        !overwrite && ispath(target) && throw(ArgumentError("target already exists: $target"))
        if overwrite
            _lattice_frohlich_atomic_rename(temporary_path, target); temporary_path = nothing
        else
            Base.Filesystem.hardlink(temporary_path, target)
        end
        return target
    finally
        temporary_path === nothing || rm(temporary_path; force=true)
    end
end

function read_lattice_frohlich_equilibrium_metadata(path::AbstractString)::Dict{String,Any}
    source = String(path)
    isfile(source) || throw(ArgumentError("metadata file does not exist: $source"))
    return TOML.parsefile(source)
end

function _lattice_frohlich_metadata_exact(field, actual, expected)
    actual == expected || throw(ArgumentError("$field does not match the reconstructed problem"))
end
function _lattice_frohlich_metadata_approximate(field, actual, expected; rtol, atol)
    isapprox(actual, expected; rtol, atol) || throw(ArgumentError("$field does not match the reconstructed problem"))
end

function validate_lattice_frohlich_equilibrium_state(
    state, metadata, config::LatticeFrohlichCurrentCorrelationConfig, decomposition, problem;
    rtol::Real=1e-12, atol::Real=1e-14,
)::TTTensor
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative"))
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    validate_lattice_frohlich_current_correlation_config(config)
    state isa TTTensor || throw(ArgumentError("equilibrium state must be a TTTensor"))
    root_density_matrix(state, problem.system)
    _validate_lattice_frohlich_equilibrium_metadata(metadata)
    for field in _LATTICE_FROHLICH_METADATA_FLOAT_FIELDS
        name = _lattice_frohlich_metadata_name(field)
        _lattice_frohlich_metadata_approximate(name, _lattice_frohlich_metadata_float(metadata, name), getfield(config, field); rtol, atol)
    end
    for field in _LATTICE_FROHLICH_METADATA_INTEGER_FIELDS
        name = _lattice_frohlich_metadata_name(field)
        _lattice_frohlich_metadata_exact(name, _lattice_frohlich_metadata_integer(metadata, name), getfield(config, field))
    end
    _lattice_frohlich_metadata_approximate("site_energies_cm", _lattice_frohlich_metadata_float_vector(metadata, "site_energies_cm"), config.site_energies_cm; rtol, atol)
    _lattice_frohlich_metadata_exact("pade_type", _lattice_frohlich_metadata_string(metadata, "pade_type"), string(config.pade_type))
    exponents = complex.(_lattice_frohlich_metadata_float_vector(metadata, "exponents_real"), _lattice_frohlich_metadata_float_vector(metadata, "exponents_imag"))
    coefficients = complex.(_lattice_frohlich_metadata_float_vector(metadata, "coefficients_real"), _lattice_frohlich_metadata_float_vector(metadata, "coefficients_imag"))
    _lattice_frohlich_metadata_approximate("exponents", exponents, decomposition.exponents; rtol, atol)
    _lattice_frohlich_metadata_approximate("coefficients", coefficients, decomposition.coefficients; rtol, atol)
    _lattice_frohlich_metadata_exact("tpsd_term_count", _lattice_frohlich_metadata_integer(metadata, "tpsd_term_count"), length(decomposition.exponents))
    rows = _lattice_frohlich_metadata_value(metadata, "frohlich_weights")
    weights = Matrix{Float64}(undef, length(rows), config.site_count)
    for (row_index, row) in pairs(rows)
        length(row) == config.site_count || throw(ArgumentError("frohlich_weights row length must equal site_count"))
        weights[row_index, :] .= Float64.(row)
    end
    size(weights) == (config.site_count, config.site_count) || throw(ArgumentError("frohlich_weights shape must equal site_count squared"))
    _lattice_frohlich_metadata_approximate("frohlich_weights", weights, problem.frohlich_weights; rtol, atol)
    _lattice_frohlich_metadata_exact("hierarchy_sizes", _lattice_frohlich_metadata_integer_vector(metadata, "hierarchy_sizes"), problem.system.nb)
    dimensions = _lattice_frohlich_metadata_integer_vector(metadata, "tt_dimensions")
    _lattice_frohlich_metadata_exact("tt_dimensions", dimensions, heom_tt_dimensions(problem.system))
    _lattice_frohlich_metadata_exact("tt_dimensions", tt_dims(state), dimensions)
    return state
end

function write_lattice_frohlich_equilibration_csv(path::AbstractString, result;
                                                   overwrite::Bool=false)::String
    target = String(path)
    !overwrite && ispath(target) && throw(ArgumentError("target already exists: $target"))
    length(result.times) == size(result.populations, 2) == length(result.trace) == length(result.maximum_rank) == length(result.mean_rank) || throw(ArgumentError("equilibration result lengths are inconsistent"))
    temporary_path = nothing
    try
        temporary_path, io = mktemp(dirname(abspath(target)))
        try
            println(io, join(["time_fs", ["population_site_$site" for site in axes(result.populations, 1)]..., "trace", "max_rank", "mean_rank"], ","))
            for index in eachindex(result.times)
                println(io, join(Any[result.times[index], result.populations[:, index]..., result.trace[index], result.maximum_rank[index], result.mean_rank[index]], ","))
            end
        finally close(io) end
        !overwrite && ispath(target) && throw(ArgumentError("target already exists: $target"))
        if overwrite
            _lattice_frohlich_atomic_rename(temporary_path, target); temporary_path = nothing
        else
            Base.Filesystem.hardlink(temporary_path, target)
        end
        return target
    finally
        temporary_path === nothing || rm(temporary_path; force=true)
    end
end

function write_lattice_frohlich_equilibration_png(path::AbstractString, result, plotter;
                                                   overwrite::Bool=false)::String
    target = String(path)
    !overwrite && ispath(target) && throw(ArgumentError("target already exists: $target"))
    temporary_path = nothing
    try
        temporary_path, io = mktemp(dirname(abspath(target))); close(io)
        png_path = "$temporary_path.png"; mv(temporary_path, png_path); temporary_path = png_path
        rm(temporary_path); plotter(temporary_path, result)
        isfile(temporary_path) || throw(ArgumentError("plotter did not write a PNG file"))
        !overwrite && ispath(target) && throw(ArgumentError("target already exists: $target"))
        if overwrite
            _lattice_frohlich_atomic_rename(temporary_path, target); temporary_path = nothing
        else
            Base.Filesystem.hardlink(temporary_path, target)
        end
        return target
    finally
        temporary_path === nothing || rm(temporary_path; force=true)
    end
end

"""
    write_lattice_frohlich_current_csv(path, result; overwrite=false) -> String

Write the complex particle-current correlation as real and imaginary
inverse-femtosecond-squared components.
"""
function write_lattice_frohlich_current_csv(path::AbstractString, result;
                                             overwrite::Bool=false)::String
    target = String(path)
    !overwrite && ispath(target) && throw(ArgumentError("target already exists: $target"))
    row_count = length(result.times)
    length(result.correlation) == row_count ||
        throw(ArgumentError("correlation length must equal times length"))
    length(result.maximum_rank) == row_count ||
        throw(ArgumentError("maximum_rank length must equal times length"))
    length(result.mean_rank) == row_count ||
        throw(ArgumentError("mean_rank length must equal times length"))

    temporary_path = nothing
    try
        temporary_path, io = mktemp(dirname(abspath(target)))
        try
            println(io, "time_fs,correlation_real_fs^-2,correlation_imag_fs^-2,max_rank,mean_rank")
            for index in eachindex(result.times)
                correlation = result.correlation[index]
                println(io, join(Any[
                    result.times[index],
                    real(correlation),
                    imag(correlation),
                    result.maximum_rank[index],
                    result.mean_rank[index],
                ], ","))
            end
        finally
            close(io)
        end
        !overwrite && ispath(target) && throw(ArgumentError("target already exists: $target"))
        if overwrite
            _lattice_frohlich_atomic_rename(temporary_path, target)
            temporary_path = nothing
        else
            Base.Filesystem.hardlink(temporary_path, target)
        end
        return target
    finally
        temporary_path === nothing || rm(temporary_path; force=true)
    end
end
