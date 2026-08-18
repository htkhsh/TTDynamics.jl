using LinearAlgebra
using Statistics
# This helper is included by the package test suite, whose isolated project does
# not declare example-only stdlibs. Resolve TOML by its standard-library UUID so
# it remains an example dependency rather than a TTDynamics package dependency.
if !isdefined(@__MODULE__, :TOML)
    @eval const TOML = Base.require(Base.PkgId(
        Base.UUID("fa267f1f-6049-4f14-aa54-33bafae1ed76"),
        "TOML",
    ))
end
using .TOML
using TTDynamics
using TTSolver
using KaisouEOM: icm2ifs

const _EQUILIBRIUM_METADATA_IDENTIFIER = "TTDynamics.HolsteinEquilibrium"
const _EQUILIBRIUM_METADATA_VERSION = 1
const _EQUILIBRIUM_HEOM_REPRESENTATION = "twin-space-v1"

const _EQUILIBRIUM_CONFIG_INTEGER_FIELDS = (
    :site_count,
    :initial_site,
    :pade_order,
    :validation_sample_count,
    :hierarchy_local_size,
    :temporal_basis_size,
    :sweep_count,
    :local_iterations,
    :kick_rank,
    :progress_interval,
)
const _EQUILIBRIUM_CONFIG_FLOAT_FIELDS = (
    :hopping_cm,
    :brownian_frequency_cm,
    :brownian_damping_cm,
    :reorganization_energy_cm,
    :temperature_K,
    :final_time_fs,
    :time_step_fs,
    :tpsd_tolerance,
    :validation_final_time_fs,
    :bcf_upper_bound_cm,
    :tamen_tolerance,
    :operator_tolerance,
    :state_rounding_tolerance,
)

_metadata_field_name(field::Symbol) = String(field)

function _metadata_value(metadata, field::AbstractString)
    haskey(metadata, field) || throw(ArgumentError("$field is missing from equilibrium metadata"))
    return metadata[field]
end

function _metadata_string(metadata, field::AbstractString)
    value = _metadata_value(metadata, field)
    value isa AbstractString || throw(ArgumentError("$field must be a string"))
    return String(value)
end

function _metadata_integer(metadata, field::AbstractString)
    value = _metadata_value(metadata, field)
    value isa Integer && !(value isa Bool) ||
        throw(ArgumentError("$field must be an integer"))
    return Int(value)
end

function _metadata_float(metadata, field::AbstractString)
    value = _metadata_value(metadata, field)
    value isa Real && !(value isa Bool) && isfinite(value) ||
        throw(ArgumentError("$field must be a finite real number"))
    return Float64(value)
end

function _metadata_float_vector(metadata, field::AbstractString)
    value = _metadata_value(metadata, field)
    value isa AbstractVector || throw(ArgumentError("$field must be an array"))
    all(element -> element isa Real && !(element isa Bool) && isfinite(element), value) ||
        throw(ArgumentError("$field must contain finite real numbers"))
    return Float64.(value)
end

function _metadata_integer_vector(metadata, field::AbstractString)
    value = _metadata_value(metadata, field)
    value isa AbstractVector || throw(ArgumentError("$field must be an array"))
    all(element -> element isa Integer && !(element isa Bool), value) ||
        throw(ArgumentError("$field must contain integers"))
    return Int.(value)
end

function _validate_toml_value(value, field::AbstractString)
    if value isa AbstractString || value isa Bool ||
       (value isa Integer && !(value isa Bool))
        return nothing
    elseif value isa AbstractFloat
        isfinite(value) || throw(ArgumentError("$field must be finite"))
        return nothing
    elseif value isa AbstractVector
        for (index, element) in pairs(value)
            _validate_toml_value(element, "$field[$index]")
        end
        return nothing
    elseif value isa AbstractDict
        for (key, element) in pairs(value)
            key isa AbstractString || throw(ArgumentError("$field dictionary keys must be strings"))
            _validate_toml_value(element, "$field.$key")
        end
        return nothing
    end
    throw(ArgumentError("$field is not TOML-safe metadata"))
end

function _validate_equilibrium_metadata(metadata)
    metadata isa AbstractDict || throw(ArgumentError("metadata must be a dictionary"))
    for (field, value) in pairs(metadata)
        field isa AbstractString || throw(ArgumentError("metadata keys must be strings"))
        _validate_toml_value(value, String(field))
    end

    _metadata_string(metadata, "identifier") == _EQUILIBRIUM_METADATA_IDENTIFIER ||
        throw(ArgumentError("identifier is unsupported"))
    _metadata_integer(metadata, "version") == _EQUILIBRIUM_METADATA_VERSION ||
        throw(ArgumentError("version is unsupported"))
    _metadata_string(metadata, "heom_representation") == _EQUILIBRIUM_HEOM_REPRESENTATION ||
        throw(ArgumentError("heom_representation is unsupported"))
    _metadata_float(metadata, "equilibration_time_fs")

    _metadata_integer(metadata, "site_count")
    _metadata_float_vector(metadata, "site_energies_cm")
    for field in _EQUILIBRIUM_CONFIG_FLOAT_FIELDS
        _metadata_float(metadata, _metadata_field_name(field))
    end
    for field in _EQUILIBRIUM_CONFIG_INTEGER_FIELDS
        _metadata_integer(metadata, _metadata_field_name(field))
    end
    _metadata_string(metadata, "pade_type")

    exponent_real = _metadata_float_vector(metadata, "exponents_real")
    exponent_imag = _metadata_float_vector(metadata, "exponents_imag")
    coefficient_real = _metadata_float_vector(metadata, "coefficients_real")
    coefficient_imag = _metadata_float_vector(metadata, "coefficients_imag")
    length(exponent_real) == length(exponent_imag) ||
        throw(ArgumentError("exponents_imag length must equal exponents_real length"))
    length(coefficient_real) == length(coefficient_imag) ||
        throw(ArgumentError("coefficients_imag length must equal coefficients_real length"))
    length(exponent_real) == length(coefficient_real) ||
        throw(ArgumentError("coefficients_real length must equal exponents_real length"))
    _metadata_integer(metadata, "tpsd_term_count") == length(exponent_real) ||
        throw(ArgumentError("tpsd_term_count must equal the decomposition array length"))
    _metadata_integer_vector(metadata, "hierarchy_sizes")
    _metadata_integer_vector(metadata, "tt_dimensions")
    return nothing
end

function _equilibrium_atomic_rename(oldpath::AbstractString, newpath::AbstractString)
    old = String(oldpath)
    new = String(newpath)
    status = ccall(:jl_fs_rename, Int32, (Cstring, Cstring), old, new)
    status < 0 && Base.uv_error("rename($(repr(old)), $(repr(new)))", status)
    return nothing
end

"""
    equilibrium_metadata(config, decomposition, problem, state;
                         equilibration_time_fs) -> Dict{String,Any}

Describe a saved twin-space Holstein HEOM equilibrium state using TOML-safe
values. Complex TPSD data is represented by parallel real and imaginary arrays.
"""
function equilibrium_metadata(config::HolsteinConfig, decomposition, problem,
                              state::TTTensor; equilibration_time_fs)::Dict{String,Any}
    metadata = Dict{String,Any}(
        "identifier" => _EQUILIBRIUM_METADATA_IDENTIFIER,
        "version" => _EQUILIBRIUM_METADATA_VERSION,
        "heom_representation" => _EQUILIBRIUM_HEOM_REPRESENTATION,
        "equilibration_time_fs" => Float64(equilibration_time_fs),
        "site_count" => config.site_count,
        "site_energies_cm" => copy(config.site_energies_cm),
        "pade_type" => string(config.pade_type),
        "exponents_real" => real.(decomposition.exponents),
        "exponents_imag" => imag.(decomposition.exponents),
        "coefficients_real" => real.(decomposition.coefficients),
        "coefficients_imag" => imag.(decomposition.coefficients),
        "tpsd_term_count" => length(decomposition.exponents),
        "hierarchy_sizes" => copy(problem.system.nb),
        "tt_dimensions" => tt_dims(state),
    )
    for field in _EQUILIBRIUM_CONFIG_FLOAT_FIELDS
        metadata[_metadata_field_name(field)] = getfield(config, field)
    end
    for field in _EQUILIBRIUM_CONFIG_INTEGER_FIELDS
        metadata[_metadata_field_name(field)] = getfield(config, field)
    end
    _validate_equilibrium_metadata(metadata)
    return metadata
end

"""
    write_equilibrium_metadata(path, metadata; overwrite=false) -> String

Atomically publish TOML metadata after writing a sibling temporary file.
"""
function write_equilibrium_metadata(path::AbstractString, metadata; overwrite::Bool=false)::String
    target = String(path)
    _validate_equilibrium_metadata(metadata)
    !overwrite && ispath(target) && throw(ArgumentError("target already exists: $target"))

    temporary_path = nothing
    try
        temporary_path, io = mktemp(dirname(abspath(target)))
        try
            TOML.print(io, metadata)
        finally
            close(io)
        end
        !overwrite && ispath(target) && throw(ArgumentError("target already exists: $target"))
        if overwrite
            _equilibrium_atomic_rename(temporary_path, target)
            temporary_path = nothing
        else
            Base.Filesystem.hardlink(temporary_path, target)
        end
        return target
    finally
        temporary_path === nothing || rm(temporary_path; force=true)
    end
end

"""
    read_equilibrium_metadata(path) -> Dict{String,Any}

Read a TOML equilibrium-state metadata sidecar without changing its contents.
Missing metadata paths are reported as `ArgumentError` so callers receive the
same application-level validation failure as malformed metadata.
"""
function read_equilibrium_metadata(path::AbstractString)::Dict{String,Any}
    source = String(path)
    isfile(source) || throw(ArgumentError("metadata file does not exist: $source"))
    return TOML.parsefile(source)
end

function _require_exact_metadata(field::AbstractString, actual, expected)
    actual == expected || throw(ArgumentError("$field does not match the reconstructed problem"))
    return nothing
end

function _require_approximate_metadata(field::AbstractString, actual, expected;
                                       rtol::Real, atol::Real)
    isapprox(actual, expected; rtol, atol) ||
        throw(ArgumentError("$field does not match the reconstructed problem"))
    return nothing
end

"""
    validate_equilibrium_state(state, metadata, config, decomposition, problem;
                               rtol=1e-12, atol=1e-14) -> TTTensor

Reject metadata or a TT state that is incompatible with the reconstructed
twin-space Holstein HEOM problem.
"""
function validate_equilibrium_state(state, metadata, config::HolsteinConfig,
                                    decomposition, problem;
                                    rtol::Real=1e-12, atol::Real=1e-14)::TTTensor
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative"))
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    state isa TTTensor || throw(ArgumentError("equilibrium state must be a TTTensor"))
    root_density_matrix(state, problem.system)
    _validate_equilibrium_metadata(metadata)

    _require_exact_metadata("site_count", _metadata_integer(metadata, "site_count"), config.site_count)
    _require_approximate_metadata(
        "site_energies_cm",
        _metadata_float_vector(metadata, "site_energies_cm"),
        config.site_energies_cm;
        rtol,
        atol,
    )
    for field in _EQUILIBRIUM_CONFIG_FLOAT_FIELDS
        name = _metadata_field_name(field)
        _require_approximate_metadata(name, _metadata_float(metadata, name), getfield(config, field); rtol, atol)
    end
    for field in _EQUILIBRIUM_CONFIG_INTEGER_FIELDS
        name = _metadata_field_name(field)
        _require_exact_metadata(name, _metadata_integer(metadata, name), getfield(config, field))
    end
    _require_exact_metadata("pade_type", _metadata_string(metadata, "pade_type"), string(config.pade_type))

    metadata_exponents_real = _metadata_float_vector(metadata, "exponents_real")
    metadata_exponents_imag = _metadata_float_vector(metadata, "exponents_imag")
    metadata_coefficients_real = _metadata_float_vector(metadata, "coefficients_real")
    metadata_coefficients_imag = _metadata_float_vector(metadata, "coefficients_imag")
    metadata_exponents = complex.(metadata_exponents_real, metadata_exponents_imag)
    metadata_coefficients = complex.(metadata_coefficients_real, metadata_coefficients_imag)
    if !isapprox(metadata_exponents, decomposition.exponents; rtol, atol)
        _require_approximate_metadata(
            "exponents_real",
            metadata_exponents_real,
            real.(decomposition.exponents);
            rtol,
            atol,
        )
        _require_approximate_metadata(
            "exponents_imag",
            metadata_exponents_imag,
            imag.(decomposition.exponents);
            rtol,
            atol,
        )
        throw(ArgumentError("exponents do not match the reconstructed problem"))
    end
    if !isapprox(metadata_coefficients, decomposition.coefficients; rtol, atol)
        _require_approximate_metadata(
            "coefficients_real",
            metadata_coefficients_real,
            real.(decomposition.coefficients);
            rtol,
            atol,
        )
        _require_approximate_metadata(
            "coefficients_imag",
            metadata_coefficients_imag,
            imag.(decomposition.coefficients);
            rtol,
            atol,
        )
        throw(ArgumentError("coefficients do not match the reconstructed problem"))
    end
    _require_exact_metadata(
        "tpsd_term_count",
        _metadata_integer(metadata, "tpsd_term_count"),
        length(decomposition.exponents),
    )
    _require_exact_metadata(
        "hierarchy_sizes",
        _metadata_integer_vector(metadata, "hierarchy_sizes"),
        problem.system.nb,
    )
    expected_dimensions = heom_tt_dimensions(problem.system)
    metadata_dimensions = _metadata_integer_vector(metadata, "tt_dimensions")
    _require_exact_metadata("tt_dimensions", metadata_dimensions, expected_dimensions)
    _require_exact_metadata("tt_dimensions", tt_dims(state), metadata_dimensions)
    return state
end

"""
    periodic_current_operator(config) -> Matrix{ComplexF64}

Return the Hermitian particle-current operator for the periodic Holstein chain.
"""
function periodic_current_operator(config::HolsteinConfig)::Matrix{ComplexF64}
    N = config.site_count
    N >= 2 || throw(ArgumentError("site_count must be at least two"))
    isfinite(config.hopping_cm) || throw(ArgumentError("hopping_cm must be finite"))
    config.hopping_cm >= 0 || throw(ArgumentError("hopping_cm must be nonnegative"))

    scale = config.hopping_cm * icm2ifs
    current = zeros(ComplexF64, N, N)
    for site in 1:(N - 1)
        current[site + 1, site] += 1im * scale
        current[site, site + 1] -= 1im * scale
    end
    if N > 2
        current[1, N] += 1im * scale
        current[N, 1] -= 1im * scale
    end
    return current
end

"""
    build_current_heom_operators(config, problem)

Embed the periodic current as a ket-side HEOM-TT action and as a root-space
observable in the canonical `[ket, bra, hierarchy...]` layout.
"""
function build_current_heom_operators(config::HolsteinConfig, problem)
    current = periodic_current_operator(config)
    N = config.site_count
    I_sys = Matrix{ComplexF64}(I, N, N)
    hierarchy_identities = [Matrix{ComplexF64}(I, n, n) for n in problem.system.nb]
    local_matrices = [current, I_sys, hierarchy_identities...]
    left_action = TTMatrix([
        reshape(ComplexF64.(matrix), 1, size(matrix, 1), size(matrix, 2), 1)
        for matrix in local_matrices
    ])
    left_action = tt_round(left_action, config.operator_tolerance)

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

    liouvillian_dimensions = tt_dims(problem.liouvillian)
    action_dimensions = tt_dims(left_action)
    action_dimensions[1] == liouvillian_dimensions[1] ||
        throw(ArgumentError("current left-action output dimensions must match the Liouvillian"))
    action_dimensions[2] == liouvillian_dimensions[2] ||
        throw(ArgumentError("current left-action input dimensions must match the Liouvillian"))
    tt_dims(observable) == heom_tt_dimensions(problem.system) ||
        throw(ArgumentError("current observable dimensions must match the HEOM system"))

    return (; current, left_action, observable)
end

"""
    propagate_cn_step(rho, liouvillian, config) -> TTTensor

Advance one Crank-Nicolson tAMEn step with the Holstein solver settings.
"""
function propagate_cn_step(rho::TTTensor, liouvillian::TTMatrix, config::HolsteinConfig;
                           trace_observable=nothing)::TTTensor
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
        rho,
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
    propagate_fixed_steps(rho, problem, config, step_count; observe)

Run a fixed number of Crank-Nicolson steps, returning the final state together
with arbitrary values returned by `observe(step, time, state)`.
"""
function propagate_fixed_steps(rho::TTTensor, problem, config::HolsteinConfig,
                               step_count::Integer; observe)
    step_count >= 0 || throw(ArgumentError("step_count must be nonnegative"))
    observations = [observe(0, 0.0, rho)]
    state = rho
    trace_observable = hasproperty(problem, :trace_observable) ?
                       problem.trace_observable : nothing
    for step in 1:step_count
        state = propagate_cn_step(
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
    normalize_heom_state(rho, trace_observable; atol=1e-14) -> TTTensor

Scale every hierarchy component by the complex root trace.
"""
function normalize_heom_state(rho::TTTensor, trace_observable::TTTensor;
                              atol::Real=1e-14)::TTTensor
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    trace_value = tt_dot(trace_observable, rho)
    isfinite(real(trace_value)) && isfinite(imag(trace_value)) ||
        throw(ArgumentError("HEOM trace must be finite"))
    abs(trace_value) > atol || throw(ArgumentError("HEOM trace magnitude must exceed atol"))

    normalized = inv(trace_value) * rho
    normalized_trace = tt_dot(trace_observable, normalized)
    isapprox(normalized_trace, 1; atol=atol) ||
        throw(ArgumentError("normalized HEOM trace is not approximately one"))
    return normalized
end

"""
    measure_heom_state(rho, trace_observable, population_observables)

Return root-space population, trace, and TT-rank diagnostics without relying
on the executable Holstein example.
"""
function measure_heom_state(rho::TTTensor, trace_observable::TTTensor,
                            population_observables)
    populations = real.([tt_dot(observable, rho) for observable in population_observables])
    trace_value = real(tt_dot(trace_observable, rho))
    ranks = tt_ranks(rho)
    return (;
        populations,
        trace=trace_value,
        maximum_rank=maximum(ranks),
        mean_rank=mean(ranks),
    )
end
