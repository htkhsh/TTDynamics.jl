using LinearAlgebra
using Statistics
using Test
using TTDynamics
using TTSolver
using KaisouEOM

if !isdefined(@__MODULE__, :HolsteinConfig)
    include("../examples/holstein/utils.jl")
end
include("../examples/holstein_current_correlation/utils.jl")
include("../examples/holstein_current_correlation/equilibrate.jl")
include("../examples/holstein_current_correlation/current_correlation.jl")

@testset "Holstein equilibration plotting loads lazily" begin
    @test !isdefined(@__MODULE__, :_save_equilibration_population_plot)
end

@testset "Holstein current plotting loads lazily" begin
    @test !isdefined(@__MODULE__, :_save_current_correlation_plot)
    @test !isdefined(@__MODULE__, :_save_current_correlation_rank_plot)
end

@testset "Holstein current rank output path" begin
    paths = _current_correlation_output_paths("diagnostics")
    @test paths.rank_png_path ==
          joinpath("diagnostics", "holstein_current_correlation_ranks.png")
end

function current_correlation_problem(config)
    H = periodic_holstein_hamiltonian(config.site_energies_cm, config.hopping_cm) *
        KaisouEOM.icm2ifs
    baths = [
        BathExp(ComplexF64[0.4], ComplexF64[0.03 - 0.02im], projector)
        for projector in site_projectors(config.site_count)
    ]
    system = HEOMTTSystem(H, NoiseExp(baths), config.hierarchy_local_size)
    liouvillian, trace_observable, population_observables =
        build_heom_liouvillian(system; tol=config.operator_tolerance)
    return (; system, liouvillian, trace_observable, population_observables)
end

function unchecked_holstein_config(config; site_count=config.site_count,
                                   hopping_cm=config.hopping_cm,
                                   progress_interval=config.progress_interval)
    values = Any[getfield(config, index) for index in 1:fieldcount(HolsteinConfig)]
    values[1] = site_count
    values[3] = hopping_cm
    values[25] = progress_interval
    return HolsteinConfig(values...)
end

function metadata_error(f)
    try
        f()
    catch error
        @test error isa ArgumentError
        return error
    end
    error("expected ArgumentError")
end

function metadata_fixture()
    config = HolsteinConfig(
        site_count=2,
        site_energies_cm=[10.0, -5.0],
        hopping_cm=2.0,
        final_time_fs=1.0,
        hierarchy_local_size=2,
        operator_tolerance=1e-14,
    )
    problem = current_correlation_problem(config)
    decomposition = (
        exponents=ComplexF64[0.1 + 0.2im, 0.3 - 0.4im],
        coefficients=ComplexF64[0.5 - 0.6im, 0.7 + 0.8im],
    )
    state = build_initial_state(problem.system, config.initial_site; tol=1e-14)
    metadata = equilibrium_metadata(
        config,
        decomposition,
        problem,
        state;
        equilibration_time_fs=1000.0,
    )
    return (; config, decomposition, problem, state, metadata)
end

function equilibration_fixture()
    config = HolsteinConfig(
        site_count=2,
        site_energies_cm=zeros(2),
        hopping_cm=2.0,
        final_time_fs=1.0,
        time_step_fs=1.0,
        hierarchy_local_size=2,
        operator_tolerance=1e-14,
        state_rounding_tolerance=1e-14,
        tamen_tolerance=1e-12,
    )
    full_problem = current_correlation_problem(config)
    state = build_initial_state(full_problem.system, config.initial_site; tol=1e-14)
    zero_liouvillian = TTMatrix([
        zeros(ComplexF64, 1, dimension, dimension, 1)
        for dimension in tt_dims(state)
    ])
    problem = (
        system=full_problem.system,
        liouvillian=zero_liouvillian,
        trace_observable=full_problem.trace_observable,
        population_observables=full_problem.population_observables,
    )
    decomposition = (
        exponents=ComplexF64[0.1 + 0.2im],
        coefficients=ComplexF64[0.5 - 0.6im],
    )
    return (; config, decomposition, problem, state)
end

@testset "Holstein current correlation utilities" begin
    config = HolsteinConfig(
        site_count=3,
        site_energies_cm=zeros(3),
        hopping_cm=2.0,
        final_time_fs=1.0,
    )
    current = periodic_current_operator(config)
    scale = config.hopping_cm * KaisouEOM.icm2ifs

    @test ishermitian(current)
    @test current[2, 1] == 1im * scale
    @test current[1, 2] == -1im * scale
    @test current[1, 3] == 1im * scale
    @test current[3, 1] == -1im * scale
    @test diag(current) == zeros(3)

    config2 = HolsteinConfig(
        site_count=2,
        site_energies_cm=zeros(2),
        hopping_cm=2.0,
        final_time_fs=1.0,
    )
    current2 = periodic_current_operator(config2)
    @test current2 == ComplexF64[0 -1im * scale; 1im * scale 0]
    @test_throws ArgumentError periodic_current_operator(
        unchecked_holstein_config(config; site_count=1),
    )
    @test_throws ArgumentError periodic_current_operator(
        unchecked_holstein_config(config; hopping_cm=-1.0),
    )
    @test_throws ArgumentError periodic_current_operator(
        unchecked_holstein_config(config; hopping_cm=NaN),
    )
end

@testset "Holstein equilibrium metadata" begin
    fixture = metadata_fixture()
    (; config, decomposition, problem, state, metadata) = fixture

    @test metadata["identifier"] == "TTDynamics.HolsteinEquilibrium"
    @test metadata["version"] == 1
    @test metadata["heom_representation"] == "twin-space-v1"
    @test metadata["equilibration_time_fs"] == 1000.0
    for field in fieldnames(HolsteinConfig)
        expected = field === :pade_type ? string(config.pade_type) : getfield(config, field)
        @test metadata[String(field)] == expected
    end
    @test metadata["site_count"] == config.site_count
    @test metadata["site_energies_cm"] == config.site_energies_cm
    @test metadata["hopping_cm"] == config.hopping_cm
    @test metadata["brownian_frequency_cm"] == config.brownian_frequency_cm
    @test metadata["brownian_damping_cm"] == config.brownian_damping_cm
    @test metadata["reorganization_energy_cm"] == config.reorganization_energy_cm
    @test metadata["temperature_K"] == config.temperature_K
    @test metadata["time_step_fs"] == config.time_step_fs
    @test metadata["pade_order"] == config.pade_order
    @test metadata["pade_type"] == string(config.pade_type)
    @test metadata["tpsd_tolerance"] == config.tpsd_tolerance
    @test metadata["hierarchy_local_size"] == config.hierarchy_local_size
    @test metadata["exponents_real"] == real.(decomposition.exponents)
    @test metadata["exponents_imag"] == imag.(decomposition.exponents)
    @test metadata["coefficients_real"] == real.(decomposition.coefficients)
    @test metadata["coefficients_imag"] == imag.(decomposition.coefficients)
    @test metadata["hierarchy_sizes"] == problem.system.nb
    @test metadata["tt_dimensions"] == tt_dims(state)

    mktempdir() do directory
        path = joinpath(directory, "equilibrium.toml")
        @test write_equilibrium_metadata(path, metadata) == path
        @test read_equilibrium_metadata(path) == metadata
        @test_throws ArgumentError write_equilibrium_metadata(path, metadata)
        replacement = deepcopy(metadata)
        replacement["equilibration_time_fs"] = 500.0
        @test write_equilibrium_metadata(path, replacement; overwrite=true) == path
        @test read_equilibrium_metadata(path) == replacement
        missing_error = metadata_error(() -> read_equilibrium_metadata(
            joinpath(directory, "missing.toml"),
        ))
        @test occursin("metadata file", sprint(showerror, missing_error))
    end

    @test validate_equilibrium_state(state, metadata, config, decomposition, problem) === state

    for (field, replacement) in (
        "heom_representation" => "vectorized-v0",
        "hopping_cm" => 3.0,
        "exponents_imag" => [0.2, -0.3],
        "hierarchy_sizes" => [3, 2],
        "tt_dimensions" => [2, 2, 3, 2],
        "identifier" => "unsupported",
        "version" => 2,
        "coefficients_real" => [NaN, 0.7],
    )
        changed = deepcopy(metadata)
        changed[field] = replacement
        error = metadata_error(() -> validate_equilibrium_state(
            state,
            changed,
            config,
            decomposition,
            problem,
        ))
        @test occursin(field, sprint(showerror, error))
    end

    old_layout = TTTensor([
        reshape(ComplexF64[1, 0, 0, 0], 1, 4, 1),
        reshape(ComplexF64[1, 0], 1, 2, 1),
        reshape(ComplexF64[1, 0], 1, 2, 1),
    ])
    old_error = metadata_error(() -> validate_equilibrium_state(
        old_layout,
        metadata,
        config,
        decomposition,
        problem,
    ))
    @test occursin("old vectorized HEOM state is unsupported", sprint(showerror, old_error))

    mktempdir() do directory
        state_path = joinpath(directory, "equilibrium.ttbin")
        metadata_path = joinpath(directory, "equilibrium.toml")
        save_tt_binary(state_path, state)
        write_equilibrium_metadata(metadata_path, metadata)
        restored = load_tt_binary(state_path)
        @test validate_equilibrium_state(
            restored,
            read_equilibrium_metadata(metadata_path),
            config,
            decomposition,
            problem,
        ).cores == state.cores

        matrix = TTMatrix([
            reshape(ComplexF64[1], 1, 1, 1, 1),
        ])
        save_tt_binary(state_path, matrix; overwrite=true)
        error = metadata_error(() -> validate_equilibrium_state(
            load_tt_binary(state_path),
            metadata,
            config,
            decomposition,
            problem,
        ))
        @test occursin("TTTensor", sprint(showerror, error))
    end
end

@testset "Holstein current twin-space operators" begin
    config = HolsteinConfig(
        site_count=2,
        site_energies_cm=zeros(2),
        hopping_cm=2.0,
        final_time_fs=1.0,
        hierarchy_local_size=2,
        operator_tolerance=1e-14,
    )
    problem = current_correlation_problem(config)
    operators = build_current_heom_operators(config, problem)
    current = operators.current
    rho = ComplexF64[0.2 + 0.3im -0.1 + 0.4im; 0.7 - 0.2im 0.8 - 0.5im]
    h1 = ComplexF64[1 + 0.2im, -0.3 + 0.4im]
    h2 = ComplexF64[-0.4 + 0.7im, 0.9 - 0.1im]
    ket_core = reshape(Matrix{ComplexF64}(I, 2, 2), 1, 2, 2)
    bra_core = reshape(rho, 2, 2, 1)
    state = TTTensor([
        ket_core,
        bra_core,
        reshape(h1, 1, 2, 1),
        reshape(h2, 1, 2, 1),
    ])
    source = operators.left_action * state
    # `tt_full` stores the first TT mode as the final dense axis.  Its column-
    # major flattening therefore places bra and hierarchy indices before ket.
    expected = kron(vec(transpose(current * rho)), kron(h1, h2))
    @test vec(tt_full(source)) ≈ expected

    root_state = TTTensor([
        ket_core,
        bra_core,
        reshape(ComplexF64[1, 0], 1, 2, 1),
        reshape(ComplexF64[1, 0], 1, 2, 1),
    ])
    @test tt_dot(operators.observable, root_state) ≈ tr(current * rho)

    auxiliary_only_state = TTTensor([
        ket_core,
        bra_core,
        reshape(ComplexF64[0, 1], 1, 2, 1),
        reshape(ComplexF64[1, 0], 1, 2, 1),
    ])
    @test tt_dot(operators.observable, auxiliary_only_state) == 0
end

@testset "Holstein state propagation and measurements" begin
    config = HolsteinConfig(
        site_count=2,
        site_energies_cm=zeros(2),
        hopping_cm=2.0,
        final_time_fs=1.0,
        hierarchy_local_size=2,
        operator_tolerance=1e-14,
        state_rounding_tolerance=1e-14,
        tamen_tolerance=1e-12,
    )
    rho = TTTensor([
        reshape(Matrix{ComplexF64}(I, 2, 2), 1, 2, 2),
        reshape(ComplexF64[1 2im; -3im 4], 2, 2, 1),
    ])
    zero_liouvillian = TTMatrix([
        zeros(ComplexF64, 1, 2, 2, 1),
        reshape(Matrix{ComplexF64}(I, 2, 2), 1, 2, 2, 1),
    ])
    @test tt_full(propagate_cn_step(rho, zero_liouvillian, config)) ≈ tt_full(rho)
    trace_observable = TTTensor([
        reshape(ComplexF64[1, 0], 1, 2, 1),
        reshape(ComplexF64[1, 0], 1, 2, 1),
    ])
    @test tt_full(propagate_cn_step(
        rho,
        zero_liouvillian,
        config;
        trace_observable,
    )) ≈ tt_full(rho)

    traced_observations = propagate_fixed_steps(
        rho,
        (; liouvillian=zero_liouvillian, trace_observable),
        config,
        1;
        observe=(step, time, state) -> (; step, time, state),
    )
    @test tt_full(traced_observations.state) ≈ tt_full(rho)
    @test [observation.step for observation in traced_observations.observations] == [0, 1]

    observations = propagate_fixed_steps(
        rho,
        (; liouvillian=zero_liouvillian),
        config,
        1;
        observe=(step, time, state) -> (; step, time, norm=norm(tt_full(state))),
    )
    @test tt_full(observations.state) ≈ tt_full(rho)
    @test [observation.step for observation in observations.observations] == [0, 1]
    @test [observation.time for observation in observations.observations] == [0.0, config.time_step_fs]
    @test_throws ArgumentError propagate_fixed_steps(rho, (; liouvillian=zero_liouvillian), config, -1; observe=(args...) -> nothing)

    problem = current_correlation_problem(config)
    scaled = (2 - 3im) * build_initial_state(problem.system, 1; tol=1e-14)
    normalized = normalize_heom_state(scaled, problem.trace_observable)
    @test tt_dot(problem.trace_observable, normalized) ≈ 1
    @test tt_full(normalized) ≈ tt_full(scaled) / (2 - 3im)
    zero_state = TTTensor([
        zeros(ComplexF64, 1, 2, 1),
        zeros(ComplexF64, 1, 2, 1),
        zeros(ComplexF64, 1, 2, 1),
        zeros(ComplexF64, 1, 2, 1),
    ])
    @test_throws ArgumentError normalize_heom_state(zero_state, problem.trace_observable)

    measurement = measure_heom_state(
        build_initial_state(problem.system, 1; tol=1e-14),
        problem.trace_observable,
        problem.population_observables,
    )
    @test measurement.populations ≈ [1.0, 0.0]
    @test measurement.trace ≈ 1.0
    @test measurement.maximum_rank >= 1
    @test measurement.mean_rank >= 1
end

@testset "Holstein lazy builders are world-age safe" begin
    mktempdir() do directory
        fixture_directory = joinpath(directory, "fixture")
        holstein_directory = joinpath(directory, "holstein")
        mkpath(fixture_directory)
        mkpath(holstein_directory)

        equilibrate_source = joinpath(
            @__DIR__,
            "..",
            "examples",
            "holstein_current_correlation",
            "equilibrate.jl",
        )
        cp(equilibrate_source, joinpath(fixture_directory, "equilibrate.jl"))
        write(
            joinpath(holstein_directory, "holstein_brownian_heomtt.jl"),
            """
            function decompose_brownian_bcf(config)
                config isa HolsteinConfig || error("unexpected configuration")
                return :decomposition
            end

            function build_holstein_heomtt(config, decomposition)
                config isa HolsteinConfig || error("unexpected configuration")
                decomposition === :decomposition || error("unexpected decomposition")
                return :problem
            end
            """,
        )

        fixture_script = joinpath(fixture_directory, "world_age_fixture.jl")
        write(
            fixture_script,
            """
            pushfirst!(LOAD_PATH, $(repr(joinpath(@__DIR__, ".."))))

            struct HolsteinConfig end
            const DEFAULT_CONFIG = HolsteinConfig()
            run_equilibration(args...; kwargs...) = nothing

            include(joinpath(@__DIR__, "equilibrate.jl"))

            function invoke_default_builders(config)
                decomposition = _default_decomposition_builder(config)
                return _default_problem_builder(config, decomposition)
            end

            @assert invoke_default_builders(HolsteinConfig()) === :problem
            """,
        )
        command = `$(Base.julia_cmd()) --startup-file=no --depwarn=error $fixture_script`
        process = run(ignorestatus(command))
        @test success(process)
    end
end

@testset "Holstein lazy plotter is world-age safe" begin
    mktempdir() do directory
        fixture_directory = joinpath(directory, "fixture")
        mkpath(fixture_directory)

        equilibrate_source = joinpath(
            @__DIR__,
            "..",
            "examples",
            "holstein_current_correlation",
            "equilibrate.jl",
        )
        cp(equilibrate_source, joinpath(fixture_directory, "equilibrate.jl"))
        write(
            joinpath(fixture_directory, "plotting.jl"),
            """
            function _save_equilibration_population_plot(path, result)
                write(path, "stub plot")
                return path
            end
            """,
        )

        fixture_script = joinpath(fixture_directory, "world_age_plotter_fixture.jl")
        write(
            fixture_script,
            """
            pushfirst!(LOAD_PATH, $(repr(joinpath(@__DIR__, ".."))))

            struct HolsteinConfig end
            const DEFAULT_CONFIG = HolsteinConfig()
            run_equilibration(args...; kwargs...) = nothing

            include(joinpath(@__DIR__, "equilibrate.jl"))

            function invoke_default_plotter(path, result)
                return _default_equilibration_plotter(path, result)
            end

            mktempdir() do directory
                path = joinpath(directory, "population.png")
                @assert invoke_default_plotter(path, :result) == path
                @assert read(path, String) == "stub plot"
            end
            """,
        )
        command = `$(Base.julia_cmd()) --startup-file=no --depwarn=error $fixture_script`
        process = run(ignorestatus(command))
        @test success(process)
    end
end

@testset "Holstein fixed-time equilibration" begin
    fixture = equilibration_fixture()
    (; config, decomposition, problem, state) = fixture

    @test DEFAULT_CONFIG isa HolsteinConfig
    result = run_equilibration(
        config,
        problem;
        initial_state=state,
        equilibration_time_fs=config.time_step_fs,
    )
    @test result.times == [0.0, 1.0]
    @test size(result.populations) == (config.site_count, 2)
    @test length(result.trace) == length(result.times)
    @test length(result.maximum_rank) == length(result.times)
    @test length(result.mean_rank) == length(result.times)
    @test result.state isa TTTensor
    @test tt_dot(problem.trace_observable, result.state) ≈ 1
    @test_throws ArgumentError run_equilibration(
        config,
        problem;
        initial_state=state,
        equilibration_time_fs=1.5,
    )
    @test_throws ArgumentError run_equilibration(
        config,
        problem;
        initial_state=state,
        equilibration_time_fs=0.0,
    )
    @test_throws ArgumentError run_equilibration(
        config,
        problem;
        initial_state=state,
        equilibration_time_fs=eps(Float64) / 2,
    )

    mktempdir() do directory
        plot_calls = Any[]
        plotter = (path, result) -> begin
            push!(plot_calls, (path, result))
            write(path, "synthetic png")
            path
        end
        outputs = save_equilibration_outputs(
            config,
            decomposition,
            problem,
            result;
            output_directory=directory,
            equilibration_time_fs=config.time_step_fs,
            plotter,
        )
        @test isfile(outputs.state_path)
        @test isfile(outputs.metadata_path)
        @test isfile(outputs.csv_path)
        @test isfile(outputs.png_path)
        @test only(plot_calls)[2] === result
        @test basename(outputs.metadata_path) == "holstein_equilibrium_metadata.toml"
        @test basename(outputs.png_path) == "holstein_equilibration_populations.png"
        @test_throws ArgumentError write_equilibration_csv(outputs.csv_path, result)
        restored = load_tt_binary(outputs.state_path)
        @test validate_equilibrium_state(
            restored,
            read_equilibrium_metadata(outputs.metadata_path),
            config,
            decomposition,
            problem,
        ).cores == result.state.cores
        @test_throws ArgumentError save_equilibration_outputs(
            config,
            decomposition,
            problem,
            result;
            output_directory=directory,
            equilibration_time_fs=config.time_step_fs,
        )
    end

    metadata_write_failure = (args...; kwargs...) -> error("forced metadata publication failure")
    mktempdir() do directory
        paths = _equilibration_output_paths(directory)
        @test_throws ErrorException save_equilibration_outputs(
            config,
            decomposition,
            problem,
            result;
            output_directory=directory,
            equilibration_time_fs=config.time_step_fs,
            metadata_writer=metadata_write_failure,
        )
        @test !ispath(paths.csv_path)
        @test !ispath(paths.state_path)
        @test !ispath(paths.metadata_path)
        @test !ispath(paths.png_path)
    end

    metadata_write_after_write_failure = (path, args...; kwargs...) -> begin
        write(path, "partial metadata")
        error("forced metadata publication failure after write")
    end
    mktempdir() do directory
        paths = _equilibration_output_paths(directory)
        @test_throws ErrorException save_equilibration_outputs(
            config,
            decomposition,
            problem,
            result;
            output_directory=directory,
            equilibration_time_fs=config.time_step_fs,
            metadata_writer=metadata_write_after_write_failure,
        )
        @test !ispath(paths.csv_path)
        @test !ispath(paths.state_path)
        @test !ispath(paths.metadata_path)
        @test !ispath(paths.png_path)
    end

    mktempdir() do directory
        paths = _equilibration_output_paths(directory)
        write(paths.png_path, "existing png")
        @test_throws ArgumentError save_equilibration_outputs(
            config,
            decomposition,
            problem,
            result;
            output_directory=directory,
            equilibration_time_fs=config.time_step_fs,
        )
        @test read(paths.png_path, String) == "existing png"
    end

    plotting_failure = (path, _) -> begin
        write(path, "partial png")
        error("forced plotting failure")
    end
    mktempdir() do directory
        paths = _equilibration_output_paths(directory)
        @test_throws ErrorException save_equilibration_outputs(
            config,
            decomposition,
            problem,
            result;
            output_directory=directory,
            equilibration_time_fs=config.time_step_fs,
            plotter=plotting_failure,
        )
        @test !ispath(paths.csv_path)
        @test !ispath(paths.state_path)
        @test !ispath(paths.metadata_path)
        @test !ispath(paths.png_path)
    end

    mktempdir() do directory
        paths = _equilibration_output_paths(directory)
        save_tt_binary(paths.state_path, state)
        @test_throws ErrorException save_equilibration_outputs(
            config,
            decomposition,
            problem,
            result;
            output_directory=directory,
            equilibration_time_fs=config.time_step_fs,
            overwrite=true,
            metadata_writer=metadata_write_failure,
        )
        @test isfile(paths.state_path)
        @test load_tt_binary(paths.state_path).cores == result.state.cores
    end

    mktempdir() do directory
        main_plot_calls = Any[]
        main_plotter = (path, result) -> begin
            push!(main_plot_calls, (path, result))
            write(path, "main synthetic png")
            path
        end
        main_result = equilibrate_main(
            config;
            equilibration_time_fs=config.time_step_fs,
            output_directory=directory,
            decomposition_builder=_ -> decomposition,
            problem_builder=(_, _) -> problem,
            equilibration_runner=(args...; kwargs...) -> result,
            plotter=main_plotter,
        )
        @test isfile(main_result.state_path)
        @test isfile(main_result.metadata_path)
        @test isfile(main_result.csv_path)
        @test isfile(main_result.png_path)
        @test only(main_plot_calls)[2] === result
    end
end

@testset "Holstein reloaded current correlation" begin
    fixture = equilibration_fixture()
    (; config, problem, state) = fixture
    operators = build_current_heom_operators(config, problem)

    result = run_current_correlation(
        state,
        config,
        problem;
        correlation_time_fs=0.0,
    )
    source = operators.left_action * state
    expected = tt_dot(operators.observable, source)
    @test result.times == [0.0]
    @test result.correlation == ComplexF64[expected]
    @test result.maximum_rank == [maximum(tt_ranks(result.state))]
    @test result.mean_rank == [mean(tt_ranks(result.state))]
    @test_throws ArgumentError run_current_correlation(
        state,
        config,
        problem;
        correlation_time_fs=0.5,
    )

    progress_config = unchecked_holstein_config(config; progress_interval=2)
    progress_records = NamedTuple[]
    progress_result = run_current_correlation(
        state,
        progress_config,
        problem;
        correlation_time_fs=5.0,
        progress_callback=record -> push!(progress_records, record),
    )
    @test [record.step for record in progress_records] == [0, 2, 4, 5]
    @test [record.step_count for record in progress_records] == fill(5, 4)
    @test [record.time_fs for record in progress_records] == [0.0, 2.0, 4.0, 5.0]
    for record in progress_records
        result_index = record.step + 1
        @test record.maximum_rank == progress_result.maximum_rank[result_index]
        @test record.mean_rank == progress_result.mean_rank[result_index]
        @test isfinite(record.elapsed_seconds)
        @test record.elapsed_seconds >= 0
    end

    progress_io = IOBuffer()
    _print_current_correlation_progress(progress_io, progress_records[2])
    progress_message = String(take!(progress_io))
    @test occursin("Step 2/5", progress_message)
    @test occursin("time=2.0 fs", progress_message)
    @test occursin("max rank=$(progress_records[2].maximum_rank)", progress_message)
    @test occursin("mean rank=$(progress_records[2].mean_rank)", progress_message)
    @test occursin("elapsed=", progress_message)

    rank_io = IOBuffer()
    _print_tt_rank_vector(rank_io, "Loaded equilibrium TT ranks", state)
    @test String(take!(rank_io)) ==
          "  Loaded equilibrium TT ranks: $(tt_ranks(state))\n"

    synthetic_result = (
        times=[0.0, 1.0],
        correlation=ComplexF64[1.5 - 2.0im, -3.0 + 4.25im],
        maximum_rank=[2, 3],
        mean_rank=[1.5, 2.0],
    )
    mktempdir() do directory
        path = joinpath(directory, "current_correlation.csv")
        @test write_current_correlation_csv(path, synthetic_result) == path
        rows = readlines(path)
        @test rows[1] == "time_fs,correlation_real_fs^-2,correlation_imag_fs^-2,max_rank,mean_rank"
        @test rows[2] == "0.0,1.5,-2.0,2,1.5"
        @test rows[3] == "1.0,-3.0,4.25,3,2.0"
        @test_throws ArgumentError write_current_correlation_csv(path, synthetic_result)
    end

    mktempdir() do directory
        paths = _current_correlation_output_paths(directory)
        correlation_calls = Any[]
        rank_calls = Any[]
        correlation_plotter = (path, value) -> begin
            push!(correlation_calls, value)
            write(path, "correlation png")
        end
        rank_plotter = (path, value) -> begin
            push!(rank_calls, value)
            write(path, "rank png")
        end
        outputs = save_current_correlation_outputs(
            paths,
            synthetic_result;
            correlation_plotter,
            rank_plotter,
        )
        @test outputs.csv_path == paths.csv_path
        @test outputs.png_path == paths.png_path
        @test outputs.rank_png_path == paths.rank_png_path
        @test only(correlation_calls) === synthetic_result
        @test only(rank_calls) === synthetic_result
        @test read(paths.png_path, String) == "correlation png"
        @test read(paths.rank_png_path, String) == "rank png"
    end

    mktempdir() do directory
        paths = _current_correlation_output_paths(directory)
        failing_rank_plotter = (path, _) -> begin
            write(path, "partial rank png")
            error("injected rank plot failure")
        end
        @test_throws ErrorException save_current_correlation_outputs(
            paths,
            synthetic_result;
            correlation_plotter=(path, _) -> write(path, "correlation png"),
            rank_plotter=failing_rank_plotter,
        )
        @test !ispath(paths.csv_path)
        @test !ispath(paths.png_path)
        @test !ispath(paths.rank_png_path)
    end

    mktempdir() do directory
        paths = _current_correlation_output_paths(directory)
        write(paths.rank_png_path, "existing rank png")
        @test_throws ArgumentError save_current_correlation_outputs(
            paths,
            synthetic_result;
            correlation_plotter=(path, _) -> write(path, "correlation png"),
            rank_plotter=(path, _) -> write(path, "rank png"),
        )
        @test !ispath(paths.csv_path)
        @test !ispath(paths.png_path)
        @test read(paths.rank_png_path, String) == "existing rank png"
    end

    mktempdir() do directory
        target = joinpath(directory, "current_correlation.png")
        result = (times=[0.0], correlation=ComplexF64[1.0 + 2.0im])
        writer = (path, _) -> write(path, "first png")
        @test write_plot_png(target, result, writer) == target
        @test read(target, String) == "first png"
        @test_throws ArgumentError write_plot_png(target, result, writer)

        rm(target)
        no_output_plotter = (path, _) -> nothing
        @test_throws ArgumentError write_plot_png(target, result, no_output_plotter)
        @test !ispath(target)

        racing_writer = (path, _) -> begin
            write(target, "concurrent png")
            write(path, "temporary png")
        end
        @test_throws ArgumentError write_plot_png(target, result, racing_writer)
        @test read(target, String) == "concurrent png"
        replacing_writer = (path, _) -> write(path, "replacement png")
        @test write_plot_png(
            target,
            result,
            replacing_writer;
            overwrite=true,
        ) == target
        @test read(target, String) == "replacement png"
    end
end

@testset "Holstein current-correlation executable guard" begin
    example_directory = joinpath(@__DIR__, "..", "examples", "holstein_current_correlation")
    manifest_path = joinpath(example_directory, "Manifest.toml")
    if isfile(manifest_path)
        correlation_script = repr(joinpath(example_directory, "current_correlation.jl"))
        output_directory = repr(joinpath(example_directory, "output"))
        command = `$(Base.julia_cmd()) --project=$example_directory -e $(
            "include($correlation_script); " *
            "@assert DEFAULT_CORRELATION_TIME_FS == 200.0; " *
            "@assert !ispath($output_directory)"
        )`
        @test success(command)
    else
        @test_skip "example environment has not been instantiated"
    end
end
