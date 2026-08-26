# Periodic Holstein model with independent Brownian baths in HEOM-TT form.
#
# This intentionally keeps the complete example in one file: physical and
# numerical parameters, model construction, bath decomposition, propagation,
# progress reporting, and result publication.

using LinearAlgebra
using Statistics
using TTSolver
using TTDynamics
using QFiND
using HEOMKit
using HEOMKit: icm2ifs
using CairoMakie

function simple_holstein_hamiltonian(site_energies::AbstractVector,
                                     hopping::Real)
    site_count = length(site_energies)
    site_count >= 2 || throw(ArgumentError("site_energies must contain at least two sites"))
    hopping >= 0 || throw(ArgumentError("hopping must be nonnegative"))

    hamiltonian = Matrix(Diagonal(ComplexF64.(site_energies)))
    for site in 1:(site_count - 1)
        hamiltonian[site, site + 1] = -hopping
        hamiltonian[site + 1, site] = -hopping
    end
    if site_count > 2
        hamiltonian[site_count, 1] = -hopping
        hamiltonian[1, site_count] = -hopping
    end
    return hamiltonian
end

function simple_holstein_site_projectors(site_count::Integer)
    site_count >= 2 || throw(ArgumentError("site_count must be at least two"))
    return [
        Matrix(Diagonal(ComplexF64[site == index for index in 1:site_count]))
        for site in 1:site_count
    ]
end

simple_holstein_output_paths(directory::AbstractString) = (
    csv=joinpath(directory, "holstein_brownian_populations.csv"),
    populations=joinpath(directory, "holstein_brownian_populations.png"),
    trace=joinpath(directory, "holstein_brownian_trace.png"),
    rank=joinpath(directory, "holstein_brownian_rank.png"),
)

function write_simple_holstein_csv(path::AbstractString, times, populations,
                                    trace_values, maximum_rank, mean_rank)
    site_count = size(populations, 1)
    header = join([
        "time_fs",
        ["population_site_$site" for site in 1:site_count]...,
        "trace",
        "max_rank",
        "mean_rank",
    ], ",")
    open(path, "w") do io
        println(io, header)
        for index in eachindex(times)
            println(io, join(Any[
                times[index], populations[:, index]..., trace_values[index],
                maximum_rank[index], mean_rank[index],
            ], ","))
        end
    end
    return String(path)
end

function save_simple_holstein_plots(paths, times, populations, trace_values,
                                    maximum_rank, mean_rank)
    population_figure = Figure(size=(900, 600))
    population_axis = Axis(
        population_figure[1, 1];
        xlabel="Time [fs]",
        ylabel="Population",
        title="Periodic Holstein HEOM-TT populations",
    )
    for site in axes(populations, 1)
        lines!(population_axis, times, populations[site, :];
               linewidth=2, label="site $site")
    end
    axislegend(population_axis; position=:rt)
    save(paths.populations, population_figure)

    trace_figure = Figure(size=(700, 450))
    trace_axis = Axis(trace_figure[1, 1];
                      xlabel="Time [fs]", ylabel="Tr[ρ]",
                      title="HEOM-TT trace conservation")
    lines!(trace_axis, times, trace_values; linewidth=2, label="trace")
    hlines!(trace_axis, [1.0]; linewidth=1, linestyle=:dash, color=:gray)
    axislegend(trace_axis; position=:rt)
    save(paths.trace, trace_figure)

    rank_figure = Figure(size=(700, 450))
    rank_axis = Axis(rank_figure[1, 1];
                     xlabel="Time [fs]", ylabel="TT rank",
                     title="HEOM-TT rank evolution")
    lines!(rank_axis, times, maximum_rank; linewidth=2, label="maximum rank")
    lines!(rank_axis, times, mean_rank; linewidth=2, label="mean rank")
    axislegend(rank_axis; position=:rt)
    save(paths.rank, rank_figure)
    return nothing
end

function main(;
    # Periodic Holstein system, in cm⁻¹ unless stated otherwise.
    site_count=5,
    site_energies_cm=zeros(site_count),
    hopping_cm=400.0,
    initial_site=1,
    # Underdamped Brownian bath and temperature.
    brownian_frequency_cm=1400.0,
    brownian_damping_cm=200.0,
    reorganization_energy_cm=600.0,
    temperature_K=300.0,
    # TPSD bath-correlation decomposition.
    pade_order=8,
    tpsd_tolerance=2e-2,
    pade_type=:Nm1,
    validation_final_time_fs=100.0,
    validation_sample_count=200,
    bcf_upper_bound_cm=10_000.0,
    # HEOM-TT and tAMEn propagation.
    hierarchy_local_size=4,
    final_time_fs=500.0,
    time_step_fs=1.0,
    temporal_basis_size=3,
    tamen_tolerance=2e-2,
    operator_tolerance=1e-10,
    state_rounding_tolerance=1e-10,
    sweep_count=3,
    local_iterations=5,
    kick_rank=4,
    progress_interval=10,
    output_directory=@__DIR__,
)
    length(site_energies_cm) == site_count ||
        throw(ArgumentError("site_energies_cm length must equal site_count"))
    1 <= initial_site <= site_count ||
        throw(ArgumentError("initial_site must index a site"))
    temporal_basis_size >= 3 && isodd(temporal_basis_size) ||
        throw(ArgumentError("temporal_basis_size must be odd and at least three"))
    step_ratio = final_time_fs / time_step_fs
    isapprox(step_ratio, round(step_ratio); atol=1e-12, rtol=1e-12) ||
        throw(ArgumentError("final_time_fs must be an integer multiple of time_step_fs"))

    println("="^68)
    println("Periodic Holstein model with independent Brownian baths (HEOM-TT)")
    println("="^68)
    println("Sites=$site_count, J=$hopping_cm cm⁻¹, T=$temperature_K K")

    # 1. Build the periodic system Hamiltonian.
    hamiltonian = simple_holstein_hamiltonian(
        site_energies_cm,
        hopping_cm,
    ) * icm2ifs

    # 2. Decompose the Brownian bath correlation function with TPSD.
    spectral_density = BrownianSD(
        brownian_frequency_cm,
        brownian_damping_cm,
        reorganization_energy_cm,
    )
    exponents, coefficients = tpsd(
        spectral_density,
        temperature_K,
        pade_order,
        tpsd_tolerance;
        pade_type,
    )
    exponents = ComplexF64.(exponents)
    coefficients = ComplexF64.(coefficients)
    isempty(exponents) && error("Brownian TPSD returned no exponential terms")
    all(>(0), real.(exponents)) ||
        error("Brownian TPSD returned a nondecaying exponential")

    validation_times = collect(range(
        0.0,
        validation_final_time_fs;
        length=validation_sample_count,
    ))
    reference_bcf = BosonicBCF(
        spectral_density,
        temperature_K;
        ub=bcf_upper_bound_cm,
    )
    reference_samples = reference_bcf.(validation_times)
    fitted_samples = bcf_approx(validation_times, exponents, coefficients)
    relative_error = norm(fitted_samples - reference_samples) / norm(reference_samples)
    println("TPSD terms per site: $(length(exponents))")
    println("TPSD validation relative error: $(round(relative_error, sigdigits=5))")

    # 3. Couple one identical Brownian bath to each site and build HEOM-TT.
    baths = [
        BathExp(exponents, coefficients, projector)
        for projector in simple_holstein_site_projectors(site_count)
    ]
    system = HEOMTTSystem(
        hamiltonian,
        NoiseExp(baths),
        hierarchy_local_size,
    )
    liouvillian, trace_observable, population_observables =
        build_heom_liouvillian(system; tol=operator_tolerance)
    rho = build_initial_state(system, initial_site; tol=operator_tolerance)
    println("TT dimensions: $(heom_tt_dimensions(system))")
    println("Initial TT ranks: $(tt_ranks(rho))")

    # 4. Propagate with one Crank-Nicolson tAMEn solve per saved step.
    step_count = round(Int, step_ratio)
    times = collect(0:step_count) .* time_step_fs
    populations = zeros(Float64, site_count, step_count + 1)
    trace_values = zeros(Float64, step_count + 1)
    maximum_rank = zeros(Int, step_count + 1)
    mean_rank = zeros(Float64, step_count + 1)
    options = Dict(
        :verb => 0,
        :nswp => sweep_count,
        :local_iters => local_iterations,
        :kickrank => kick_rank,
        :time_scheme => "CN",
        :time_error_damp => 100.0,
        :obs => [trace_observable],
    )

    measure! = function(index, state)
        populations[:, index] .= real.([
            tt_dot(observable, state) for observable in population_observables
        ])
        trace_values[index] = real(tt_dot(trace_observable, state))
        ranks = tt_ranks(state)
        maximum_rank[index] = maximum(ranks)
        mean_rank[index] = mean(ranks)
        return nothing
    end
    print_progress = function(step, index)
        population_text = join([
            "P$site=$(round(populations[site, index], digits=6))"
            for site in 1:site_count
        ], ", ")
        println("step $step/$step_count, t=$(times[index]) fs: " *
                "$population_text, trace=$(round(trace_values[index], digits=8)), " *
                "max rank=$(maximum_rank[index])")
    end

    measure!(1, rho)
    print_progress(0, 1)
    for index in 2:(step_count + 1)
        space_time_state = tkron(
            rho,
            tt_ones(temporal_basis_size; T=ComplexF64),
        )
        space_time_state, time_grid, _, _ = tamen(
            space_time_state,
            time_step_fs * liouvillian,
            tamen_tolerance,
            options,
        )
        rho = extract_snapshot(space_time_state, time_grid, 1.0, "CN")
        rho = tt_round(rho, state_rounding_tolerance)
        measure!(index, rho)

        step = index - 1
        if step % progress_interval == 0 || step == step_count
            print_progress(step, index)
        end
    end

    # 5. Publish population, trace, and rank diagnostics.
    mkpath(output_directory)
    paths = simple_holstein_output_paths(output_directory)
    write_simple_holstein_csv(
        paths.csv,
        times,
        populations,
        trace_values,
        maximum_rank,
        mean_rank,
    )
    save_simple_holstein_plots(
        paths,
        times,
        populations,
        trace_values,
        maximum_rank,
        mean_rank,
    )
    println("Maximum absolute trace drift: $(maximum(abs.(trace_values .- 1.0)))")
    println("Wrote: $(paths.csv)")
    println("Wrote: $(paths.populations)")
    println("Wrote: $(paths.trace)")
    println("Wrote: $(paths.rank)")

    return (; times, populations, trace=trace_values, maximum_rank, mean_rank,
            state=rho, system, paths)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
