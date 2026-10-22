using LinearAlgebra
using CairoMakie
using QFiND
using HEOMKit: icm2ifs, BathExp, NoiseExp
using TTDynamics
using TTSolver

isdefined(@__MODULE__, :DEFAULT_CONFIG) || include(joinpath(@__DIR__, "config.jl"))
isdefined(@__MODULE__, :peierls_current_action) || include(joinpath(@__DIR__, "model.jl"))

_kron_all(factors) = reduce(kron, factors)

function _gibbs_density(hamiltonian, temperature_K)
    temperature_K > 0 || throw(ArgumentError("temperature_K must be positive"))
    energies, vectors = eigen(Hermitian(hamiltonian))
    β = QFiND.ħ * 1e15 / (QFiND.kb * temperature_K)
    weights = exp.(-β .* (energies .- minimum(energies)))
    weights ./= sum(weights)
    return vectors * Diagonal(weights) * vectors'
end

function _heom_gibbs_state(system::HEOMTTSystem, temperature_K)
    n = size(system.H_sys, 1)
    density = _gibbs_density(system.H_sys, temperature_K)
    ket = reshape(Matrix{ComplexF64}(I, n, n), 1, n, n)
    bra = reshape(ComplexF64.(density), n, n, 1)
    vacuum = [reshape(ComplexF64[1; zeros(size - 1)], 1, size, 1)
              for size in system.nb]
    return TTTensor([ket, bra, vacuum...])
end

function _tfd_reference(config, electronic, bonds, differences, ω, g, nthermal,
                        times_fs, dt)
    n = config.site_count
    ancilla_identity = Matrix{ComplexF64}(I, n, n)
    density = _gibbs_density(electronic, config.temperature_K)
    energies, vectors = eigen(Hermitian(density))
    purification = vec(vectors * Diagonal(sqrt.(max.(energies, 0))) * vectors')
    tfd_frequencies = Float64[]
    tfd_coefficients = Float64[]
    tfd_baths = BosonicTFD[]
    for site in 1:n
        append!(tfd_frequencies, [ω, -ω])
        append!(tfd_coefficients,
                [sqrt(2) * g * sqrt(nthermal + 1), sqrt(2) * g * sqrt(nthermal)])
        push!(tfd_baths, BosonicTFD(
            [ω, -ω],
            [sqrt(2) * g * sqrt(nthermal + 1), sqrt(2) * g * sqrt(nthermal)],
            kron(ancilla_identity, bonds[site] - bonds[mod1(site - 1, n)]),
        ))
    end
    tfd_system = SBSystem(kron(ancilla_identity, electronic), BosonicEnv(tfd_baths))
    mode_count = length(tfd_frequencies)
    dimensions = fill(2, mode_count)
    tfd_hamiltonian, _ = tt_sbham(tfd_system; basis_sizes=dimensions,
                                  unit_conv=1.0)
    dense_hamiltonian = tt_full(tfd_hamiltonian)

    local_identity = Matrix{ComplexF64}(I, 2, 2)
    annihilation = ComplexF64[0 1; 0 0]
    bath_identity = _kron_all(fill(local_identity, mode_count))
    coordinates = Matrix{ComplexF64}[]
    for site in 1:n
        coordinate = zeros(ComplexF64, size(bath_identity))
        first_mode = 2site - 1
        for (mode, factor) in ((first_mode, g * sqrt(nthermal + 1)),
                               (first_mode + 1, g * sqrt(nthermal)))
            factors = fill(local_identity, mode_count)
            factors[mode] = factor .* (annihilation + annihilation')
            coordinate .+= _kron_all(factors)
        end
        push!(coordinates, coordinate)
    end
    dense_current = zeros(ComplexF64, n^2 * size(bath_identity, 1),
                          n^2 * size(bath_identity, 1))
    hopping = config.hopping_cm * icm2ifs
    for site in 1:n
        electronic_current = kron(ancilla_identity, -im * differences[site])
        bath_current = -hopping .* bath_identity + coordinates[site] -
                       coordinates[mod1(site + 1, n)]
        dense_current .+= kron(electronic_current, bath_current)
    end
    vacuum = ComplexF64[1, 0]
    tfd_state = reduce(kron, vcat([purification], fill(vacuum, mode_count)))
    source = dense_current * tfd_state
    tfd_step = exp(-im * dt * Hermitian(dense_hamiltonian))
    tfd_correlation = ComplexF64[]
    tfd_state_now = copy(tfd_state)
    tfd_source_now = copy(source)
    for index in eachindex(times_fs)
        push!(tfd_correlation, dot(tfd_state_now, dense_current * tfd_source_now))
        if index != lastindex(times_fs)
            tfd_state_now = tfd_step * tfd_state_now
            tfd_source_now = tfd_step * tfd_source_now
        end
    end
    return (; correlation=tfd_correlation,
            dimension=size(dense_hamiltonian, 1))
end

function finite_oscillator_reference(config; frequency_cm=250.0,
                                     coupling_cm=20.0, oscillator_size=4,
                                     times_fs=collect(0.0:1.0:100.0))
    n = config.site_count
    n == 3 || throw(ArgumentError("the dense reference uses a three-site ring"))
    oscillator_size >= 2 || throw(ArgumentError("oscillator_size must be at least two"))
    length(times_fs) >= 2 && first(times_fs) == 0 ||
        throw(ArgumentError("times_fs must start at zero and contain at least two points"))
    intervals = diff(Float64.(times_fs))
    all(>(0), intervals) &&
        all(isapprox.(intervals, first(intervals); rtol=1e-12, atol=1e-14)) ||
        throw(ArgumentError("times_fs must be uniformly spaced and increasing"))
    d = oscillator_size
    Id = Matrix{ComplexF64}(I, d, d)
    Ie = Matrix{ComplexF64}(I, n, n)
    occupation = diagm(0 => ComplexF64.(0:(d - 1)))
    annihilation = diagm(1 => sqrt.(ComplexF64.(1:(d - 1))))
    ω = frequency_cm * icm2ifs
    g = coupling_cm * icm2ifs
    t = config.hopping_cm * icm2ifs
    β = QFiND.ħ * 1e15 / (QFiND.kb * config.temperature_K)
    nthermal = inv(expm1(β * ω))
    weights = exp.(-β * ω .* (0:(d - 1)))
    weights ./= sum(weights)
    thermal = Matrix(Diagonal(ComplexF64.(weights)))

    oscillator_identity = _kron_all(fill(Id, n))
    bath_density = _kron_all(fill(thermal, n))
    coordinates = [_kron_all([site == mode ?
                              g * (annihilation + annihilation') : Id
                              for site in 1:n]) for mode in 1:n]
    free_bath = sum((_kron_all([site == mode ? ω * occupation : Id
                               for site in 1:n]) for mode in 1:n))

    # Construct the reference directly from site hopping matrices, independent
    # of the example's bond and current builders.
    bonds = Matrix{ComplexF64}[]
    differences = Matrix{ComplexF64}[]
    forwards = Matrix{ComplexF64}[]
    for site in 1:n
        neighbor = mod1(site + 1, n)
        forward = zeros(ComplexF64, n, n)
        forward[neighbor, site] = 1
        push!(forwards, forward)
        push!(bonds, forward + forward')
        push!(differences, forward - forward')
    end
    electronic = Matrix(Diagonal(ComplexF64.(config.site_energies_cm))) * icm2ifs
    for bond in bonds
        electronic -= t * bond
    end
    couplings = [bonds[site] - bonds[mod1(site - 1, n)] for site in 1:n]
    interaction_by_bath = sum((kron(couplings[site], coordinates[site])
                               for site in 1:n))
    interaction_by_bond = sum((kron(bonds[site],
                                    coordinates[site] - coordinates[mod1(site + 1, n)])
                               for site in 1:n))
    interaction_error = norm(interaction_by_bath - interaction_by_bond)
    Hamiltonian = kron(electronic, oscillator_identity) +
                  kron(Ie, free_bath) + interaction_by_bath
    current = sum((kron(-im * differences[site],
                        -t * oscillator_identity + coordinates[site] -
                        coordinates[mod1(site + 1, n)]) for site in 1:n))
    function flux_hamiltonian(phase)
        free = kron(Matrix(Diagonal(ComplexF64.(config.site_energies_cm))) *
                    icm2ifs, oscillator_identity) + kron(Ie, free_bath)
        return free + sum((kron(exp(im * phase) * forwards[site] +
                                  exp(-im * phase) * forwards[site]',
                                  -t * oscillator_identity + coordinates[site] -
                                  coordinates[mod1(site + 1, n)]) for site in 1:n))
    end
    phase_step = 1e-5
    flux_derivative_error = norm(current +
        (flux_hamiltonian(phase_step) - flux_hamiltonian(-phase_step)) /
        (2phase_step))
    electron_density = _gibbs_density(electronic, config.temperature_K)
    physical_density = kron(electron_density, bath_density)
    physical_source = current * physical_density
    physical_correlation = ComplexF64[]
    physical_state = copy(physical_source)
    physical_step = exp(-im * first(intervals) * Hermitian(Hamiltonian))
    for index in eachindex(times_fs)
        push!(physical_correlation, tr(current * physical_state))
        index == lastindex(times_fs) ||
            (physical_state = physical_step * physical_state * physical_step')
    end

    # A single undamped harmonic mode has an exact two-exponential thermal BCF.
    # A minute damping avoids a zero-real-part sorting singularity in NoiseExp.
    exponents = ComplexF64[1e-12 + im * ω, 1e-12 - im * ω]
    coefficients = ComplexF64[g^2 * (nthermal + 1), g^2 * nthermal]
    noise = NoiseExp([BathExp(exponents, coefficients, coupling)
                      for coupling in couplings])
    system = HEOMTTSystem(electronic, noise, 2)
    liouvillian, trace_observable, _ =
        build_heom_liouvillian(system; tol=config.operator_tolerance)
    current_action = peierls_current_action(config, system)
    initial = _heom_gibbs_state(system, config.temperature_K)
    source = vec(tt_full(current_action * initial))
    dense_generator = tt_full(liouvillian)
    dense_current = tt_full(current_action)
    root_trace = vec(tt_full(trace_observable))
    heom_state = copy(source)
    heom_step = exp(first(intervals) * dense_generator)
    heom_correlation = ComplexF64[]
    for index in eachindex(times_fs)
        push!(heom_correlation, dot(root_trace, dense_current * heom_state))
        index == lastindex(times_fs) || (heom_state = heom_step * heom_state)
    end
    tfd = _tfd_reference(config, electronic, bonds, differences, ω, g, nthermal,
                         times_fs, first(intervals))
    heom_error = maximum(abs.(heom_correlation - physical_correlation))
    tfd_error = maximum(abs.(tfd.correlation - physical_correlation))
    method_difference = maximum(abs.(tfd.correlation - heom_correlation))
    return (; times_fs, physical_correlation, heom_correlation,
            tfd_correlation=tfd.correlation,
            maximum_error=heom_error, maximum_tfd_error=tfd_error,
            maximum_heom_tfd_difference=method_difference,
            interaction_error, flux_derivative_error,
            physical_dimension=size(Hamiltonian, 1),
            heom_dimension=size(dense_generator, 1), tfd_dimension=tfd.dimension)
end

function validate_dense_main()
    config = merge(DEFAULT_CONFIG, (;
        site_count=3, site_energies_cm=zeros(3), hopping_cm=50.0,
        temperature_K=100.0, operator_tolerance=1e-14))
    result = finite_oscillator_reference(config)
    println("Three-site Peierls CCF: finite oscillators, HEOM, and TFD")
    println("  physical dimension = $(result.physical_dimension)")
    println("  HEOM dimension = $(result.heom_dimension)")
    println("  TFD dimension = $(result.tfd_dimension)")
    println("  interaction identity error = $(result.interaction_error)")
    println("  flux derivative error = $(result.flux_derivative_error)")
    println("  max |HEOM - finite oscillator| = $(result.maximum_error)")
    println("  max |TFD - finite oscillator| = $(result.maximum_tfd_error)")
    println("  max |TFD - HEOM| = $(result.maximum_heom_tfd_difference)")
    println("  time_fs     finite oscillator                 HEOM-TT                         TFD")
    shown_indices = unique(vcat(collect(1:10:length(result.times_fs)),
                                [lastindex(result.times_fs)]))
    for index in shown_indices
        println("  $(result.times_fs[index])  " *
                "$(result.physical_correlation[index])  " *
                "$(result.heom_correlation[index])  " *
                "$(result.tfd_correlation[index])")
    end
    output_directory = joinpath(@__DIR__, "output")
    mkpath(output_directory)
    plot_path = joinpath(output_directory, "peierls_dense_validation.png")
    save_validation_plot(plot_path, result)
    println("  wrote $plot_path")
    return result
end

function save_validation_plot(path::AbstractString, result)
    figure = Figure(size=(980, 900))
    title = "Three-site Peierls CCF: finite oscillators, HEOM, and TFD"
    real_axis = Axis(figure[1, 1]; xlabel="Time (fs)", ylabel="Re C(t) (fs⁻²)",
                     title)
    lines!(real_axis, result.times_fs, real.(result.physical_correlation);
           linewidth=2.5, label="Finite oscillators")
    lines!(real_axis, result.times_fs, real.(result.heom_correlation);
           linewidth=2, linestyle=:dash, label="HEOM")
    lines!(real_axis, result.times_fs, real.(result.tfd_correlation);
           linewidth=2, linestyle=:dot, label="TFD")
    axislegend(real_axis; position=:rt)

    imaginary_axis = Axis(figure[2, 1]; xlabel="Time (fs)",
                          ylabel="Im C(t) (fs⁻²)")
    lines!(imaginary_axis, result.times_fs, imag.(result.physical_correlation);
           linewidth=2.5, label="Finite oscillators")
    lines!(imaginary_axis, result.times_fs, imag.(result.heom_correlation);
           linewidth=2, linestyle=:dash, label="HEOM")
    lines!(imaginary_axis, result.times_fs, imag.(result.tfd_correlation);
           linewidth=2, linestyle=:dot, label="TFD")
    axislegend(imaginary_axis; position=:rb)

    error_axis = Axis(figure[3, 1]; xlabel="Time (fs)",
                      ylabel="Absolute difference from finite oscillators (fs⁻²)",
                      yscale=log10)
    heom_errors = abs.(result.heom_correlation - result.physical_correlation)
    tfd_errors = abs.(result.tfd_correlation - result.physical_correlation)
    lines!(error_axis, result.times_fs, max.(heom_errors, eps(Float64));
           linewidth=2, label="HEOM")
    lines!(error_axis, result.times_fs, max.(tfd_errors, eps(Float64));
           linewidth=2, label="TFD")
    axislegend(error_axis; position=:lt)
    linkxaxes!(real_axis, imaginary_axis, error_axis)
    save(String(path), figure)
    return String(path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    validate_dense_main()
end
