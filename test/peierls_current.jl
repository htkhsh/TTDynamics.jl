using Test

module PeierlsValidation
include(joinpath(@__DIR__, "..", "examples", "peierls", "validate_dense.jl"))
include(joinpath(@__DIR__, "..", "examples", "peierls", "utils.jl"))
end

@testset "Peierls current and finite oscillator" begin
    result = PeierlsValidation.finite_oscillator_reference(
        merge(PeierlsValidation.DEFAULT_CONFIG,
              (; site_count=3, site_energies_cm=zeros(3),
                 hopping_cm=50.0, temperature_K=100.0,
                 operator_tolerance=1e-14));
        times_fs=[0.0, 100.0])
    @test result.physical_dimension == 192
    @test result.heom_dimension == 576
    @test result.interaction_error < 1e-12
    @test result.flux_derivative_error < 1e-10
    @test result.times_fs[end] == 100.0
    @test result.maximum_error < 1e-5
    @test result.tfd_dimension == result.heom_dimension
    @test result.tfd_correlation[1] ≈ result.heom_correlation[1] atol=1e-10
    @test result.maximum_heom_tfd_difference < 3e-6
    @test abs(imag(result.heom_correlation[1])) < 1e-12
    @test real(result.heom_correlation[1]) > 0

    config = merge(PeierlsValidation.DEFAULT_CONFIG,
                   (; site_count=3, site_energies_cm=zeros(3),
                      hopping_cm=50.0, temperature_K=100.0,
                      operator_tolerance=1e-14))
    example = PeierlsValidation
    ω = 250.0 * example.icm2ifs
    g = 20.0 * example.icm2ifs
    β = example.QFiND.ħ * 1e15 / (example.QFiND.kb * config.temperature_K)
    nthermal = inv(expm1(β * ω))
    noise = example.NoiseExp([
        example.BathExp(ComplexF64[1e-12 + im * ω, 1e-12 - im * ω],
                        ComplexF64[g^2 * (nthermal + 1), g^2 * nthermal],
                        coupling)
        for coupling in example.peierls_couplings(3)
    ])
    system = example.HEOMTTSystem(
        example.peierls_hamiltonian(config) * example.icm2ifs, noise, 2)
    action = example.peierls_current_action(config, system)
    adjoint_action = example.TTMatrix([
        permutedims(conj(core), (1, 3, 2, 4)) for core in action.cores])
    @test example.tt_full(adjoint_action) ≈ example.tt_full(action)' atol=1e-12
    _, trace_observable, _ = example.build_heom_liouvillian(system; tol=1e-14)
    observable = adjoint_action * trace_observable
    state = example.build_initial_state(system, 1)
    source = action * state
    @test example.tt_dot(observable, source) ≈
          example.tt_dot(trace_observable, action * source) atol=1e-12

    metadata = example.equilibrium_metadata(config, (; system), state;
                                            equilibration_time_fs=1.0)
    @test metadata["initial_site"] == 1
    @test example.validate_equilibrium_metadata(metadata, config,
                                                (; system), state) === nothing
    @test_throws ArgumentError example.validate_equilibrium_metadata(
        metadata, merge(config, (; initial_site=2)), (; system), state)
end
