using Test

module PeierlsBrownianTFDValidation
const VALIDATION_SCRIPT = joinpath(@__DIR__, "..", "examples", "peierls",
                                   "validate_brownian_tfd.jl")
isfile(VALIDATION_SCRIPT) && include(VALIDATION_SCRIPT)
end

@testset "Peierls Brownian TFD bath discretization" begin
    @test isdefined(PeierlsBrownianTFDValidation, :brownian_validation_config)
    @test isdefined(PeierlsBrownianTFDValidation, :discretize_brownian_bath)
    if isdefined(PeierlsBrownianTFDValidation, :brownian_validation_config)
        config = PeierlsBrownianTFDValidation.brownian_validation_config(
            heom_tamen_tolerance=1e-5, tfd_tamen_tolerance=3e-6)
        @test config.hierarchy_local_size == 20
        @test config.reorganization_energy_cm == 50.0
        @test config.temperature_K == 300.0
        @test config.heom_tamen_tolerance == 1e-5
        @test config.tfd_tamen_tolerance == 3e-6
    end
    if isdefined(PeierlsBrownianTFDValidation, :discretize_brownian_bath)
        bath = PeierlsBrownianTFDValidation.discretize_brownian_bath(
            frequency_cm=50.0, damping_cm=50.0,
            reorganization_energy_cm=150.0, temperature_K=300.0,
            omega_min_cm=-500.0, omega_max_cm=800.0,
            n_frequency=300, identification_time_fs=50.0,
            n_time=64, identification_tolerance=0.1,
        )
        @test length(bath.frequencies_cm) > 1
        @test length(bath.frequencies_cm) == length(bath.coefficients_cm)
        @test isfinite(bath.relative_bcf_error)
        @test bath.relative_bcf_error < 0.15

        config = merge(PeierlsBrownianTFDValidation.DEFAULT_CONFIG, (; 
            site_count=3, site_energies_cm=zeros(3), hopping_cm=50.0,
            temperature_K=300.0, initial_condition=:gibbs,
        ))
        model = PeierlsBrownianTFDValidation.build_brownian_tfd_model(
            config, bath; boson_size=2)
        @test PeierlsBrownianTFDValidation.tt_dims(model.initial_state) ==
              [9; fill(2, 3 * length(bath.frequencies_cm))]
        @test isapprox(PeierlsBrownianTFDValidation.norm(model.initial_state), 1.0;
                       atol=1e-12)
        zero_time = PeierlsBrownianTFDValidation.run_brownian_tfd_correlation(
            model, config; duration_fs=0.0)
        @test length(zero_time.correlation) == 1
        @test abs(imag(first(zero_time.correlation))) < 1e-10
    end
end
