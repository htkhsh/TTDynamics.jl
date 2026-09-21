using LinearAlgebra
using Test
using TTSolver

if !isdefined(@__MODULE__, :TTDynamics)
    include(joinpath(@__DIR__, "..", "src", "TTDynamics.jl"))
end
using .TTDynamics
if !isdefined(@__MODULE__, :twin_permutation)
    include(joinpath(@__DIR__, "multicore_test_utils.jl"))
end

function state_test_correlation()
    metadata = CorrelationFitMetadata(0, 0, 0, 0, 0.1, (0.0, 1.0), [1.0], 1.0, Inf, 1.0, String[])
    ExponentialCorrelation(ComplexF64[0.7], ComplexF64[0.2], ComplexF64[0.2], metadata)
end

@testset "multicore HEOM initial density states" begin
    H = tt_mkron(ComplexF64[0 1; 1 0], ComplexF64.(Diagonal(0:2)))
    system = MultiCoreHEOMTTSystem(H, MultiCoreBathCoupling[], Int[])
    rho1 = ComplexF64[0.7 0.1im; -0.1im 0.3]
    rho2 = ComplexF64[0.2 0 0; 0 0.5 0.1im; 0 -0.1im 0.3]
    state = build_twin_heom_initial_state(system, [rho1, rho2]; tol=1e-14, density_tolerance=1e-12)
    @test tt_dims(state) == [2, 2, 3, 3]
    @test tt_ranks(state)[2] == rank(rho1)
    @test tt_ranks(state)[4] == rank(rho2)
    P = twin_permutation([2, 3])
    expected_interleaved = P * vec(kron(rho1, rho2))
    @test vec(tt_full(state)) ≈ expected_interleaved atol=1e-11
    # `tt_svd` consumes the internal little-endian ordering; convert from the
    # public interleaved (kron-compatible) ordering before decomposing.
    correlated_input = vec(permutedims(reshape(expected_interleaved, 3, 3, 2, 2), 4:-1:1))
    correlated = tt_svd(correlated_input, [2, 2, 3, 3]; tol=1e-14)
    rebuilt = build_twin_heom_initial_state(system, correlated; tol=1e-14)
    @test vec(tt_full(rebuilt)) ≈ expected_interleaved atol=1e-11
    near_psd = ComplexF64[1 + 1e-13 0; 0 -1e-13]
    @test tt_dims(build_twin_heom_initial_state(MultiCoreHEOMTTSystem(tt_matrix(zeros(2, 2)), MultiCoreBathCoupling[], Int[]), [near_psd]; density_tolerance=1e-12)) == [2, 2]
    @test_throws ArgumentError build_twin_heom_initial_state(system, [rho1])
    @test_throws DimensionMismatch build_twin_heom_initial_state(system, [rho1, Matrix{ComplexF64}(I, 2, 2) / 2])
    @test_throws ArgumentError build_twin_heom_initial_state(system, [rho1, ComplexF64[1.1 0 0; 0 -0.1 0; 0 0 0]], density_tolerance=1e-12)
    @test_throws ArgumentError build_twin_heom_initial_state(system, [rho1, rho2]; density_tolerance=-1.0)
    open_system = MultiCoreHEOMTTSystem(H, [MultiCoreBathCoupling(1, Matrix{ComplexF64}(I, 2, 2), state_test_correlation())], [2])
    open_state = build_twin_heom_initial_state(open_system, [rho1, rho2])
    @test tt_dims(open_state) == [2, 2, 3, 3, 2]
    hierarchy_view = reshape(tt_full(open_state), 2, :)
    @test hierarchy_view[2, :] ≈ zeros(size(hierarchy_view, 2)) atol=1e-12
    correlated_open = build_twin_heom_initial_state(open_system, correlated; tol=1e-14)
    @test tt_dims(correlated_open) == [2, 2, 3, 3, 2]
    wrong_physical_state = product_tt_state([ComplexF64[1, 0], ComplexF64[1, 0], ComplexF64[1, 0], ComplexF64[1, 0]])
    @test_throws DimensionMismatch build_twin_heom_initial_state(system, wrong_physical_state)
end

@testset "multicore roots and observables" begin
    H = tt_mkron(zeros(ComplexF64, 2, 2), zeros(ComplexF64, 3, 3))
    system = MultiCoreHEOMTTSystem(H, MultiCoreBathCoupling[], Int[])
    rho1 = ComplexF64[0.6 0.2im; -0.2im 0.4]
    rho2 = ComplexF64[0.3 0.1 0; 0.1 0.4 -0.05im; 0 0.05im 0.3]
    state = build_twin_heom_initial_state(system, [rho1, rho2]; tol=1e-14)
    root = root_twin_state(state, system)
    @test tt_dims(root) == [2, 2, 3, 3]
    trace_observable = build_twin_trace_observable(system; tol=1e-14)
    @test twin_expectation(trace_observable, state, system) ≈ 1 atol=1e-11
    nonhermitian = ComplexF64[0 1; 0 0]
    observable = build_twin_local_observable(system, 1, nonhermitian; tol=1e-14)
    expected = tr(kron(nonhermitian, Matrix{ComplexF64}(I, 3, 3)) * kron(rho1, rho2))
    @test twin_expectation(observable, state, system) ≈ expected atol=1e-11
    dense = root_density_matrix(state, system; max_dimension=10)
    @test dense ≈ kron(rho1, rho2) atol=1e-11
    @test_throws ArgumentError root_density_matrix(state, system; max_dimension=5)

    open_system = MultiCoreHEOMTTSystem(
        tt_matrix(zeros(ComplexF64, 2, 2)),
        [MultiCoreBathCoupling(1, Matrix{ComplexF64}(I, 2, 2), state_test_correlation())],
        [2],
    )
    open_state = build_twin_heom_initial_state(
        open_system, [ComplexF64[0.7 0.1; 0.1 0.3]]; tol=1e-14,
    )
    modified_cores = deepcopy(open_state.cores)
    modified_cores[end][:, 1, :] .*= 2
    modified_cores[end][:, 2, :] .*= 3
    modified_state = TTTensor(modified_cores)
    @test vec(tt_full(root_twin_state(modified_state, open_system))) ≈
          2 .* vec(tt_full(root_twin_state(open_state, open_system))) atol=1e-11
end
