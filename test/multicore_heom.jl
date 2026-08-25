using LinearAlgebra
using Test
using TTSolver

if !isdefined(@__MODULE__, :TTDynamics)
    include(joinpath(@__DIR__, "..", "src", "TTDynamics.jl"))
end
using .TTDynamics

@testset "scaled multi-core HEOM component equation" begin
    metadata = CorrelationFitMetadata(
        0,
        0,
        0,
        0,
        0.1,
        (0.0, 1.0),
        [1.0],
        1.0,
        Inf,
        1.0,
        String[],
    )
    rate = 0.7 + 0.2im
    forward = 0.31 - 0.17im
    backward = -0.08 + 0.23im
    fit = ExponentialCorrelation([rate], [forward], [backward], metadata)
    scale = only(fit.scales)
    V = ComplexF64[0.4 0.7im; -0.2 1.1]
    VL = liouville_left(V)
    VR = liouville_right(V)
    physical_L = ComplexF64[
        -0.3+0.1im 0.2im 0.0 0.4
        0.1 -0.5-0.2im -0.3im 0.0
        0.0 0.7 0.2+0.4im -0.1im
        -0.2im 0.0 0.6 -0.8
    ]
    physical_L_tt = TTMatrix([reshape(physical_L, 1, 4, 4, 1)])
    system = MultiCoreHEOMTTSystem(
        physical_L_tt,
        [4],
        [LocalBathCoupling(1, V, fit)],
        [3],
    )

    expected = zeros(ComplexF64, 12, 12)
    I4 = Matrix{ComplexF64}(I, 4, 4)
    for n in 0:2
        nblock = (n + 1):3:12
        expected[nblock, nblock] .+= physical_L - n * rate * I4
        if n < 2
            upblock = (n + 2):3:12
            expected[nblock, upblock] .+=
                -1im * scale * sqrt(n + 1) * (VL - VR)
        end
        if n > 0
            downblock = n:3:12
            expected[nblock, downblock] .+=
                -1im * sqrt(n) / scale * (forward * VL - backward * VR)
        end
    end

    density = vec(ComplexF64[0.6 0.2+0.1im; -0.3im 0.4])
    probe = kron(density, ComplexF64[0.2, -0.5im, 0.7])
    dense_generator = tt_full(build_multicore_heom_generator(system))
    @test dense_generator ≈ expected atol=1e-11
    @test dense_generator * probe ≈ expected * probe atol=1e-11
end

@testset "multi-core HEOM vacuum state and root contractions" begin
    metadata = CorrelationFitMetadata(
        0,
        0,
        0,
        0,
        0.1,
        (0.0, 1.0),
        [1.0, 0.5],
        1.0,
        Inf,
        1.0,
        String[],
    )
    fit = ExponentialCorrelation(
        [0.4 + 0.1im, 0.9 - 0.2im],
        [0.2 + 0.05im, -0.03 + 0.07im],
        [0.18 - 0.02im, 0.04 + 0.06im],
        metadata,
    )
    physical_L = TTMatrix([
        reshape(ComplexF64.(diagm(0 => [0.0, -0.1, -0.2, -0.3])), 1, 4, 4, 1),
        reshape(ComplexF64.(diagm(0 => collect(0.0:-0.05:-0.4))), 1, 9, 9, 1),
    ])
    coupling = LocalBathCoupling(
        2,
        ComplexF64[0.5 0.1im 0.0; -0.1im -0.2 0.3; 0.0 0.3 0.7],
        fit,
    )
    system = MultiCoreHEOMTTSystem(physical_L, [4, 9], [coupling], [2, 3])

    physical_before = deepcopy(physical_L.cores)
    rates_before = copy(fit.rates)
    forward_before = copy(fit.coeff_forward)
    backward_before = copy(fit.coeff_backward)
    scales_before = copy(fit.scales)
    generator = build_multicore_heom_generator(system)
    @test tt_dims(generator) == ([4, 9, 2, 3], [4, 9, 2, 3])
    @test physical_L.cores == physical_before
    @test fit.rates == rates_before
    @test fit.coeff_forward == forward_before
    @test fit.coeff_backward == backward_before
    @test fit.scales == scales_before

    rho1 = ComplexF64[0.7 0.1im; -0.1im 0.3]
    rho2 = ComplexF64[0.2 0.0 0.0; 0.0 0.5 0.1im; 0.0 -0.1im 0.3]
    initial = build_multicore_heom_initial_state(system, [rho1, rho2])
    root = root_ado(initial, system)
    trace_observable = TTTensor([
        reshape(vec(Matrix{ComplexF64}(I, 2, 2)), 1, 4, 1),
        reshape(vec(Matrix{ComplexF64}(I, 3, 3)), 1, 9, 1),
    ])
    local_observable = TTTensor([
        reshape(vec(ComplexF64[1 0; 0 0]), 1, 4, 1),
        reshape(vec(Matrix{ComplexF64}(I, 3, 3)), 1, 9, 1),
    ])
    @test tt_dims(initial) == [4, 9, 2, 3]
    @test tt_dims(root) == [4, 9]
    @test tt_dot(trace_observable, root) ≈ 1
    @test root_expectation(initial, system, trace_observable) ≈ 1
    @test root_expectation(initial, system, local_observable) ≈ rho1[1, 1]

    closed_system = MultiCoreHEOMTTSystem(
        physical_L,
        [4, 9],
        LocalBathCoupling[],
        Int[],
    )
    closed_generator = build_multicore_heom_generator(closed_system)
    closed_initial = build_multicore_heom_initial_state(closed_system, [rho1, rho2])
    @test tt_dims(closed_generator) == ([4, 9], [4, 9])
    @test tt_full(closed_generator) ≈ tt_full(physical_L) atol=1e-11
    @test tt_dims(root_ado(closed_initial, closed_system)) == [4, 9]
    @test root_expectation(closed_initial, closed_system, trace_observable) ≈ 1
end

@testset "multi-core HEOM types" begin
    A = ComplexF64[1 2im; 3 4]
    rho = ComplexF64[0.4 0.1im; -0.1im 0.6]
    @test liouville_left(A) * vec(rho) ≈ vec(A * rho)
    @test liouville_right(A) * vec(rho) ≈ vec(rho * A)

    metadata = CorrelationFitMetadata(
        0,
        0,
        0,
        0,
        0.1,
        (0.0, 1.0),
        [1.0],
        1.0,
        Inf,
        1.0,
        String[],
    )
    fit = ExponentialCorrelation([1 + 0im], [0.2 + 0.1im], [0.2 - 0.1im], metadata)
    @test fit.scales ≈ [sqrt(abs(0.2 + 0.1im))]
    @test_throws ArgumentError ExponentialCorrelation(
        [-1 + 0im],
        [1im],
        [-1im],
        metadata,
    )
    @test_throws ArgumentError ExponentialCorrelation(
        ComplexF64[-1 + 0im],
        ComplexF64[1im],
        ComplexF64[-1im],
        [1.0],
        metadata,
    )
end

@testset "multi-core HEOM dimensions and validation" begin
    metadata = CorrelationFitMetadata(
        0,
        0,
        0,
        0,
        0.1,
        (0.0, 1.0),
        [1.0],
        1.0,
        Inf,
        1.0,
        String[],
    )
    fit = ExponentialCorrelation(
        [1 + 0im, 2 + 0im],
        [0.2 + 0.1im, 0.05 + 0.02im],
        [0.2 - 0.1im, 0.05 - 0.02im],
        metadata,
    )
    physical_L = TTMatrix([
        reshape(Matrix{ComplexF64}(I, 4, 4), 1, 4, 4, 1),
        reshape(Matrix{ComplexF64}(I, 9, 9), 1, 9, 9, 1),
    ])
    couplings = [LocalBathCoupling(2, Matrix{ComplexF64}(I, 3, 3), fit)]
    system = MultiCoreHEOMTTSystem(physical_L, [4, 9], couplings, [3, 3])

    @test multicore_heom_dimensions(system) == [4, 9, 3, 3]
    @test_throws ArgumentError LocalBathCoupling(0, Matrix{ComplexF64}(I, 3, 3), fit)
    @test_throws ArgumentError MultiCoreHEOMTTSystem(
        physical_L,
        [4, 9],
        [LocalBathCoupling(3, Matrix{ComplexF64}(I, 3, 3), fit)],
        [3, 3],
    )
    @test_throws ArgumentError MultiCoreHEOMTTSystem(physical_L, [4, 9], couplings, [0])

    closed_system = MultiCoreHEOMTTSystem(physical_L, [4, 9], LocalBathCoupling[], Int[])
    @test multicore_heom_dimensions(closed_system) == [4, 9]
    @test isempty(closed_system.couplings)
    @test isempty(closed_system.hierarchy_sizes)
end
