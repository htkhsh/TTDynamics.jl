using LinearAlgebra
using Test
using TTSolver
using HEOMKit
include(joinpath(@__DIR__, "multicore_test_utils.jl"))

if !isdefined(@__MODULE__, :TTDynamics)
    include(joinpath(@__DIR__, "..", "src", "TTDynamics.jl"))
end
using .TTDynamics

@testset "MultiCore Redfield tAMEn propagation" begin
    h = tt_matrix(ComplexF64[0.2 0.0; 0.0 -0.2])
    system = MultiCoreRedfieldTTSystem(h, [ComplexF64[0.2 0.0; 0.0 -0.2]],
                                       MultiCoreBathCoupling[])
    state = product_tt_state([ComplexF64[1.0, 0.0], ComplexF64[1.0, 0.0]])
    trajectory = propagate_multicore_redfield_tamen(
        system, state, [0.0, 0.01]; temporal_basis_size=2,
        tolerance=1e-3, state_tolerance=1e-8,
        options=Dict(:verb => 0, :nswp => 1, :local_iters => 2),
    )
    @test length(trajectory) == 2
    @test tt_dims(trajectory[end]) == [2, 2]
end

function redfield_test_correlation()
    metadata = CorrelationFitMetadata(
        0, 0, 0, 0, 0.1, (0.0, 1.0), [1.0], 1.0, Inf, 1.0, String[],
    )
    ExponentialCorrelation([0.8], [0.25 + 0.04im], [0.15 + 0.02im], metadata)
end

@testset "MultiCore local Redfield dissipator" begin
    h1 = ComplexF64[0.2 0.3; 0.3 -0.1]
    h2 = ComplexF64[0.0 0.2; 0.2 0.4]
    v1 = ComplexF64[1 0; 0 -1]
    v2 = ComplexF64[0 1; 1 0]
    h = tt_mkron(h1, h2)
    correlation = redfield_test_correlation()
    couplings = [
        MultiCoreBathCoupling(1, v1, correlation),
        MultiCoreBathCoupling(2, v2, correlation),
    ]
    system = MultiCoreRedfieldTTSystem(h, [h1, h2], couplings)

    dissipator = build_multicore_redfield_dissipator(system; tol=1e-13)
    @test tt_dims(dissipator) == ([2, 2, 2, 2], [2, 2, 2, 2])

    baths = [RedfieldBath(BathExp(correlation.rates,
                                  correlation.coeff_forward,
                                  coupling.operator)) for coupling in couplings]
    jumps = [global_redfield_jump(local_h, coupling.operator, bath)
             for (local_h, coupling, bath) in zip([h1, h2], couplings, baths)]
    identity2 = Matrix{ComplexF64}(I, 2, 2)
    expected = zeros(ComplexF64, 16, 16)
    for (subsystem, coupling, jump) in zip(1:2, couplings, jumps)
        s = coupling.operator
        left = subsystem == 1 ? kron(s * jump, identity2) : kron(identity2, s * jump)
        phi_left = subsystem == 1 ? kron(jump, identity2) : kron(identity2, jump)
        right_s = subsystem == 1 ? kron(transpose(s), identity2) : kron(identity2, transpose(s))
        right_phi = subsystem == 1 ? kron(transpose(jump'), identity2) : kron(identity2, transpose(jump'))
        right_product = subsystem == 1 ? kron(transpose(jump' * s), identity2) : kron(identity2, transpose(jump' * s))
        ket_s = subsystem == 1 ? kron(s, identity2) : kron(identity2, s)
        expected .-= kron(kron(identity2, identity2), left)
        expected .+= kron(right_s, phi_left)
        expected .-= kron(right_product, kron(identity2, identity2))
        expected .+= kron(right_phi, ket_s)
    end
    permutation = twin_permutation([2, 2])
    expected_interleaved = permutation * expected * permutation'
    @test tt_full(dissipator) ≈ expected_interleaved atol=1e-11

    generator = build_multicore_redfield_generator(system; tol=1e-13)
    @test tt_full(generator) ≈ tt_full(build_multicore_liouvillian(h; tol=1e-13)) + expected_interleaved atol=1e-11
end
