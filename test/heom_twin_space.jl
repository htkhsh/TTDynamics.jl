using LinearAlgebra
using Test
using TTDynamics
using TTSolver
using KaisouEOM

function twin_test_system(; nb=2)
    H = ComplexF64[0.3 0.7 + 0.2im; 0.7 - 0.2im -0.4]
    V = ComplexF64[0.6 0.1 + 0.3im; 0.1 - 0.3im -0.2]
    # A real exponent preserves a single hierarchy mode in NoiseExp while the
    # complex coefficient and coupling retain complex-algebra coverage.
    bath = BathExp(ComplexF64[0.4], ComplexF64[0.03 - 0.02im], V)
    HEOMTTSystem(H, NoiseExp([bath]), nb)
end

function twin_coefficient_matrix(state::TTTensor)
    @assert length(state.cores) == 2
    ket_core, bra_core = state.cores
    nket = size(ket_core, 2)
    nbra = size(bra_core, 2)
    return ComplexF64[
        (reshape(ket_core[1, ket, :], 1, :) * bra_core[:, bra, :])[1]
        for ket in 1:nket, bra in 1:nbra
    ]
end

function dense_twin_liouvillian(system::HEOMTTSystem)
    @assert TTDynamics.n_bcf(system) == 1
    @assert length(system.nb) == 1

    Ns = TTDynamics.nsys(system)
    nb = only(system.nb)
    Is = Matrix{ComplexF64}(I, Ns, Ns)
    Ib = Matrix{ComplexF64}(I, nb, nb)
    n = diagm(0 => ComplexF64.(0:(nb - 1)))
    bplus = diagm(-1 => sqrt.(ComplexF64.(1:(nb - 1))))
    bminus = diagm(1 => sqrt.(ComplexF64.(1:(nb - 1))))

    H = system.H_sys
    V = only(system.noise.V)
    gamma = only(TTDynamics.γ(system))
    c1 = only(TTDynamics.c1(system))
    c2 = only(TTDynamics.c2(system))
    sqrt_abs_c = sqrt(only(system.noise.abs_coeff))

    hamiltonian_left = kron(H, Is, Ib)
    hamiltonian_right = kron(Is, transpose(H), Ib)
    decay = gamma * kron(Is, Is, n)
    upward_left = sqrt_abs_c * kron(V, Is, bminus)
    upward_right = sqrt_abs_c * kron(Is, transpose(V), bminus)
    downward_left = (c1 / sqrt_abs_c) * kron(V, Is, bplus)
    downward_right = (c2 / sqrt_abs_c) * kron(Is, conj(V), bplus)

    return -1im * (hamiltonian_left - hamiltonian_right) - decay -
           1im * (upward_left - upward_right) -
           1im * (downward_left - downward_right)
end

@testset "HEOM twin-space states" begin
    system = twin_test_system()

    @test heom_tt_dimensions(system) == [2, 2, 2]

    rho = TTDynamics._build_twin_initial_state(system, 2; tol=1e-14)
    @test tt_dims(rho) == [2, 2, 2]
    @test root_density_matrix(rho, system) ≈ ComplexF64[0 0; 0 1] atol=1e-12
    @test tt_ranks(rho) == [1, 1, 1, 1]
    @test_throws ArgumentError TTDynamics._build_twin_initial_state(system, 0)
    @test_throws ArgumentError TTDynamics._build_twin_initial_state(system, 3)
end

@testset "HEOM twin-space system validation" begin
    system = twin_test_system()
    bath = first(system.noise.baths)
    noise = NoiseExp([bath])

    @test_throws ArgumentError HEOMTTSystem(zeros(ComplexF64, 2, 3), noise, 2)
    @test_throws ArgumentError HEOMTTSystem(zeros(ComplexF64, 0, 0), noise, 2)
    @test_throws ArgumentError HEOMTTSystem(ComplexF64[0 NaN; 0 0], noise, 2)
    @test_throws ArgumentError HEOMTTSystem(zeros(ComplexF64, 2, 2), noise, [2, 2])
    @test_throws ArgumentError HEOMTTSystem(zeros(ComplexF64, 2, 2), noise, 0)

    mismatched_bath = BathExp(ComplexF64[0.4], ComplexF64[0.03 - 0.02im],
                              zeros(ComplexF64, 3, 3))
    @test_throws ArgumentError HEOMTTSystem(zeros(ComplexF64, 2, 2), NoiseExp([mismatched_bath]), 2)
end

@testset "HEOM twin-space local products" begin
    tensor = TTDynamics._local_product_tensor([
        ComplexF64[1 + im, 2],
        ComplexF64[3im, -1, 4],
    ])
    @test tt_dims(tensor) == [2, 3]
    @test vec(tt_full(tensor)) == kron(ComplexF64[1 + im, 2], ComplexF64[3im, -1, 4])
    @test_throws ArgumentError TTDynamics._local_product_tensor(Vector{ComplexF64}[])
    @test_throws ArgumentError TTDynamics._local_product_tensor([ComplexF64[]])

    matrices = [
        ComplexF64[1 + im 2; -3im 4],
        ComplexF64[0 1im -2],
    ]
    matrix = TTDynamics._local_product_matrix(matrices)
    @test tt_dims(matrix) == ([2, 1], [2, 3])
    @test tt_full(matrix) == kron(matrices[1], matrices[2])
    @test_throws ArgumentError TTDynamics._local_product_matrix(Matrix{ComplexF64}[])
    @test_throws ArgumentError TTDynamics._local_product_matrix([zeros(ComplexF64, 0, 1)])
end

@testset "HEOM twin-space left and right actions" begin
    rho = ComplexF64[0.2 + 0.3im -0.1 + 0.4im; 0.7 - 0.2im 0.8 - 0.5im]
    A = ComplexF64[0.4 + 0.2im -0.3im; 0.6 -0.1 + 0.5im]
    B = ComplexF64[-0.2 + 0.7im 0.9; -0.4im 0.3 - 0.1im]
    identity2 = Matrix{ComplexF64}(I, 2, 2)
    state = TTTensor([
        reshape(identity2, 1, 2, 2),
        reshape(rho, 2, 2, 1),
    ])

    left_result = TTDynamics._local_product_matrix([A, identity2]) * state
    right_result = TTDynamics._local_product_matrix([identity2, transpose(B)]) * state
    adjoint_right_result = TTDynamics._local_product_matrix([identity2, conj(B)]) * state

    @test twin_coefficient_matrix(left_result) ≈ A * rho
    @test twin_coefficient_matrix(right_result) ≈ rho * B
    @test twin_coefficient_matrix(adjoint_right_result) ≈ rho * B'
end

@testset "HEOM twin-space root extraction" begin
    system = twin_test_system()
    ket_core = zeros(ComplexF64, 1, 2, 2)
    ket_core[1, 1, :] = ComplexF64[1 + 2im, 2 - im]
    ket_core[1, 2, :] = ComplexF64[-1im, 1 - 3im]
    bra_core = zeros(ComplexF64, 2, 2, 2)
    bra_core[:, 1, :] = ComplexF64[1, 2im, 2 - im, -1]
    bra_core[:, 2, :] = ComplexF64[2im, 1 - im, -2, 1 + 2im]
    auxiliary_core = zeros(ComplexF64, 2, 2, 1)
    auxiliary_core[:, 1, 1] = ComplexF64[1 - im, 2im]
    auxiliary_core[:, 2, 1] = ComplexF64[3 + 2im, -4im]
    state = TTTensor([ket_core, bra_core, auxiliary_core])

    expected_root = zeros(ComplexF64, 2, 2)
    for ket in 1:2, bra in 1:2, left_rank in 1:2, right_rank in 1:2
        expected_root[ket, bra] += ket_core[1, ket, left_rank] *
                                   bra_core[left_rank, bra, right_rank] *
                                   auxiliary_core[right_rank, 1, 1]
    end
    @test root_density_matrix(state, system) ≈ expected_root

    old_layout = TTTensor([
        reshape(ComplexF64[1, 0, 0, 0], 1, 4, 1),
        reshape(ComplexF64[1, 0], 1, 2, 1),
    ])
    old_error = try
        root_density_matrix(old_layout, system)
        nothing
    catch error
        error
    end
    @test old_error isa ArgumentError
    @test occursin("old vectorized HEOM state is unsupported", sprint(showerror, old_error))

    wrong_layout = TTTensor([
        reshape(ComplexF64[1, 0], 1, 2, 1),
        reshape(ComplexF64[1, 0, 0], 1, 3, 1),
        reshape(ComplexF64[1, 0], 1, 2, 1),
    ])
    wrong_error = try
        root_density_matrix(wrong_layout, system)
        nothing
    catch error
        error
    end
    @test wrong_error isa ArgumentError
    @test occursin("[2, 2, 2]", sprint(showerror, wrong_error))
    @test occursin("[2, 3, 2]", sprint(showerror, wrong_error))
end

@testset "HEOM twin-space observables" begin
    system = twin_test_system()
    rho = ComplexF64[0.2 + 0.3im -0.1 + 0.4im; 0.7 - 0.2im 0.8 - 0.5im]
    ket_core = reshape(Matrix{ComplexF64}(I, 2, 2), 1, 2, 2)
    bra_core = reshape(rho, 2, 2, 1)
    vacuum = reshape(ComplexF64[1, 0], 1, 2, 1)
    state = TTTensor([ket_core, bra_core, vacuum])

    trace_observable = TTDynamics._twin_trace_observable(system; tol=1e-14)
    @test tt_dims(trace_observable) == [2, 2, 2]
    @test tt_dot(trace_observable, state) ≈ tr(rho)

    for site in 1:2
        population = TTDynamics._twin_population_observable(system, site; tol=1e-14)
        @test tt_dims(population) == [2, 2, 2]
        @test tt_dot(population, state) ≈ rho[site, site]
    end
    @test_throws ArgumentError TTDynamics._twin_population_observable(system, 0)
    @test_throws ArgumentError TTDynamics._twin_population_observable(system, 3)
end

@testset "HEOM twin-space Liouvillian" begin
    system = twin_test_system()
    dense_reference = dense_twin_liouvillian(system)

    liouvillian, trace_observable, populations =
        build_heom_liouvillian(system; tol=1e-14)

    @test tt_dims(liouvillian) == ([2, 2, 2], [2, 2, 2])
    @test tt_full(liouvillian) ≈ dense_reference rtol=1e-12 atol=1e-12
    @test tt_dims(trace_observable) == [2, 2, 2]
    @test all(population -> tt_dims(population) == [2, 2, 2], populations)

    rho = ComplexF64[0.2 + 0.3im -0.1 + 0.4im; 0.7 - 0.2im 0.8 - 0.5im]
    state = TTTensor([
        reshape(Matrix{ComplexF64}(I, 2, 2), 1, 2, 2),
        reshape(rho, 2, 2, 1),
        reshape(ComplexF64[1, 0], 1, 2, 1),
    ])
    @test tt_dot(trace_observable, state) ≈ tr(rho)
    @test [tt_dot(population, state) for population in populations] ≈ diag(rho)

    dense_trace = vec(tt_full(trace_observable))
    @test dense_trace' * tt_full(liouvillian) ≈ zeros(ComplexF64, 1, 8) atol=1e-12

    rho0 = build_initial_state(system, 1; tol=1e-14)
    @test tt_dims(rho0) == [2, 2, 2]
    propagated = tt_ksl(rho0, (-1) * liouvillian, 1e-3;
                        symmetric=true, expmethod=:taylor, tol=1e-12, rmax=4)
    @test tt_dot(trace_observable, propagated) ≈ 1.0 atol=1e-9
end
