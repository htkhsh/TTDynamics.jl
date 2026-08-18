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
