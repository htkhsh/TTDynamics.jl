using LinearAlgebra
using Test
using TTDynamics
using TTSolver
using HEOMKit

@testset "HEOM extended left actions" begin
    H = ComplexF64[0 0.2; 0.2 0.1]
    V1 = ComplexF64[1 0; 0 -1]
    V2 = ComplexF64[0 1; 1 0]
    baths = [
        BathExp(ComplexF64[0.3], ComplexF64[0.08 + 0.02im], V1),
        BathExp(ComplexF64[0.4], ComplexF64[0.05 - 0.01im], V2),
    ]
    system = HEOMTTSystem(H, NoiseExp(baths), [3, 2])
    A = ComplexF64[0.1 0.2im; -0.2im 0.3]
    I2 = Matrix{ComplexF64}(I, 2, 2)
    I3 = Matrix{ComplexF64}(I, 3, 3)
    bminus3 = diagm(1 => sqrt.(ComplexF64[1, 2]))
    bplus3 = diagm(-1 => sqrt.(ComplexF64[1, 2]))
    bminus2 = diagm(1 => ComplexF64[1])
    bplus2 = diagm(-1 => ComplexF64[1])

    left = build_heom_left_action(system, A; tol=1e-14)
    @test tt_full(left) ≈ kron(A, I2, I3, I2) atol=1e-12
    @test_throws DimensionMismatch build_heom_left_action(system, ones(3, 3))

    noise = system.noise
    bath1 = build_heom_bath_coordinate_left_action(system, 1; tol=1e-14)
    dense1 = kron(I2, I2,
                  sqrt(noise.abs_coeff[1]) * bminus3 +
                  noise.c1[1] / sqrt(noise.abs_coeff[1]) * bplus3,
                  I2)
    @test tt_full(bath1) ≈ dense1 atol=1e-12

    bath2 = build_heom_bath_coordinate_left_action(system, 2; tol=1e-14)
    dense2 = kron(I2, I2, I3,
                  sqrt(noise.abs_coeff[2]) * bminus2 +
                  noise.c1[2] / sqrt(noise.abs_coeff[2]) * bplus2)
    @test tt_full(bath2) ≈ dense2 atol=1e-12
    @test_throws ArgumentError build_heom_bath_coordinate_left_action(system, 0)
    @test_throws ArgumentError build_heom_bath_coordinate_left_action(system, 3)

    # The current source creates a first-tier ADO from the root. Conversely,
    # its root observation reads a nonzero first-tier ADO.
    root = kron(ComplexF64[1, 0], ComplexF64[1, 0],
                ComplexF64[1, 0, 0], ComplexF64[1, 0])
    first_tier = kron(ComplexF64[1, 0], ComplexF64[1, 0],
                      ComplexF64[0, 1, 0], ComplexF64[1, 0])
    @test first_tier' * tt_full(bath1) * root ≈
          noise.c1[1] / sqrt(noise.abs_coeff[1])
    @test root' * tt_full(bath1) * first_tier ≈ sqrt(noise.abs_coeff[1])

    zero_scale = HEOMTTSystem(H, NoiseExp(BathExp(ComplexF64[0.3],
        ComplexF64[0.1im], V1)), 2)
    @test_throws ArgumentError build_heom_bath_coordinate_left_action(zero_scale, 1)
end
