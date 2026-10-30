using LinearAlgebra
using Test
using TTDynamics
using TTSolver
using HEOMKit

# The asymmetric joint moments distinguish r=+1 from r=-1, and multiple
# exponentials per bath catch missing terms or incorrect ADO scaling.
function electron_lattice_fixture()
    projectors = [Matrix(Diagonal(ComplexF64.(1:3 .== j))) for j in 1:3]
    baths = [BathExp(ComplexF64[0.3, 0.7], ComplexF64[2, 8], projectors[1]),
             BathExp(ComplexF64[0.4], ComplexF64[4.5], projectors[2]),
             BathExp(ComplexF64[0.5], ComplexF64[12.5], projectors[3])]
    system = HEOMTTSystem(zeros(3, 3), NoiseExp(baths), 3)
    p = [0.2, 0.3, 0.5]
    joint = ComplexF64[1 2 4; 3 5 7; 6 8 9]
    vacuum = [ComplexF64[1, 0, 0] for _ in system.nb]
    function diagonal_ado(values, occupations)
        TTTensor([reshape(Matrix{ComplexF64}(I, 3, 3), 1, 3, 3),
                  reshape(Matrix(Diagonal(values)), 3, 3, 1),
                  [reshape(v, 1, 3, 1) for v in occupations]...])
    end
    state = diagonal_ado(ComplexF64.(p), vacuum)
    for bath in 1:3
        modes = system.noise.jstart_bath[bath] .+ (0:system.noise.nterms_bath[bath]-1)
        for k in modes
            occupations = copy(vacuum)
            occupations[k] = ComplexF64[0, 1, 0]
            state += diagonal_ado(joint[:, bath] / (length(modes) *
                                  sqrt(system.noise.abs_coeff[k])), occupations)
        end
    end
    # Higher tiers must not leak into either the mean or the joint moment.
    occupations = copy(vacuum)
    occupations[1] = ComplexF64[0, 0, 1]
    state += diagonal_ado(ComplexF64[100, 200, 300], occupations)
    return (; system, state, p, joint)
end

@testset "Connected periodic electron–lattice correlation from first-tier ADOs" begin
    f = electron_lattice_fixture()
    @test isdefined(TTDynamics, :electron_lattice_correlation)
    if isdefined(TTDynamics, :electron_lattice_correlation)
        m = electron_lattice_correlation(f.state, f.system)
        @test m.distances == [0, 1, 2]
        @test m.populations ≈ f.p atol=1e-12
        @test m.joint_moments ≈ f.joint atol=1e-11
        @test m.bath_mean ≈ [10, 15, 20] atol=1e-11
        @test m.uncentered_correlation ≈ [15, 15, 15] atol=1e-11
        @test m.correlation ≈ [-1.5, 1, 0.5] atol=1e-11
        @test sum(m.correlation) ≈ 0 atol=1e-11
        @test electron_lattice_correlation(2.7 * f.state, f.system).correlation ≈
              m.correlation atol=1e-11

        # Complex moments must retain their phase (no conjugation of an ADO).
        complex_state = f.state + 0.3im * (f.state -
            TTTensor([reshape(ComplexF64[1, 0, 0], 1, 3, 1),
                      reshape(ComplexF64[1, 0, 0], 1, 3, 1),
                      [reshape(ComplexF64[1, 0, 0], 1, 3, 1) for _ in f.system.nb]...]))
        cm = electron_lattice_correlation(complex_state, f.system)
        @test cm.joint_moments ≈ (1 + 0.3im) * f.joint atol=1e-11
        @test any(abs.(imag.(cm.correlation)) .> 0.1)

        root = build_initial_state(f.system, 1)
        @test electron_lattice_correlation(root, f.system).correlation ≈ zeros(3)
        # Factorized electronic state with nonzero bath mean cancels exactly.
        factors = [reshape(ComplexF64[1, 0.2, 0], 1, 3, 1) for _ in f.system.nb]
        factorized = TTTensor([reshape(Matrix{ComplexF64}(I, 3, 3), 1, 3, 3),
                              reshape(Matrix(Diagonal(ComplexF64.(f.p))), 3, 3, 1),
                              factors...])
        fm = electron_lattice_correlation(factorized, f.system)
        @test all(abs.(fm.bath_mean) .> 0)
        @test fm.correlation ≈ zeros(3) atol=1e-12
        @test_throws ArgumentError electron_lattice_correlation(0 * root, f.system)
        small = HEOMTTSystem(f.system.H_sys, f.system.noise, 1)
        @test_throws ArgumentError electron_lattice_correlation(build_initial_state(small, 1), small)
        @test_throws ArgumentError electron_lattice_correlation(root, small)
        one_bath = HEOMTTSystem(f.system.H_sys, NoiseExp([first(f.system.noise.baths)]), 3)
        @test_throws ArgumentError electron_lattice_correlation(build_initial_state(one_bath, 1), one_bath)
    end
end
