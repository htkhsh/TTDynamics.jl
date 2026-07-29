using TTDynamics
using TTSolver
using Test

@testset "TTDynamics.jl" begin
    # Write your tests here.
end

include("../examples/tfd/ksl/utils.jl")

@testset "KSL example utilities" begin
    @test admissible_tt_ranks([2, 3, 4]) == [1, 2, 4, 1]
    @test admissible_tt_ranks([2, 3, 4], 3) == [1, 2, 3, 1]
    @test_throws ArgumentError admissible_tt_ranks(Int[], 30)
    @test_throws ArgumentError admissible_tt_ranks([2, 0, 4], 30)
    @test_throws ArgumentError admissible_tt_ranks([2, 3, 4], 0)

    x0 = initial_tt_state([SpinSpace(0.5), BosonSpace(3), BosonSpace(4)];
                          state_index=1, site=1)
    target_ranks = admissible_tt_ranks(tt_dims(x0), 30)
    padded = pad_tt_ranks(x0, target_ranks)

    @test tt_ranks(padded) == target_ranks
    @test tt_full(padded) == tt_full(x0)
    @test_throws ArgumentError pad_tt_ranks(x0, [1, 2, 1])
    @test_throws ArgumentError pad_tt_ranks(x0, [2, 2, 4, 1])
    @test_throws ArgumentError pad_tt_ranks(padded, [1, 1, 1, 1])

    spin_superposition = product_tt_state([
        ComplexF64[1 / sqrt(2), 1im / sqrt(2)],
        ComplexF64[1, 0, 0],
    ])
    @test first_site_populations(spin_superposition) ≈ [0.5, 0.5]
end

@testset "KSL entangled first-site populations" begin
    # This catches an implementation that drops complex off-diagonal bond
    # environments while contracting the bath sites.
    cores = [
        reshape(ComplexF64[1 + 2im, 2 - im, -1im, 1 - 3im], 1, 2, 2),
        reshape(
            ComplexF64[
                1, 1im, 2 - im, -1,
                2im, 1 - im, -2, 1 + 2im,
            ],
            2,
            2,
            2,
        ),
        reshape(ComplexF64[1 - im, 2im, -1, 1 + im, 2 - 2im, -2im], 2, 3, 1),
    ]
    psi = TTTensor(cores)
    dense_psi = tt_full(psi)
    # tt_full returns kron-compatible axes, so the first TT core is the final
    # dense axis.  This dense expansion is independent of the observable
    # contraction below.
    dense_populations = [sum(abs2, @view dense_psi[:, :, state]) for state in 1:2]

    @test first_site_populations(psi) ≈ dense_populations
end

@testset "KSL padded-state evolution" begin
    sigma_x = ComplexF64[0 1; 1 0]
    number = ComplexF64[0 0; 0 1]
    identity2 = Matrix{ComplexF64}(I, 2, 2)
    H = tt_mkron(sigma_x, identity2) + 0.25 * tt_mkron(identity2, number)
    x0 = product_tt_state([
        ComplexF64[1, 0],
        ComplexF64[1, 0],
    ])
    ranks = admissible_tt_ranks(tt_dims(x0), 30)
    x0 = pad_tt_ranks(x0, ranks)
    x1 = tt_ksl(x0, 1im * H, 0.1; symmetric=true, rmax=30)

    @test norm(x1) ≈ norm(x0) rtol=1e-11
end
