using LinearAlgebra
using Test
using TTSolver
using HEOMKit

if !isdefined(@__MODULE__, :TTDynamics)
    include(joinpath(@__DIR__, "..", "src", "TTDynamics.jl"))
end
using .TTDynamics

@testset "shared tAMEn propagation contract" begin
    h = tt_matrix(ComplexF64[0.2 0.0; 0.0 -0.2])
    system = MultiCoreRedfieldTTSystem(h, [ComplexF64[0.2 0.0; 0.0 -0.2]],
                                       MultiCoreBathCoupling[])
    state = product_tt_state([ComplexF64[1.0, 0.0], ComplexF64[1.0, 0.0]])
    options = Dict{Symbol,Any}(:verb => 0, :nswp => 1, :local_iters => 2)
    original_options = copy(options)
    callback_events = Tuple{Int,Float64}[]

    trajectory = propagate_multicore_redfield_tamen(
        system, state, [0.0, 0.01, 0.02]; temporal_basis_size=2,
        tolerance=1e-3, state_tolerance=1e-8, preserve_trace=true,
        options=options,
        callback=(step, time, _) -> push!(callback_events, (step, time)),
    )
    @test length(trajectory) == 3
    @test tt_dims(trajectory[end]) == [2, 2]
    @test options == original_options
    @test callback_events == [(0, 0.0), (1, 0.01), (2, 0.02)]

    one_point = propagate_multicore_redfield_tamen(
        system, state, [0.0]; options=Dict(:verb => 0),
    )
    @test one_point == [state]
    @test_throws ArgumentError propagate_multicore_redfield_tamen(
        system, state, [0.0, 0.01]; preserve_trace=true,
        options=Dict(:verb => 0, :correct_2_norm => true),
    )
    @test_throws ArgumentError propagate_multicore_redfield_tamen(
        system, state, [0.0, 0.01]; conserved_observables=[tt_ones([2, 2])],
        options=Dict(:verb => 0, :obs => [tt_ones([2, 2])]),
    )
end
