using Test
using CairoMakie

include(joinpath(@__DIR__, "..", "examples", "common", "plot_utils.jl"))

@testset "Common example plotting utilities" begin
    reference_times = [0.0, 0.5, 1.0]
    reference_populations = [1.0 0.8 0.6; 0.0 0.2 0.4]
    method_times = [0.0, 1.0]
    method_populations = [1.0 0.61; 0.0 0.39]

    population_figure = plot_spin_boson_comparison(
        reference_times,
        reference_populations,
        method_times,
        method_populations;
        reference_method="HEOM",
        method="KSL",
    )
    population_axis = population_figure.content[1]
    @test length(population_axis.scene.plots) == 4
    @test [plot.label[] for plot in population_axis.scene.plots] ==
          ["HEOM P(up)", "HEOM P(down)", "KSL P(up)", "KSL P(down)"]
    @test getindex.(population_axis.scene.plots[1][1][], 1) == reference_times
    @test getindex.(population_axis.scene.plots[3][1][], 1) == method_times

    norm_figure = plot_norm_conservation(
        method_times,
        [1.0, 1.01];
        method="KSL",
    )
    norm_axis = norm_figure.content[1]
    norm_points = norm_axis.scene.plots[1][1][]
    @test norm_axis.yscale[] === log10
    @test getindex.(norm_points, 2) ≈ [eps(Float64), 0.01]
    @test norm_axis.scene.plots[1].label[] == "KSL |norm - 1|"

    trace_figure = try
        plot_norm_conservation(
            method_times,
            [1.0, 0.999];
            method="HEOM-TT",
            quantity="Tr[ρ_sys]",
        )
    catch error
        error
    end
    @test trace_figure isa Figure
    if trace_figure isa Figure
        trace_axis = trace_figure.content[1]
        @test trace_axis.ylabel[] == "|Tr[ρ_sys] - 1|"
        @test trace_axis.scene.plots[1].label[] ==
              "HEOM-TT |Tr[ρ_sys] - 1|"
    end

    rank_history = [[1, 2, 3, 1], [1, 3, 4, 1]]
    rank_figure = plot_tt_rank_history(method_times, rank_history; method="KSL")
    rank_axis = rank_figure.content[1]
    @test length(rank_axis.scene.plots) == 3
    @test [plot.label[] for plot in rank_axis.scene.plots] ==
          ["bond 1", "bond 2", "maximum rank"]
    @test getindex.(rank_axis.scene.plots[end][1][], 2) == [3, 4]
end
