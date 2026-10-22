# Common plotting utilities for TTDynamics examples
# Uses CairoMakie for publication-quality figures

using CairoMakie
using DelimitedFiles

function _save_example_figure(filename, figure)
    if filename !== nothing
        save(filename, figure)
        println("Saved: $filename")
    end
    return figure
end

"""
    plot_spin_boson_comparison(reference_times, reference_populations,
                               method_times, method_populations;
                               reference_method="Reference", method="TT",
                               title="Spin-Boson Population Dynamics",
                               xlabel="Time", filename=nothing)

Compare two-state spin populations produced on independent time grids.
Each population array must have shape `2 × length(times)`.
"""
function plot_spin_boson_comparison(
    reference_times,
    reference_populations,
    method_times,
    method_populations;
    reference_method::String="Reference",
    method::String="TT",
    title::String="Spin-Boson Population Dynamics",
    xlabel::String="Time",
    filename::Union{String,Nothing}=nothing,
)
    size(reference_populations) == (2, length(reference_times)) ||
        throw(DimensionMismatch("reference populations must have size 2 × length(reference_times)"))
    size(method_populations) == (2, length(method_times)) ||
        throw(DimensionMismatch("method populations must have size 2 × length(method_times)"))

    figure = Figure(size=(900, 600))
    axis = Axis(
        figure[1, 1];
        xlabel,
        ylabel="Population",
        title,
    )
    lines!(axis, reference_times, real.(reference_populations[1, :]);
           label="$reference_method P(up)", linewidth=3, color=:blue)
    lines!(axis, reference_times, real.(reference_populations[2, :]);
           label="$reference_method P(down)", linewidth=3, color=:red)
    lines!(axis, method_times, real.(method_populations[1, :]);
           label="$method P(up)", linewidth=2, linestyle=:dash, color=:cyan)
    lines!(axis, method_times, real.(method_populations[2, :]);
           label="$method P(down)", linewidth=2, linestyle=:dash, color=:orange)
    axislegend(axis; position=:rt)

    return _save_example_figure(filename, figure)
end

"""
    plot_spin_boson_populations(ts, p_up_exact, p_up_tt, p_dn_exact, p_dn_tt;
                                 method="TT", title_suffix="", filename=nothing)

Plot spin populations (P(↑) and P(↓)) comparing exact and TT solutions.

# Arguments
- `ts`: Time array
- `p_up_exact`, `p_up_tt`: Spin-up populations (exact and TT)
- `p_dn_exact`, `p_dn_tt`: Spin-down populations (exact and TT)
- `method`: Method name for legend (default: "TT")
- `title_suffix`: Additional text for title (default: "")
- `filename`: If provided, save figure to this file

# Returns
- `fig`: CairoMakie Figure object
"""
function plot_spin_boson_populations(
    ts, p_up_exact, p_up_tt, p_dn_exact, p_dn_tt;
    method::String="TT",
    title_suffix::String="",
    filename::Union{String, Nothing}=nothing
)
    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1];
        xlabel="t",
        ylabel="Population",
        title="Spin-Boson" * (isempty(title_suffix) ? "" : " ($title_suffix)")
    )

    # Exact solutions (solid lines)
    lines!(ax, ts, p_up_exact; label="exact P(↑)", linewidth=2, color=:blue)
    lines!(ax, ts, p_dn_exact; label="exact P(↓)", linewidth=2, color=:red)

    # TT solutions (dashed lines)
    lines!(ax, ts, p_up_tt; label="$method P(↑)", linewidth=2, linestyle=:dash, color=:cyan)
    lines!(ax, ts, p_dn_tt; label="$method P(↓)", linewidth=2, linestyle=:dash, color=:orange)

    axislegend(ax; position=:rt)

    return _save_example_figure(filename, fig)
end

"""
    plot_norm_conservation(ts, norm_exact, norm_tt;
                           method="TT", filename=nothing)

Plot absolute norm errors comparing exact and TT solutions on a logarithmic
scale. Exact zeros are clipped to `eps(Float64)` for plotting only.

# Arguments
- `ts`: Time array
- `norm_exact`, `norm_tt`: Norms (exact and TT)
- `method`: Method name for legend (default: "TT")
- `filename`: If provided, save figure to this file

# Returns
- `fig`: CairoMakie Figure object
"""
function plot_norm_conservation(
    ts, norm_exact, norm_tt;
    method::String="TT",
    filename::Union{String, Nothing}=nothing
)
    fig = Figure(size=(600, 400))
    ax = Axis(fig[1, 1];
        xlabel="t",
        ylabel="|Norm - 1|",
        title="Norm Conservation Error",
        yscale=log10,
    )

    plot_floor = eps(Float64)
    exact_error = max.(abs.(norm_exact .- 1), plot_floor)
    tt_error = max.(abs.(norm_tt .- 1), plot_floor)
    lines!(ax, ts, exact_error; label="exact |norm - 1|", linewidth=2, color=:blue)
    lines!(ax, ts, tt_error; label="$method |norm - 1|", linewidth=2,
           linestyle=:dash, color=:orange)

    axislegend(ax; position=:rt)

    return _save_example_figure(filename, fig)
end

"""
    plot_norm_conservation(ts, norms; method="TT", target=1.0,
                           quantity="norm",
                           title=nothing,
                           xlabel="Time", filename=nothing)

Plot `abs(norms - target)` on a logarithmic scale. Exact zeros are clipped to
`eps(Float64)` for plotting only; the input data are not modified.
"""
function plot_norm_conservation(
    ts,
    norms;
    method::String="TT",
    target::Real=1.0,
    quantity::String="norm",
    title::Union{String,Nothing}=nothing,
    xlabel::String="Time",
    filename::Union{String,Nothing}=nothing,
)
    length(norms) == length(ts) ||
        throw(DimensionMismatch("norms and times must have the same length"))

    target_label = isone(target) ? "1" : string(target)
    errors = max.(abs.(norms .- target), eps(Float64))
    figure = Figure(size=(600, 400))
    axis = Axis(
        figure[1, 1];
        xlabel,
        ylabel="|$quantity - $target_label|",
        title=isnothing(title) ? "$method Norm Conservation Error" : title,
        yscale=log10,
    )
    lines!(axis, ts, errors; label="$method |$quantity - $target_label|", linewidth=2,
           color=:blue)
    axislegend(axis; position=:rt)

    return _save_example_figure(filename, figure)
end

"""
    plot_tt_rank_history(ts, rank_history; method="TT",
                         title=nothing,
                         xlabel="Time", filename=nothing)

Plot every internal TT bond rank together with the maximum rank.
"""
function plot_tt_rank_history(
    ts,
    rank_history;
    method::String="TT",
    title::Union{String,Nothing}=nothing,
    xlabel::String="Time",
    filename::Union{String,Nothing}=nothing,
)
    length(rank_history) == length(ts) ||
        throw(DimensionMismatch("rank history and times must have the same length"))
    isempty(rank_history) && throw(ArgumentError("rank history must not be empty"))
    rank_count = length(first(rank_history))
    rank_count >= 3 || throw(ArgumentError("rank history must contain at least one internal bond"))
    all(ranks -> length(ranks) == rank_count, rank_history) ||
        throw(DimensionMismatch("all rank-history entries must have the same length"))

    figure = Figure(size=(900, 600))
    axis = Axis(
        figure[1, 1];
        xlabel,
        ylabel="TT rank",
        title=isnothing(title) ? "$method TT Rank History" : title,
    )
    for bond_index in 2:(rank_count - 1)
        lines!(
            axis,
            ts,
            getindex.(rank_history, bond_index);
            label="bond $(bond_index - 1)",
        )
    end
    lines!(
        axis,
        ts,
        maximum.(rank_history);
        label="maximum rank",
        color=:black,
        linewidth=3,
    )
    axislegend(axis; position=:rt)

    return _save_example_figure(filename, figure)
end

"""
    save_spin_boson_data(filename, ts, p_up_exact, p_up_tt, p_dn_exact, p_dn_tt,
                         norm_exact, norm_tt)

Save spin-boson simulation data to CSV file.

Columns: t, p_up_exact, p_up_tt, p_dn_exact, p_dn_tt, norm_exact, norm_tt
"""
function save_spin_boson_data(
    filename, ts, p_up_exact, p_up_tt, p_dn_exact, p_dn_tt, norm_exact, norm_tt
)
    data = hcat(ts, p_up_exact, p_up_tt, p_dn_exact, p_dn_tt, norm_exact, norm_tt)
    writedlm(filename, data, ',')
    println("Saved: $filename")
end

"""
    compute_spin_populations(psi_vec, nb)

Compute spin-up and spin-down populations from wavefunction.

For kron(spin, boson) ordering:
- spin-up:   indices 1:nb
- spin-down: indices nb+1:2*nb

# Arguments
- `psi_vec`: Vector of wavefunctions at each time step
- `nb`: Number of boson states

# Returns
- `p_up`: Spin-up population at each time step
- `p_dn`: Spin-down population at each time step
"""
function compute_spin_populations(psi_vec::Vector{<:AbstractVector}, nb::Integer)
    p_up = [sum(abs2, v[1:nb]) for v in psi_vec]
    p_dn = [sum(abs2, v[(nb + 1):end]) for v in psi_vec]
    return p_up, p_dn
end

export plot_spin_boson_populations, plot_spin_boson_comparison,
       plot_norm_conservation, plot_tt_rank_history, save_spin_boson_data,
       compute_spin_populations
