# Common plotting utilities for TTSolver examples
# Uses CairoMakie for publication-quality figures

using CairoMakie
using DelimitedFiles

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

    if filename !== nothing
        save(filename, fig)
        println("Saved: $filename")
    end

    return fig
end

"""
    plot_norm_conservation(ts, norm_exact, norm_tt;
                           method="TT", filename=nothing)

Plot norm conservation comparing exact and TT solutions.

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
        ylabel="Norm",
        title="Norm Conservation"
    )

    lines!(ax, ts, norm_exact; label="exact", linewidth=2, color=:blue)
    lines!(ax, ts, norm_tt; label=method, linewidth=2, linestyle=:dash, color=:orange)

    axislegend(ax; position=:rt)

    if filename !== nothing
        save(filename, fig)
        println("Saved: $filename")
    end

    return fig
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

export plot_spin_boson_populations, plot_norm_conservation,
       save_spin_boson_data, compute_spin_populations
