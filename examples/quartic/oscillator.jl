using LinearAlgebra

"""
    QuarticMode

Projected local quartic-oscillator basis and operators.
"""
struct QuarticMode
    energies::Vector{Float64}
    hamiltonian::Matrix{ComplexF64}
    q::Matrix{ComplexF64}
    q2::Matrix{ComplexF64}
    p::Matrix{ComplexF64}
    transform::Matrix{ComplexF64}
end

function build_quartic_mode(config::QuarticConfig)
    validate_quartic_config(config)

    d_raw = config.d_raw
    a = zeros(ComplexF64, d_raw, d_raw)
    for n in 1:(d_raw - 1)
        a[n, n + 1] = sqrt(n)
    end

    q_raw = (a + a') / sqrt(2 * config.basis_frequency)
    p_raw = -1im * sqrt(config.basis_frequency / 2) * (a - a')
    h_raw = config.omega_p / 2 * (p_raw * p_raw) +
            config.K2 / 2 * (q_raw * q_raw) +
            config.K4 / 4 * (q_raw * q_raw * q_raw * q_raw)
    eigenbasis = eigen(Hermitian(h_raw))
    energies = Float64.(eigenbasis.values[1:config.d_keep])
    U = ComplexF64.(eigenbasis.vectors[:, 1:config.d_keep])

    project(A) = ComplexF64.(U' * A * U)
    h_projected = Matrix(Diagonal(ComplexF64.(energies)))
    q_projected = project(q_raw)
    q2_projected = project(q_raw * q_raw)
    p_projected = project(p_raw)
    return QuarticMode(energies, h_projected, q_projected, q2_projected, p_projected, U)
end

function quartic_thermal_density(mode::QuarticMode, temperature)
    temperature = Float64(temperature)
    isfinite(temperature) || throw(ArgumentError("temperature must be finite"))
    temperature > 0 || throw(ArgumentError("temperature must be positive"))

    shifted_energies = mode.energies .- minimum(mode.energies)
    weights = exp.(-shifted_energies ./ temperature)
    partition = sum(weights)
    partition > 0 || throw(ArgumentError("thermal partition function must be positive"))
    return ComplexF64.(Matrix(Diagonal(weights ./ partition)))
end
