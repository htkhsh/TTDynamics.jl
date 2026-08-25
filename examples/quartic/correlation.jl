using QuadGK

abstract type AbstractSpectralDensity end

struct CriticallyDampedBrownian <: AbstractSpectralDensity
    lambda::Float64
    gamma::Float64

    function CriticallyDampedBrownian(lambda, gamma)
        λ = Float64(lambda)
        Γ = Float64(gamma)
        isfinite(λ) || throw(ArgumentError("lambda must be finite"))
        isfinite(Γ) || throw(ArgumentError("gamma must be finite"))
        λ >= 0 || throw(ArgumentError("lambda must be nonnegative"))
        Γ > 0 || throw(ArgumentError("gamma must be positive"))
        return new(λ, Γ)
    end
end

spectral_density(bath::CriticallyDampedBrownian, ω) =
    4 * bath.lambda * bath.gamma^3 * ω / (ω^2 + bath.gamma^2)^2

xcothx(x) = abs(x) < 1e-4 ? 1 + x^2 / 3 - x^4 / 45 : x * coth(x)

function spectral_density_over_omega(bath::CriticallyDampedBrownian, ω)
    if iszero(ω)
        return 4 * bath.lambda / bath.gamma
    end
    return spectral_density(bath, ω) / ω
end

function bath_correlation(
    bath::CriticallyDampedBrownian,
    t,
    temperature;
    omega_integration_max=nothing,
    rtol=1e-8,
    atol=1e-10,
    maxevals=100_000,
)
    τ = Float64(t)
    βinv = Float64(temperature)
    isfinite(τ) || throw(ArgumentError("time must be finite"))
    isfinite(βinv) || throw(ArgumentError("temperature must be finite"))
    βinv > 0 || throw(ArgumentError("temperature must be positive"))
    rtol > 0 || throw(ArgumentError("rtol must be positive"))
    atol > 0 || throw(ArgumentError("atol must be positive"))
    maxevals > 0 || throw(ArgumentError("maxevals must be positive"))

    upper = if omega_integration_max === nothing
        Inf
    else
        ωmax = Float64(omega_integration_max)
        isfinite(ωmax) || throw(ArgumentError("omega_integration_max must be finite or nothing"))
        ωmax > 0 || throw(ArgumentError("omega_integration_max must be positive"))
        ωmax
    end

    β = inv(βinv)
    thermal_integrand(ω) = begin
        x = β * ω / 2
        prefactor = (2 / β) * spectral_density_over_omega(bath, ω) * xcothx(x)
        return prefactor * cos(ω * τ) / π
    end
    dissipative_integrand(ω) = -spectral_density(bath, ω) * sin(ω * τ) / π

    real_part, _ = quadgk(thermal_integrand, 0.0, upper; rtol=rtol, atol=atol, maxevals=maxevals)
    imag_part, _ = quadgk(dissipative_integrand, 0.0, upper; rtol=rtol, atol=atol, maxevals=maxevals)
    return ComplexF64(real_part, imag_part)
end

function sample_bath_correlation(
    bath::CriticallyDampedBrownian,
    temperature,
    times;
    omega_integration_max=nothing,
    rtol=1e-8,
    atol=1e-10,
    maxevals=100_000,
)
    return ComplexF64[
        bath_correlation(
            bath,
            t,
            temperature;
            omega_integration_max=omega_integration_max,
            rtol=rtol,
            atol=atol,
            maxevals=maxevals,
        ) for t in times
    ]
end
