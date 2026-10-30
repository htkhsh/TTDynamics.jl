"""
    electron_lattice_correlation(state, system::HEOMTTSystem)

Measure the equal-time connected correlation on a periodic single-electron
lattice with one independent bath per site (bath order must match site order):
`C_nB(r) = sum_j (<n_j B_mod(j+r,N)> - <n_j><B_mod(j+r,N)>)`.

The bath coordinate is the one in `H_int = sum_l S_l B_l`. In the scaled
`NoiseExp` hierarchy its first moment is
`R_l = sum_{k in bath l} sqrt(abs_coeff[k]) * rho[e_k]`.
Thus `<n_j B_l> = R_l[j,j] / Tr(rho[0])` and
`<B_l> = Tr(R_l) / Tr(rho[0])`. No electronic coupling weights are applied
again, including for nonlocal Fröhlich couplings.

Returns `distances` (`0:N-1`), `correlation`, `uncentered_correlation`,
`joint_moments` (site × bath), `bath_mean`, and `populations`. Expectations
are normalized by the root trace and retain their complex numerical parts.
Coordinates and correlations have the same units as the bath coordinate
(`fs^-1` for the Holstein and lattice Fröhlich examples).

Only root and first-tier TT slices are contracted; the full hierarchy is
never expanded. Every hierarchy mode must have size at least two. Use a
physical HEOM density state, rather than a current-inserted source state.
"""
function electron_lattice_correlation(state::TTTensor, system::HEOMTTSystem)
    _validate_heom_state(state, system)
    n = nsys(system)
    noise = system.noise
    noise.nbath == n || throw(ArgumentError(
        "electron–lattice correlation requires one bath per site in site order",
    ))
    all(>=(2), system.nb) || throw(ArgumentError(
        "electron–lattice correlation requires first-tier ADOs (all hierarchy sizes >= 2)",
    ))
    all(>(0), noise.nterms_bath) || throw(ArgumentError("each bath must have exponential terms"))
    all(s -> isfinite(s) && s >= 0, noise.abs_coeff) ||
        throw(ArgumentError("HEOM scaling must be finite and nonnegative"))

    # Vacuum suffixes are reused for each single excitation.
    modes = length(system.nb)
    right = Vector{Vector{ComplexF64}}(undef, modes + 1)
    right[end] = ones(ComplexF64, 1)
    for k in modes:-1:1
        right[k] = state.cores[k + 2][:, 1, :] * right[k + 1]
    end
    ket, bra = state.cores[1:2]
    left = zeros(ComplexF64, n, size(bra, 3))
    for j in 1:n
        left[j, :] = vec(reshape(ket[1, j, :], 1, :) * bra[:, j, :])
    end
    populations = left * right[1]
    trace = sum(populations)
    isfinite(trace) && !iszero(trace) ||
        throw(ArgumentError("root trace must be finite and nonzero"))
    populations ./= trace

    joint_moments = zeros(ComplexF64, n, n)
    for bath in 1:n
        first_mode = noise.jstart_bath[bath]
        for k in first_mode:(first_mode + noise.nterms_bath[bath] - 1)
            core = state.cores[k + 2]
            joint_moments[:, bath] .+= sqrt(noise.abs_coeff[k]) .*
                ((left * core[:, 2, :]) * right[k + 1])
            left = left * core[:, 1, :]
        end
    end
    joint_moments ./= trace
    all(isfinite, populations) && all(isfinite, joint_moments) ||
        throw(ArgumentError("electron–lattice moments contain nonfinite values"))
    bath_mean = vec(sum(joint_moments; dims=1))
    distances = collect(0:(n - 1))
    uncentered_correlation = ComplexF64[
        sum(joint_moments[j, mod1(j + r, n)] for j in 1:n) for r in distances
    ]
    correlation = ComplexF64[
        uncentered_correlation[r + 1] -
        sum(populations[j] * bath_mean[mod1(j + r, n)] for j in 1:n)
        for r in distances
    ]
    return (; distances, correlation, uncentered_correlation,
            joint_moments, bath_mean, populations)
end
