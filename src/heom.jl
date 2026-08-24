"""
    HEOMTTSystem

HEOM-TT system defined using NoiseExp from HEOMKit.

# Fields
- `H_sys`: System Hamiltonian
- `noise`: NoiseExp object containing BCF exponentials and coupling operators
- `nb`: Maximum hierarchy depth per mode (Vector{Int}, one per BCF term)

# Accessors
- `nsys(sys)`: System dimension
- `n_bcf(sys)`: Number of BCF terms
- `γ(sys)`: Decay rates (negated exponents for HEOM convention)
- `c1(sys)`, `c2(sys)`: sqrt coefficients
"""
struct HEOMTTSystem
    H_sys::Matrix{ComplexF64}
    noise::NoiseExp
    nb::Vector{Int}

    function HEOMTTSystem(H_sys::Matrix{ComplexF64}, noise::NoiseExp, nb::Vector{Int})
        _validate_heom_system(H_sys, noise, nb)
        new(H_sys, noise, nb)
    end
end

# Constructors
function _validate_heom_system(H_sys::AbstractMatrix, noise::NoiseExp, nb::Vector{Int})
    size(H_sys, 1) == size(H_sys, 2) && !isempty(H_sys) ||
        throw(ArgumentError("H_sys must be a nonempty square matrix"))
    all(isfinite, H_sys) || throw(ArgumentError("H_sys must be finite"))
    length(nb) == noise.nterms ||
        throw(ArgumentError("nb length must equal the number of bath terms"))
    all(>(0), nb) || throw(ArgumentError("all hierarchy sizes must be positive"))
    for (index, coupling) in pairs(noise.V)
        size(coupling) == size(H_sys) ||
            throw(ArgumentError("bath coupling $index must match H_sys dimensions"))
        all(isfinite, coupling) ||
            throw(ArgumentError("bath coupling $index must be finite"))
    end
    return nothing
end

function HEOMTTSystem(H_sys::AbstractMatrix, noise::NoiseExp, nb::Vector{Int})
    return HEOMTTSystem(ComplexF64.(H_sys), noise, Int.(nb))
end

# Convenience constructor: uniform nb for all modes
function HEOMTTSystem(H_sys::AbstractMatrix, noise::NoiseExp, nb::Int)
    return HEOMTTSystem(H_sys, noise, fill(nb, noise.nterms))
end

# Accessor functions
nsys(sys::HEOMTTSystem) = size(sys.H_sys, 1)
n_bcf(sys::HEOMTTSystem) = sys.noise.nterms

"""    heom_tt_dimensions(system) -> Vector{Int}

Return the canonical twin-space HEOM mode dimensions
`[ket, bra, hierarchy...]`.
"""
heom_tt_dimensions(system::HEOMTTSystem) =
    [nsys(system), nsys(system), system.nb...]

"""Decay rates (HEOM convention: positive real part, C(t) = Σ c exp(-γt))"""
γ(sys::HEOMTTSystem) = ComplexF64.(sys.noise.γ)

"""c1 coefficients from NoiseExp"""
c1(sys::HEOMTTSystem) = ComplexF64.(sys.noise.c1)

"""c2 coefficients from NoiseExp"""
c2(sys::HEOMTTSystem) = ComplexF64.(sys.noise.c2)

function _local_product_tensor(vectors::AbstractVector{<:AbstractVector})
    isempty(vectors) && throw(ArgumentError("at least one local vector is required"))
    all(vector -> !isempty(vector), vectors) ||
        throw(ArgumentError("local vectors must be nonempty"))
    all(vector -> all(isfinite, vector), vectors) ||
        throw(ArgumentError("local vectors must be finite"))
    cores = [reshape(ComplexF64.(vector), 1, length(vector), 1) for vector in vectors]
    return TTTensor(cores)
end

function _local_product_matrix(matrices::AbstractVector{<:AbstractMatrix})
    isempty(matrices) && throw(ArgumentError("at least one local matrix is required"))
    all(matrix -> !isempty(matrix), matrices) ||
        throw(ArgumentError("local matrices must be nonempty"))
    all(matrix -> all(isfinite, matrix), matrices) ||
        throw(ArgumentError("local matrices must be finite"))
    cores = [reshape(ComplexF64.(matrix), 1, size(matrix, 1), size(matrix, 2), 1)
             for matrix in matrices]
    return TTMatrix(cores)
end

function _build_twin_initial_state(system::HEOMTTSystem, init_state::Int;
                                   tol::Float64=1e-12)
    Ns = nsys(system)
    1 <= init_state <= Ns ||
        throw(ArgumentError("initial state must be in 1:$Ns"))

    ket = zeros(ComplexF64, Ns)
    bra = zeros(ComplexF64, Ns)
    ket[init_state] = 1
    bra[init_state] = 1
    vacuum = [ComplexF64[1; zeros(ComplexF64, n - 1)] for n in system.nb]

    return tt_round(_local_product_tensor([ket, bra, vacuum...]), tol)
end

function _validate_heom_state(state::TTTensor, system::HEOMTTSystem)
    expected = heom_tt_dimensions(system)
    actual = tt_dims(state)
    actual == expected && return nothing

    old_layout = [nsys(system)^2, system.nb...]
    actual == old_layout && throw(ArgumentError(
        "old vectorized HEOM state is unsupported; regenerate it in twin-space format",
    ))
    throw(ArgumentError("HEOM state dimensions must be $expected; got $actual"))
end

"""    root_density_matrix(state, system) -> Matrix{ComplexF64}

Extract the root auxiliary density operator by fixing every hierarchy index to
its vacuum state, without expanding the hierarchy tensor.
"""
function root_density_matrix(state::TTTensor, system::HEOMTTSystem)::Matrix{ComplexF64}
    _validate_heom_state(state, system)
    cores = state.cores

    hierarchy_environment = ones(ComplexF64, 1)
    for core in Iterators.reverse(cores[3:end])
        hierarchy_environment = ComplexF64.(core[:, 1, :]) * hierarchy_environment
    end

    Ns = nsys(system)
    density = zeros(ComplexF64, Ns, Ns)
    ket_core = cores[1]
    bra_core = cores[2]
    for ket in 1:Ns, bra in 1:Ns
        density[ket, bra] = (
            reshape(ComplexF64.(ket_core[1, ket, :]), 1, :) *
            ComplexF64.(bra_core[:, bra, :]) * hierarchy_environment
        )[1]
    end
    all(isfinite, density) || throw(ArgumentError(
        "reconstructed root density matrix contains nonfinite values for " *
        "HEOM state with dimensions $(tt_dims(state))",
    ))
    return density
end

function _twin_trace_observable(system::HEOMTTSystem; tol::Float64=1e-12)
    Ns = nsys(system)
    ket_core = zeros(ComplexF64, 1, Ns, Ns)
    bra_core = zeros(ComplexF64, Ns, Ns, 1)
    for index in 1:Ns
        ket_core[1, index, index] = 1
        bra_core[index, index, 1] = 1
    end
    vacuum = [ComplexF64[1; zeros(ComplexF64, n - 1)] for n in system.nb]
    cores = [ket_core, bra_core,
             [reshape(vector, 1, length(vector), 1) for vector in vacuum]...]
    return tt_round(TTTensor(cores), tol)
end

function _twin_population_observable(system::HEOMTTSystem, site::Int;
                                     tol::Float64=1e-12)
    Ns = nsys(system)
    1 <= site <= Ns || throw(ArgumentError("population site must be in 1:$Ns"))

    ket = zeros(ComplexF64, Ns)
    bra = zeros(ComplexF64, Ns)
    ket[site] = 1
    bra[site] = 1
    vacuum = [ComplexF64[1; zeros(ComplexF64, n - 1)] for n in system.nb]
    return tt_round(_local_product_tensor([ket, bra, vacuum...]), tol)
end

function _heom_local_matrix(identity_factors::Vector{Matrix{ComplexF64}}, replacements...)
    factors = copy(identity_factors)
    for (index, factor) in replacements
        checkbounds(Bool, factors, index) ||
            throw(ArgumentError("HEOM local replacement core index $index is out of range"))
        expected = size(identity_factors[index])
        actual = size(factor)
        core_context = index == 1 ? "ket" : index == 2 ? "bra" : "hierarchy $(index - 2)"
        actual == expected || throw(ArgumentError(
            "HEOM local replacement for $core_context core $index must have dimensions " *
            "$expected; got $actual",
        ))
        factors[index] = ComplexF64.(factor)
    end
    return _local_product_matrix(factors)
end

"""    build_heom_liouvillian(params; tol=1e-12) → (XOP, Ix, Pop)"""
function build_heom_liouvillian(sys::HEOMTTSystem; tol::Float64=1e-12)
    Ns = nsys(sys)
    nb = sys.nb
    gamma = γ(sys)
    H = sys.H_sys
    noise = sys.noise
    nbcf = n_bcf(sys)

    println("Building HEOM Liouvillian in TT format...")

    Is = Matrix{ComplexF64}(I, Ns, Ns)
    hierarchy_identities = [Matrix{ComplexF64}(I, n, n) for n in nb]
    identity_factors = [Is, Is, hierarchy_identities...]
    n_ops = [diagm(0 => ComplexF64.(0:(n - 1))) for n in nb]
    bplus_ops = [diagm(-1 => sqrt.(ComplexF64.(1:(n - 1)))) for n in nb]
    bminus_ops = [diagm(1 => sqrt.(ComplexF64.(1:(n - 1)))) for n in nb]

    hamiltonian_left = _heom_local_matrix(identity_factors, 1 => H)
    hamiltonian_right = _heom_local_matrix(identity_factors, 2 => transpose(H))
    hamiltonian = hamiltonian_left - hamiltonian_right
    decay_accumulator = nothing
    upward_accumulator = nothing
    downward_accumulator = nothing

    for k in 1:nbcf
        hierarchy_mode = k + 2
        V = noise.V[k]
        sqrt_abs_c = sqrt(noise.abs_coeff[k])

        decay = _heom_local_matrix(
            identity_factors,
            hierarchy_mode => gamma[k] * n_ops[k],
        )
        upward_left = _heom_local_matrix(
            identity_factors,
            1 => sqrt_abs_c * V,
            hierarchy_mode => bminus_ops[k],
        )
        upward_right = _heom_local_matrix(
            identity_factors,
            2 => sqrt_abs_c * transpose(V),
            hierarchy_mode => bminus_ops[k],
        )

        decay_accumulator = isnothing(decay_accumulator) ?
                            decay : decay_accumulator + decay
        decay_accumulator = tt_round(decay_accumulator, tol)

        upward = upward_left - upward_right
        upward_accumulator = isnothing(upward_accumulator) ?
                             upward : upward_accumulator + upward

        if abs(noise.c1[k]) > tol
            downward_left = _heom_local_matrix(
                identity_factors,
                1 => (noise.c1[k] / sqrt_abs_c) * V,
                hierarchy_mode => bplus_ops[k],
            )
            downward_accumulator = isnothing(downward_accumulator) ?
                                   downward_left : downward_accumulator + downward_left
        end
        if abs(noise.c2[k]) > tol
            downward_right = _heom_local_matrix(
                identity_factors,
                2 => (noise.c2[k] / sqrt_abs_c) * conj(V),
                hierarchy_mode => bplus_ops[k],
            )
            downward_accumulator = isnothing(downward_accumulator) ?
                                   -downward_right : downward_accumulator - downward_right
        end

        if k % 10 == 0
            upward_accumulator = tt_round(upward_accumulator, tol)
            !isnothing(downward_accumulator) &&
                (downward_accumulator = tt_round(downward_accumulator, tol))
        end
    end

    !isnothing(upward_accumulator) &&
        (upward_accumulator = tt_round(upward_accumulator, tol))
    !isnothing(downward_accumulator) &&
        (downward_accumulator = tt_round(downward_accumulator, tol))

    liouvillian = (-1im) * hamiltonian
    !isnothing(decay_accumulator) &&
        (liouvillian = liouvillian - decay_accumulator)
    !isnothing(upward_accumulator) &&
        (liouvillian = liouvillian - 1im * upward_accumulator)
    !isnothing(downward_accumulator) &&
        (liouvillian = liouvillian - 1im * downward_accumulator)
    liouvillian = tt_round(liouvillian, tol)
    println("  XOP number of cores: $(length(liouvillian.cores))")

    trace_observable = _twin_trace_observable(sys; tol)
    populations = [_twin_population_observable(sys, site; tol) for site in 1:Ns]
    return liouvillian, trace_observable, populations
end

"""    build_initial_state(params, init_state; tol=1e-12) → TTTensor"""
function build_initial_state(params::HEOMTTSystem, init_state::Int; tol::Float64=1e-12)
    return _build_twin_initial_state(params, init_state; tol)
end
