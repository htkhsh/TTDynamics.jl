"""Parameters for Bosonic TFD representation."""
struct BosonicTFD
    nmode::Int
    freq::Vector{Float64}
    coeff::Vector{Float64}
    V::Matrix{ComplexF64}
end

"""Construct BosonicTFD from frequency and coefficient parameters."""
function BosonicTFD(freq::Vector, coeff::Vector, V::AbstractMatrix)
    return BosonicTFD(length(freq), Float64.(freq), Float64.(coeff), ComplexF64.(V))
end

"""Unified boson parameters from multiple environments."""
struct BosonicEnv
    nenv::Int
    envs::Vector{BosonicTFD}         # List of baths
    freq::Vector{Float64}
    coeff::Vector{Float64}
    nmode::Int
    V::Vector{Matrix{ComplexF64}}     # List of interaction operators (nbath)
end


"""Construct BosonicEnv from multiple BosonicTFD objects."""
function BosonicEnv(envs::Vector{BosonicTFD})
    nenvs = length(envs)
    nmode = sum(b.nmode for b in envs)
    freq = vcat([b.freq for b in envs]...)
    coeff = vcat([b.coeff for b in envs]...)
    V = [b.V for b in envs]
    
    return BosonicEnv(nenvs, envs, freq, coeff, nmode, V)
end

function BosonicEnv(env::BosonicTFD)
    return BosonicEnv([env])
end

"""Spin-boson system with system Hamiltonian and bosonic environment."""
struct SBSystem
    hsys::Matrix{ComplexF64}        # System Hamiltonian
    benv::BosonicEnv                 # Bosonic environment
end

"""Construct SBSystem from system Hamiltonian and BosonicEnv."""
function SBSystem(hsys::AbstractMatrix, benv::BosonicEnv)
    return SBSystem(ComplexF64.(hsys), benv)
end

"""Construct SBSystem from system Hamiltonian and single bath parameters."""
function SBSystem(hsys::AbstractMatrix, bfreq::Vector, bcoeff::Vector, bV::AbstractMatrix)
    benv = BosonicEnv(BosonicTFD(bfreq, bcoeff, bV))
    return SBSystem(ComplexF64.(hsys), benv)
end

"""Construct SBSystem from system Hamiltonian and multiple bath parameters."""
function SBSystem(hsys::AbstractMatrix, bfreqs::Vector{Vector}, bcoeffs::Vector{Vector}, bVs::Vector{AbstractMatrix})
    envs = BosonicTFD[]
    for (bfreq, bcoeff, bV) in zip(bfreqs, bcoeffs, bVs)
        push!(envs, BosonicTFD(bfreq, bcoeff, bV))
    end
    benv = BosonicEnv(envs)
    return SBSystem(ComplexF64.(hsys), benv)
end

"""
    build_sbham(sys; basis_sizes=nothing, unit_conv=1.0, threshold=0.999, nb_default=10, nb_max=30)

Build spin-boson Hamiltonian in TT format. Returns `(H_tt, basis_sizes)`.
"""
function tt_sbham(sys::SBSystem;
                basis_sizes::Union{Nothing, AbstractVector{Int}}=nothing,
                threshold::Real=0.999,
                nb_min::Int=10,
                nb_max::Int=30,
                unit_conv::Real=1.0)
    hsys = sys.hsys
    benv = sys.benv
    nsys = size(hsys, 1)
    n_modes = benv.nmode
    
    # Estimate basis sizes if not provided
    if isnothing(basis_sizes)
        basis_sizes = estimate_basis_sizes(benv.freq, benv.coeff; threshold=threshold, nb_min=nb_min, nb_max=nb_max)
    else
        @assert length(basis_sizes) == n_modes "basis_sizes must match number of modes ($n_modes)"
        basis_sizes = collect(basis_sizes)
    end
    
    # Unit conversion
    hsys = ComplexF64.(hsys) * unit_conv
    ω = Float64.(benv.freq) * unit_conv
    g = Float64.(benv.coeff) * unit_conv
    
    # Identity matrices for each space
    Id_sys = Matrix{ComplexF64}(I, nsys, nsys)
    # Create BosonSpace for each mode and extract operators
    boson_spaces = [BosonSpace(basis_sizes[i]) for i in 1:n_modes]
    Id_bosons = [Matrix{ComplexF64}(I, basis_sizes[i], basis_sizes[i]) for i in 1:n_modes]
    x_ops = [Matrix(boson_spaces[i].x) for i in 1:n_modes] 
    n_ops = [Matrix(boson_spaces[i].n) for i in 1:n_modes]
    
    # Build Hamiltonian in TT format H = H_sys ⊗ I_bath + I_sys ⊗ Σᵢ ωᵢ nᵢ + Σⱼ Vⱼ ⊗ Σᵢ gⱼᵢ xᵢ
    # System term: H_sys ⊗ I₁ ⊗ I₂ ⊗ ... ⊗ Iₙ
    H_tt = tt_mkron(hsys, Id_bosons...)
    
    # Bath energy terms I_sys ⊗ ... ⊗ ωᵢ nᵢ ⊗ ... 
    for i in 1:n_modes
        Ids_n = copy(Id_bosons)
        Ids_n[i] = n_ops[i]
        H_tt = H_tt + ω[i] * tt_mkron(Id_sys, Ids_n...)
    end
    
    # Coupling terms for each environment
    mode_offset = 0
    for (env_idx, env) in enumerate(benv.envs)
        V = ComplexF64.(benv.V[env_idx])
        for j in 1:env.nmode
            i = mode_offset + j
            Ids_x = copy(Id_bosons)
            Ids_x[i] = x_ops[i]
            H_tt = H_tt + g[i] * tt_mkron(V, Ids_x...)
        end
        mode_offset += env.nmode
    end
    H_tt = tt_round(H_tt, 1e-12)
    
    return H_tt, basis_sizes
end