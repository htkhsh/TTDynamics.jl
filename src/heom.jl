"""
    HEOMTTSystem

HEOM-TT system defined using NoiseExp from KaisouEOM.

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
end

# Constructors
function HEOMTTSystem(H_sys::AbstractMatrix, noise::NoiseExp, nb::Vector{Int})
    @assert length(nb) == noise.nterms "nb must have length equal to number of BCF terms"
    return HEOMTTSystem(ComplexF64.(H_sys), noise, nb)
end

# Convenience constructor: uniform nb for all modes
function HEOMTTSystem(H_sys::AbstractMatrix, noise::NoiseExp, nb::Int)
    return HEOMTTSystem(ComplexF64.(H_sys), noise, fill(nb, noise.nterms))
end

# Accessor functions
nsys(sys::HEOMTTSystem) = size(sys.H_sys, 1)
n_bcf(sys::HEOMTTSystem) = sys.noise.nterms

"""Decay rates (HEOM convention: positive real part, C(t) = Σ c exp(-γt))"""
γ(sys::HEOMTTSystem) = ComplexF64.(sys.noise.γ)

"""c1 coefficients from NoiseExp"""
c1(sys::HEOMTTSystem) = ComplexF64.(sys.noise.c1)

"""c2 coefficients from NoiseExp"""
c2(sys::HEOMTTSystem) = ComplexF64.(sys.noise.c2)

"""    build_heom_liouvillian(params; tol=1e-10) → (XOP, Ix, Pop)"""
function build_heom_liouvillian(sys::HEOMTTSystem; tol::Float64=1e-12)
    
    # Use accessor functions
    Ns = nsys(sys)
    nb = sys.nb
    γ_val = γ(sys)
    H_sys_val = sys.H_sys
    noise = sys.noise
    nbcf = n_bcf(sys)
    
    println("Building HEOM Liouvillian in TT format...")
    
    # System identity in super-operator space (nsys² × nsys²)
    Is_super = Matrix{ComplexF64}(I, Ns^2, Ns^2)
    Is = Matrix{ComplexF64}(I, Ns, Ns)
    
    # Bath identity matrices for each mode
    Ib = [Matrix{ComplexF64}(I, nb[k], nb[k]) for k in 1:nbcf]
    
    # Bath ladder operators (as dense matrices)
    n_ops = [diagm(0 => ComplexF64.(0:nb[k]-1)) for k in 1:nbcf]
    bp_ops = [diagm(-1 => sqrt.(ComplexF64.(1:nb[k]-1))) for k in 1:nbcf]  # Creation
    bm_ops = [diagm(1 => sqrt.(ComplexF64.(1:nb[k]-1))) for k in 1:nbcf]   # Annihilation
    
    # =========================================
    # Decay operator G = I_sys ⊗ Σₖ γₖ nₖ
    # =========================================
    G = nothing
    for k in 1:nbcf
        Ib_n = copy(Ib)
        Ib_n[k] = n_ops[k]
        Gk = γ_val[k] * tt_mkron(Is_super, Ib_n...)
        if isnothing(G)
            G = Gk
        else
            G = G + Gk
            G = tt_round(G, tol)
        end
    end
    
    # =========================================
    # Relaxation operators L±
    # =========================================
    Lplus = nothing
    Lmins = nothing
    
    for k in 1:nbcf
        Sk = ComplexF64.(noise.V[k])
        sqrt_abs_c = sqrt(noise.abs_coeff[k])
        c1k = noise.c1[k]
        c2k = noise.c2[k]
        
        # Bath operators with bₖ⁻ or bₖ⁺ at position k
        Ib_bm = copy(Ib); Ib_bm[k] = bm_ops[k]
        Ib_bp = copy(Ib); Ib_bp[k] = bp_ops[k]
        
        # L⁺: [V, •] = Vρ - ρV, coefficient sqrt(|c|)
        Vx_k = kron(Sk, Is) - kron(Is, transpose(Sk))
        Lplus_k = tt_mkron(sqrt_abs_c * Vx_k, Ib_bm...)
        Lplus = isnothing(Lplus) ? Lplus_k : Lplus + Lplus_k
        
        # L⁻: c1/sqrt(|c|) * Vρ ⊗ b⁺ - c2/sqrt(|c|) * ρV† ⊗ b⁺
        if abs(c1k) > tol
            Lmins_k = tt_mkron((c1k / sqrt_abs_c) * kron(Sk, Is), Ib_bp...)
            Lmins = isnothing(Lmins) ? Lmins_k : Lmins + Lmins_k
        end
        if abs(c2k) > tol
            Lmins_k = tt_mkron((c2k / sqrt_abs_c) * kron(Is, conj(Sk)), Ib_bp...)
            Lmins = isnothing(Lmins) ? (-1) * Lmins_k : Lmins - Lmins_k
        end
        
        # Round periodically
        if k % 10 == 0
            !isnothing(Lplus) && (Lplus = tt_round(Lplus, tol))
            !isnothing(Lmins) && (Lmins = tt_round(Lmins, tol))
        end
    end
    println("  Built L⁺ and L⁻ (relaxation) operators")
    
    # L_sys = -i[H, ρ] → -i(H ⊗ I - I ⊗ H) vec(ρ)  (for Hermitian H)
    # Note: vec(Aρ) = (A ⊗ I) vec(ρ), vec(ρA†) = (I ⊗ A†) vec(ρ)
    L_sys_mat = kron(H_sys_val, Is) - kron(Is, H_sys_val)
    L_sys_full = tt_mkron(L_sys_mat, Ib...)
    println("  Built L_sys (system Liouvillian)")
    
    XOP = (-1im) * L_sys_full - G
    if !isnothing(Lplus)
        XOP = XOP - (1im) * Lplus
    end
    if !isnothing(Lmins)
        XOP = XOP - (1im) * Lmins
    end
    XOP = tt_round(XOP, tol)
    println("  XOP number of cores: $(length(XOP.cores))")
    
    # =========================================
    # Build auxiliary operators
    # =========================================
    
    # Trace operator: |Tr⟩ = Σᵢ |i,i⟩ ⊗ |0,0,...,0⟩
    I1_vec = zeros(ComplexF64, Ns^2)
    for i in 1:Ns
        idx = (i-1)*Ns + i  # Index of |i,i⟩ in vectorized form
        I1_vec[idx] = 1.0
    end
    Ix = TTTensor([reshape(I1_vec, 1, Ns^2, 1)])
    for k in 1:nbcf
        v0_k = zeros(ComplexF64, nb[k])
        v0_k[1] = 1.0
        v0_k_tt = TTTensor([reshape(v0_k, 1, nb[k], 1)])
        Ix = tkron(Ix, v0_k_tt)
    end
    Ix = tt_round(Ix, tol)
    
    # Population operators
    Pop = TTTensor{ComplexF64}[]
    for k in 1:Ns
        pop_k = zeros(ComplexF64, Ns, Ns)
        pop_k[k, k] = 1.0
        pop_k_vec = vec(pop_k)
        
        pop_k_tt = TTTensor([reshape(pop_k_vec, 1, Ns^2, 1)])
        for j in 1:nbcf
            v0_j = zeros(ComplexF64, nb[j])
            v0_j[1] = 1.0
            v0_j_tt = TTTensor([reshape(v0_j, 1, nb[j], 1)])
            pop_k_tt = tkron(pop_k_tt, v0_j_tt)
        end
        push!(Pop, tt_round(pop_k_tt, tol))
    end
    
    return XOP, Ix, Pop
end

"""    build_initial_state(params, init_state; tol=1e-12) → TTTensor"""
function build_initial_state(params::HEOMTTSystem, init_state::Int; tol::Float64=1e-12)
    # Use accessor functions
    Ns = nsys(params)
    nb = params.nb
    nbcf = n_bcf(params)
    
    # Initial density matrix: |init_state⟩⟨init_state|
    rho0 = zeros(ComplexF64, Ns, Ns)
    rho0[init_state, init_state] = 1.0
    rho0_vec = vec(rho0)
    
    # Vectorized density matrix in TT format
    rho0_tt = TTTensor([reshape(rho0_vec, 1, Ns^2, 1)])
    
    # Tensor product with |0,0,...,0⟩ for hierarchy
    for k in 1:nbcf
        v0_k = zeros(ComplexF64, nb[k])
        v0_k[1] = 1.0
        v0_k_tt = TTTensor([reshape(v0_k, 1, nb[k], 1)])
        rho0_tt = tkron(rho0_tt, v0_k_tt)
    end
    
    return tt_round(rho0_tt, tol)
end
