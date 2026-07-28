abstract type PhysicalQuantity end

struct Operator{T<:Complex} <: PhysicalQuantity
    matrix::Matrix{T}
end

struct Hamiltonian{T<:Complex} <: PhysicalQuantity
    matrix::Matrix{T}
end

Operator(mat::AbstractMatrix) = Operator(Matrix{ComplexF64}(mat))
Hamiltonian(mat::AbstractMatrix) = Hamiltonian(Matrix{ComplexF64}(mat))

Base.size(op::Operator) = size(op.matrix)
Base.getindex(op::Operator, I...) = op.matrix[I...]
Base.Matrix(op::Operator) = op.matrix
Base.size(h::Hamiltonian) = size(h.matrix)
Base.getindex(h::Hamiltonian, I...) = h.matrix[I...]
Base.Matrix(h::Hamiltonian) = h.matrix

# Operator arithmetic
Base.:+(a::Operator, b::Operator) = Operator(a.matrix + b.matrix)
Base.:-(a::Operator, b::Operator) = Operator(a.matrix - b.matrix)
Base.:-(a::Operator) = Operator(-a.matrix)
Base.:*(a::Operator, b::Operator) = Operator(a.matrix * b.matrix)
Base.:*(a::Number, b::Operator) = Operator(a * b.matrix)
Base.:*(a::Operator, b::Number) = Operator(a.matrix * b)
Base.:/(a::Operator, b::Number) = Operator(a.matrix / b)
Base.adjoint(op::Operator) = Operator(adjoint(op.matrix))
Base.:^(op::Operator, n::Integer) = Operator(op.matrix^n)

# Hamiltonian arithmetic
Base.:+(a::Hamiltonian, b::Hamiltonian) = Hamiltonian(a.matrix + b.matrix)
Base.:-(a::Hamiltonian, b::Hamiltonian) = Hamiltonian(a.matrix - b.matrix)
Base.:-(a::Hamiltonian) = Hamiltonian(-a.matrix)
Base.:*(a::Hamiltonian, b::Hamiltonian) = Hamiltonian(a.matrix * b.matrix)
Base.:*(a::Number, b::Hamiltonian) = Hamiltonian(a * b.matrix)
Base.:*(a::Hamiltonian, b::Number) = Hamiltonian(a.matrix * b)
Base.:/(a::Hamiltonian, b::Number) = Hamiltonian(a.matrix / b)
Base.adjoint(h::Hamiltonian) = Hamiltonian(adjoint(h.matrix))
Base.:^(h::Hamiltonian, n::Integer) = Hamiltonian(h.matrix^n)

struct SpinSpace
    S::Float64
    Sx::Operator{ComplexF64}
    Sy::Operator{ComplexF64}
    Sz::Operator{ComplexF64}
    Sp::Operator{ComplexF64}
    Sm::Operator{ComplexF64}
end

struct BosonSpace
    dim::Int
    a::Operator{ComplexF64}
    adag::Operator{ComplexF64}
    n::Operator{ComplexF64}
    x::Operator{ComplexF64}
    p::Operator{ComplexF64}
end

dim(space::SpinSpace) = Int(round(2 * space.S + 1))
dim(space::BosonSpace) = space.dim

function SpinSpace(S::Real)
    d = Int(round(2 * S + 1))
    Sp_mat = zeros(ComplexF64, d, d)
    Sz_mat = zeros(ComplexF64, d, d)
    for i in 1:d
        m = S - (i - 1)
        Sz_mat[i, i] = m
        if i >= 2
            Sp_mat[i - 1, i] = sqrt(S * (S + 1) - m * (m + 1))
        end
    end
    Sm_mat = adjoint(Sp_mat)
    Sx_mat = (Sp_mat + Sm_mat) / 2
    Sy_mat = (Sp_mat - Sm_mat) / (2im)
    SpinSpace(Float64(S), Operator(Sx_mat), Operator(Sy_mat), Operator(Sz_mat), Operator(Sp_mat), Operator(Sm_mat))
end

function BosonSpace(dim::Int)
    a_mat = zeros(ComplexF64, dim, dim)
    n_mat = zeros(ComplexF64, dim, dim)
    for nval in 0:dim-1
        n_mat[nval + 1, nval + 1] = nval
        if nval >= 1
            a_mat[nval, nval + 1] = sqrt(nval)
        end
    end
    adag_mat = adjoint(a_mat)
    x_mat = (a_mat + adag_mat) / sqrt(2)
    p_mat = (a_mat - adag_mat) / (im * sqrt(2))
    BosonSpace(dim, Operator(a_mat), Operator(adag_mat), Operator(n_mat), Operator(x_mat), Operator(p_mat))
end

# Pauli matrices
struct PauliMatrices
    σx::Operator{ComplexF64}
    σy::Operator{ComplexF64}
    σz::Operator{ComplexF64}
    σp::Operator{ComplexF64}
    σm::Operator{ComplexF64}
    I::Operator{ComplexF64}
end

function Base.getproperty(p::PauliMatrices, s::Symbol)
    if s === :sigmax || s === :sigma_x
        return getfield(p, :σx)
    elseif s === :sigmay || s === :sigma_y
        return getfield(p, :σy)
    elseif s === :sigmaz || s === :sigma_z
        return getfield(p, :σz)
    elseif s === :sigmap || s === :sigma_p
        return getfield(p, :σp)
    elseif s === :sigmam || s === :sigma_m
        return getfield(p, :σm)
    else
        return getfield(p, s)
    end
end

Base.propertynames(::PauliMatrices) = (:σx, :σy, :σz, :σp, :σm, :I, :sigmax, :sigmay, :sigmaz, :sigmap, :sigmam, :sigma_x, :sigma_y, :sigma_z, :sigma_p, :sigma_m)

const PAULI = PauliMatrices(
    Operator(ComplexF64[0 1; 1 0]),
    Operator(ComplexF64[0 -im; im 0]),
    Operator(ComplexF64[1 0; 0 -1]),
    Operator(ComplexF64[0 1; 0 0]),
    Operator(ComplexF64[0 0; 1 0]),
    Operator(ComplexF64[1 0; 0 1])
)