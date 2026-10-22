using LinearAlgebra
using Test
using TTSolver
using TTDynamics

@testset "separated Hamiltonians with no static term" begin
    X = tt_matrix(ComplexF64[0 1; 1 0])
    Z = tt_matrix(ComplexF64[1 0; 0 -1])
    terms = [identity => X]
    H = TimeDependentHamiltonian(terms)
    @test tt_full(H(1.3)) ≈ ComplexF64[0 1.3; 1.3 0]
    @test tt_full(TimeDependentHamiltonian(0, terms)(1.3)) ≈ tt_full(H(1.3))
    @test tt_full(TimeDependentHamiltonian(nothing, terms)(1.3)) ≈ tt_full(H(1.3))
    @test tt_full(TimeDependentHamiltonian(Z, terms)(1.3)) ≈ ComplexF64[1 1.3; 1.3 -1]
    @test tt_full(TimeDependentHamiltonian(Z)(1.3)) ≈ tt_full(Z)
    @test_throws ArgumentError TimeDependentHamiltonian(0, [])
    @test_throws ArgumentError TimeDependentHamiltonian(1, terms)
    @test_throws ArgumentError TimeDependentHamiltonian([2 => X])
    @test_throws DimensionMismatch TimeDependentHamiltonian(Z, [identity => tt_eye(3)])
    @test_throws DimensionMismatch TimeDependentHamiltonian(tt_matrix(ones(2, 3)))
    @test_throws ArgumentError TimeDependentHamiltonian(tt_matrix(fill(NaN, 2, 2)))
    @test_throws ArgumentError H(Inf)
    @test_throws ArgumentError TimeDependentHamiltonian([(t -> Inf) => X])(0.0)
end

@testset "driven Schrödinger evolution uses absolute time" begin
    X = tt_matrix(ComplexF64[0 1; 1 0])
    H = TimeDependentHamiltonian([identity => X])
    initial = tt_unit([2], 1; T=ComplexF64)
    times = [1.2, 1.2, 1.35, 1.7]
    saved = deepcopy(X.cores)
    options = Dict(:verb => 0, :nswp => 5, :local_iters => 20,
                   :correct_2_norm => false, :time_error_damp => 10)
    for method in (:ksl, :cheb, :cn)
        events = Tuple{Int,Float64}[]
        callback = (step, time, _) -> push!(events, (step, time))
        if method == :ksl
            trajectory = propagate_ksl(H, initial, times; callback)
        else
            trajectory = propagate_tamen(H, initial, times;
                temporal_basis_size=5, tolerance=1e-7,
                options=merge(options, Dict(:time_scheme => String(method))),
                workspace=TAMEnWorkspace(ComplexF64), callback)
        end
        @test length(trajectory) == length(times)
        @test trajectory[1] === initial
        @test events == [(0, 1.2), (1, 1.2), (2, 1.35), (3, 1.7)]
        for (time, state) in zip(times, trajectory)
            phase = (time^2 - times[1]^2) / 2
            expected = ComplexF64[cos(phase), -1im * sin(phase)]
            @test vec(tt_full(state)) ≈ expected atol=(method == :ksl ? 1e-10 : 2e-6)
        end
    end
    @test X.cores == saved
    @test !haskey(options, :time_scheme)
end

@testset "constant and zero Hamiltonian propagation" begin
    X = tt_matrix(ComplexF64[0 1; 1 0])
    initial = tt_unit([2], 1; T=ComplexF64)
    options = Dict(:verb => 0, :nswp => 4, :local_iters => 10,
                   :correct_2_norm => false)
    for H in (X, TimeDependentHamiltonian(X), TimeDependentHamiltonian(0, [(t -> 1) => X]))
        ksl_state = last(propagate_ksl(H, initial, [2.0, 2.2]))
        @test vec(tt_full(ksl_state)) ≈ exp(-0.2im * tt_full(X)) * ComplexF64[1, 0] atol=1e-10
        tamen_state = last(propagate_tamen(H, initial, [2.0, 2.2];
            temporal_basis_size=5, tolerance=1e-7, options))
        @test vec(tt_full(tamen_state)) ≈ vec(tt_full(ksl_state)) atol=2e-6
    end
    zero_H = TimeDependentHamiltonian([(t -> 0) => X])
    for propagate in (propagate_ksl, propagate_tamen)
        trajectory = propagate(zero_H, initial, [0.0, 0.1])
        @test vec(tt_full(last(trajectory))) ≈ ComplexF64[1, 0] atol=1e-10
        @test propagate(X, initial, [2.0]) == [initial]
        @test_throws ArgumentError propagate(X, initial, [])
        @test_throws ArgumentError propagate(X, initial, [1.0, 0.0])
        @test_throws ArgumentError propagate(X, initial, [0.0, Inf])
        @test_throws DimensionMismatch propagate(X, tt_unit([3]), [0.0, 0.1])
        @test_throws ArgumentError propagate(X, initial, [0.0, 0.1]; tolerance=0)
        @test_throws ArgumentError propagate(TimeDependentHamiltonian([(t -> NaN) => X]), initial, [0.0, 0.1])
    end
    @test_throws ArgumentError propagate_ksl(X, initial, [0.0, 0.1]; rmax=0)
    @test_throws ArgumentError propagate_tamen(X, initial, [0.0, 0.1]; temporal_basis_size=2, options=Dict(:time_scheme => "cn"))
end

@testset "noncommuting driven two-site Hamiltonian" begin
    X = ComplexF64[0 1; 1 0]
    Y = ComplexF64[0 -im; im 0]
    Z = ComplexF64[1 0; 0 -1]
    I2 = Matrix{ComplexF64}(I, 2, 2)
    H = TimeDependentHamiltonian(0.2 * tt_mkron(Z, Z),
        [cos => tt_mkron(X, I2), sin => tt_mkron(Y, I2)])
    initial = TTTensor([reshape(I2 / sqrt(2), 1, 2, 2), reshape(I2, 2, 2, 1)])
    start, stop = 1.1, 1.5
    effective = kron(X, I2) + 0.2 * kron(Z, Z) - 0.5 * kron(Z, I2)
    expected = kron(exp(-0.5im * stop * Z), I2) *
               exp(-1im * (stop - start) * effective) *
               kron(exp(0.5im * start * Z), I2) * ComplexF64[1, 0, 0, 1] / sqrt(2)
    errors = Float64[]
    for intervals in (4, 8)
        states = propagate_ksl(H, initial, range(start, stop; length=intervals + 1))
        push!(errors, norm(vec(tt_full(last(states))) - expected))
    end
    @test errors[2] < errors[1] / 3
    @test errors[2] < 2e-4
    options = Dict(:verb => 0, :nswp => 8, :local_iters => 30,
                   :correct_2_norm => false, :time_error_damp => 10)
    for scheme in ("cheb", "cn")
        states = propagate_tamen(H, initial, [start, 1.3, stop];
            temporal_basis_size=5, tolerance=1e-7,
            options=merge(options, Dict(:time_scheme => scheme)))
        @test vec(tt_full(last(states))) ≈ expected atol=3e-6
    end
end
