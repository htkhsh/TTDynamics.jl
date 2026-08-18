using TTDynamics
using TTSolver
using Test

@testset "TT binary I/O" begin
    @testset "exact round trips" begin
        for T in (Float32, Float64, ComplexF32, ComplexF64)
            values(::Type{S}, n) where {S<:AbstractFloat} = S.(1:n) ./ S(7)
            values(::Type{Complex{S}}, n) where {S<:AbstractFloat} =
                Complex{S}.(S.(1:n) ./ S(7), -S.(n:-1:1) ./ S(11))

            tensor = TTTensor([
                reshape(values(T, 12), 1, 3, 4),
                reshape(values(T, 40), 4, 5, 2),
                reshape(values(T, 4), 2, 2, 1),
            ])
            matrix = TTMatrix([
                reshape(values(T, 24), 1, 2, 3, 4),
                reshape(values(T, 80), 4, 5, 2, 2),
                reshape(values(T, 12), 2, 3, 2, 1),
            ])

            mktempdir() do directory
                for (name, original) in (("tensor", tensor), ("matrix", matrix))
                    path = joinpath(directory, "$name.ttbin")
                    @test save_tt_binary(path, original) == path
                    restored = load_tt_binary(path)
                    @test typeof(restored) === typeof(original)
                    @test eltype(first(restored.cores)) === T
                    @test tt_dims(restored) == tt_dims(original)
                    @test tt_ranks(restored) == tt_ranks(original)
                    @test restored.cores == original.cores
                end
            end
        end
    end

    @testset "version 1 header" begin
        mktempdir() do directory
            path = joinpath(directory, "header.ttbin")
            save_tt_binary(path, TTTensor([reshape(Float32[1], 1, 1, 1)]))
            bytes = read(path)
            @test bytes[1:8] == UInt8[codeunits("TTDYNBIN")...]
            @test bytes[9:10] == UInt8[0x01, 0x00]
            @test bytes[11:14] == UInt8[0x01, 0x01, 0x01, 0x00]
        end
    end
end
