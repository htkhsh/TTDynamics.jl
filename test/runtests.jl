using TTDynamics
using TTSolver
using Test
using HEOMKit

@testset "TTDynamics.jl" begin
    # Core package tests only. Holstein and Fröhlich examples are maintained
    # and run separately in their own example environments.
end

include("tt_io.jl")
include("multicore_heom.jl")
include("heom_twin_space.jl")
include("quartic.jl")
