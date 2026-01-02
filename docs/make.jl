using TTDynamics
using Documenter

DocMeta.setdocmeta!(TTDynamics, :DocTestSetup, :(using TTDynamics); recursive=true)

makedocs(;
    modules=[TTDynamics],
    authors="htkhsh",
    sitename="TTDynamics.jl",
    format=Documenter.HTML(;
        canonical="https://takahashi.github.io/TTDynamics.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/takahashi/TTDynamics.jl",
    devbranch="main",
)
