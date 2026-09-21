holstein_output_paths(directory::AbstractString) = (
    csv=joinpath(directory, "holstein_brownian_populations.csv"),
)

function save_holstein_results(output_directory, config::HolsteinConfig, result)
    mkpath(output_directory)
    paths = holstein_output_paths(output_directory)
    write_holstein_population_csv(paths.csv, result)
    return (; csv_path=paths.csv)
end
