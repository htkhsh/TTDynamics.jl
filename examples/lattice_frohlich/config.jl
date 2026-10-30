"""Default controls for the periodic lattice Fröhlich HEOM-TT example."""
const DEFAULT_CONFIG = (
    output_directory="output-n15-f2-g300",
    site_count=15,
    site_energies_cm=zeros(Float64, 15),
    hopping_cm=4000.0,
    brownian_frequency_cm=160.0,
    brownian_damping_cm=300.0,
    reorganization_energy_cm=400.0,
    frohlich_range=2.0,
    temperature_K=300.0,
    blas_threads=8,
    decomposition_method=:psd,
    initial_condition=:localized,
    initial_site=1,
    time_step_fs=0.5,
    pade_order=0,
    tpsd_tolerance=2e-2,
    pade_type=:Nm1,
    hierarchy_local_size=5,
    temporal_basis_size=3,
    equilibration_tamen_tolerance=1e-2,
    correlation_tamen_tolerance=2e-2,
    operator_tolerance=1e-10,
    state_rounding_tolerance=1e-10,
    sweep_count=3,
    local_iterations=5,
    kick_rank=4,
    progress_interval=10,
)

const EQUILIBRATION_TIME_FS = 700.0
const CORRELATION_TIME_FS = 800.0
