"""Controls for the periodic pure-Peierls HEOM-TT example."""
const DEFAULT_CONFIG = (
    output_directory="output",
    site_count=5,
    site_energies_cm=zeros(Float64, 5),
    hopping_cm=800.0,
    bath_model=:drude,
    brownian_frequency_cm=50.0,
    brownian_damping_cm=90.0,
    drude_cutoff_cm=50.0,
    reorganization_energy_cm=100.0,
    temperature_K=300.0,
    blas_threads=4,
    initial_condition=:localized,
    initial_site=1,
    time_step_fs=0.5,
    decomposition_method=:tpsd,
    pade_order=5,
    tpsd_tolerance=1e-2,
    pade_type=:Nm1,
    hierarchy_local_size=15,
    temporal_basis_size=3,
    equilibration_tamen_tolerance=7e-3,
    correlation_tamen_tolerance=2e-2,
    operator_tolerance=1e-10,
    state_rounding_tolerance=1e-10,
    sweep_count=3,
    local_iterations=5,
    kick_rank=4,
    progress_interval=10,
)

const EQUILIBRATION_TIME_FS = 1000.0
const CORRELATION_TIME_FS = 200.0
