"""Default controls for the periodic Holstein HEOM-TT example."""
const DEFAULT_CONFIG = (
    site_count=7,
    site_energies_cm=zeros(Float64, 7),
    hopping_cm=4000.0,
    brownian_frequency_cm=160.0,
    brownian_damping_cm=100.0,
    reorganization_energy_cm=400.0,
    temperature_K=300.0,
    blas_threads=8,
    decomposition_method=:psd,
    initial_condition=:localized,
    initial_site=1,
    time_step_fs=0.5,
    pade_order=0,
    tpsd_tolerance=2e-2,
    pade_type=:Nm1,
    hierarchy_local_size=4,
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

const EQUILIBRATION_TIME_FS = 1000.0
const CORRELATION_TIME_FS = 200.0
