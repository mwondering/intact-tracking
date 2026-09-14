"""Additional deployable measurements, never simulator-state substitutes."""

from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.utils.noise.noise_cfg import GaussianNoiseCfg

IMU_HISTORY = "adaptation_imu_accel_history"


def measured_imu_acceleration(env):
    """Existing pelvis accelerometer: local specific force, including gravity.

    Read the robot XML's raw sensor buffer, not root velocity or acceleration
    reconstructed from privileged state. Observation-manager noise is applied
    before inserting this reading into its causal history.
    """
    address = getattr(env, "_adaptation_imu_address", None)
    if address is None:
        model = env.sim.mj_model
        ids = [
            i for i in range(model.nsensor)
            if model.sensor(i).name.split("/")[-1] == "imu_lin_acc"
        ]
        if len(ids) != 1 or int(model.sensor_dim[ids[0]]) != 3:
            raise ValueError("Expected one existing 3D pelvis IMU accelerometer")
        address = int(model.sensor_adr[ids[0]])
        env._adaptation_imu_address = address
    return env.sim.data.sensordata[:, address : address + 3]


def configure_context_sensors(cfg):
    """Keep every existing observation and DR event unchanged; append noisy IMU."""
    cfg.observations[IMU_HISTORY] = ObservationGroupCfg(
        terms={
            "specific_force": ObservationTermCfg(
                func=measured_imu_acceleration,
                noise=GaussianNoiseCfg(std=0.2),
                history_length=50,
            )
        },
        concatenate_terms=True,
        enable_corruption=True,
    )
