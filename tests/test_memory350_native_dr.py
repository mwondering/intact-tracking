from types import SimpleNamespace

import pytest
import torch

from intact_tracking.memory350_native_dr import native_nominal_ids, native_metric_schema, CapturedAction
from intact_tracking.memory350_dr_center_rollout import normalize_dr_metric
from intact_tracking.rollout.nominal import NominalPairRollout
from intact_tracking.memory350_response_window import ResponseWindowCollector
from intact_tracking.environment.mdp.sp import substep_tracking_cache
from test_memory350_response_window import FakeA, FakeB


def test_nominal_minimum_and_both_train_validation_partitions():
    ids = native_nominal_ids(8192, .1)
    assert len(ids) == 820 and len(ids.unique()) == 820
    assert (ids < 8064).sum() == 807 and (ids >= 8064).sum() == 13
    assert (ids >= 0).all() and (ids < 8192).all()
    with pytest.raises(ValueError):
        native_nominal_ids(1, .1)


def test_native_metric_includes_gain_factors_without_payload_dimensions():
    names = ['base_com/com_offset/torso_link/x', 'base_mass/relative_mass/torso_link',
             'foot_friction/friction/shared/0', 'motor_params_implicit/kp_scale/hip_joint',
             'motor_params_implicit/kd_scale/hip_joint', 'motor_params_implicit/armature_scale/hip_joint',
             'motor_params_implicit/kp_scale/knee_joint']
    params = {'base_com': {'ranges': {0: [-.075, .075]}}, 'base_mass': {'ranges': [-1., 1.]},
              'foot_friction': {'ranges': [.3, 2.]}, 'motor_params_implicit': {
                  'stiffness_range': {'.*': [.8, 1.2]}, 'damping_range': {'.*': [.8, 1.2]},
                  'armature_range': {'.*': [.8, 1.2]}}}
    schema = native_metric_schema(names, params, {'torso_link': 10.})
    assert len(schema['groups']) == 6
    for indices in schema['groups'].values():
        assert sum(schema['coordinate_weights'][i] for i in indices) == pytest.approx(1/6)
    metric = normalize_dr_metric(torch.tensor([schema['lower'], schema['upper']]), schema)
    torch.testing.assert_close((metric[1]-metric[0]).norm(), torch.tensor(1.))


def test_response_collector_replays_substeps_but_replay_keeps_mean_action():
    class SubstepA(FakeA):
        def step(self, **kwargs):
            batch = super().step(**kwargs)
            trace = batch['joint_target'][:, None].repeat(1, 4, 1)
            trace += torch.arange(4)[None, :, None]
            batch['joint_target_substeps'] = trace
            batch['joint_target'] = trace.mean(1)
            return batch
    admitted = []
    b = FakeB()
    ResponseWindowCollector()(SubstepA(), b, SimpleNamespace(add_step=admitted.append))
    assert b.calls[0][1].shape == (1, 10, 4, 29)
    torch.testing.assert_close(b.calls[0][1][0, 0, :, 0], torch.arange(4).float())
    assert admitted[0]['joint_target'].shape == (1, 29)
    assert admitted[0]['joint_target'][0, 0] == 1.5


def test_nominal_replays_each_physics_substep_in_order(monkeypatch):
    import intact_tracking.rollout.nominal as module
    applied = []
    robot = SimpleNamespace(set_joint_position_target=lambda value, **kwargs: applied.append(value.clone()))
    sim = SimpleNamespace(step=lambda: None, forward=lambda: None)
    scene = SimpleNamespace(write_data_to_sim=lambda: None, update=lambda **kwargs: None)
    env = SimpleNamespace(cfg=SimpleNamespace(decimation=4), sim=sim, scene=scene, physics_dt=.005)
    obj = object.__new__(NominalPairRollout)
    obj.env, obj.robot, obj.config, obj._env_ids = env, robot, SimpleNamespace(horizon=2), torch.arange(1)
    monkeypatch.setattr(module, '_robot_raw_state', lambda _: torch.zeros(1, 71))
    targets = torch.arange(8.).reshape(1, 2, 4, 1).expand(1, 2, 4, 29)
    result = obj._step_joint_targets(targets)
    assert result.shape == (1, 2, 71)
    assert [float(x[0, 0]) for x in applied] == list(range(8))


def test_new_reward_accumulators_do_not_cancel_opposite_signed_substeps():
    obj = object.__new__(substep_tracking_cache)
    obj.decimation = 2
    obj._substep_count = 0
    obj._joint_pos = torch.zeros(1, 2, 2)
    obj._joint_vel = torch.zeros_like(obj._joint_pos)
    obj._joint_acc_sq_sum = torch.zeros(1, 2)
    obj._joint_power_abs_sum = torch.zeros(1, 2)
    obj._contact_found = torch.zeros(1, 2, 2, dtype=torch.bool)
    obj._joint_torque_samples = None
    obj._read_contact_found = lambda: torch.zeros(1, 2, dtype=torch.bool)
    obj._metric_value = torch.zeros(1)
    data = SimpleNamespace(joint_pos=torch.zeros(1, 2), joint_vel=torch.tensor([[3., 4.]]),
                           joint_acc=torch.tensor([[2., 3.]]), qfrc_actuator=torch.tensor([[2., 2.]]))
    obj.asset = SimpleNamespace(data=data)
    obj(None)
    data.joint_acc.neg_()
    data.joint_vel.neg_()
    obj(None)
    torch.testing.assert_close(obj.joint_acc_squared_average(), torch.tensor([[4., 9.]]))
    torch.testing.assert_close(obj.joint_power_absolute_average(), torch.tensor([[6., 8.]]))
