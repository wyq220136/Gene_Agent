"""抓碗后放入必须按旧阶段顺序，且消费每段真实状态。"""

import numpy as np
import pytest


def test_object_goal_is_independent_of_grasp_and_entry_does_not_rotate_bowl():
    """不同抓法必须把同一个碗目标反算成不同 TCP，不能复制固定 TCP。"""
    from scipy.spatial.transform import Rotation
    from gene_agent.tasks.policies.microwave_task.operations.bowl import object_relative_insert_targets
    goal = np.eye(4)
    goal[:3, :3] = Rotation.from_euler('x', 90, degrees=True).as_matrix()
    goal[:3, 3] = [.50, .33, .13]
    tcp_goals = []
    for angle in (20., 40.):
        attachment = np.eye(4)
        attachment[:3, :3] = Rotation.from_euler('xy', [90, angle], degrees=True).as_matrix()
        attachment[:3, 3] = [.065, 0., 0.]
        targets = object_relative_insert_targets(goal, attachment, [1., 0., 0.])
        tcp_goals.append(targets['insert'])
        np.testing.assert_allclose(targets['insert'] @ attachment, goal, atol=1e-12)
        before = targets['preplace'] @ attachment
        np.testing.assert_allclose(before[:3, :3], goal[:3, :3], atol=1e-12)
        np.testing.assert_allclose(goal[:3, 3] - before[:3, 3], [.16, 0., -.04])
        np.testing.assert_allclose(targets['retreat_1'][:3, 3] - targets['insert'][:3, 3],
                                   [-.06, 0., 0.], atol=1e-12)
        np.testing.assert_allclose(targets['retreat_2'], targets['preplace'])
    assert not np.allclose(tcp_goals[0], tcp_goals[1])


def test_insert_clearance_tilt_flattens_tcp_without_exceeding_stability_margin():
    from scipy.spatial.transform import Rotation
    from gene_agent.tasks.policies.microwave_task.operations.bowl import (
        bounded_insert_tilt_for_tcp_clearance,
    )

    goal = np.eye(4)
    # Bowl 局部 +Y 指向世界 +Z，进炉方向取世界 +X。
    goal[:3, :3] = np.column_stack((
        [1., 0., 0.], [0., 0., 1.], [0., -1., 0.],
    ))
    goal[:3, 3] = [.43, .33, .13]
    nominal_tcp = np.eye(4)
    nominal_tcp[:3, :3] = Rotation.from_euler('y', 15., degrees=True).as_matrix()
    attachment = np.linalg.inv(nominal_tcp) @ goal

    adjusted, evidence = bounded_insert_tilt_for_tcp_clearance(
        goal, attachment, [1., 0., 0.],
        max_tcp_table_angle_deg=60., max_object_tilt_deg=12.,
    )

    np.testing.assert_allclose(adjusted[:3, 3], goal[:3, 3])
    assert abs(evidence['applied_object_tilt_deg']) == pytest.approx(12.)
    assert evidence['nominal_tcp_table_angle_deg'] == pytest.approx(75.)
    assert evidence['final_tcp_table_angle_deg'] == pytest.approx(63.)
    assert evidence['target_met'] is False
    bowl_axis = adjusted[:3, 1]
    bowl_tilt = np.rad2deg(np.arccos(np.clip(bowl_axis @ [0., 0., 1.], -1., 1.)))
    assert bowl_tilt == pytest.approx(12.)


def test_insert_clearance_tilt_keeps_already_flat_tcp_unchanged():
    from scipy.spatial.transform import Rotation
    from gene_agent.tasks.policies.microwave_task.operations.bowl import (
        bounded_insert_tilt_for_tcp_clearance,
    )

    goal = np.eye(4)
    goal[:3, :3] = np.column_stack((
        [1., 0., 0.], [0., 0., 1.], [0., -1., 0.],
    ))
    tcp = np.eye(4)
    tcp[:3, :3] = Rotation.from_euler('y', 35., degrees=True).as_matrix()
    attachment = np.linalg.inv(tcp) @ goal
    adjusted, evidence = bounded_insert_tilt_for_tcp_clearance(
        goal, attachment, [1., 0., 0.],
    )
    np.testing.assert_allclose(adjusted, goal)
    assert evidence['nominal_tcp_table_angle_deg'] == pytest.approx(55.)
    assert evidence['applied_object_tilt_deg'] == 0.
    assert evidence['target_met'] is True


def test_insert_contact_guard_rejects_wrist_and_bowl_cabinet_contacts():
    import mujoco
    from gene_agent.tasks.policies.microwave_task.simulation_bowl.insert import insertion_contact_check
    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="microwave"><geom name="microwave_turntable_collision" type="box" size=".1 .1 .01"/>
        <geom name="ceiling" pos="0 0 .3" type="box" size=".1 .1 .01"/></body>
      <body name="arm" pos=".3 0 .2"><freejoint/>
        <geom name="wrist" type="sphere" size=".02"/></body>
      <body name="bowl" pos="0 0 .029"><freejoint/>
        <geom name="bowl_bottom" type="sphere" size=".02"/></body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    metadata = {'group_root_bodies': {'right': 'arm'},
                'grippers': {'right': {'pad_geom_names': []}}}
    mujoco.mj_forward(model, data)
    assert insertion_contact_check(model, data, metadata)['success']
    data.qpos[:3] = [0, 0, .285]
    mujoco.mj_forward(model, data)
    check = insertion_contact_check(model, data, metadata)
    assert not check['success']
    assert any('wrist' in row['geom_names'] for row in check['forbidden_contacts'])
    data.qpos[:3] = [.3, 0, .2]
    data.qpos[7:10] = [0, 0, .285]
    mujoco.mj_forward(model, data)
    check = insertion_contact_check(model, data, metadata)
    assert not check['success']
    assert any('ceiling' in row['geom_names'] for row in check['forbidden_contacts'])


def test_insert_contact_guard_forbids_pad_bowl_contact_after_release():
    import mujoco
    from gene_agent.tasks.policies.microwave_task.simulation_bowl.insert import insertion_contact_check

    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="microwave"><geom name="microwave_turntable_collision"
        type="box" size=".1 .1 .01"/></body>
      <body name="arm" pos=".3 0 .2"><freejoint/>
        <geom name="right_gripper_negative_pad" type="sphere" size=".02"/></body>
      <body name="bowl" pos=".3 0 .2"><freejoint/>
        <geom name="bowl_wall" type="sphere" size=".02"/></body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    metadata = {
        'group_root_bodies': {'right': 'arm'},
        'grippers': {'right': {'pad_geom_names': ['right_gripper_negative_pad']}},
    }
    mujoco.mj_forward(model, data)
    held = insertion_contact_check(
        model, data, metadata, allow_pad_bowl_contact=True)
    released = insertion_contact_check(
        model, data, metadata, allow_pad_bowl_contact=False)
    assert held['success'] is True
    assert released['success'] is False
    assert released['forbidden_contacts'][0]['geom_names'] == [
        'right_gripper_negative_pad', 'bowl_wall']


def test_released_retreat_lifts_only_inward_pad_above_bowl_rim():
    import mujoco
    from gene_agent.tasks.policies.microwave_task.simulation_bowl.insert import (
        raise_retreat_for_released_pad_clearance,
    )

    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="bowl">
        <geom name="bowl_visual" pos="0 0 .20" type="sphere" size=".10"
          contype="0" conaffinity="0"/>
        <geom name="bowl_wall" pos="0 0 .03"
          type="box" size=".03 .03 .01"/>
      </body>
      <body name="arm">
        <geom name="right_gripper_positive_pad" pos="-.04 0 .01"
          type="box" size=".005 .005 .01"/>
        <geom name="right_gripper_negative_pad" pos=".04 0 .035"
          type="box" size=".005 .005 .01"/>
      </body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    targets = {name: np.eye(4) for name in (
        'preplace', 'insert', 'retreat_1', 'retreat_2')}
    targets['insert'][2, 3] = .10
    targets['retreat_1'][0, 3] = -.06
    targets['retreat_1'][2, 3] = .10
    targets['preplace'][2, 3] = .14
    targets['retreat_2'][2, 3] = .14
    metadata = {'grippers': {'right': {'pad_geom_names': [
        'right_gripper_positive_pad', 'right_gripper_negative_pad']}}}

    adjusted, evidence = raise_retreat_for_released_pad_clearance(
        model, data, metadata, targets, np.array([1., 0., 0.]),
        clearance_margin_m=.002,
    )

    # 碗沿最高点 .04m，靠炉内指垫最低点 .025m，再加2mm净空。
    assert evidence['critical_pad']['geom_name'] == 'right_gripper_negative_pad'
    assert evidence['bowl_collision_geom_count'] == 1
    assert evidence['bowl_collision_top_world_z_m'] == pytest.approx(.04)
    assert evidence['geometric_required_lift_m'] == pytest.approx(.017)
    assert evidence['minimum_lift_m'] == pytest.approx(0.)
    assert evidence['required_lift_m'] == pytest.approx(.017)
    np.testing.assert_allclose(adjusted['retreat_lift'][:3, 3], [0., 0., .117])
    assert adjusted['retreat_1'][2, 3] == pytest.approx(.117)
    np.testing.assert_allclose(adjusted['retreat_2'], targets['retreat_2'])

    default_adjusted, default_evidence = raise_retreat_for_released_pad_clearance(
        model, data, metadata, targets, np.array([1., 0., 0.]),
    )
    assert default_evidence['clearance_margin_m'] == pytest.approx(.005)
    assert default_evidence['required_lift_m'] == pytest.approx(.020)
    assert default_adjusted['retreat_lift'][2, 3] == pytest.approx(.120)
    np.testing.assert_allclose(default_adjusted['retreat_2'], targets['retreat_2'])


def test_released_retreat_respects_configured_minimum_lift():
    import mujoco
    from gene_agent.tasks.policies.microwave_task.simulation_bowl.insert import (
        raise_retreat_for_released_pad_clearance,
    )

    model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="bowl"><geom name="bowl_wall" pos="0 0 .03"
        type="box" size=".03 .03 .01"/></body>
      <body name="arm">
        <geom name="right_gripper_positive_pad" pos="-.04 0 .01"
          type="box" size=".005 .005 .01"/>
        <geom name="right_gripper_negative_pad" pos=".04 0 .035"
          type="box" size=".005 .005 .01"/>
      </body>
    </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    targets = {name: np.eye(4) for name in (
        'preplace', 'insert', 'retreat_1', 'retreat_2')}
    targets['insert'][2, 3] = .10
    targets['preplace'][2, 3] = .14
    targets['retreat_2'][2, 3] = .14
    metadata = {'grippers': {'right': {'pad_geom_names': [
        'right_gripper_positive_pad', 'right_gripper_negative_pad']}}}

    adjusted, evidence = raise_retreat_for_released_pad_clearance(
        model, data, metadata, targets, np.array([1., 0., 0.]),
        clearance_margin_m=.002, minimum_lift_m=.03,
    )

    assert evidence['geometric_required_lift_m'] == pytest.approx(.017)
    assert evidence['minimum_lift_m'] == pytest.approx(.03)
    assert evidence['required_lift_m'] == pytest.approx(.03)
    assert adjusted['retreat_lift'][2, 3] == pytest.approx(.13)
    assert adjusted['retreat_1'][2, 3] == pytest.approx(.13)
    np.testing.assert_allclose(adjusted['retreat_2'], targets['retreat_2'])


def test_released_retreat_executes_vertical_lift_before_pose_exit(tmp_path):
    from gene_agent.tasks.policies.microwave_task.simulation_bowl.insert import (
        execute_insertion_stages,
    )

    calls = []

    class Runtime:
        def __init__(self):
            self.index = 0

        def query(self, **_kwargs):
            return {'state_id': f'actual-{self.index}'}

        def simulation_attachments(self):
            return []

        def execute_gripper_command(self, **_kwargs):
            self.index += 1
            return {'success': True}

        def execute_hold(self, **_kwargs):
            self.index += 1
            return {'success': True}

    class Bridge:
        def __call__(self, stage, _target, state_id):
            calls.append((stage, state_id))
            runtime.index += 1
            return {'success': True}

    def prepare(targets, _state):
        adjusted = {name: pose.copy() for name, pose in targets.items()}
        adjusted['retreat_lift'] = adjusted['insert'].copy()
        adjusted['retreat_lift'][2, 3] += .012
        adjusted['retreat_1'][2, 3] += .012
        return adjusted, {'retreat_strategy': 'vertical_then_horizontal'}

    runtime = Runtime()
    targets = {name: np.eye(4) for name in (
        'preplace', 'insert', 'retreat_1', 'retreat_2')}
    result = execute_insertion_stages(
        runtime, Bridge(), targets, tmp_path, release_width_m=.08,
        prepare_released_retreat=prepare,
    )

    assert result['success'] is True
    assert result['stage_order'][-3:] == [
        'retreat_lift', 'retreat_1', 'retreat_2']
    assert calls[-3:] == [
        ('retreat_lift', 'actual-4'),
        ('retreat_1', 'actual-5'),
        ('retreat_2', 'actual-6'),
    ]


def test_retreat_rotation_keeps_old_stage_positions_and_other_targets():
    from gene_agent.tasks.policies.microwave_task.operations.bowl import rotate_retreat_keyposes

    poses = {name: np.eye(4) for name in (
        "preplace", "insert", "retreat_1", "retreat_2")}
    rotated = rotate_retreat_keyposes(poses, {
        "retreat_1": (0., -10., 0.), "retreat_2": (0., -10., 0.)})

    np.testing.assert_allclose(rotated["preplace"], np.eye(4))
    np.testing.assert_allclose(rotated["insert"], np.eye(4))
    np.testing.assert_allclose(rotated["retreat_1"][:3, 3], [0., 0., 0.])
    assert rotated["retreat_1"][0, 2] == pytest.approx(-np.sin(np.deg2rad(10)))
    np.testing.assert_allclose(rotated["retreat_2"], rotated["retreat_1"])
    np.testing.assert_allclose(poses["retreat_1"], np.eye(4))
    with pytest.raises(ValueError, match="局部旋转"):
        rotate_retreat_keyposes(poses, {"retreat_1": (0., 20., 0.)})


def test_physical_insert_uses_actual_states_and_releases_before_retreat(tmp_path):
    from gene_agent.tasks.policies.microwave_task.simulation_bowl.insert import execute_insertion_stages

    calls = []

    class Runtime:
        def __init__(self):
            self.index = 0
            self.attached = True

        def query(self, *, robot_groups=(), object_names=()):
            return {"state_id": f"actual-{self.index}",
                    "robots": {"right": {"T_world_tcp": np.eye(4).tolist()}},
                    "objects": {"bowl": {"T_world_object": np.eye(4).tolist()}},
                    "simulation_attachments": self.simulation_attachments()}

        def simulation_attachments(self):
            return ([{"object_instance_id": "bowl"}] if self.attached else [])

        def execute_gripper_command(self, **kwargs):
            calls.append(("release", kwargs["planned_state_id"]))
            self.index += 1
            return {"success": True}

        def release_simulation_object(self, group, object_name):
            calls.append(("detach", f"actual-{self.index}"))
            assert (group, object_name) == ("right", "bowl")
            self.attached = False
            self.index += 1

        def execute_hold(self, **kwargs):
            calls.append(("settle", kwargs["planned_state_id"]))
            self.index += 1
            return {"success": True}

    class Bridge:
        def __call__(self, stage, target, state_id):
            calls.append((stage, state_id))
            runtime.index += 1
            return {"success": True}

    runtime = Runtime()
    targets = {name: np.eye(4) for name in (
        "preplace", "insert", "retreat_1", "retreat_2")}
    result = execute_insertion_stages(runtime, Bridge(), targets, tmp_path,
                                      release_width_m=.08)

    assert result["success"] is True
    assert result["stage_order"] == [
        "preplace", "insert", "release", "detach", "retreat_1", "retreat_2"]
    assert calls == [
        ("preplace", "actual-0"), ("insert", "actual-1"),
        ("release", "actual-2"), ("detach", "actual-3"),
        ("settle", "actual-4"), ("retreat_1", "actual-5"),
        ("retreat_2", "actual-6"),
    ]
    assert runtime.attached is False


def test_physical_insert_stops_before_release_if_linear_insert_fails(tmp_path):
    from gene_agent.tasks.policies.microwave_task.simulation_bowl.insert import execute_insertion_stages

    class Runtime:
        def query(self, *, robot_groups=(), object_names=()):
            return {"state_id": "held", "simulation_attachments": []}

        def execute_gripper_command(self, **_kwargs):
            raise AssertionError("直线插入失败后不应松爪")

    class Bridge:
        def __call__(self, stage, target, state_id):
            if stage == "insert":
                return {"success": False, "plan": {"status": "linear_path_failed"}}
            return {"success": True}

    targets = {name: np.eye(4) for name in (
        "preplace", "insert", "retreat_1", "retreat_2")}
    result = execute_insertion_stages(Runtime(), Bridge(), targets, tmp_path,
                                      release_width_m=.08)
    assert result["success"] is False
    assert result["failure_stage"] == "insert"


def test_physical_insert_records_snapshot_error_without_losing_finished_stages(tmp_path):
    """关节物理超限等异常需要保留已完成前缀和实际状态。"""
    import json
    from gene_agent.tasks.policies.microwave_task.simulation_bowl.insert import execute_insertion_stages

    class Runtime:
        def query(self, *, robot_groups=(), object_names=()):
            return {"state_id": "actual", "robots": {"right": {
                "gripper": {"width_m": .0804}}}, "objects": {"bowl": {}},
                "simulation_attachments": []}

    class Bridge:
        def __call__(self, stage, target, state_id):
            if stage == "preplace":
                raise ValueError("Actual gripper opening is outside declared range")

    targets = {name: np.eye(4) for name in (
        "preplace", "insert", "retreat_1", "retreat_2")}
    result = execute_insertion_stages(Runtime(), Bridge(), targets, tmp_path,
                                      release_width_m=.08)
    assert result["failure_stage"] == "preplace"
    assert result["stages"]["preplace"]["exception_type"] == "ValueError"
    saved = json.loads((tmp_path / "insert_result.json").read_text())
    assert saved["observations"]["held_start"]["robots"]["right"]["gripper"][
        "width_m"] == .0804


def test_footprint_rejects_bowl_rim_outside_region_even_when_center_is_inside():
    from gene_agent.tasks.policies.microwave_task.simulation_bowl.insert import box_footprint_margins

    region = np.eye(4)
    centers = np.array([[0.09, 0., 0.], [0., 0., 0.]])
    result = box_footprint_margins(region, (.10, .10), centers, (.02, .01))
    assert result["success"] is False
    assert result["minimum_margin_m"] < 0
    safe = box_footprint_margins(region, (.10, .10), centers - [.03, 0., 0.],
                                 (.005, .005))
    assert safe["success"] is True


def test_cavity_footprint_uses_actual_collision_shape_and_closed_door_plane():
    """地板稍短于门板时，不能用外接球或地板前沿误判炉腔占用。"""
    import mujoco
    from gene_agent.tasks.policies.microwave_task.simulation_bowl.insert import cavity_footprint_margins

    model = mujoco.MjModel.from_xml_string("""
    <mujoco><worldbody>
      <geom name="microwave_floor_collision" type="box" size=".1 .1 .01"/>
      <body name="microwave_door" pos="0 .115 0">
        <joint name="microwave_door_joint" type="hinge" axis="0 0 1"/>
        <geom name="microwave_door_panel_collision" type="box" size=".1 .01 .1"/>
      </body>
      <body name="bowl" pos="0 .095 .04">
        <joint name="bowl_free" type="free"/>
        <geom name="bowl_box" type="box" size=".01 .007 .01"/>
      </body>
    </worldbody></mujoco>
    """)
    data = mujoco.MjData(model)
    data.qpos[model.joint("microwave_door_joint").qposadr[0]] = .5
    mujoco.mj_forward(model, data)
    bowl_geoms = np.array([model.geom("bowl_box").id])
    result = cavity_footprint_margins(model, data, bowl_geoms)
    assert result["success"] is True
    assert result["minimum_margin_m"] == pytest.approx(.003)
    assert result["front_boundary_source"] == "closed_door_panel"

    data.qpos[model.joint("bowl_free").qposadr[0] + 1] = .11
    mujoco.mj_forward(model, data)
    outside = cavity_footprint_margins(model, data, bowl_geoms)
    assert outside["success"] is False
    assert outside["minimum_margin_m"] == pytest.approx(-.012)


def test_bowl_upright_check_rejects_mid_trajectory_tilt_even_if_final_upright():
    from gene_agent.tasks.policies.microwave_task.simulation_bowl.insert import evaluate_bowl_upright_path

    upright = np.eye(4)
    upright[:3, :3] = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    tilted = np.eye(4)
    tilted[:3, :3] = np.array([
        [1., 0., 0.],
        [0., np.cos(np.deg2rad(35)), -np.sin(np.deg2rad(35))],
        [0., np.sin(np.deg2rad(35)), np.cos(np.deg2rad(35))],
    ]) @ upright[:3, :3]

    def sample(pose):
        return {"objects": {"bowl": {"T_world_object": pose.tolist()}}}

    observations = {"held_start": sample(upright),
                    "insert": sample(upright), "release": sample(upright)}
    stages = {
        "preplace": {"execution": {"replay_samples": [sample(upright)]}},
        "insert": {"execution": {"replay_samples": [sample(tilted), sample(upright)]}},
        "release": {"replay_samples": [sample(upright)]},
    }
    result = evaluate_bowl_upright_path(observations, stages, max_tilt_deg=15.)
    assert result["success"] is False
    assert result["worst_stage"] == "insert"
    assert result["maximum_observed_tilt_deg"] == pytest.approx(35.)


def test_retreat_motion_rejects_mid_trajectory_bowl_bump_that_returns_near_start():
    from gene_agent.tasks.policies.microwave_task.simulation_bowl.insert import evaluate_bowl_retreat_motion

    def sample(x):
        pose = np.eye(4)
        pose[0, 3] = x
        return {"objects": {"bowl": {"T_world_object": pose.tolist()}}}

    observations = {"settled": sample(0.), "retreat_2": sample(.004)}
    stages = {
        "retreat_1": {"execution": {"replay_samples": [sample(0.)]}},
        "retreat_2": {"execution": {"replay_samples": [
            sample(0.), sample(.008), sample(.004)]}},
    }
    result = evaluate_bowl_retreat_motion(observations, stages,
                                           max_translation_m=.005)
    assert result["success"] is False
    assert result["maximum_observed_translation_m"] == pytest.approx(.008)
    assert result["final_translation_m"] == pytest.approx(.004)


def test_insert_depth_offset_keeps_old_shape_and_uses_microwave_frame():
    from gene_agent.tasks.policies.microwave_task.operations.bowl import offset_insert_keyposes

    microwave = np.eye(4)
    microwave[:3, :3] = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    poses = {name: np.eye(4) for name in (
        "preplace", "insert", "retreat_1", "retreat_2")}
    shifted = offset_insert_keyposes(poses, microwave, depth_offset_m=.02)

    # 此旋转下炉体局部 +Y 对应世界 +X；全部仍保持原来的旋转矩阵。
    for name in ("preplace", "insert", "retreat_1", "retreat_2"):
        np.testing.assert_allclose(shifted[name][:3, 3], [.02, 0., 0.])
        np.testing.assert_allclose(shifted[name][:3, :3], poses[name][:3, :3])
    # 微调插入深度必须整体搬动四个旧位姿，保持撤离段原有的相对位移。
    np.testing.assert_allclose(shifted["retreat_2"][:3, 3] - shifted["retreat_1"][:3, 3],
                               poses["retreat_2"][:3, 3] - poses["retreat_1"][:3, 3])
    np.testing.assert_allclose(offset_insert_keyposes(poses, microwave, 0.)["insert"],
                               poses["insert"])

@pytest.mark.parametrize('scope,expected', [
    ('preplace',['preplace']),('insert',['preplace','insert']),
    ('release',['preplace','insert','release','detach','settle'])])
def test_insert_debug_stop_keeps_release_atomic(tmp_path, scope, expected):
    from gene_agent.tasks.policies.microwave_task.simulation_bowl.insert import execute_insertion_stages
    from types import SimpleNamespace
    calls=[]
    attached=[True]
    def action(name):
        calls.append(name)
        return {'success':True,'executed':True}
    def detach(*args):
        action('detach'); attached[0]=False
    runtime=SimpleNamespace(query=lambda **k:{'state_id':str(len(calls))},
        simulation_attachments=lambda:[{}] if attached[0] else [],
        execute_gripper_command=lambda **k:action('release'),release_simulation_object=detach,
        execute_hold=lambda **k:action('settle'))
    # 真实适配按 object_instance_id 选择 bowl。
    runtime.simulation_attachments=lambda:[{'object_instance_id':'bowl'}] if attached[0] else []
    class Bridge:
        def __call__(self,n,*a): return action(n)
        def plan_and_execute_candidates(self,n,*a): return action(n)
    r=execute_insertion_stages(runtime,Bridge(),{n:np.eye(4) for n in
        ('preplace','insert','retreat_1','retreat_2')},tmp_path,release_width_m=.08,stop_after=scope)
    assert calls == expected and r['success']
    assert r['requested_scope'] == scope
    assert r['actual_final']['state_id'] == str(len(calls))
    if scope == 'release': assert not attached[0]
