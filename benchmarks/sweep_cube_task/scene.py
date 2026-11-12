"""清扫初态采样、应用与落稳检查；沿用原任务数值和过滤规则。"""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from gene_agent.scene.tabletop_randomization import sample_tabletop_layout

OBJECTS = ("brush", "cube", "cube_red", "cube_blue", "dustpan")

def place_initial_objects(self, *, randomize: bool, seed: int) -> dict:
    if self.scene.get("placement_mode") == "fixed_scene_keyframe":
        if randomize:
            raise ValueError("此任务使用固定场景 keyframe；请用独立场景样本评测")
        sample = {"poses_xyyaw_rad": {}, "seed": seed,
                  "source": "scene_xml_keyframe_preserved_full_orientation"}
        for name in OBJECTS:
            body = self.data.body(name)
            matrix = body.xmat.reshape(3, 3)
            sample["poses_xyyaw_rad"][name] = [float(body.xpos[0]),
                float(body.xpos[1]), float(np.arctan2(matrix[1, 0], matrix[0, 0]))]
        return {"mode": "fixed_scene_keyframe", "sample": sample,
                "settled": None, "valid_initial_state": None}
    config = self.scene["randomization"]
    nominal = {}
    footprints = {}
    for name in OBJECTS:
        pose = np.asarray(self.scene["objects"][name]["T_world_object"], dtype=float)
        nominal[name] = (float(pose[0, 3]), float(pose[1, 3]),
                         float(np.arctan2(pose[1, 0], pose[0, 0])))
        footprints[name] = tuple(self.scene["objects"][name]["extents_xyz_m"][:2])
    ranges = (config["ranges_xyyaw_rad"] if randomize else
              {name: (0., 0., 0.) for name in OBJECTS})
    policy_record=None
    if 'placement_rules' in self.scene:
        from gene_agent.scene.placement import sample_placement
        matrices={n:np.asarray(self.scene['objects'][n]['T_world_object']) for n in OBJECTS}
        policy_record=sample_placement(matrices,self.scene['placement_rules'],
            seed=seed,randomize=randomize,footprints=footprints)
        sample={'seed':seed,'poses_xyyaw_rad':{n:[p[0][3],p[1][3],float(np.arctan2(p[1][0],p[0][0]))]
                for n,p in policy_record['sampled_poses_world'].items()}}
    else:
        sample = sample_tabletop_layout(
            nominal, ranges, footprints,
            table_bounds_xy=tuple(config["table_bounds_xy_m"]), seed=seed,
            min_clearance_m=float(config["min_clearance_m"]),
            max_attempts=int(config["max_attempts"]))
    for name, (x, y, yaw) in sample["poses_xyyaw_rad"].items():
        joint_name = f"{name}_free"
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            if policy_record:
                pose=np.asarray(policy_record['sampled_poses_world'][name])
                bid=self.model.body(name).id
                self.model.body_pos[bid]=pose[:3,3]
                q=Rotation.from_matrix(pose[:3,:3]).as_quat()
                self.model.body_quat[bid]=q[[3,0,1,2]]
                continue
            if any(abs(value) > 1e-12 for value in ranges[name]):
                raise ValueError(f"固定物体 {name} 不能采样非零位姿范围")
            continue
        address = int(self.model.jnt_qposadr[joint_id])
        self.data.qpos[address:address + 2] = [x, y]
        self.data.qpos[address + 3:address + 7] = [
            np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)]
        if policy_record:
            pose=np.asarray(policy_record['sampled_poses_world'][name])
            self.data.qpos[address:address+3]=pose[:3,3]
            q=Rotation.from_matrix(pose[:3,:3]).as_quat()
            self.data.qpos[address+3:address+7]=q[[3,0,1,2]]
        elif name in self.scene.get('initial_pose_overrides', {}):
            # 场景编辑器显式保存了完整位姿；随机扰动只叠加世界系 XY/yaw，
            # 保留人工设置的高度和 roll/pitch，不能退化为仅 yaw 的姿态。
            pose = np.asarray(self.scene['initial_pose_overrides'][name], dtype=float)
            yaw0 = np.arctan2(pose[1, 0], pose[0, 0])
            delta = Rotation.from_euler('z', yaw-yaw0).as_matrix()
            q = Rotation.from_matrix(delta @ pose[:3, :3]).as_quat()
            self.data.qpos[address + 2] = pose[2, 3]
            self.data.qpos[address + 3:address + 7] = q[[3, 0, 1, 2]]
    mujoco.mj_forward(self.model, self.data)
    return {"mode": "randomized" if randomize else "nominal",
            "sample": sample, "settled": None, "valid_initial_state": None,
            "policy": policy_record}


def record_settled_placement(self) -> None:
    sample = self.placement["sample"]["poses_xyyaw_rad"]
    settled = {}
    policy=self.placement.get('policy') or {}
    invalid = list(policy.get('invalid_reasons',[]))
    for name in OBJECTS:
        body = self.data.body(name)
        x, y = float(body.xpos[0]), float(body.xpos[1])
        rotation = body.xmat.reshape(3, 3)
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        drift = float(np.linalg.norm(np.array([x, y]) - sample[name][:2]))
        settled[name] = {"xyyaw_rad": [x, y, yaw], "z_m": float(body.xpos[2]),
                         "xy_drift_from_sample_m": drift}
        if drift > .03:
            invalid.append({"object": name, "reason": "excess_settling_drift",
                            "drift_m": drift})
    names = set(OBJECTS)
    for index in range(self.data.ncon):
        contact = self.data.contact[index]
        one = self.model.body(int(self.model.geom_bodyid[int(contact.geom1)])).name
        two = self.model.body(int(self.model.geom_bodyid[int(contact.geom2)])).name
        if one in names and two in names and one != two and contact.dist < -.003:
            invalid.append({"objects": [one, two], "reason": "initial_object_penetration",
                            "penetration_m": -float(contact.dist)})
    self.placement.update(settled=settled, valid_initial_state=not invalid,
                          invalid_reasons=invalid)

