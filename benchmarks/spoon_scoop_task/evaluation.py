"""勺子盛球、入目标碗与整批转移的独立判据。"""
import numpy as np

def _body_pose(body):
    pose = np.eye(4)
    pose[:3, :3] = body.xmat.reshape(3, 3)
    pose[:3, 3] = body.xpos
    return pose

def balls_in_spoon(local):
    return {name for name, xyz in local.items()
            if .029 <= xyz[0] <= .073 and abs(xyz[1]) <= .016
            and .006 <= xyz[2] <= .032}

def pour_postcondition(settled_success, cargo_ids, transferred):
    return bool(settled_success and cargo_ids and transferred == cargo_ids)

def balls_in_target(data, target_name, ball_names) -> set[str]:
    bowl = _body_pose(data.body(target_name))
    inverse = np.linalg.inv(bowl)
    inside = set()
    for name in ball_names:
        local = inverse[:3, :3] @ data.body(name).xpos + inverse[:3, 3]
        if np.linalg.norm(local[:2]) <= .035 and .007 <= local[2] <= .11:
            inside.add(name)
    return inside

