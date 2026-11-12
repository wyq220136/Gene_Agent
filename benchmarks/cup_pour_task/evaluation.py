"""杯子任务的独立几何判据；保留旧阈值，不能由 policy 改写。"""
import numpy as np

MIN_BALLS_IN_TARGET = 5
MAX_BALLS_OUTSIDE = 5

def pour_postcondition(transferred: set[str], spilled: set[str]) -> bool:
    return (len(transferred) >= MIN_BALLS_IN_TARGET and
            len(spilled) <= min(MAX_BALLS_OUTSIDE, len(transferred)))


def placement_postcondition(xy_error_m, orientation_error_rad, height_m, attachments):
    """沿用原放回容差：2 cm、10°、底面高度 15 mm，且已解除辅助夹持。"""
    return bool(xy_error_m <= .02 and orientation_error_rad <= np.deg2rad(10)
                and not attachments and height_m <= .015)

def _body_pose(body):
    pose = np.eye(4)
    pose[:3, :3] = body.xmat.reshape(3, 3)
    pose[:3, 3] = body.xpos
    return pose

def ball_membership(data, cup_name, target_name, ball_names) -> dict[str, set[str]]:
    cup_inverse = np.linalg.inv(_body_pose(data.body(cup_name)))
    bowl_inverse = np.linalg.inv(_body_pose(data.body(target_name)))
    in_cup, in_bowl = set(), set()
    for name in ball_names:
        position = np.r_[data.body(name).xpos, 1.]
        local_cup = (cup_inverse @ position)[:3]
        local_bowl = (bowl_inverse @ position)[:3]
        if np.linalg.norm(local_cup[:2]) <= .039 and -.01 <= local_cup[2] <= .105:
            in_cup.add(name)
        elif np.linalg.norm(local_bowl[:2]) <= .045 and .004 <= local_bowl[2] <= .11:
            in_bowl.add(name)
    return {"in_cup": in_cup, "in_target": in_bowl,
            "outside": set(ball_names) - in_cup - in_bowl}

