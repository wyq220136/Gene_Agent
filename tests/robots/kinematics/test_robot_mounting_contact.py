"""MuJoCo 的本体安装接触必须与已验证的 cuRobo 安装例外一致。"""

import mujoco

from gene_agent.simulation import build_scene


def test_body_proxy_does_not_push_mounted_arm_at_pick_pregrasp(tmp_path):
    model, data, metadata = build_scene(tmp_path / "model")
    group = "right"
    # 固定的旧执行末端姿态曾出现 robot_body_body ↔ right_link_1 接触，
    # 令第1关节偏离目标约0.13rad。这里复现该姿态并检查碰撞对本身。
    positions = (-0.3990773446, 0.5000944658, 2.2983480140,
                 2.1283981968, 2.0354253171, -1.6681462700, 1.5840943115)
    for name, value in zip(metadata["robot"]["joint_groups"][group], positions):
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    contacts = {
        frozenset((model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]))
        for contact in data.contact[:data.ncon]
    }
    body = model.body("robot_body").id
    assert frozenset((body, model.body("right_link_1").id)) not in contacts
    assert frozenset((model.body("right_base_link").id,
                      model.body("right_link_1").id)) not in contacts
    assert metadata["robot"]["mounting_contact_exceptions"]["robot_links"] == [
        "base_link", "link_1",
    ]
