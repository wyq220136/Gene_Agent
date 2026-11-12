"""物理状态是事实来源；JSON 快照用于阅读，积分状态用于同模型复现。"""
from pathlib import Path
import hashlib
import mujoco
import numpy as np

STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


def model_fingerprint(model):
    """哈希编译后模型，包含物理参数和实际 mesh；同维度不代表同一系统。"""
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    return hashlib.sha256(buffer.tobytes()).hexdigest()


def integration_state(model, data):
    state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC))
    mujoco.mj_getState(model, data, state, STATE_SPEC)
    return state


def state_id(model, data):
    return hashlib.sha256(integration_state(model, data).tobytes()).hexdigest()


def _name(model, kind, index):
    return mujoco.mj_id2name(model, kind, index) or f'unnamed_{index}'


def _transform(position, rotation):
    matrix = np.eye(4)
    matrix[:3, :3] = rotation.reshape(3, 3)
    matrix[:3, 3] = position
    return matrix.tolist()


def snapshot(model, data):
    """返回米/弧度、world frame 的 JSON 数据；调用者需暂停同时写 data 的线程。"""
    # 在副本计算派生量，避免读取 mj_step 后尚未更新的运动学缓存或改写 warmstart。
    copied = mujoco.MjData(model)
    mujoco.mj_copyData(copied, model, data)
    mujoco.mj_forward(model, copied)
    contacts = []
    for i in range(copied.ncon):
        contact = copied.contact[i]
        force = np.zeros(6)
        mujoco.mj_contactForce(model, copied, i, force)
        contacts.append({
            'geom_ids': [int(contact.geom1), int(contact.geom2)],
            'geom_names': [_name(model, mujoco.mjtObj.mjOBJ_GEOM, int(j)) for j in (contact.geom1, contact.geom2)],
            'body_names': [_name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[j])) for j in (contact.geom1, contact.geom2)],
            'dist_m': float(contact.dist), 'position_world_m': contact.pos.tolist(),
            'contact_frame_world': contact.frame.reshape(3, 3).tolist(),
            'force_contact_frame': force.tolist(),
        })
    return {
        'schema_version': 1, 'state_id': state_id(model, data), 'sim_time': float(data.time),
        'qpos': data.qpos.tolist(), 'qvel': data.qvel.tolist(), 'ctrl': data.ctrl.tolist(),
        'act': data.act.tolist(), 'qacc': copied.qacc.tolist(),
        'joint_names': [_name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)],
        'joints': [{'name': _name(model, mujoco.mjtObj.mjOBJ_JOINT, i),
                    'type': int(model.jnt_type[i]), 'qpos_address': int(model.jnt_qposadr[i]),
                    'dof_address': int(model.jnt_dofadr[i])} for i in range(model.njnt)],
        'bodies': {_name(model, mujoco.mjtObj.mjOBJ_BODY, i): {'id': i, 'T_world_body': _transform(copied.xpos[i], copied.xmat[i])} for i in range(model.nbody)},
        'sites': {_name(model, mujoco.mjtObj.mjOBJ_SITE, i): {'id': i, 'T_world_site': _transform(copied.site_xpos[i], copied.site_xmat[i])} for i in range(model.nsite)},
        'contacts': contacts,
    }


def save_state(model, data, path):
    """保存完整 mjSTATE_INTEGRATION；模型本身仍须使用相同配置/资产重建。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as stream:
        np.savez(stream, state=integration_state(model, data), spec=int(STATE_SPEC),
                 mujoco_version=mujoco.__version__, model_fingerprint=model_fingerprint(model),
                 dimensions=[model.nq, model.nv, model.na, model.nu],
                 eq_data=model.eq_data.copy())
    return str(path)


def restore_state(model, data, path):
    """恢复状态并刷新派生量，随后再次写回积分量以保留原始 warmstart。"""
    with np.load(path, allow_pickle=False) as archive:
        state = archive['state']
        if int(archive['spec']) != int(STATE_SPEC) or str(archive['mujoco_version']) != mujoco.__version__:
            raise ValueError('State format or MuJoCo version mismatch')
        if not np.array_equal(archive['dimensions'], [model.nq, model.nv, model.na, model.nu]):
            raise ValueError('State model dimensions mismatch')
        # weld 的相对位姿写在模型 eq_data 中；先临时恢复，再核对完整模型指纹。
        # 指纹不匹配时撤销模型改动，避免调用方收到异常后留下半恢复状态。
        previous_eq_data = model.eq_data.copy()
        if 'eq_data' in archive:
            if archive['eq_data'].shape != model.eq_data.shape:
                raise ValueError('State equality dimensions mismatch')
            model.eq_data[:] = archive['eq_data']
        if 'model_fingerprint' not in archive or str(archive['model_fingerprint']) != model_fingerprint(model):
            model.eq_data[:] = previous_eq_data
            raise ValueError('State model fingerprint mismatch')
        if state.size != mujoco.mj_stateSize(model, STATE_SPEC):
            raise ValueError('Integration state size mismatch')
        mujoco.mj_setState(model, data, state, STATE_SPEC)
        mujoco.mj_forward(model, data)
        mujoco.mj_setState(model, data, state, STATE_SPEC)
