# 开发环境

Windows 原生环境及 2026-10-27 的 GPU 规划验收见 [Windows 仿真环境说明](../../user/environment/windows_simulation_environment.md)；以下为原 Linux 机器记录，其 CUDA 12.1 版本锁不适用于本机 Blackwell 显卡。

受众：开发 Agent 与本地验收者。环境名称固定为 `curobo`，仅在此环境安装依赖。

## 机器与选型

- Linux，RTX 4090 24 GB，驱动 550.163.01；系统 CUDA toolkit 12.1。
- 新建 conda 环境 `/home/ubuntu/miniconda3/envs/curobo`，Python 3.11。
- PyTorch 2.5.1 使用官方 cu121 wheel，保持与现有驱动兼容，不升级系统驱动。
- cuRobo 从 `third_partys/curobo` 安装；固定提交
  `78fd485fa82d9b9a063fb4985e371814587e666a`。旧项目使用新版 MotionPlanner API，
  不能直接照搬旧版 0.7/0.8 安装说明。
- MuJoCo 自动反馈使用 EGL 离屏渲染；不全局设置 `MUJOCO_GL`，人工 viewer 保留桌面后端。

## 检查入口

```bash
conda activate curobo
cd /home/ubuntu/Documents/yixuan/Gene_Agent
python tools/diagnostics/check_environment.py --output runs/environment-20261004
```

入口独立检查 CUDA 运算、MuJoCo 落球接触与 RGB 图像、cuRobo API 导入。
摘要打印 PASS/FAIL；完整异常和版本保存到 `environment.json`。
图像为 `mujoco_environment.png`。退出码非零意味着至少一项失败。

这些检查不是机器人验收。Realman FK/TCP、碰撞资产、完整轨迹、夹爪接触和任务成功需后续分别验证。

## 安装与验证结果

2026-10-04 已完成以下验证：

- PyTorch 2.5.1+cu121：RTX 4090 实际矩阵运算通过。
- MuJoCo 3.14.0：落球接触通过，球心高度 0.059633 m；EGL 图像已生成并实际查看。
- cuRobo：MotionPlanner、MotionPlannerCfg、ToolPoseCriteria 导入通过。
- cuRobo 官方 forward_kinematics --test：单次、1000 组批量 GPU FK、梯度计算通过，退出码 0。
  使用官方 Franka 资产，仅验证环境，不表示 Realman 或其他型号迁移验收完成。
- pip check：No broken requirements found。

[结构化结果](../../../runs/environment-20261004/environment.json)、
[MuJoCo 图像](../../../runs/environment-20261004/mujoco_environment.png)、
[GPU FK 日志](../../../runs/environment-20261004/curobo_fk.log)、
[依赖版本锁](../../../environment/requirements.lock.txt)、
[conda 创建记录](../../../environment/conda-history.yml)。

未验证：GPU IK/完整轨迹、Realman 模型、桌面 viewer、物理抓碗和旋钮任务。

## 已处理的兼容问题

1. 当前 shell 的 PYTHONPATH 注入 ROS Python 3.10 路径。新环境使用 Python 3.11，
   因此通过 conda 环境变量配置仅在 curobo 激活时设置 PYTHONPATH 为空、PYTHONNOUSERSITE=1。
   不修改 ROS 安装或全局 shell 配置；直接调用环境 Python 时也需传入这两个变量。
2. Warp 1.6.2 不支持 cuRobo 调用的 wp.func(module=...)。核对接口后固定到 Warp 1.12.1，
   重新检查导入和 GPU FK 均通过。修复前 traceback 保留于
   [首次检查结果](../../../runs/environment-20261004/environment-before-warp-fix.json)。
   版本兼容依据：[Warp 官方兼容说明](https://nvidia.github.io/warp/v1.12/user_guide/compatibility.html)。
3. 使用 cuda-core 1.2.1、cuda-bindings 12.9.9，配合 PyTorch 的 CUDA 12.1 runtime/NVRTC。
   未安装会额外拉取整套 toolkit 的 cuda-core[cu12] extra；以实际 GPU FK 检查验证该组合。

## 复现说明

先按 conda-history.yml 创建环境，再恢复 third_partys/curobo 的上述固定提交，
安装 requirements.lock.txt。PyTorch cu121 wheel 来自官方源
https://download.pytorch.org/whl/cu121；其他依赖本轮使用清华 PyPI 镜像。
锁文件中 cuRobo 为固定 Git 提交的 editable 安装，PyTorch 临时下载路径已改为版本号与官方索引；
版本锁不是跨操作系统通用锁。当前安装使用 third_partys/curobo 源码。
重建后必须重新执行检查入口与以下 GPU FK 检查：

```bash
python -m curobo.examples.getting_started.forward_kinematics --test
python -m pip check
```
