# Windows 原生 MuJoCo + cuRobo 环境

2026-10-27 已在本机 Windows、RTX PRO 1000 Blackwell Laptop GPU（8 GB）上安装并验证，不使用 WSL。

## 启动

打开 **Anaconda Prompt**：

```bat
conda activate curobo
cd /d C:\work\Agentic_Robot\Gene_Agent
```

环境位置为 `C:\Users\jyx04\anaconda3\envs\curobo`。普通终端如果尚未初始化 Conda，也可直接调用该目录下的 `python.exe`。

运行基础环境检查：

```bat
set MUJOCO_GL=glfw
python tools/diagnostics/check_environment.py --output runs/environment-windows-check
```

运行 cuRobo GPU 示例：

```bat
python -m curobo.examples.getting_started.forward_kinematics --test
python -m curobo.examples.getting_started.inverse_kinematics --test
python -m curobo.examples.getting_started.motion_planning --mode pose --output-dir runs/curobo-windows-demo
```

环境激活时已设置 `MUJOCO_GL=glfw`、`PYTHONNOUSERSITE=1` 和 `PYTHONUTF8=1`。直接调用环境 Python 时，需要自行设置 `MUJOCO_GL=glfw`，避免现有检查入口默认选择 Linux 的 EGL。

## 实际版本与验收

| 组件 | 版本 |
|---|---|
| Python | 3.11.18 |
| PyTorch | 2.11.0+cu128 |
| PyTorch CUDA runtime | 12.8 |
| MuJoCo | 3.14.0 |
| cuRobo | 0.8.0.post1.dev43，提交 `78fd485fa82d9b9a063fb4985e371814587e666a` |
| Warp | 1.12.1 |
| cuda-core / cuda-bindings | 1.2.1 / 12.9.10 |

实际通过：CUDA 矩阵计算、MuJoCo 落球接触及 GLFW 渲染、cuRobo API 导入、GPU 单次/1000 组批量 FK 和梯度计算、单目标 IK、官方 Franka 带碰撞场景的 pose 轨迹规划，以及 `pip check`。MuJoCo 截图已打开检查。批量 FK 本次约 2.79 ms；规划返回 61 个轨迹点，时长约 1.53 s，并保存了实际规划结果和关节轨迹。

证据位于 本机验收目录 `runs/environment-windows-20261027/`（不随 Git 分发，可能需重新生成）：`verification.json`、`environment.json`、`mujoco_environment.png`、`curobo_fk.log`、`curobo_ik.log`、`curobo_motion.log`、`curobo_planning_result.json`、`curobo_trajectory.pt`。

本轮验证使用官方 Franka 示例，没有运行完整微波炉 Task。当前 Windows 工作副本缺少完整 Task 默认配置 `data/task_1/simulation/mujoco_task1_physical.json`；环境可用不代表该任务配置和全部资产已验收。

## 重建与安装记录

使用 [Windows Conda 定义](../../../environment/conda-windows.yml) 和 [Windows 依赖版本锁](../../../environment/requirements.windows.lock.txt)，不要使用 Linux CUDA 12.1 的旧锁。先克隆 `NVlabs/curobo` 到 `third_partys/curobo` 并切换到上述固定提交，再在项目根目录运行：

```bat
conda env create -f environment/conda-windows.yml
conda activate curobo
python -m pip install -r environment/requirements.windows.lock.txt
```

本机原 Anaconda 的 launcher 安装记录缺少 `paths_data`，导致创建环境失败；已备份原记录，并依据官方包缓存逐文件校验 SHA256 后恢复。备份为 `runs/conda-launchers-metadata-before.json`。大包下载使用官方 PyTorch/PyPI 文件并校验 SHA256；下载缓存保留在 `runs/downloads/`。
