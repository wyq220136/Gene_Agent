# 感知模型本机资源与环境

受众：后续维护感知运行环境的开发者。状态：三个环境均已配置；本文记录 2026-10-06 的运行证据和 2026-10-08 复核的本机版本，并给出修复、重装与验证入口。原项目的 [FoundationPose 安装踩坑记录](foundationpose踩坑记录.md) 已原文复制进本仓库；它包含旧路径和针对特定报错的临时处理，使用前应对照本文。

## 固定位置

| 模块 | conda 环境 | 源码 | 权重 |
| --- | --- | --- | --- |
| SAM3 | `sam3`，Python 3.12.13，PyTorch 2.7.1+cu126 | `third_party/sam3` | `/home/ubuntu/Documents/yixuan/storage/ckpts/SAM3/sam3.pt` |
| SAM3D Objects | `sam3d`，Python 3.11.16，PyTorch 2.5.1+cu121 | `third_party/sam-3d-objects`、`third_party/dinov2` | `/home/ubuntu/Documents/yixuan/storage/ckpts/SAM3D/checkpoints/` |
| FoundationPose | `foundationpose`，Python 3.10.20，PyTorch 2.0.0.post304 / CUDA 11.8 | `gene_agent/perception_modules/pose_est/fp_pose_est/FoundationPose` | `/home/ubuntu/Documents/yixuan/storage/ckpts/FoundationPose/` |

代码以 `gene_agent/perception_modules/paths.py` 的 `CHECKPOINT_ROOT` 解析权重位置。FoundationPose 源码下的 `weights` 是指向该目录的符号链接；scorer 与 refiner 分别读取 `2024-01-11-20-02-45`、`2023-10-28-18-33-37`。模型服务不是 GT 仿真的启动前置条件。

## 已运行的检查

- SAM3：官方源码提交 `2345a4a`；`sam3.pt` 大小 3,450,062,241 字节，SHA256 与 Hugging Face LFS 元数据 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e` 一致。PyTorch 2.7.1+cu126 在 RTX 4090 上实际执行了 CUDA 张量；模型加载后，256×256 测试图的图像编码与文本提示前向成功。空白图得到零个 mask 属预期测试输入结果，不代表真实图像分割质量。
- FoundationPose：从本机 `ti` 环境复制出独立环境，保留可工作的 PyTorch 2.0.0.post304 + CUDA 11.8 与 PyTorch3D 0.7.5；补齐源码依赖，SciPy 固定 1.12.0、NumPy 固定 1.26.4、`warp-lang` 固定 1.0.2。已导入 `estimater`、`mycpp`、`nvdiffrast`，加载官方 scorer/refiner 权重，初始化 CUDA context，并对三角形实际执行 CUDA 栅格化。随后使用 `task_1` 副本的真实 RGB-D/mesh 完成 Bowl 和 Microwave 核心首帧估计；位姿精度尚无真值评估。
- SAM3D：现有 `pipeline.yaml` 引用的六组 checkpoint 和对应 YAML 均存在且非空；PyTorch 2.5.1+cu121、PyTorch3D CUDA 扩展及 Kaolin 在 RTX 4090 上通过运行检查。`notebook/inference.py` 完整导入通过，项目的 `Sam3DReconstructor()` 从指定目录加载六组权重并在 CUDA 上初始化成功。随后使用 `task_1` 副本的 Bowl RGB-D 与新 mask 成功导出 OBJ；几何精度尚无真值评估。

## 环境约束

- SAM3 代码中的图像推理需在 `torch.autocast("cuda", dtype=torch.bfloat16)` 内调用。现有 `seg_server.py` 与 `batch_seg_mask.py` 已使用该上下文。当前源码还需 `einops`、`pycocotools`、`psutil` 和提供 `pkg_resources` 的 setuptools；这些已安装。
- FoundationPose 的 `build_all_conda.sh` 只构建 `mycpp`，需在已激活的 `foundationpose` 环境中运行；已针对 Python 3.10 显式设置 CMake 的 Python 路径。系统 Boost、Eigen、CUDA 11.8 已检查可用。
- SAM3D 的两份生成器 YAML 中四处 DINOv2 构造已指向 `third_party/dinov2`，并设 `pretrained: false`；六组现有 checkpoint 含 DINO 参数，加载时严格恢复，避免启动时访问 GitHub。DINOv2 源码来自官方 `main` 归档，下载文件 SHA256 为 `b54b724b91cf3c97ce3ba657e6e1588b06433f4933ff68750871ffe564f35d93`。MoGe 使用 SAM3D 官方依赖指定的提交 `a8c37341bc0325ca99b9d57981cc3bb2bd3e255b`。运行时将系统的 ROS `PYTHONPATH` 清除，并正确激活 conda 以设置 `CONDA_PREFIX`。
- [SAM3D 官方设置说明](https://github.com/facebookresearch/sam-3d-objects/blob/main/doc/setup.md)要求至少 32 GB 显存；本机 RTX 4090 为 24 GB。模型加载成功不代表完整重建能在 24 GB 中运行。现有 `ss_encoder.safetensors` 是零字节，但当前 `pipeline.yaml` 不引用它。当前 Hugging Face 账号访问 `facebook/sam-3d-objects` 的补下载请求返回“需要审批”；需要补齐或更新官方资源时，用户需先获得该模型仓库访问权限。

## 重新配置与修复顺序

三个模型的 PyTorch/CUDA 扩展版本不同，应继续使用三个独立 conda 环境；GT 仿真仍使用 `curobo`。本机已验证的源码分别为 SAM3 提交 `2345a4a`、SAM3D Objects 提交 `86448b5`。这些版本与现有权重和代码配套；上游当前默认安装命令可能已变化。重装时先固定项目源码及权重，再按各自源码中的安装说明创建环境，最后核对下列本机版本与 CUDA 扩展。**仅能 import、能加载权重、能完成 GPU 推理是不同检查层级。**

### SAM3

- 当前环境还装有 `einops` 0.8.2、`pycocotools` 2.0.11、`psutil` 7.2.2、`setuptools` 81.0.0。若报 `pkg_resources` 缺失，检查 setuptools。源码安装步骤见 [当前检出的 SAM3 README](../../../third_party/sam3/README.md)，但它推荐的最新 PyTorch/CUDA 不等于本机上述可用组合。
- 权重放在表中的 `sam3.pt`，不放进源码或 Git。本机文件为 3,450,062,241 字节；复制后可用上文 SHA256 核验。该模型仓库可能要求访问权限。
- 图像编码和文本提示推理使用 `torch.autocast("cuda", dtype=torch.bfloat16)`；`seg_server.py` 已封装。空白图零个 mask 仅用于环境冒烟检查，不能作为真实分割效果判断。

### SAM3D Objects

- [当前检出的官方 setup](../../../third_party/sam-3d-objects/doc/setup.md)提供 `environments/default.yml` 和开发、PyTorch3D、推理依赖的分步安装说明。当前本机对应 PyTorch3D 0.7.8、Kaolin 0.17.0、NumPy 1.26.4。PyTorch3D 需实测 CUDA 扩展，单纯 import 不够。
- 现有生成器 YAML 的 DINOv2 路径和 `pretrained: false` 是离线加载现有权重所需的项目调整。更换源码或 checkpoint 后复核这四处配置及参数恢复结果。不要让初始化隐式从 GitHub 拉 DINOv2。MoGe 使用上游依赖指定提交 `a8c37341bc0325ca99b9d57981cc3bb2bd3e255b`。
- 正常激活 `sam3d` 环境以设置 `CONDA_PREFIX`，清除 ROS 注入的 `PYTHONPATH`。官方说明要求至少 32 GB 显存；本机 24 GB RTX 4090 完成过 Bowl 单例 OBJ 导出，仍需对新的输入检查显存和几何质量。现有零字节 `ss_encoder.safetensors` 未被当前 `pipeline.yaml` 引用；更换配置时重新检查。补下载 `facebook/sam-3d-objects` 曾遇到仓库访问审批。

### FoundationPose

本机 `foundationpose` 是从既有 `ti` 环境克隆后补齐依赖得到的可用环境，并非跨机器可直接复用的安装命令。其主要包版本为 NumPy 1.26.4、SciPy 1.12.0、`warp-lang` 1.0.2、`nvdiffrast` 0.4.0、`pybind11` 2.12.0、setuptools 81.0.0；Boost、Eigen 和 CUDA 11.8 在本机可用。源码内 `requirements.txt` 的 Torch pin 与当前可用的 `torch.__version__` 不完全相同；**不要在现有环境中无条件执行整份 requirements 安装**。

原 [踩坑记录](foundationpose踩坑记录.md) 中的 `pose_param_est/fp_pose_est`、`pipeline/.../weights`、`/home/ubuntu/anaconda3/envs/FP` 是旧项目路径。当前源码和权重路径见上表。原文中的 `--no-build-isolation`、匹配 Python/CUDA 的 PyTorch3D wheel、Boost、`pkg_resources`、Windows 换行、`crypt.h`、GCC 与输入尺度建议，应在对应报错出现时使用。尤其复制 `crypt.h` 到旧环境是临时处理，应先检查本机 Python 开发头文件和构建器路径。

当前源码的 `build_all_conda.sh` 只编译 `mycpp`，并把 CMake Python 指向激活环境。需要重新编译时：

```bash
conda activate foundationpose
cd /home/ubuntu/Documents/yixuan/Gene_Agent/gene_agent/perception_modules/pose_est/fp_pose_est/FoundationPose
CMAKE_PREFIX_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/pybind11/share/cmake/pybind11" bash build_all_conda.sh
```

编译前检查 `which python`、`CONDA_PREFIX` 和 CMake 路径是否属于同一环境。`weights` 链接须指向统一 checkpoint 目录，两个模型子目录下各有 `model_best.pth` 和 `config.yml`。

## 轻量检查与服务启动

从项目根目录运行。以下 import/文件检查不做完整推理；`PYTHONNOUSERSITE=1` 用于避免用户 site-packages 混入环境。
2026-10-08 本次文档核对时，三个环境的包导入正常，但当前执行会话的 `torch.cuda.is_available()` 返回 `False`；本次没有重跑 GPU 推理。上面的 GPU 结果来自 2026-10-06 已记录的本机运行。

```bash
PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/sam3/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/foundationpose/bin/python -c 'import torch, scipy, numpy; print(torch.__version__, scipy.__version__, numpy.__version__)'
PYTHONPATH= PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/sam3d/bin/python -c 'import torch, kaolin, pytorch3d; print(torch.__version__, torch.cuda.is_available())'
test -s /home/ubuntu/Documents/yixuan/storage/ckpts/SAM3/sam3.pt
test -s /home/ubuntu/Documents/yixuan/storage/ckpts/SAM3D/checkpoints/pipeline.yaml
test -s /home/ubuntu/Documents/yixuan/storage/ckpts/FoundationPose/2024-01-11-20-02-45/model_best.pth
test -s /home/ubuntu/Documents/yixuan/storage/ckpts/FoundationPose/2023-10-28-18-33-37/model_best.pth
```

GPU 可见时，以下命令分别在三个终端运行，默认端口为 18080、18081、18083。服务输入示例见 [感知模块说明](../../../gene_agent/perception_modules/readme.md)。SAM3D 的完整模型加载还需正确激活 conda，不能只调用环境内 Python：

```bash
PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/sam3/bin/python -m gene_agent.perception_modules.segmentation.seg_server --device cuda
PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/foundationpose/bin/python -m gene_agent.perception_modules.pose_est.pose_server --device cuda:0
conda activate sam3d
PYTHONPATH= PYTHONNOUSERSITE=1 python -m gene_agent.perception_modules.recon_3d.recon_server --device cuda
```

## 复现入口

```bash
conda activate sam3d
unset PYTHONPATH
PYTHONNOUSERSITE=1 python -c 'from gene_agent.perception_modules.recon_3d.recon_core import Sam3DReconstructor; print(Sam3DReconstructor().is_loaded)'
```

在项目根目录运行；该命令会加载模型并占用显存。`sam3`、`foundationpose` 同样用 `conda activate` 进入各自环境。GT 仿真仍使用 `curobo`。

## 上游来源

- [SAM3 官方仓库](https://github.com/facebookresearch/sam3)
- [SAM3D Objects 官方仓库](https://github.com/facebookresearch/sam-3d-objects)
- [FoundationPose 官方仓库](https://github.com/NVlabs/FoundationPose)
- [FoundationPose 安装踩坑记录原文副本](foundationpose踩坑记录.md)，来源：`/home/ubuntu/Documents/yixuan/gene_real_demo/docs/foundationpose踩坑记录.md`。
