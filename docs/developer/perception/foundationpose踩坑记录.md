# 安装流程

python == 3.9 / 3.10
cuda == 11.8
其他版本都不行


如果报错了再来检查下面的问题：
可能需要从官网安装boost，并把它添加到环境变量中；或者有权限的情况下可以：`apt-get install -y libboost-all-dev`


```bash
cd pose_param_est/fp_pose_est
git clone https://github.com/NVlabs/FoundationPose.git
cd Foundationpose
# 注释掉foundationpose build_all_conda.sh的10-12行，不build mycuda了。如果出问题，可以把涉及到pip install的部分使用--no-build-isolation参数
```

运行python -m pip install -r requirements.txt --no-build-isolation的时候可以查看是否能查找到指定的torch版本，如果python3.10不行，需要换成3.9

```bash
conda install conda-forge::eigen=3.4.0 -y
# install dependencies
python -m pip install -r requirements.txt --no-build-isolation
# Install NVDiffRast
python -m pip install --quiet --no-cache-dir  --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git

# PyTorch3D 这里对应的是py3.10版本
python -m pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt200/download.html
# 或者
# git clone https://github.com/facebookresearch/pytorch3d.git
# cd pytorch3d && pip install -e . --no-build-isolation

CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh
```
最后下载FoundationPose的模型权重到`pipeline/pose_param_est/fp_pose_est/FoundationPose/weights/`目录

# 可能遇到的环境问题

- 如果遇到`ModuleNotFoundError: No module named 'pkg_resources'`: 可以执行`python -m pip install "setuptools<82.0.0" `
- 如果缺失pybind11，可以运行`pip install pybind11`
- 从windows复制到linux，`sed -i 's/\r$//' build_all_conda.sh`
- 如果报错里出现缺少crypt.h文件: `cp /usr/include/crypt.h /home/ubuntu/anaconda3/envs/FP/include/python3.9/`
- nvdiffrast build失败:
```
查找问题：
python setup.py build 2>&1 | grep "error:"

如果gcc出现问题，解决方案是：
conda install -c conda-forge gcc_linux-64=11 gxx_linux-64=11

可参考https://github.com/NVlabs/nvdiffrast/issues/238
```


# out of memory

- 把输入的图片尺度改小
- 把输入的mesh简化面数

# 估计效果差

- 检查深度单位
- 检查经过foundationpose resize+过滤处理之后的mask是否包含白色的像素
