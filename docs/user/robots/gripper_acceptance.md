# 夹爪仿真验收

受众：使用者。当前已通过固定右腕台架验收的 `parallel_slide_proxy`：这是依据现有指端mesh构造的平行滑动近似，
不是已标定的真实连杆机构，也不连接真机。

从项目根目录使用 `curobo` 环境：

```bash
PYTHONPATH= PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/curobo/bin/python tools/diagnostics/check_gripper_actuation.py
```

结果写入 `runs/gripper-actuation/`：开合动画 `actuation.gif`、`open.png`、`closed.png`、
`holding.png`、松爪负例 `released.png` 和包括实际开口/接触力/方块高度的 `acceptance.json`。

验收分两部分：

- 固定手腕，实际动力学开到0、7、41、80毫米，检查开口误差、两指对称和TCP不漂移。
- 20毫米宽、20克的自由方块先放在支撑台上，闭爪后撤走支撑，靠接触摩擦保持2.5秒。
  不绑定、不移动方块；掉落超过4毫米或缺少双指接触即失败。

无载允许开口误差0.5毫米、对称误差0.25毫米；接触力必须有限，单接触点峰值不超过100牛。
100牛是台架异常检测阈值，不是这副夹爪的标定承载能力。驱动力限制由机器人代理配置给定。
台架成功也不表示机械臂抓碗成功；长指形状、垫片摩擦、质量和驱动参数仍有建模假设。

有桌面显示时，可打开循环开合视图：

```bash
PYTHONPATH= PYTHONNOUSERSITE=1 /home/ubuntu/miniconda3/envs/curobo/bin/python tools/diagnostics/check_gripper_actuation.py --viewer
```

窗口关闭后退出。无界面验收采用EGL；交互窗口需要桌面显示。

本次结果（`acceptance.json`）：

- 四个无载目标实测分别为0、7、41、80毫米；两指对称，TCP位置/方向漂移为0。
- 独立box内表面测量与公开夹爪API的site测量一致。
- 撤走支撑后保持2.5秒，方块最大下降2.83毫米；终态双指接触，没有台面/掌部等额外支撑。
- 采用0.8秒闭合斜坡，单接触点瞬态峰值2.85牛。每指执行器限力10牛并不限制碰撞瞬态接触峰值；20毫米方块的
  最终指间距18.78毫米反映软接触压入，是当前代理精度限制。
- 随后打开夹爪，方块在重力下落约77毫米至已撤低的支撑台，松爪负例通过，排除了隐藏绑定保持。

已实际查看开、闭、握持近景和松爪帧。交互viewer入口已提供，但本轮无界面运行未人工打开窗口。

构建方法、原始资产位置、上轮返工原因以及visual/collision分离要求，详见
[开发交接文档](../../developer/robots/gripper_implementation.md)。2026-10-05已恢复原连杆/指端STL，移除可见代理滑轨和方块。连杆随指平移仍是视觉近似，
尚未标定真实机构。正常视图只显示外观，碰撞代理用独立叠图检查。


外观验收：`python tools/diagnostics/check_gripper_visual.py`。
查看 `runs/gripper-visual-comparison/comparison_view105.png` 与 `comparison_view195.png`：
每行依次为0/40/80mm；三列分别是固定原装配、修改前代理、恢复后的外观。
原装配为固定约70mm开口，仅作外观参考。开合动画为`visual_sweep.gif`，碰撞叠图为
`collision_overlay_40mm.png`；物理夹持动画另见`runs/gripper-visual-restored/actuation.gif`。
