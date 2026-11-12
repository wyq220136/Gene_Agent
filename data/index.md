# 数据输入地图

原始采集、标定和唯一证据不能因为缺索引就删除。这里不罗列本机运行输出。

| 数据 | 使用边界 |
|---|---|
| [task_1/index.md](task_1/index.md) | 微波炉当前配置消费者；原始 objs/camera/RGB-D 保留原路径 |
| [calibration/README.md](calibration/README.md) | 标定操作与采集保存规则 |
| [realworld/brush_cube_dustpan_0/](realworld/brush_cube_dustpan_0/) | 旧刷名义场景：scene.xml 装配模型，scene.json 记录来源，task.json 定义目标 |
| 同场景 assets/、concepts/、robot/ | mesh、功能几何与机器人来源；由场景引用，整包维护 |

新场景构建见 [用户文档](../docs/user/index.md)。运行结果放 runs；临时实验放 runs/tmp，均不跟踪。
