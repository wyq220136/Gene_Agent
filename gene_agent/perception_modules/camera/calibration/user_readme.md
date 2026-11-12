# 用户自己记录的标定方案

瑞尔曼单臂

第三视角相机


conda activate sam3
python -m gene_agent.perception_modules.camera.camera_server   --serial 412622271769 --endpoint tcp://127.0.0.1:18082   --width 1280 --height 720 --stream-fps 30 --max-fps 10

conda activate sam3
python -m gene_agent.perception_modules.camera.camera_server \
  --serial 317622072840 --endpoint tcp://127.0.0.1:18083 \
  --width 640 --height 480 --stream-fps 30 --max-fps 10



手腕相机继续保持原来的 1280×720，这样可以复用现有手眼结果。两台相机不需要使用相同分辨率。

conda activate zehao
python -m gene_agent.perception_modules.camera.calibration moving-board \
  --handeye data/calibration/latest/handeye/handeye.json \
  --ip 192.168.1.18 --group right \
  --wrist-endpoint tcp://127.0.0.1:18082 \
  --third-endpoint tcp://127.0.0.1:18083 \
  --third-serial 317622072840 \
  --poses 8 --validation-poses 2 --visualize