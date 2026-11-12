"""自动截图 Knowledge Viser 工具，等待 WebGL 与侧栏控件均就绪。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen

import numpy as np
from PIL import Image
from websockets.sync.client import connect

from gene_agent.visualization.motion_debug.capture import DEFAULT_CHROME, DEFAULT_WINDOW_SIZE, _free_port, _stop, _validate_png
from gene_agent.visualization.pick_bowl.capture import _devtools_page, _save_cdp_screenshot, _wait_marker
from gene_agent.visualization.grasp_tuner.feedback import read_tuner_state


ROOT = Path(__file__).resolve().parents[3]


def _temporary_chrome_profile() -> tempfile.TemporaryDirectory:
    """Chrome 子进程退出时可能短暂回写 profile，不让清理竞态否定已完成的截图。"""
    return tempfile.TemporaryDirectory(
        prefix="gene-knowledge-chrome-",
        ignore_cleanup_errors=True,
    )


def _scene_material_ready(path: Path, *, require_gripper: bool, view: str = "bowl") -> bool:
    """检查碗与候选夹爪 mesh 的材质已上传，避免把临时白模当成反馈。"""
    pixels = np.asarray(Image.open(path).convert("RGB"))[:, :1000, :].astype(np.int16)
    # 微波炉全景中右侧机械臂也有蓝青色，不能让它充当“碗已上传材质”
    # 的证据。只检查桌面中央的碗活动区域；白色占位碗必须等待。
    if view in {"microwave_handle", "microwave_closeup"}:
        # 门上操作的近景故意不包含桌上的碗；用门板暗色材质和夹爪橙色共同确认
        # WebGL 已加载，不能继续以“看见青色碗”为成功条件。
        channels = pixels.max(axis=2) - pixels.min(axis=2)
        door = ((pixels.max(axis=2) < 165) & (pixels.min(axis=2) > 25)
                & (channels < 55))
        if int(door.sum()) < 5000:
            return False
    else:
        bowl_pixels = pixels[300:800, 300:850, :] if view == "microwave" else pixels
        cyan = (
            (bowl_pixels[:, :, 0] < 125)
            & (bowl_pixels[:, :, 1] > 110) & (bowl_pixels[:, :, 1] < 210)
            & (bowl_pixels[:, :, 2] > 120) & (bowl_pixels[:, :, 2] < 220)
            & (bowl_pixels[:, :, 1] - bowl_pixels[:, :, 0] > 35)
        )
        if int(cyan.sum()) < 1000:
            return False
    if require_gripper:
        if view in {"microwave", "microwave_handle", "microwave_closeup"}:
            # 默认只显示接在腕部的真实橙色夹爪，目标金色 ghost 可关闭。
            # 不能继续以 ghost 的金色像素作为加载完成条件。
            orange = ((pixels[:, :, 0] > 115) & (pixels[:, :, 1] > 55)
                      & (pixels[:, :, 1] < 190) & (pixels[:, :, 2] < 140)
                      & (pixels[:, :, 0] - pixels[:, :, 1] > 25)
                      & (pixels[:, :, 1] - pixels[:, :, 2] > 15))
            return int(orange.sum()) >= 200
        # Bowl 视角避开左上角机器人；夹爪侧视则检查整个三维画面，
        # 否则相机移动后会把已正确加载的材质误判为缺失。
        local = pixels[260:650, 500:980, :] if view == "bowl" else pixels
        gold = ((local[:, :, 0] > 170) & (local[:, :, 1] > 110)
                & (local[:, :, 1] < 225) & (local[:, :, 2] < 135)
                & (local[:, :, 0] - local[:, :, 2] > 80))
        return int(gold.sum()) >= 1000
    return True


def _microwave_material_view(state: dict) -> str:
    """选择与当前相机构图一致的材质检查规则。

    门上的近景不会拍到桌面中央的碗，因此不能继续以“碧色碗已上传”
    作为 WebGL 材质就绪的条件。
    """
    if not state.get("robot_overview", True) and state.get("mode") in {"handle_pull", "door_push"}:
        return "microwave_closeup"
    return "microwave"


def _wait_ui(socket_url: str, expected_text: str, timeout_s: float = 20.) -> None:
    """Viser WebGL ready 后再等待 React 控件挂载，避免侧栏半屏截图。"""
    with connect(socket_url, open_timeout=10., close_timeout=5.) as ws:
        deadline = time.monotonic() + timeout_s
        identifier = 500
        while time.monotonic() < deadline:
            identifier += 1
            ws.send(json.dumps({"id":identifier,"method":"Runtime.evaluate","params":{"expression":"document.body.innerText","returnByValue":True}}))
            while True:
                answer = json.loads(ws.recv(timeout=5.))
                if answer.get("id") == identifier:
                    break
            value = ((answer.get("result") or {}).get("result") or {}).get("value") or ""
            if expected_text in value:
                return
            time.sleep(.1)
    raise TimeoutError(f"Viser 控件未出现：{expected_text}")


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool",choices=("lab","scene","microwave"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--run",type=Path)
    parser.add_argument("--kind")
    parser.add_argument("--mode",choices=("handle_grasp","handle_pull","door_push","knob_grasp","knob_turn"),default="handle_grasp")
    parser.add_argument("--waypoint",type=int,default=0,help="微波炉路径 waypoint；-1 为终点")
    parser.add_argument("--ik-backend",choices=("mujoco","curobo"),help="微波炉 Knowledge 的 IK 求解器")
    parser.add_argument("--view",choices=("bowl","gripper","overview"))
    parser.add_argument("--check-ik",action="store_true",help="场景截图前运行纯位姿 IK")
    parser.add_argument("--restore-state",type=Path,help="从已有截图清单恢复参数、候选与相机")
    parser.add_argument("--url",help="连接已经打开的 Viser 服务，截取当前滑块状态")
    parser.add_argument("--state-file",type=Path,help="该服务的 --state-out JSON；与 --url 一起使用")
    args=parser.parse_args()
    if bool(args.url) != bool(args.state_file):
        parser.error("--url 与 --state-file 必须同时提供")
    if args.url and args.restore_state:
        parser.error("连接实时服务时无需 --restore-state；请在服务启动时恢复")
    restored=read_tuner_state(args.restore_state,tool="microwave_knowledge" if args.tool=="microwave" else args.tool) if args.restore_state else None
    run=args.run or (Path(restored["run"]) if restored and args.tool in {"scene","microwave"} else ROOT/"runs/pick-bowl-planning-final")
    kind=args.kind or ((restored or {}).get("kind") if args.tool=="lab" else None) or "ring"
    view=args.view or ((restored or {}).get("view") if args.tool=="scene" else None) or "bowl"
    output=args.output.expanduser().resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    log=output.with_suffix(".log")
    port=_free_port() if args.url is None else None
    url=args.url or f"http://127.0.0.1:{port}/"
    state_file=(args.state_file or output.with_suffix(".state.json")).expanduser().resolve()
    if args.tool=="lab":
        command=[sys.executable,"-u",str(ROOT/"tools/editors/view_grasp_knowledge.py"),"--kind",kind,"--port",str(port),"--state-out",str(state_file)]
        marker="[knowledge-lab] scene-ready:"
        ui_text="抓取参数滑块"
    elif args.tool=="microwave":
        selected_mode=(restored or {}).get("mode",args.mode)
        command=[sys.executable,"-u",str(ROOT/"tools/editors/view_microwave_knowledge.py"),"--run",str(run),"--mode",selected_mode,"--waypoint",str(args.waypoint),"--port",str(port),"--state-out",str(state_file)]
        if args.ik_backend:
            command.extend(["--ik-backend",args.ik_backend])
        marker="[microwave-knowledge] scene-ready:"
        ui_text="微波炉 Knowledge 调参"
    else:
        command=[sys.executable,"-u",str(ROOT/"tools/editors/tune_scene_grasp.py"),"--run",str(run),"--view",view,"--port",str(port),"--state-out",str(state_file)]
        if args.check_ik:
            command.append("--check-ik")
        marker="[scene-grasp-tuner] scene-ready:"
        ui_text="TCP 后处理偏移"
    if restored:
        command.extend(("--restore-state",str(args.restore_state)))
    environment=os.environ.copy();environment["PYTHONPATH"]=str(ROOT)
    with log.open("w",encoding="utf-8") as stream:
        server=subprocess.Popen(command,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT) if args.url is None else None
        try:
            if server is not None:
                _wait_marker(server,log,"已就绪：",30.)
            with urlopen(url,timeout=5.) as response:
                if response.status!=200:raise RuntimeError(f"Viser HTTP {response.status}")
            with _temporary_chrome_profile() as profile:
                debug_port=_free_port()
                chrome=[str(DEFAULT_CHROME),"--headless=new","--no-sandbox","--ignore-gpu-blocklist","--enable-gpu","--use-angle=gl-egl","--disable-dev-shm-usage","--remote-allow-origins=*","--force-device-scale-factor=1",f"--window-size={DEFAULT_WINDOW_SIZE[0]},{DEFAULT_WINDOW_SIZE[1]}",f"--user-data-dir={profile}",f"--remote-debugging-port={debug_port}",url]
                browser=subprocess.Popen(chrome,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
                try:
                    socket_url=_devtools_page(debug_port,15.)
                    if server is not None:
                        _wait_marker(browser,log,marker,30.)
                    _wait_ui(socket_url,ui_text)
                    # Viser 的 mesh/材质在 WebGL 首帧之后仍可能继续上传；
                    # 等待有界的资源收敛，避免截图出现白色物体或缺少示意夹爪。
                    # 微波炉包含较多独立门板 mesh；碗先着色并不意味着门板也已上传。
                    time.sleep(15.0 if args.tool=="microwave" else 4.0)
                    before=state_file.read_bytes()
                    _save_cdp_screenshot(socket_url,output)
                    if (args.tool=="scene" and view in {"bowl","gripper"}) or args.tool=="microwave":
                        require_gripper = args.tool=="microwave" or json.loads(before).get("gripper_visual")=="mujoco_mesh"
                        current_state=json.loads(before)
                        material_view = (_microwave_material_view(current_state)
                                         if args.tool=="microwave" else view or "overview")
                        # Viser 的连接就绪与 WebGL 材质上传不是同一事件。
                        # 白色占位 mesh 会误导视觉调参；对已知青色 Bowl 等待真实材质。
                        deadline=time.monotonic()+15.
                        while not _scene_material_ready(output, require_gripper=require_gripper, view=material_view) and time.monotonic()<deadline:
                            time.sleep(1.)
                            _save_cdp_screenshot(socket_url,output)
                        if not _scene_material_ready(output, require_gripper=require_gripper, view=material_view):
                            raise RuntimeError("场景 visual mesh 仍是白色占位材质，截图不可信")
                    after=state_file.read_bytes()
                    if before!=after:
                        raise RuntimeError("截图时界面参数发生变化；请重试，以免图片和状态清单不一致")
                finally:_stop(browser)
            _validate_png(output,DEFAULT_WINDOW_SIZE)
        finally:
            if server is not None:
                _stop(server)
    state=json.loads(after)
    expected_tool="microwave_knowledge" if args.tool=="microwave" else args.tool
    if state.get("tool")!=expected_tool:
        raise ValueError(f"状态文件属于 {state.get('tool')!r}，请求截取 {args.tool!r}")
    manifest={"tool":args.tool,"image":str(output),"image_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),"log":str(log),"size_px":list(DEFAULT_WINDOW_SIZE),"state_file":str(state_file),"state_sha256":hashlib.sha256(after).hexdigest(),"state":state}
    output.with_suffix(".json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"[knowledge-capture] 截图：{output}")
    print(f"[knowledge-capture] 清单：{output.with_suffix('.json')}")


if __name__=="__main__":main()
