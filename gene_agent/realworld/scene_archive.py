"""现场静态模型、感知输入与冻结请求的可携带包；不调用设备或推进物理。"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import xml.etree.ElementTree as ET

import numpy as np


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".pending")
    temp.write_text(json.dumps(_plain(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SceneArchive:
    def __init__(self, root, *, metadata=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest = dict(schema_version=1, format="gene_agent_realworld_scene",
            metadata=_plain(metadata or {}), scenes={}, selections=[], requests=[],
            files={}, file_map={}, missing=[], perception_status="not_collected",
            environment=dict(python=sys.version, platform=platform.platform()),
            claim="冻结的现场观测与规划代理；未推进物理，未证明现场接触或任务成功")
        project = Path(__file__).resolve().parents[2]
        self.manifest["source_code_hashes"] = {
            str(path.relative_to(project)): _hash(path)
            for path in sorted((project / "gene_agent").rglob("*.py"))}
        self.manifest["environment"].update(numpy=np.__version__,
            torch=getattr(sys.modules.get("torch"), "__version__", None))
        self._scanned = set()
        self.save()

    def _register(self, path, *, source=None):
        path = Path(path).resolve()
        relative = str(path.relative_to(self.root))
        self.manifest["files"][relative] = dict(sha256=_hash(path), size_bytes=path.stat().st_size,
                                                source_path=str(source) if source else None)
        return relative

    def add_file(self, source, *, scan=False):
        source = Path(source).expanduser().resolve()
        if not source.is_file():
            if str(source) not in self.manifest["missing"]:
                self.manifest["missing"].append(str(source))
            return None
        key = str(source)
        if key not in self.manifest["file_map"]:
            digest = _hash(source)
            relative = f"assets/{digest}{source.suffix.lower()}"
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                destination.write_bytes(source.read_bytes())
            self._register(destination, source=source)
            self.manifest["file_map"][key] = relative
        if scan and key not in self._scanned:
            self._scanned.add(key)
            if source.suffix.lower() == ".json":
                try:
                    self._scan_paths(json.loads(source.read_text(encoding="utf-8")))
                except (ValueError, UnicodeError):
                    pass
            elif source.suffix.lower() in {".xml", ".urdf"}:
                self._scan_xml(source)
        return self.manifest["file_map"][key]

    def _scan_paths(self, value):
        if isinstance(value, dict):
            for key, child in value.items():
                if isinstance(key, str) and key.startswith("/") and Path(key).is_file():
                    self.add_file(key, scan=True)
                self._scan_paths(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                self._scan_paths(child)
        elif isinstance(value, str) and value.startswith("/") and Path(value).is_file():
            self.add_file(value, scan=True)

    @staticmethod
    def _xml_file(source, element, compiler):
        field = "filename" if "filename" in element.attrib else "file"
        raw = element.get(field)
        if not raw or raw.startswith("package://"):
            return field, None
        path = Path(raw)
        directory = compiler.get("texturedir" if element.tag == "texture" else "meshdir", compiler.get("assetdir", ""))
        candidates = [path] if path.is_absolute() else [source.parent / directory / path, source.parent / path]
        return field, next((p.resolve() for p in candidates if p.is_file()), candidates[0].resolve())

    def _scan_xml(self, source):
        try:
            tree = ET.parse(source)
        except ET.ParseError:
            return
        compiler = tree.getroot().find("compiler")
        attrs = {} if compiler is None else compiler.attrib
        for element in tree.iter():
            if "file" not in element.attrib and "filename" not in element.attrib:
                continue
            _, path = self._xml_file(source, element, {} if element.tag == "include" else attrs)
            if path is not None:
                self.add_file(path, scan=True)

    def add_scene(self, scene, *, label="initial"):
        """展开 include、复制有效几何，保留两臂/夹爪/门角 qpos 和原快照。"""
        import mujoco
        scene = _plain(scene)
        if label in self.manifest["scenes"]:
            raise ValueError("同名现场不能覆盖")
        source = Path(scene["scene_xml_path"]).resolve()
        model = mujoco.MjModel.from_xml_path(str(source))
        state = np.asarray(scene["snapshot"]["qpos_snapshot"], dtype=float)
        if state.shape != (model.nq,) or not np.isfinite(state).all():
            raise ValueError("冻结现场缺少与模型一致的有限 qpos")
        folder = self.root / "scenes" / label
        folder.mkdir(parents=True, exist_ok=False)
        expanded = folder / "model.xml"
        # mj_saveLastXML 会把部分数值截为六位有效数字；展开源 include
        # 并保留原始属性文本，避免复制现场时产生微米级坐标漂移。
        def portable_tree(path, inherited=None, active=()):
            path = Path(path).resolve()
            if path in active:
                raise ValueError("场景 include 存在循环")
            tree = ET.parse(path)
            root = tree.getroot()
            compiler = root.find("compiler")
            attrs = {**(inherited or {}), **(dict(compiler.attrib) if compiler is not None else {})}
            for parent in list(root.iter()):
                for element in list(parent):
                    if element.tag == "include":
                        _, child_path = self._xml_file(path, element, {})
                        child = portable_tree(child_path, attrs, (*active, path))
                        index = list(parent).index(element)
                        parent.remove(element)
                        for offset, replacement in enumerate(list(child)):
                            parent.insert(index + offset, replacement)
                    elif "file" in element.attrib:
                        field, asset = self._xml_file(path, element, attrs)
                        if asset is None or not asset.is_file():
                            raise ValueError(f"场景几何无法冻结: {element.attrib}")
                        relative = self.add_file(asset, scan=True)
                        element.set(field, os.path.relpath(self.root / relative, folder))
            if compiler is not None:
                for field in ("meshdir", "texturedir", "assetdir"):
                    compiler.attrib.pop(field, None)
                compiler.set("strippath", "false")
            return root
        ET.ElementTree(portable_tree(source)).write(expanded, encoding="utf-8", xml_declaration=True)
        restored = mujoco.MjModel.from_xml_path(str(expanded))
        data = mujoco.MjData(restored)
        data.qpos[:] = state
        mujoco.mj_forward(restored, data)
        if (restored.nq, restored.ngeom, restored.nbody) != (model.nq, model.ngeom, model.nbody):
            raise ValueError("可携带模型拓扑与现场不一致")
        original_data = mujoco.MjData(model)
        original_data.qpos[:] = state
        mujoco.mj_forward(model, original_data)
        if not np.allclose(data.geom_xpos, original_data.geom_xpos, atol=1e-10, rtol=0):
            raise ValueError("可携带模型几何 pose 与原现场不一致")
        payload = folder / "scene.json"
        _write(payload, scene)
        self._scan_paths(scene)
        entry = dict(scene_ref=self._register(payload), model_ref=self._register(expanded),
            snapshot_sha256=scene["snapshot"]["snapshot_sha256"],
            source_frame_number=scene["provenance"].get("source_frame_number"),
            physics_stepped=False, velocity_status="not_measured; MuJoCo defaults used only for static FK")
        self.manifest["scenes"][label] = entry
        self.manifest["environment"]["mujoco"] = mujoco.__version__
        self.save()
        return label

    def add_perception(self, scene, sources):
        for source in sources:
            source = Path(source)
            if source.is_dir():
                for path in sorted(source.rglob("*")):
                    if (path.is_file() and not path.is_symlink()
                            and not path.resolve().is_relative_to(self.root)):
                        self.add_file(path, scan=True)
            else:
                self.add_file(source, scan=True)
        if scene:
            for value in scene.get("provenance", {}).get("source_paths", {}).values():
                path = Path(value)
                if not path.is_file():
                    self.add_file(path)
                    continue
                if path.is_file():
                    self.add_file(path, scan=True)
                    # 同一运行帧的 RGB、depth、camera 及 pose 记录。
                    for sibling in path.parent.iterdir():
                        if sibling.is_file():
                            self.add_file(sibling, scan=True)
        camera = (scene or {}).get("provenance", {}).get("source_paths", {}).get("camera")
        self.manifest["perception_status"] = "unavailable"
        if camera and Path(camera).is_file():
            record = json.loads(Path(camera).read_text(encoding="utf-8"))
            frame = (scene or {}).get("provenance", {}).get("source_frame_number")
            match = record.get("frame_number") == frame
            # source_paths.camera 在 tracked/camera，RGB-D 在 capture；显式沿
            # capture_provenance 收齐输入，不能把 tracked 同目录当作原始帧。
            origin_path = Path(camera).parent.parent / "capture_provenance.json"
            if origin_path.is_file():
                origin = json.loads(origin_path.read_text(encoding="utf-8"))
                siblings = [Path(origin["source_files"][name]) for name in ("rgb.png", "depth.png", "camera.json")]
                self.add_file(origin_path, scan=True)
            else:
                siblings = [Path(camera).parent / name for name in ("rgb.png", "depth.png", "camera.json")]
            for path in siblings:
                self.add_file(path)
            context = siblings[-1].parent / "viewer_context.json"
            if context.is_file():
                self.add_file(context)
            raw_camera = json.loads(siblings[-1].read_text()) if siblings[-1].is_file() else {}
            match = match and raw_camera.get("frame_number") == frame
            self.manifest["initial_frame_files"] = [self.manifest["file_map"].get(str(p.resolve())) for p in siblings]
            self.manifest["perception_status"] = ("captured" if match and all(p.is_file() for p in siblings)
                                                   else "frame_mismatch" if not match else "partial")
        self.save()

    def add_selection(self, scene, candidates, parameters, profile_path, seed):
        label = f"selection_{len(self.manifest['selections']):04d}"
        self.add_scene(scene, label=label)
        profile = self.add_file(profile_path)
        payload = dict(scene_id=label, candidates=_plain(candidates), parameters=_plain(parameters),
                       profile_ref=profile, profile_sha256=_hash(profile_path), seed=int(seed))
        path = self.root / "selections" / f"{label}.json"
        _write(path, payload)
        self.manifest["selections"].append(self._register(path))
        self.save()
        return str(path)

    def add_requests(self, directory):
        for path in sorted(Path(directory).glob("*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            self._scan_paths(value.get("replay_config", {}))
            self._scan_paths(value.get("snapshot", {}))
            target = self.root / "motion_requests" / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
            self.manifest["requests"].append(self._register(target, source=path))

    def save(self):
        self.manifest["status"] = ("partial" if self.manifest["missing"] or self.manifest.get("recording_errors") or self.manifest["perception_status"] != "captured" else "complete") if "initial" in self.manifest["scenes"] else "initial_scene_unavailable"
        payload = deepcopy(self.manifest)
        payload["manifest_payload_sha256"] = _digest(payload)
        _write(self.root / "manifest.json", payload)
        return self.root / "manifest.json"


def verify_scene_archive(path):
    path = Path(path).resolve()
    path = path / "manifest.json" if path.is_dir() else path
    root = path.parent
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = manifest.pop("manifest_payload_sha256", None)
    if manifest.get("format") != "gene_agent_realworld_scene" or _digest(manifest) != expected:
        raise ValueError("现场包 manifest 格式或 hash 不匹配")
    for relative, record in manifest["files"].items():
        file = (root / relative).resolve()
        if not file.is_relative_to(root) or not file.is_file() or _hash(file) != record["sha256"]:
            raise ValueError(f"现场包文件缺失、越界或 hash 改变: {relative}")
    return root, manifest


def load_scene_archive(path, *, scene_id="initial"):
    """在当前 MuJoCo 恢复保存状态；不读 SDK、不 reset/step。"""
    import mujoco
    root, manifest = verify_scene_archive(path)
    if scene_id not in manifest["scenes"]:
        raise ValueError(f"没有保存场景 {scene_id}；感知失败时不能伪造初态")
    entry = manifest["scenes"][scene_id]
    scene = json.loads((root / entry["scene_ref"]).read_text(encoding="utf-8"))
    mapping = {old: str(root / new) for old, new in manifest["file_map"].items()}
    def relocate(value):
        if isinstance(value, dict):
            return {key: relocate(child) for key, child in value.items()}
        if isinstance(value, list):
            return [relocate(child) for child in value]
        return mapping.get(value, value) if isinstance(value, str) else value
    scene = relocate(scene)
    scene["scene_xml_path"] = str(root / entry["model_ref"])
    model = mujoco.MjModel.from_xml_path(scene["scene_xml_path"])
    data = mujoco.MjData(model)
    qpos = np.asarray(scene["snapshot"]["qpos_snapshot"], dtype=float)
    if qpos.shape != (model.nq,) or not np.isfinite(qpos).all():
        raise ValueError("包中的现场关节状态无效")
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    scene["offline_replay"] = dict(source_manifest=str(root / "manifest.json"),
        scene_id=scene_id, source_snapshot_sha256=entry["snapshot_sha256"], physics_stepped=False)
    return scene, model, data


def prepare_relocated_request(bundle, request_id, output):
    """仅重定位已验签资产地址；内容、目标、batch 与求解策略必须保留。"""
    from gene_agent.motion.curobo.diagnostics import digest, dependency_signature
    import curobo
    root, manifest = verify_scene_archive(bundle)
    entries = manifest["requests"]
    if isinstance(request_id, int) or str(request_id).isdigit():
        index = int(request_id)
        if not 0 <= index < len(entries):
            raise ValueError("冻结请求下标超出范围")
        relative = entries[index]
    else:
        matches = [ref for ref in entries if Path(ref).name == str(request_id)]
        if len(matches) != 1:
            raise ValueError("请求须为包内唯一文件名或下标")
        relative = matches[0]
    raw = json.loads((root / relative).read_text(encoding="utf-8"))
    for key, field in (("replay_config", "replay_config_sha256"), ("request", "request_sha256"),
                       ("snapshot", "snapshot_payload_sha256"), ("requests", "requests_sha256")):
        if key in raw and digest(raw[key]) != raw.get(field):
            raise ValueError(f"原冻结请求 {key} hash 不匹配")
    old_environment = raw["environment"]
    dependency_root = Path(curobo.__file__).resolve().parent
    if sys.version != old_environment.get("python"):
        raise ValueError("Python 版本不同；可导入场景，但严格请求重跑需原版本")
    if not raw.get("dependency_hashes") or dependency_signature(str(dependency_root), fresh=True) != raw["dependency_hashes"]:
        raise ValueError("cuRobo 完整源码/配置版本不一致；不能用路径重定位替换依赖")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    mapping = {old: str(root / new) for old, new in manifest["file_map"].items()}
    mapping[old_environment["curobo_module"]] = str(Path(curobo.__file__).resolve())
    changes = []
    # 原 URDF 也验过内容；派生副本只改 mesh 文件地址，不改任何数值/拓扑。
    for old, new in list(mapping.items()):
        if Path(old).suffix.lower() != ".urdf":
            continue
        tree = ET.parse(new)
        modified = False
        for element in tree.iter():
            if "filename" not in element.attrib:
                continue
            field, source = SceneArchive._xml_file(Path(old), element, {})
            if source is not None and str(source) in mapping:
                element.set(field, mapping[str(source)])
                modified = True
        if modified:
            path = output / "relocated_assets" / Path(new).name
            path.parent.mkdir(parents=True, exist_ok=True)
            tree.write(path, encoding="utf-8", xml_declaration=True)
            mapping[old] = str(path)
            changes.append(dict(kind="urdf_asset_addresses", source_path=old,
                                source_sha256=_hash(new), relocated_path=str(path), relocated_sha256=_hash(path)))
    project = Path(__file__).resolve().parents[2]
    for old in raw.get("code_hashes", {}):
        marker = "/gene_agent/motion/"
        if marker in old:
            mapping[old] = str(project / "gene_agent/motion" / old.split(marker, 1)[1])
    def relocate(value):
        if isinstance(value, dict):
            return {mapping.get(key, key): relocate(child) for key, child in value.items()}
        if isinstance(value, list):
            return [relocate(child) for child in value]
        return mapping.get(value, value) if isinstance(value, str) else value
    value = relocate(raw)
    value["environment"] = dict(old_environment, executable=sys.executable, curobo_module=str(Path(curobo.__file__).resolve()))
    value["dependency_root"] = str(dependency_root)
    for old in raw["asset_hashes"]:
        new = mapping.get(old, old)
        if Path(old).suffix.lower() == ".urdf" and new != mapping.get(old_environment["curobo_module"]):
            value["asset_hashes"][new] = _hash(new)
    kin = value["replay_config"].get("robot_config", {}).get("kinematics", {})
    if kin.get("urdf_path"):
        kin["asset_root_path"] = str(Path(kin["urdf_path"]).parent)
    for key, field in (("replay_config", "replay_config_sha256"), ("request", "request_sha256"),
                       ("snapshot", "snapshot_payload_sha256"), ("requests", "requests_sha256")):
        if key in value:
            value[field] = digest(value[key])
    value["relocation"] = dict(source_bundle=str(root / relative), source_sha256=_hash(root / relative),
        original_environment=old_environment, address_changes=changes,
        scope="只改变已验签文件地址；原 targets、batch、seed、mode、碰撞策略和容差保留")
    target = output / "relocated_request.json"
    _write(target, value)
    return target
