"""单物体真实场景采集：冻结一帧后试 prompt，再使用确认的 mask。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from gene_agent.perception_modules.camera.live_rgbd_protocol import FramePacket
from gene_agent.perception_modules import interactive_realworld_perception as flow


def test_capture_rejects_wrong_camera_before_saving(tmp_path: Path, monkeypatch) -> None:
    class WrongCamera:
        def __init__(self, endpoint):
            assert endpoint == "tcp://camera"

        def info(self):
            return {"ok": True, "serial": "other-camera"}

        def get_frame(self):
            raise AssertionError("不应读取错误相机的帧")

        def close(self):
            pass

    monkeypatch.setattr(flow, "CameraClient", WrongCamera)
    with pytest.raises(ValueError, match="相机序列号不一致"):
        flow.capture_once(tmp_path / "hammer_0", endpoint="tcp://camera", expected_serial="chosen-camera")
    assert not (tmp_path / "hammer_0").exists()


def test_capture_saves_aligned_millimeter_depth_and_camera_metadata(tmp_path: Path) -> None:
    packet = FramePacket(
        rgb=np.full((2, 3, 3), 80, dtype=np.uint8),
        depth=np.array([[1.0, 0.0, 1.25], [0.5, 2.0, 0.75]], dtype=np.float32),
        frame_number=17,
        timestamp_ms=1234.5,
        serial="camera-123",
        intrinsic_matrix=np.array([[600, 0, 1], [0, 601, 1], [0, 0, 1]], dtype=float),
    )

    paths = flow.save_capture(packet, tmp_path / "hammer_0")

    assert Image.open(paths.depth).getpixel((0, 0)) == 1000
    assert Image.open(paths.depth).getpixel((2, 0)) == 1250
    assert json.loads(paths.camera.read_text())["depth_unit"] == "millimeter"
    assert json.loads(paths.camera.read_text())["serial"] == "camera-123"
    assert np.array_equal(np.asarray(Image.open(paths.rgb)), packet.rgb)


def test_rejected_prompt_keeps_capture_and_only_accepted_mask_becomes_final(
    tmp_path: Path, monkeypatch,
) -> None:
    object_dir = tmp_path / "hammer_0"
    packet = FramePacket(
        rgb=np.full((2, 3, 3), 80, dtype=np.uint8),
        depth=np.ones((2, 3), dtype=np.float32),
        frame_number=17,
        timestamp_ms=1234.5,
        serial="camera-123",
        intrinsic_matrix=np.eye(3),
    )
    paths = flow.save_capture(packet, object_dir)
    before = paths.rgb.read_bytes()
    prompts = []

    def fake_segment(server, rgb_path, items, *, timeout):
        prompts.append(items[0]["prompt"])
        assert Path(rgb_path) == paths.rgb
        assert timeout == 300.0
        mask = np.zeros((2, 3), dtype=np.uint8)
        mask[0, len(prompts) - 1] = 255
        path = Path(items[0]["mask_path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(path)
        return 200, {"ok": True, "success": True, "results": [{"ok": True}]}

    answers = iter(["the red hammer", "y"])
    monkeypatch.setattr(flow, "segment_image", fake_segment)
    selected = flow.choose_mask(
        paths, category="hammer", initial_prompt="hammer", seg_server="http://test",
        input_fn=lambda _: next(answers), preview_fn=lambda _: None,
    )

    assert prompts == ["hammer", "the red hammer"]
    assert paths.rgb.read_bytes() == before
    assert selected.prompt == "the red hammer"
    assert Image.open(selected.mask).getpixel((1, 0)) == 255
    assert Image.open(selected.mask).getpixel((0, 0)) == 0
    assert len(json.loads((object_dir / "prompt_trials.json").read_text())) == 2
    assert (object_dir / "prompt_trials" / "overlay_001.png").exists()
    assert (object_dir / "prompt_trials" / "overlay_002.png").exists()


def test_reconstruction_request_uses_frozen_capture_and_confirmed_mask(
    tmp_path: Path, monkeypatch,
) -> None:
    object_dir = tmp_path / "brush_0"
    packet = FramePacket(
        rgb=np.zeros((2, 2, 3), dtype=np.uint8),
        depth=np.ones((2, 2), dtype=np.float32),
        frame_number=1,
        timestamp_ms=1.0,
        serial="camera-456",
        intrinsic_matrix=np.eye(3),
    )
    paths = flow.save_capture(packet, object_dir)
    paths.mask.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ones((2, 2), dtype=np.uint8) * 255).save(paths.mask)
    received = {}

    def fake_post(server, payload, *, timeout):
        received.update(payload)
        simplified = object_dir / "meshes" / "obj_mesh_simplified.obj"
        simplified.parent.mkdir(parents=True, exist_ok=True)
        simplified.write_text("o brush\n")
        return {"ok": True, "output_path": payload["output_path"],
                "simplified_output_path": str(simplified)}

    monkeypatch.setattr(flow, "post_reconstruction", fake_post)
    flow.reconstruct_mesh(paths, server="http://test")

    assert received["depth_path"] == str(paths.depth)
    assert received["mask_path"] == str(paths.mask)
    assert received["depth_scale"] == 0.001
    assert received["simplify_mesh"] is True
    assert received["visualize"] is False
    assert received["use_foundationpose"] is False


def test_foundationpose_uses_scaled_mesh_and_releases_session(tmp_path: Path, monkeypatch) -> None:
    paths = flow.paths_for(tmp_path / "hammer_0")
    paths.simplified_mesh.parent.mkdir(parents=True)
    paths.simplified_mesh.write_text("o hammer\n")
    calls = []

    class FakePoseClient:
        def __init__(self, server, timeout):
            assert server == "http://pose"

        def register(self, **kwargs):
            assert kwargs["mesh_path"] == paths.simplified_mesh
            assert kwargs["mask_path"] == paths.mask
            calls.append("register")
            return {"ok": True, "T_camera_object": np.eye(4).tolist()}

        def stop(self, session_id):
            calls.append("stop")

        def unload_model(self):
            calls.append("unload_pose")

    monkeypatch.setattr(flow, "PoseClient", FakePoseClient)
    monkeypatch.setattr(flow, "unload_model", lambda server: calls.append("unload"))

    pose = flow.estimate_pose(paths, pose_server="http://pose", recon_server="http://recon")

    np.testing.assert_array_equal(pose, np.eye(4))
    np.testing.assert_array_equal(np.load(paths.pose), np.eye(4))
    assert calls == ["unload", "register", "stop", "unload_pose"]


def test_stage_transition_releases_seg_and_recon_even_when_reconstruction_fails(
    tmp_path: Path, monkeypatch,
) -> None:
    paths = flow.paths_for(tmp_path / "brush_0")
    calls = []
    monkeypatch.setattr(flow, "unload_seg_model", lambda server: calls.append("seg_unload"))
    monkeypatch.setattr(flow, "unload_model", lambda server: calls.append("recon_unload"))

    def fail_reconstruction(paths, *, server):
        calls.append("reconstruct")
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(flow, "reconstruct_mesh", fail_reconstruction)
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        flow.run_stages(paths, seg_server="seg", recon_server="recon", pose_server="pose",
                        skip_pose=True, reuse_mesh=False)
    assert calls == ["seg_unload", "reconstruct", "recon_unload"]


def test_reuse_mask_requires_confirmed_identical_candidate(tmp_path: Path) -> None:
    paths = flow.paths_for(tmp_path / "brush_0")
    paths.object_dir.mkdir()
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(paths.rgb)
    Image.fromarray(np.ones((2, 2), dtype=np.uint16) * 1000).save(paths.depth)
    paths.camera.write_text(json.dumps({"serial": "camera-456", "intrinsic_matrix": np.eye(3).tolist()}))
    Image.fromarray(np.array([[255, 0], [0, 0]], dtype=np.uint8)).save(paths.mask)
    candidate = paths.object_dir / "prompt_trials" / "mask_001.png"
    candidate.parent.mkdir()
    candidate.write_bytes(paths.mask.read_bytes())
    (paths.object_dir / "prompt_trials.json").write_text(json.dumps([
        {"prompt": "a brush", "mask_path": str(candidate), "accepted": True}
    ]))

    assert flow.validate_reused_mask(paths, serial="camera-456") == "a brush"
    Image.fromarray(np.zeros((2, 2), dtype=np.uint8)).save(paths.mask)
    with pytest.raises(ValueError, match="mask"):
        flow.validate_reused_mask(paths, serial="camera-456")


def test_accepting_new_prompt_supersedes_old_acceptance_and_records_capture_hashes(
    tmp_path: Path, monkeypatch,
) -> None:
    paths = flow.paths_for(tmp_path / "brush_0")
    paths.object_dir.mkdir()
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(paths.rgb)
    Image.fromarray(np.ones((2, 2), dtype=np.uint16) * 1000).save(paths.depth)
    paths.camera.write_text(json.dumps({"serial": "camera-456", "intrinsic_matrix": np.eye(3).tolist()}))
    first = paths.object_dir / "prompt_trials" / "mask_001.png"
    first.parent.mkdir()
    Image.fromarray(np.ones((2, 2), dtype=np.uint8) * 255).save(first)
    paths.mask.write_bytes(first.read_bytes())
    (paths.object_dir / "prompt_trials.json").write_text(json.dumps([
        {"prompt": "old", "mask_path": str(first), "accepted": True}
    ]))

    def fake_segment(server, rgb_path, items, *, timeout):
        Image.fromarray(np.array([[255, 0], [0, 0]], dtype=np.uint8)).save(items[0]["mask_path"])
        return 200, {"ok": True, "success": True, "results": [{"ok": True}]}

    monkeypatch.setattr(flow, "segment_image", fake_segment)
    flow.choose_mask(paths, category="brush", initial_prompt="new", seg_server="seg",
                     input_fn=lambda _: "y", preview_fn=lambda _: None)
    trials = json.loads((paths.object_dir / "prompt_trials.json").read_text())
    assert [trial["accepted"] for trial in trials] == [False, True]
    assert set(trials[-1]["capture_sha256"]) == {"rgb", "depth", "camera"}
    assert flow.validate_reused_mask(paths, serial="camera-456") == "new"
    paths.rgb.write_bytes(paths.rgb.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="采集"):
        flow.validate_reused_mask(paths, serial="camera-456")
