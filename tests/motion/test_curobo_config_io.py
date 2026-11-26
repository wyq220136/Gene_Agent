"""静态 YAML 可复用解析，源内容和每次动态几何仍必须独立。"""
import hashlib
import os
import yaml


def test_unchanged_robot_yaml_parses_once_and_returns_independent_copies(tmp_path, monkeypatch):
    from gene_agent.motion.curobo.config_io import load_robot_yaml
    path = tmp_path/'robot.yml'
    path.write_text('kinematics:\n  lock_joints: {finger: 0.04}\n')
    actual_parse = yaml.safe_load
    parsed = []
    def count_parse(value):
        parsed.append(value)
        return actual_parse(value)
    monkeypatch.setattr(yaml, 'safe_load', count_parse)
    first, digest = load_robot_yaml(path)
    first['kinematics']['lock_joints']['finger'] = 0.01
    second, again = load_robot_yaml(path)
    assert second['kinematics']['lock_joints']['finger'] == 0.04
    assert digest == again == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(parsed) == 1


def test_same_size_mtime_content_change_invalidates_robot_yaml_cache(tmp_path):
    from gene_agent.motion.curobo.config_io import load_robot_yaml
    path = tmp_path/'robot.yml'
    path.write_text('kinematics: {finger: 0.04}\n')
    stamp = path.stat()
    _, old_hash = load_robot_yaml(path)
    path.write_text('kinematics: {finger: 0.01}\n')
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    config, new_hash = load_robot_yaml(path)
    assert config['kinematics']['finger'] == 0.01
    assert new_hash != old_hash
