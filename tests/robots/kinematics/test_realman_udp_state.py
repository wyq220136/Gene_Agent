"""UMI 式 UDP 状态与单 SDK 生命周期的离线验证，不连接设备。"""

import unittest
import ctypes
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def fake_udp_vendor_types(monkeypatch):
    """只验证回调与生命周期；厂商 ABI 仍需另外验收，不导入真实 SDK。"""
    class JointStatus(ctypes.Structure):
        _fields_ = [("joint_position", ctypes.c_float * 7)]

    class JointPacket(ctypes.Structure):
        _fields_ = [("arm_ip", ctypes.c_char * 16), ("errCode", ctypes.c_int),
                    ("joint_status", JointStatus)]

    class PushConfig:
        def __init__(self, *, cycle, enable, port, force_coordinate, ip):
            self.cycle = cycle
            self.enable = enable
            self.port = port
            self.force_coordinate = force_coordinate
            self.ip = ip.encode("ascii")

    interface = ModuleType("Robotic_Arm.rm_robot_interface")
    interface.rm_realtime_arm_joint_state_t = JointPacket
    interface.rm_realtime_push_config_t = PushConfig
    monkeypatch.setitem(sys.modules, "Robotic_Arm", ModuleType("Robotic_Arm"))
    monkeypatch.setitem(sys.modules, "Robotic_Arm.rm_robot_interface", interface)


def packet(ip=b"192.168.1.18", code=0, degrees=None):
    return SimpleNamespace(arm_ip=ip, errCode=code,
                           joint_status=SimpleNamespace(joint_position=[90.0] * 7 if degrees is None else degrees))


class UDPStateTest(unittest.TestCase):
    def stream(self):
        from gene_agent.realworld.devices.arms.realman.realtime_state import RealmanUDPState
        return RealmanUDPState("192.168.1.18", target_ip="192.168.1.10", read_timeout_s=0.01)

    def test_valid_packet_converts_degrees_once_and_cannot_be_consumed_twice(self):
        from gene_agent.realworld.devices.arms.realman.adapter import RealmanStateReadError

        stream = self.stream()
        stream.accept_packet(packet())
        np.testing.assert_allclose(stream.read_joints(), [np.pi / 2] * 7)
        with self.assertRaises(RealmanStateReadError):
            stream.read_joints()
        stream.accept_packet(packet(degrees=[45.0] * 7))
        np.testing.assert_allclose(stream.read_joints(), [np.pi / 4] * 7)

    def test_wrong_robot_bad_packet_nonfinite_and_old_state_are_not_used(self):
        from gene_agent.realworld.devices.arms.realman.adapter import RealmanStateReadError

        for bad in (packet(ip=b"192.168.1.19"), packet(code=-3), packet(degrees=[np.nan] * 7),
                    packet(degrees=[0.0] * 6)):
            with self.subTest(packet=bad):
                stream = self.stream()
                stream.accept_packet(bad)
                with self.assertRaises(RealmanStateReadError):
                    stream.read_joints()
        stream = self.stream()
        with patch("gene_agent.realworld.devices.arms.realman.realtime_state.time.monotonic", return_value=0):
            stream.accept_packet(packet())
        with self.assertRaises(RealmanStateReadError):
            stream.read_joints()

    def test_udp_provider_replaces_tcp_polling_in_shared_adapter(self):
        from gene_agent.realworld.devices.arms.realman.adapter import RealmanAdapter

        stream = self.stream()
        stream.accept_packet(packet())
        sdk = SimpleNamespace(rm_get_joint_degree=lambda: self.fail("UDP 模式不能查询TCP关节"))
        adapter = RealmanAdapter(sdk, group="right", joint_names=tuple(f"right_joint_{i}" for i in range(1, 8)),
                                 T_flange_tcp=np.eye(4), joint_state_reader=stream.read_joints)
        np.testing.assert_allclose(adapter.read_joint_state().positions_rad, [np.pi / 2] * 7)

    class SDK:
        def __init__(self, set_return=0):
            self.callback = None
            self.configuration = None
            self.set_return = set_return
            self.set_count = 0

        def rm_realtime_arm_state_call_back(self, callback):
            self.callback = callback

        def rm_set_realtime_push(self, config):
            from Robotic_Arm.rm_robot_interface import rm_realtime_arm_joint_state_t
            self.configuration = config
            self.set_count += 1
            value = rm_realtime_arm_joint_state_t()
            value.arm_ip = b"192.168.1.18"
            for i in range(7):
                value.joint_status.joint_position[i] = 90.0
            self.callback(value)
            return self.set_return

        def rm_get_realtime_push(self):
            c = self.configuration
            return 0, {"enable": True, "ip": c.ip.decode(), "port": c.port, "cycle": c.cycle}

    def test_start_verifies_packet_and_ambiguous_configuration_ack_then_closes_callback(self):
        for code in (0, -2):
            with self.subTest(code=code):
                stream = self.stream()
                sdk = self.SDK(set_return=code)
                stream.start(sdk, startup_timeout_s=0.02)
                try:
                    np.testing.assert_allclose(stream.read_joints(), [np.pi / 2] * 7)
                    self.assertEqual(sdk.configuration.port, 8888)
                    self.assertEqual(sdk.configuration.cycle, 2)
                    self.assertEqual(sdk.set_count, 1)
                    first_callback = sdk.callback
                finally:
                    stream.close()
                self.assertIsNot(sdk.callback, first_callback)

    def test_single_process_second_stream_cannot_replace_active_callback(self):
        first = self.stream()
        first_sdk = self.SDK()
        second = self.stream()
        second_sdk = self.SDK()
        first.start(first_sdk, startup_timeout_s=0.02)
        try:
            with self.assertRaises(RuntimeError):
                second.start(second_sdk, startup_timeout_s=0.02)
            self.assertEqual(second_sdk.set_count, 0)
            np.testing.assert_allclose(first.read_joints(), [np.pi / 2] * 7)
        finally:
            first.close()
            second.close()

    def test_sdk_receive_timeout_is_set_before_connecting(self):
        from gene_agent.realworld.devices.arms.realman.connection import connect_realman_read_only

        calls = []
        sdk = SimpleNamespace(rm_set_timeout=lambda ms: calls.append(("timeout", ms)),
                              rm_create_robot_arm=lambda *a: (calls.append(("connect",)), SimpleNamespace(id=1))[1],
                              rm_delete_robot_arm=lambda: 0)
        with connect_realman_read_only("192.168.1.18", group="right", sdk=sdk):
            pass
        self.assertEqual(calls, [("timeout", 2000), ("connect",)])

    def test_udp_startup_failure_closes_sdk_without_tcp_fallback(self):
        from gene_agent.realworld.devices.arms.realman.connection import connect_realman_read_only

        calls = []
        sdk = SimpleNamespace(rm_set_timeout=lambda ms: None,
                              rm_create_robot_arm=lambda *a: SimpleNamespace(id=1),
                              rm_delete_robot_arm=lambda: calls.append("delete"))
        with patch("gene_agent.realworld.devices.arms.realman.realtime_state.RealmanUDPState.start",
                   side_effect=RuntimeError("no fresh UDP")):
            with self.assertRaisesRegex(RuntimeError, "no fresh UDP"):
                connect_realman_read_only("192.168.1.18", group="right", sdk=sdk,
                                          state_source="udp", udp_target_ip="192.168.1.200")
        self.assertEqual(calls, ["delete"])

    def test_startup_missing_packets_releases_callback_for_next_connection(self):
        stream = self.stream()
        sdk = self.SDK()
        sdk.rm_set_realtime_push = lambda config: 0
        from gene_agent.realworld.devices.arms.realman.adapter import RealmanStateReadError
        with self.assertRaises(RealmanStateReadError):
            stream.start(sdk, startup_timeout_s=0.01)
        replacement = self.stream()
        try:
            replacement.start(self.SDK(), startup_timeout_s=0.02)
            np.testing.assert_allclose(replacement.read_joints(), [np.pi / 2] * 7)
        finally:
            replacement.close()


if __name__ == "__main__":
    unittest.main()
