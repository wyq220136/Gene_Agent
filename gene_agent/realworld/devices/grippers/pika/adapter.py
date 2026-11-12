"""Pika 串口位置控制薄适配；沿用 umi_data_train 的 mm ↔ m 约定。"""
from __future__ import annotations

import math
import time


class PikaGripperAdapter:
    def __init__(self, sdk, *, max_width_m=.09, max_state_age_s=.5):
        self.sdk = sdk
        self.max_width_m = float(max_width_m)
        self.max_state_age_s = float(max_state_age_s)

    @classmethod
    def connect(cls, port):
        # 仅显式真机执行才 import 和连接；构造上层任务不会使能电机。
        from pika.gripper import Gripper
        class TimestampedGripper(Gripper):
            last_motor_update = 0.
            def _data_callback(self, data):
                super()._data_callback(data)
                if 'motor' in data:
                    self.last_motor_update = time.monotonic()
        sdk = TimestampedGripper(port)
        try:
            if not sdk.connect() or not sdk.enable():
                raise RuntimeError(f'Pika 连接/使能失败：{port}')
            time.sleep(1.)
            adapter = cls(sdk)
            adapter.read_width_m()
            return adapter
        except BaseException:
            sdk.disconnect()
            raise

    def read_width_m(self):
        if not self.sdk.is_connected or time.monotonic()-self.sdk.last_motor_update > self.max_state_age_s:
            raise RuntimeError('Pika 电机状态过期或串口已断开')
        width = float(self.sdk.get_gripper_distance())*.001
        if not math.isfinite(width) or not 0 <= width <= self.max_width_m:
            raise ValueError(f'Pika 返回的实测宽度无效：{width} m')
        return width

    def command_width_m(self, width_m, *, blocking=True):
        target = float(width_m)
        if not math.isfinite(target) or not 0 <= target <= self.max_width_m:
            raise ValueError('Pika 目标开口超出范围')
        self.read_width_m()
        if not self.sdk.set_gripper_distance(target*1000):
            raise RuntimeError('Pika 串口位置命令失败')
        if not blocking:
            return
        # 闭合会被物体阻挡，稳定宽度只能说明停稳，不能当夹持成功。
        deadline = time.monotonic()+4.
        previous = self.read_width_m(); stable = 0
        for _ in range(100):
            time.sleep(.05)
            actual = self.read_width_m()
            stable = stable+1 if abs(actual-previous) < .0003 else 0
            previous = actual
            if stable >= 8:
                if target > .001 and abs(actual-target) > .003:
                    raise RuntimeError(f'Pika 未达到目标开口：实际{actual:.4f}，目标{target:.4f}')
                return
            if time.monotonic() > deadline:
                break
        raise RuntimeError('Pika 开口未稳定')

    def close(self):
        # 不自动松爪/disable，避免退出时掉落已经抓住的物体。
        self.sdk.disconnect()
