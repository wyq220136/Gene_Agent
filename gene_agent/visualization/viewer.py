"""Linux passive viewer 生命周期：退出请求之后等待真实渲染线程释放资源。"""
import queue
import sys
import threading
import mujoco


class _JoinedViewer:
    def __init__(self, handle, thread, errors, timeout):
        self._handle, self._thread = handle, thread
        self._errors, self._timeout = errors, timeout

    def __getattr__(self, name):
        return getattr(self._handle, name)

    def close(self):
        # 原 Handle.close 只设 exitrequest；is_running=False 不表示 GLFW 已析构。
        # 等待 _launch_internal 真正返回，其中包含 render_loop、side_thread.join、destroy。
        self._handle.close()
        self._thread.join(self._timeout)
        if self._thread.is_alive():
            raise TimeoutError('MuJoCo viewer render thread did not stop within timeout')
        if self._errors:
            raise RuntimeError('MuJoCo viewer thread failed') from self._errors[0]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def open_viewer(model, data, *, shutdown_timeout=10.0, camera_preset='overview', target_body='bowl'):
    """返回带有界 join 的 passive handle；必须用 with 或显式 close。

    当前适配限定 Linux。MuJoCo 公共 launch_passive 不暴露其 daemon 线程，
    因此只在此处复用本地 _launch_internal，避免全局 monkeypatch 或猜测线程列表。
    该私有入口需随 MuJoCo 升级重跑桌面子进程回归；不修改安装包或 GLFW 配置。
    """
    if sys.platform != 'linux':
        raise NotImplementedError('Joined passive viewer currently supports Linux only')
    if shutdown_timeout <= 0:
        raise ValueError('shutdown_timeout must be positive')
    import mujoco.viewer
    mujoco.mj_forward(model, data)
    handles = queue.Queue(maxsize=1)
    errors = []

    def run():
        try:
            mujoco.viewer._launch_internal(model, data, run_physics_thread=False, handle_return=handles)
        except BaseException as error:
            errors.append(error)
            # 初始化失败时唤醒调用线程；启动后失败则由 close 检查 errors。
            try:
                handles.put_nowait(error)
            except queue.Full:
                pass

    thread = threading.Thread(target=run, name='gene-agent-mujoco-viewer', daemon=True)
    thread.start()
    # 等待场景载入的公共 launch_passive 本来也阻塞；错误通过队列显式传回。
    handle = handles.get()
    if isinstance(handle, BaseException):
        thread.join(shutdown_timeout)
        raise RuntimeError('MuJoCo viewer could not start') from handle
    result = _JoinedViewer(handle, thread, errors, shutdown_timeout)
    # 无任务物体的小场景仍可用于生命周期回归；其camera由MuJoCo自动适配。
    if camera_preset is not None and mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body) >= 0:
        from .style import camera_presets, configure_camera
        try:
            with result.lock():
                configure_camera(result.cam, camera_presets(model, data, target_body)[camera_preset])
        except Exception:
            result.close()
            raise
    return result
