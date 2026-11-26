"""Linux inode 变更事件用于安全哈希复用；资源不可用时回退重读。"""
import ctypes
import os
from pathlib import Path
import struct


class MeshFileWatch:
    _event = struct.Struct("iIII")
    _mask = 0x00000C0E  # MODIFY/ATTRIB/CLOSE_WRITE/DELETE_SELF/MOVE_SELF。

    def __init__(self):
        self.fd = -1
        self.epoch = 0
        self.files = {}
        self.versions = {}
        self.unavailable = set()
        self.enabled_ever = False
        try:
            self.libc = ctypes.CDLL(None, use_errno=True)
            self.libc.inotify_init1.argtypes = [ctypes.c_int]
            self.libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
            self.fd = self.libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        except (AttributeError, OSError):
            self.fd = -1

    def close(self):
        if self.fd >= 0:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = -1
        self.files.clear()
        self.versions.clear()
        self.unavailable.clear()

    def __del__(self):
        self.close()

    def _drain(self):
        while self.fd >= 0:
            try:
                data = os.read(self.fd, 65536)
            except BlockingIOError:
                return
            except OSError:
                self.close()
                return
            if not data:
                self.close()
                return
            offset = 0
            while offset < len(data):
                wd, mask, cookie, size = self._event.unpack_from(data, offset)
                offset += self._event.size + size
                self.versions[wd] = self.versions.get(wd, 0) + 1
                if mask & (0x4000 | 0x8000 | 0x400 | 0x800 | 0x2000):
                    self.epoch += 1  # overflow/失效/移动使已有 token 全失效。
                    if mask & 0x8000:
                        self.files = {path: entry for path, entry in self.files.items() if entry[0] != wd}

    def token(self, path):
        if self.fd < 0:
            return None
        path = str(Path(path).resolve())
        if path in self.unavailable:
            return None
        value = os.stat(path)
        identity = (value.st_dev, value.st_ino)
        known = self.files.get(path)
        if known is None or known[1] != identity:
            wd = self.libc.inotify_add_watch(self.fd, os.fsencode(path), self._mask)
            if wd < 0:
                self.unavailable.add(path)
                return None
            self.enabled_ever = True
            self.files[path] = (wd, identity)
        self._drain()
        known = self.files.get(path)
        if self.fd < 0 or known is None:
            return None
        # 直接监测 inode；经另一个硬链接修改也会使此 token 失效。
        wd = known[0]
        return (self.epoch, wd, self.versions.get(wd, 0))
