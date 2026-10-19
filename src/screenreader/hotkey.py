"""全局快捷键（键盘组合键 + 鼠标侧键）。

**为什么不能用 Qt 的 QShortcut / QShortcutEvent**：
`QShortcut` 只在**本程序有键盘焦点**时触发。但这个程序的使用场景是
"游戏全屏在前台，用户按一下快捷键让读屏读一段" —— 那时候焦点全在游戏上，
Qt 根本收不到按键。鼠标侧键（XBUTTON1/2）连 Qt 事件都到不了（Qt 只把它当
`BackButton`/`ForwardButton`，且同样要求焦点）。

所以只能走 Windows 的低级钩子：

    WH_KEYBOARD_LL   -> 全局键盘
    WH_MOUSE_LL      -> 全局鼠标（用来收侧键）

钩子回调运行在**它自己的线程**里（不是 Qt 主线程），所以回调里绝对不能碰
界面控件 —— 这里只把"命中了哪个动作"塞进一个线程安全队列，
由界面用 QTimer 轮询取走。这样跨线程是安全的。

**不引第三方依赖**：`keyboard` / `pynput` / `pywin32` 都没有装，
而且它们都会额外带一堆东西。这里用 ctypes 直接调 user32，几十行就够。

支持的写法（大小写不敏感，用 `+` 连接）：

    "ctrl+alt+s"      键盘组合键
    "ctrl+shift+f9"
    "mouse4"          鼠标侧键 1（后退键）
    "mouse5"          鼠标侧键 2（前进键）

单独一个字母/功能键也能用（如 "f8"），但**不建议**——会和游戏按键冲突。
"""

from __future__ import annotations

import ctypes
import logging
import queue
import sys
import threading
from ctypes import wintypes
from typing import Dict, Optional, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- 常量 --
WH_KEYBOARD_LL = 13
WH_MOUSE_LL = 14

WM_KEYDOWN = 0x0100
WM_SYSKEYDOWN = 0x0104
WM_XBUTTONDOWN = 0x020B
WM_QUIT = 0x0012

MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008

VK_SHIFT, VK_CONTROL, VK_MENU, VK_LWIN, VK_RWIN = 0x10, 0x11, 0x12, 0x5B, 0x5C

# 允许的名字 -> 虚拟键码
_NAMED = {
    "space": 0x20, "tab": 0x09, "enter": 0x0D, "return": 0x0D,
    "esc": 0x1B, "escape": 0x1B, "backspace": 0x08, "delete": 0x2E,
    "insert": 0x2D, "home": 0x24, "end": 0x23,
    "pageup": 0x21, "pagedown": 0x22,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "`": 0xC0, "-": 0xBD, "=": 0xBB, "[": 0xDB, "]": 0xDD,
    "\\": 0xDC, ";": 0xBA, "'": 0xDE, ",": 0xBC, ".": 0xBE, "/": 0xBF,
}
for _i in range(1, 25):
    _NAMED[f"f{_i}"] = 0x6F + _i


def _vk_name(vk: int) -> str:
    """虚拟键码 -> 我们自己的写法（`parse_binding` 的反向）。"""
    if 0x41 <= vk <= 0x5A:                 # A-Z
        return chr(vk).lower()
    if 0x30 <= vk <= 0x39:                 # 0-9
        return chr(vk)
    if 0x70 <= vk <= 0x87:                 # F1-F24
        return f"f{vk - 0x6F}"
    for name, code in _NAMED.items():
        if code == vk:
            return name
    return ""


def parse_binding(spec: str) -> Optional[Tuple[frozenset, int]]:
    """把 "ctrl+alt+s" / "mouse4" 解析成 (修饰键集合, vk)。

    鼠标侧键用负数 vk 表示（-1 = 侧键1，-2 = 侧键2），这样键盘和鼠标
    共用同一套比较逻辑。解析不出来返回 None。
    """
    if not spec:
        return None
    parts = [p.strip().lower() for p in spec.replace(" ", "").split("+") if p.strip()]
    if not parts:
        return None

    mods = set()
    key = None
    for p in parts:
        if p in ("ctrl", "control"):
            mods.add(MOD_CONTROL)
        elif p == "alt":
            mods.add(MOD_ALT)
        elif p == "shift":
            mods.add(MOD_SHIFT)
        elif p in ("win", "super", "cmd"):
            mods.add(MOD_WIN)
        elif p in ("mouse4", "xbutton1", "侧键1"):
            key = (-1, 0)
        elif p in ("mouse5", "xbutton2", "侧键2"):
            key = (-2, 0)
        elif p in _NAMED:
            key = (0, _NAMED[p])
        elif len(p) == 1 and (p.isalpha() or p.isdigit()):
            key = (0, ord(p.upper()))
        else:
            return None
    if key is None:
        return None
    return frozenset(mods), key[0] if key[0] else key[1]


class _KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", wintypes.DWORD),
        ("scanCode", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_void_p),
    ]


class _MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("pt", wintypes.POINT),
        ("mouseData", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_void_p),
    ]


_LRESULT = ctypes.c_ssize_t
_HOOKPROC = ctypes.WINFUNCTYPE(_LRESULT, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)


class HotkeyListener:
    """全局快捷键监听。命中后把动作名放进 `events` 队列。"""

    def __init__(self) -> None:
        self.events: "queue.Queue[str]" = queue.Queue()
        self._bindings: Dict[str, Tuple[frozenset, int]] = {}
        self._lock = threading.RLock()
        self._threads: list = []
        self._hooks: list = []
        self._procs: list = []
        self._running = False
        self._user32 = None
        self._kernel32 = None
        # 防止按住不放狂触发：记录上一次触发时间
        self._last_fire = 0.0
        self._cooldown = 0.35
        # **捕获模式**：设置快捷键时用它录"下一次按键"和鼠标侧键。
        self.captured: "queue.Queue[str]" = queue.Queue()
        self._capturing = False
        # **逃生键**：框选层显示期间打开，保证用户按 Esc 一定能退出。
        self._escape_on = False

    # ------------------------------------------------------------------ #
    def begin_capture(self) -> None:
        """开始录下一个按键/侧键。

        用全局钩子录，而不是 Qt 的 keyPressEvent/mousePressEvent：
        Qt 只在控件有焦点时收得到，而且**鼠标侧键在很多鼠标驱动下不会
        转成 Qt 的 BackButton/ForwardButton**（用户实测前进侧键设不上）。
        """
        while not self.captured.empty():
            try:
                self.captured.get_nowait()
            except Exception:
                break
        self._capturing = True

    def end_capture(self) -> None:
        self._capturing = False

    def set_escape(self, on: bool) -> None:
        """框选层显示/关闭时调用，控制"全局 Esc 取消"。"""
        self._escape_on = bool(on)

    # ------------------------------------------------------------------ #
    @property
    def supported(self) -> bool:
        return sys.platform == "win32"

    def set_bindings(self, bindings: Dict[str, str]) -> None:
        """设置绑定：{动作名: "ctrl+alt+s"}。解析失败的会被忽略并记日志。"""
        parsed: Dict[str, Tuple[frozenset, int]] = {}
        for name, spec in (bindings or {}).items():
            got = parse_binding(spec)
            if got is None:
                log.warning("快捷键写法看不懂，已忽略：%s = %r", name, spec)
                continue
            parsed[name] = got
        with self._lock:
            self._bindings = parsed
        log.info("全局快捷键：%s", {k: v for k, v in (bindings or {}).items() if k in parsed})

    # ------------------------------------------------------------------ #
    def start(self) -> bool:
        """启动钩子。失败返回 False（不影响主程序继续用）。"""
        if not self.supported:
            log.info("非 Windows，跳过全局快捷键")
            return False
        if self._running:
            return True
        try:
            self._user32 = ctypes.WinDLL("user32", use_last_error=True)
            self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            self._user32.SetWindowsHookExW.restype = wintypes.HHOOK
            self._user32.SetWindowsHookExW.argtypes = [
                ctypes.c_int, _HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD,
            ]
            self._user32.CallNextHookEx.restype = _LRESULT
            self._user32.CallNextHookEx.argtypes = [
                wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM,
            ]
        except Exception as e:
            log.warning("加载 user32 失败，全局快捷键不可用：%s", e)
            return False

        self._running = True
        ok_kb = self._start_hook(WH_KEYBOARD_LL)
        ok_ms = self._start_hook(WH_MOUSE_LL)
        if not (ok_kb or ok_ms):
            self._running = False
            return False
        log.info("全局快捷键已启用（键盘=%s 鼠标=%s）", ok_kb, ok_ms)
        return True

    def _start_hook(self, kind: int) -> bool:
        proc = _HOOKPROC(self._callback)
        thread = threading.Thread(target=self._pump, args=(kind, proc),
                                  name=f"hotkey-{kind}", daemon=True)
        thread.start()
        self._threads.append(thread)
        self._procs.append(proc)
        # 等钩子真正装上（最多 1 秒）
        for _ in range(100):
            if len(self._hooks) > 0:
                return True
            threading.Event().wait(0.01)
        return False

    def _pump(self, kind: int, proc) -> None:
        """钩子线程：装钩子 + 跑消息循环（钩子必须活在有消息泵的线程里）。

        **hMod 必须传 0，不能传 `GetModuleHandleW(None)`。**
        实测：传 exe 的 handle 会 `SetWindowsHookExW` 失败、`GetLastError()=126`
        （ERROR_MOD_NOT_FOUND）——因为本项目的 `.venv\\python.exe` 是个
        **重定向桩**，真正跑 Python 的模块不在那个 exe 里，Windows 就找不到
        包含回调函数的模块。传 0 表示"钩子过程就在调用进程里"，
        对 WH_KEYBOARD_LL / WH_MOUSE_LL 这正是要求。
        """
        try:
            hook = self._user32.SetWindowsHookExW(kind, proc, 0, 0)
            if not hook:
                err = ctypes.get_last_error()
                # 保险起见再试一次 exe handle（有些环境反过来才行）
                hmod = self._kernel32.GetModuleHandleW(None)
                hook = self._user32.SetWindowsHookExW(kind, proc, hmod, 0)
                if not hook:
                    log.warning("SetWindowsHookEx 失败 kind=%s err=%s", kind, err)
                    return
                log.info("钩子 kind=%s 用 exe handle 装上（hMod=0 失败）", kind)
            self._hooks.append(hook)
            msg = wintypes.MSG()
            while self._running:
                r = self._user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if r in (0, -1):
                    break
                self._user32.TranslateMessage(ctypes.byref(msg))
                self._user32.DispatchMessageW(ctypes.byref(msg))
        except Exception as e:
            log.warning("钩子线程异常 kind=%s：%s", kind, e)

    # ------------------------------------------------------------------ #
    def _callback(self, ncode, wparam, lparam):
        try:
            if ncode >= 0:
                if wparam in (WM_KEYDOWN, WM_SYSKEYDOWN):
                    vk = ctypes.cast(lparam, ctypes.POINTER(_KBDLLHOOKSTRUCT)).contents.vkCode
                    self._match(vk, is_mouse=False)
                elif wparam == WM_XBUTTONDOWN:
                    data = ctypes.cast(lparam, ctypes.POINTER(_MSLLHOOKSTRUCT)).contents.mouseData
                    # 高字 = 哪个侧键（1 = 后退，2 = 前进）
                    btn = (data >> 16) & 0xFFFF
                    if btn in (1, 2):
                        self._match(-btn, is_mouse=True)
        except Exception:
            pass
        return self._user32.CallNextHookEx(None, ncode, wparam, lparam)

    def _match(self, vk: int, is_mouse: bool = False) -> None:
        """看这次按键命中什么：先看捕获模式，再看逃生键，最后看绑定。"""
        import time as _t

        # ① **捕获模式**：正在设置快捷键，把这一下录下来，不触发任何绑定
        if self._capturing:
            spec = self._spec_of(vk, is_mouse)
            if spec:
                self._capturing = False
                self.captured.put(spec)
            return

        # ② **逃生键**：框选层显示期间，Esc 一定要能退出。
        #    框选层是全屏置顶窗口，一旦失了焦点 Qt 就收不到 Esc，
        #    用户会被困在框选界面里（实测严重到只能重启电脑）。
        if self._escape_on and not is_mouse and vk == 0x1B:
            self.events.put("__escape__")
            return

        with self._lock:
            if not self._bindings:
                return
            for name, (mods, want) in self._bindings.items():
                if want != vk:
                    continue
                if not self._mods_ok(mods):
                    continue
                now = _t.time()
                if now - self._last_fire < self._cooldown:
                    return
                self._last_fire = now
                self.events.put(name)
                return

    def _spec_of(self, vk: int, is_mouse: bool) -> str:
        """把一次按键翻译成 "ctrl+alt+s" / "mouse4" 这种写法。"""
        if is_mouse:
            return {-1: "mouse4", -2: "mouse5"}.get(vk, "")
        if vk in (VK_SHIFT, VK_CONTROL, VK_MENU, VK_LWIN, VK_RWIN):
            return ""            # 只按了修饰键，等真正的按键
        u = self._user32
        parts = []
        try:
            if u.GetAsyncKeyState(VK_CONTROL) & 0x8000:
                parts.append("ctrl")
            if u.GetAsyncKeyState(VK_MENU) & 0x8000:
                parts.append("alt")
            if u.GetAsyncKeyState(VK_SHIFT) & 0x8000:
                parts.append("shift")
            if (u.GetAsyncKeyState(VK_LWIN) & 0x8000) or (u.GetAsyncKeyState(VK_RWIN) & 0x8000):
                parts.append("win")
        except Exception:
            pass
        name = _vk_name(vk)
        if not name:
            return ""
        parts.append(name)
        return "+".join(parts)

    def _mods_ok(self, mods: frozenset) -> bool:
        """当前按住的修饰键，是否正好等于要求的那一组。

        要求"正好等于"而不是"包含"：否则绑了 `ctrl+s` 的时候，
        按 `ctrl+shift+s` 也会命中，容易误触。
        """
        u = self._user32
        down = set()
        if u.GetAsyncKeyState(VK_CONTROL) & 0x8000:
            down.add(MOD_CONTROL)
        if u.GetAsyncKeyState(VK_MENU) & 0x8000:
            down.add(MOD_ALT)
        if u.GetAsyncKeyState(VK_SHIFT) & 0x8000:
            down.add(MOD_SHIFT)
        if (u.GetAsyncKeyState(VK_LWIN) & 0x8000) or (u.GetAsyncKeyState(VK_RWIN) & 0x8000):
            down.add(MOD_WIN)
        return down == set(mods)

    # ------------------------------------------------------------------ #
    def stop(self) -> None:
        self._running = False
        for h in self._hooks:
            try:
                self._user32.UnhookWindowsHookEx(h)
            except Exception:
                pass
        self._hooks.clear()

    def poll(self) -> Optional[str]:
        """取一个待处理的动作名（界面用 QTimer 轮询这个）。"""
        try:
            return self.events.get_nowait()
        except queue.Empty:
            return None
