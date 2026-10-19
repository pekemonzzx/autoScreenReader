"""测试 TTS 服务的自动启动路径。

**为什么专门写这个**：`start_tts_service()` 里用了 `subprocess.STDOUT`，
但 `runtime.py` 从来没 `import subprocess`。开发机上永远发现不了 ——
因为服务都是我手动起的，`ensure_tts_service()` 一看端口已经有人监听就直接返回，
那条自动启动的代码**一次都没执行过**。

到全新机器上才炸：日志里只有一句 `启动失败：name 'subprocess' is not defined`，
界面表现是「整段模式点了没反应 / 一直在合成失败」。
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    mark = "OK" if cond else "FAIL"
    print(f"  [{mark}] {name}" + (f"    {detail}" if detail else ""))


def _fake_runtime(base: Path) -> Path:
    """造一个假装装好了的运行时 + 声库。"""
    root = base / "gsv"
    (root / "runtime").mkdir(parents=True, exist_ok=True)
    (root / "runtime" / "python.exe").write_bytes(b"MZ fake")
    voices = base / "voices" / "fakevoice"
    voices.mkdir(parents=True, exist_ok=True)
    (voices / "voice.json").write_text(
        '{"name":"fakevoice","display":"假声库","version":"v2ProPlus"}', encoding="utf-8")
    (voices / "tts_infer.yaml").write_text(
        "custom:\n  device: cuda\n  version: v2ProPlus\n", encoding="utf-8")
    (base / "data").mkdir(parents=True, exist_ok=True)
    return root


def _free_port() -> int:
    """找一个**确定没人监听**的端口。

    测试不能假设 9880 是空闲的 —— 开发机上 TTS 服务经常正跑着，
    那样 start_tts_service() 会走 "already: True" 短路，
    根本不会执行到我们要测的那几行（`subprocess.STDOUT` 就是在那里）。
    """
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_start_tts_service() -> None:
    print("\n[TTS 服务自动启动路径]")
    from screenreader import runtime as R
    from screenreader.config import AppConfig

    port = _free_port()
    print(f"  使用空闲端口 {port}")

    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        root = _fake_runtime(base)

        cfg = AppConfig()
        cfg.tts.runtime_root = str(root)
        cfg.tts.api_url = f"http://127.0.0.1:{port}"

        calls = []

        class FakeProc:
            pid = 4242

        def fake_popen(cmd, **kw):
            calls.append((cmd, kw))
            # 真进程会一直占着这个句柄；假进程不关的话临时目录删不掉，
            # 测试最后会以 WinError 32 收尾（不影响结论但很难看）
            try:
                if kw.get("stdout"):
                    kw["stdout"].close()
            except Exception:
                pass
            return FakeProc()

        # 拦截真正拉起进程；但 **不拦 subprocess.STDOUT** ——
        # 那个常量求值才是原来炸掉的地方
        import screenreader.proc as proc
        real = proc.popen_hidden
        proc.popen_hidden = fake_popen
        # runtime.py 里是函数内 `from .proc import popen_hidden`，所以改模块属性即可
        try:
            res = R.start_tts_service(cfg, gsv_root=root, voices_root=base / "voices",
                                      say=lambda *a: None)
        finally:
            proc.popen_hidden = real

        check("调用没有抛异常（原来这里 NameError）", True)
        check("返回 ok", res.get("ok") is True, str(res))
        check("确实发起了进程", len(calls) == 1, f"{len(calls)} 次")
        if calls:
            cmd, kw = calls[0]
            check("命令用了运行时里的 python",
                  "python.exe" in cmd[0], str(cmd[:2]))
            check("带上了 -c 指定推理配置",
                  "-c" in cmd and str(base / "voices" / "fakevoice" / "tts_infer.yaml") in cmd,
                  str(cmd))
            check("stderr 合并到 stdout（用的是 subprocess.STDOUT）",
                  kw.get("stderr") == __import__("subprocess").STDOUT,
                  repr(kw.get("stderr")))
            check("工作目录是运行时根", str(kw.get("cwd")) == str(root), str(kw.get("cwd")))

        # 运行时不存在时要给出明确错误，而不是崩
        cfg2 = AppConfig()
        cfg2.tts.runtime_root = str(base / "不存在")
        cfg2.tts.api_url = f"http://127.0.0.1:{_free_port()}"
        res2 = R.start_tts_service(cfg2, gsv_root=base / "不存在",
                                   voices_root=base / "voices", say=lambda *a: None)
        check("运行时缺失时返回 ok=False", res2.get("ok") is False, str(res2))
        check("错误信息里带路径", "python.exe" in res2.get("message", ""), res2.get("message", ""))


def test_stop_service_tree() -> None:
    """停服务要能**连子进程一起**清掉。

    用户反馈"关闭后 python 进程还在"。原因很可能是 api_v2.py 派生了子进程：
    真正监听端口的是子进程，我们记录的父 PID 和 netstat 看到的对不上，
    于是被判成"不是本程序启动的"而拒绝清理。

    这里造一个"父进程 -> 子进程监听端口"的场景来验证。
    """
    import socket
    import subprocess
    import time

    from screenreader import runtime as R

    print("\n[TTS 服务清理：父子进程]")

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    print(f"  用空闲端口 {port}")

    # 父进程只负责起子进程，**自己不监听**
    parent_code = (
        "import subprocess,sys,time;"
        "child='''import socket,time\n"
        f"s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)\n"
        f"s.bind((\"127.0.0.1\",{port}));s.listen(1)\n"
        "time.sleep(600)''';"
        "subprocess.Popen([sys.executable,'-c',child]);"
        "time.sleep(600)"
    )
    parent = subprocess.Popen([sys.executable, "-c", parent_code],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    holder = None
    for _ in range(60):
        time.sleep(0.25)
        holder = R.service_pids(port)
        if holder:
            break

    check("子进程确实在监听", bool(holder), str(holder))
    check("监听的不是父进程（模拟真实情况）",
          bool(holder) and int(holder[0]) != parent.pid,
          f"父 {parent.pid} vs 监听 {holder}")

    # 只把我们记下的**父 PID** 交给 stop_service
    ok = R.stop_service(port, only_pid=parent.pid)
    time.sleep(1.0)
    left = R.service_pids(port)
    print(f"    stop_service 返回 {ok}；端口残留 {left}")
    check("端口被清理干净", not left, str(left))

    # 收尾
    try:
        parent.kill()
    except Exception:
        pass


def test_shutdown_all_kills_service() -> None:
    """完整停机（关程序走的那条路）必须真的把服务进程清掉。

    这里不启动真服务（太慢），用假进程 + 假端口验证调用链：
      _shutdown_all -> _stop_tts_service -> kill_service_processes
    并且确认 **_stop_tts_service 里不再有 NameError** ——
    用户实测过"确认关闭了但 python.exe 还在"，根因就是那里漏了
    `from .. import runtime as R`，而关闭流程把 NameError 吞进了日志。
    """
    import inspect

    print("\n[关闭时的完整停机]")
    from screenreader.ui.main_window import MainWindow

    src = inspect.getsource(MainWindow._stop_tts_service)
    check("_stop_tts_service 里有 runtime 导入",
          "import runtime as R" in src, src.strip().splitlines()[0][:60])
    check("_stop_tts_service 用的是三重保险",
          "kill_service_processes" in src)

    # _shutdown_all 必须真的调用服务清理
    src2 = inspect.getsource(MainWindow._shutdown_all)
    check("_shutdown_all 会清服务", "_stop_tts_service" in src2)
    check("_shutdown_all 先停播放", "stop_playback" in src2)
    check("_shutdown_all 会停引擎", "engine.stop()" in src2)

    # closeEvent 必须先完整停机再退出
    src3 = inspect.getsource(MainWindow.closeEvent)
    check("closeEvent 调 _shutdown_all", "_shutdown_all" in src3)
    check("closeEvent 会复核残留", "left" in src3)
    check("closeEvent 先确认再关", "_confirm_close" in src3)

    # 假环境跑一遍：没有服务时也应当安全返回
    from screenreader.config import AppConfig

    class Fake:
        pass

    f = Fake()
    f.cfg = AppConfig()
    f.cfg.tts.api_url = "http://127.0.0.1:1"      # 不可能有服务
    res = MainWindow._stop_tts_service(f)
    check("没服务时安全返回", isinstance(res, dict) and res.get("ok") is True, str(res))


def test_imports_complete() -> None:
    """静态扫描：不能再出现「用了但没导入」的模块名。"""
    print("\n[模块导入完整性]")
    import ast
    import builtins

    src = ROOT / "src"
    problems = []
    for f in sorted(src.rglob("*.py")):
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError as e:
            problems.append(f"{f.name}: 语法错误 {e}")
            continue

        imported, defined, used = set(), set(dir(builtins)), set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {(a.asname or a.name.split(".")[0]) for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                imported |= {(a.asname or a.name) for a in node.names if a.name != "*"}
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined.add(node.name)
                defined |= {a.arg for a in node.args.args + node.args.kwonlyargs}
                if node.args.vararg:
                    defined.add(node.args.vararg.arg)
                if node.args.kwarg:
                    defined.add(node.args.kwarg.arg)
            elif isinstance(node, ast.ClassDef):
                defined.add(node.name)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                defined.add(node.name)
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                for it in node.items:
                    if isinstance(it.optional_vars, ast.Name):
                        defined.add(it.optional_vars.id)
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                if isinstance(node.target, ast.Name):
                    defined.add(node.target.id)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                defined.add(node.id)
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                used.add(node.value.id)

        for name in sorted(used - imported - defined):
            if name in ("self", "cls") or name.isupper() or name.startswith("__"):
                continue
            problems.append(f"{f.relative_to(ROOT)}: 用了 {name}.xxx 但没有导入")

    check("没有「用了但没导入」的模块名", not problems,
          "; ".join(problems[:3]) if problems else "扫描通过")


def main() -> int:
    for fn in (test_imports_complete, test_start_tts_service,
               test_stop_service_tree, test_shutdown_all_kills_service):
        fn()
    print("\n" + "=" * 60)
    print(f"通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    if FAIL:
        for f in FAIL:
            print(f"  FAIL: {f}")
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
