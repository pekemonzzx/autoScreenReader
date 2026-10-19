"""TTS 推理运行时：按需下载、自动配置。

为什么要把运行时和主程序分开：
  * 主程序（界面 + OCR + 引擎）只有 ~200 MB，适合做安装包；
  * TTS 运行时（裁剪后 9 GB）体积大、且只有装了才用得上，做成**首次启动按需下载**，
    用户拿到安装包的第一印象就是"小、装得快"。

发布方做一次：
    python -m screenreader runtime pack --out D:\\build\\gsv     # 分卷 + manifest.json
    # 把 D:\\build\\gsv 整个传到你的网盘/服务器，把 base_url 填进 manifest.json

使用方：
    python -m screenreader runtime install --manifest <url 或本地路径>
    python -m screenreader runtime install --from D:\\下载\\gsv     # 离线：指向已解压的目录
    python -m screenreader runtime status
    python -m screenreader runtime configure                       # 修隐形引用 + 写回配置

装完后 `configure()` 会：
  1. 重写 `users.pth`——否则运行时还会去原打包者的绝对路径找模块（本机能跑、目标机崩）
  2. 把所有声库档案的 `train.gsv_root` 指向新位置
  3. 校验关键文件确实在
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

log = logging.getLogger(__name__)

MANIFEST_NAME = "runtime_manifest.json"
MARKER_NAME = ".runtime_installed.json"
# 分卷大小：单个压缩包别太大，网盘/浏览器上传都容易出问题
DEFAULT_PART_MB = 1900

ProgressFn = Callable[[str, int, int], None]
StopFn = Callable[[], bool]


class InstallCancelled(Exception):
    """用户取消了安装。"""


def runtime_root() -> Path:
    from .config import _app_data_base

    return _app_data_base() / "gsv"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def human_eta(seconds: float) -> str:
    """把剩余时间说成人话。

    不足 1 分钟要按**秒**显示 —— 之前一律显示"剩余约 0 分钟"，
    用户根本不知道还要等 5 秒还是 55 秒。
    """
    if seconds < 0:
        return ""
    if seconds < 60:
        return f"剩余约 {max(1, int(round(seconds)))} 秒"
    if seconds < 3600:
        return f"剩余约 {int(seconds / 60)} 分钟"
    if seconds < 24 * 3600:
        return f"剩余约 {seconds / 3600:.1f} 小时"
    return ""      # 超过一天就不显示（速度估算刚开始时不准）


def dir_size(p: Path) -> int:
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


# --------------------------------------------------------------------------- #
@dataclass
class RuntimePart:
    name: str
    url: str = ""
    sha256: str = ""
    size: int = 0


@dataclass
class RuntimeManifest:
    runtime_version: str = "1"
    base_url: str = ""
    parts: List[RuntimePart] = field(default_factory=list)
    total_size: int = 0
    note: str = ""

    @classmethod
    def load(cls, src) -> "RuntimeManifest":
        """src 可以是本地路径、http(s) URL、或已经是 dict。"""
        if isinstance(src, dict):
            raw = src
        else:
            s = str(src)
            if s.startswith(("http://", "https://")):
                with _open_url(s, timeout=60) as r:
                    raw = json.loads(r.read().decode("utf-8"))
            else:
                raw = json.loads(Path(s).read_text(encoding="utf-8"))
        m = cls(
            runtime_version=str(raw.get("runtime_version", "1")),
            base_url=raw.get("base_url", ""),
            total_size=int(raw.get("total_size", 0)),
            note=raw.get("note", ""),
        )
        m.parts = [
            RuntimePart(
                name=p["name"], url=p.get("url", ""),
                sha256=p.get("sha256", ""), size=int(p.get("size", 0)),
            )
            for p in raw.get("parts", [])
        ]
        return m

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    def resolve_url(self, part: RuntimePart) -> str:
        if part.url:
            return part.url
        if self.base_url:
            return self.base_url.rstrip("/") + "/" + part.name
        return part.name


# --------------------------------------------------------------------------- #
def _pid_on_port(port: int) -> Optional[int]:
    """谁在监听这个端口（Windows 上用 netstat，不依赖 psutil）。"""
    try:
        r = subprocess.run(
            ["netstat", "-ano", "-p", "TCP"],
            capture_output=True, text=True, errors="replace",
            creationflags=0x08000000,      # CREATE_NO_WINDOW，别闪黑框
        )
    except Exception as e:
        log.warning("netstat 失败：%s", e)
        return None
    for line in (r.stdout or "").splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0].upper() == "TCP" \
                and parts[1].endswith(f":{port}") and parts[3].upper() == "LISTENING":
            try:
                return int(parts[4])
            except ValueError:
                continue
    return None


def _image_of(pid: int) -> str:
    try:
        r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                           capture_output=True, text=True, errors="replace",
                           creationflags=0x08000000)
        line = (r.stdout or "").strip().splitlines()
        if line:
            return line[0].split()[0].strip('"').lower()
    except Exception:
        pass
    return ""


def _kill_tree(pid: int) -> bool:
    """连同子进程一起干掉。"""
    try:
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       capture_output=True, text=True, errors="replace",
                       creationflags=0x08000000)
        return True
    except Exception as e:
        log.warning("taskkill %s 失败：%s", pid, e)
        return False


def stop_service(port: int = 9880, only_pid: Optional[int] = None) -> bool:
    """停掉监听在 `port` 上的 TTS 服务。

    为什么不能只按 PID 精确匹配：`api_v2.py` 有可能**派生子进程**，
    真正监听端口的是子进程 —— 我们记下的父 PID 和 netstat 看到的对不上，
    于是被判成"不是本程序启动的"而拒绝清理，用户就看到 python 进程还留着。

    所以策略是：
      1. 先干掉我们记下的那个**进程树**（/T 连子进程）
      2. 再看端口还在不在；还在就按端口找到 PID
      3. 那个 PID 只要还是 python.exe，就认定是我们的服务，一起清掉
         （不是 python 的进程一律不动，避免误杀）
    """
    stopped = False

    # 1) 先杀我们自己记下的那个进程树
    if only_pid:
        if _kill_tree(only_pid):
            stopped = True
        time.sleep(0.4)

    # 2) 看端口还在不在
    pid = _pid_on_port(port)
    if pid is None:
        return stopped

    # 3) 还在 -> 按端口再清一次
    if only_pid and pid != only_pid:
        img = _image_of(pid)
        if "python" not in img:
            log.info("端口 %s 上是 PID %s（%s），不是 python，保留不动", port, pid, img or "?")
            return stopped
        log.info("端口 %s 上是 PID %s（子进程？），一并清掉", port, pid)
    _kill_tree(pid)
    stopped = True
    log.info("已停止 TTS 服务 PID=%s（端口 %s）", pid, port)
    return stopped


def pythons_under(root: Path) -> List[int]:
    """找出可执行文件位于 `root` 之下的 python.exe —— 就是我们的推理服务。

    为什么不能只靠端口找：`api_v2.py` 会派生子进程，父子 PID 对不上；
    某些情况下端口检测也可能失灵。按**可执行文件路径**找是最可靠的 ——
    那个 9GB 运行时目录下的 python.exe 只可能是我们拉起来的。

    优先用 PowerShell 的 CIM 查询（能拿到 ExecutablePath）；失败就用 WMIC。
    """
    root_s = str(root).rstrip("\\/").lower()
    pids: List[int] = []
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Where-Object { $_.ExecutablePath -and "
        f"$_.ExecutablePath.ToLower().StartsWith('{root_s}') }} | "
        "ForEach-Object { $_.ProcessId }"
    )
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True, text=True, errors="replace",
            creationflags=0x08000000, timeout=20,
        )
        for line in (r.stdout or "").splitlines():
            line = line.strip()
            if line.isdigit():
                pids.append(int(line))
    except Exception as e:
        log.warning("按路径找 python 失败（powershell）：%s", e)
    return pids


def kill_service_processes(port: int = 9880, only_pid: Optional[int] = None,
                           runtime: Optional[Path] = None,
                           timeout_s: float = 6.0) -> Dict:
    """尽最大努力清掉 TTS 服务，返回 {'killed': [...], 'left': [...]}。

    用户反馈过一次"关了但还是有 python.exe 留着"，所以这里做三重保险：

      1. 干掉我们记下的那个**进程树**（/T 连子进程）
      2. 反复按端口找持有者并干掉，直到端口释放或超时
      3. 还不行就**按可执行文件路径清扫** —— 运行时目录下的 python.exe
         只可能是我们的服务，跟端口/父子关系都无关

    每一步都写日志，出问题时能看出卡在哪一步。
    """
    killed: List[int] = []

    # 1) 我们记下的进程树
    if only_pid:
        if _kill_tree(only_pid):
            killed.append(only_pid)
            log.info("已杀进程树 PID=%s", only_pid)
        time.sleep(0.3)

    # 2) 反复按端口清，直到释放
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        pid = _pid_on_port(port)
        if pid is None:
            break
        if pid not in killed:
            img = _image_of(pid)
            if img and "python" not in img:
                log.info("端口 %s 上是 %s（非 python），不动它", port, img)
                break
            _kill_tree(pid)
            killed.append(pid)
            log.info("按端口清掉 PID=%s（%s）", pid, img or "?")
        time.sleep(0.4)

    # 3) 兜底：按运行时路径清扫
    if _pid_on_port(port) is not None and runtime:
        for pid in pythons_under(Path(runtime)):
            if pid in killed:
                continue
            _kill_tree(pid)
            killed.append(pid)
            log.info("按运行时路径清掉 PID=%s", pid)
        time.sleep(0.4)

    left = service_pids(port)
    return {"killed": killed, "left": left}



def service_pids(port: int = 9880) -> List[int]:
    """端口上还有没有残留（供自检/测试用）。"""
    p = _pid_on_port(port)
    return [p] if p else []


def status(root: Optional[Path] = None) -> Dict:
    root = Path(root) if root else runtime_root()
    marker = root / MARKER_NAME
    info = {
        "root": str(root),
        "installed": False,
        "version": "",
        "size": 0,
        "missing": [],
    }
    if marker.exists():
        try:
            info.update(json.loads(marker.read_text(encoding="utf-8")))
            info["installed"] = True
        except Exception:
            pass
    if root.exists():
        info["size"] = dir_size(root)
    if info["installed"]:
        info["missing"] = check_files(root)
        if info["missing"]:
            info["installed"] = False
    return info


REQUIRED = [
    "runtime/python.exe",
    "api_v2.py",
    "GPT_SoVITS/TTS_infer_pack/TTS.py",
    "GPT_SoVITS/pretrained_models/chinese-hubert-base",
    "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
    "tools/audio_sr.py",
    "tools/AP_BWE_main",
]


def check_files(root: Path) -> List[str]:
    return [r for r in REQUIRED if not (root / r.replace("/", os.sep)).exists()]


# --------------------------------------------------------------------------- #
def repair_users_pth(root: Path, say=print) -> None:
    """重写 users.pth 指向自己。

    集成包的 users.pth 硬编码了打包者路径；不改的话运行时**还会去原路径找模块**，
    表现为"开发机一切正常、拷到目标机就崩"，而且自动验证也会被骗过。
    """
    pth = root / "runtime" / "Lib" / "site-packages" / "users.pth"
    if not pth.exists():
        say("    （没有 users.pth，跳过）")
        return
    subdirs = ["GPT_SoVITS/BigVGAN", "tools", "tools/asr", "GPT_SoVITS", "tools/uvr5"]
    lines = [str(root)]
    for s in subdirs:
        if (root / s.replace("/", os.sep)).exists():
            lines.append(f"{root}/{s}")
    pth.write_text("\n".join(lines) + "\n", encoding="utf-8")
    say(f"    已重写 users.pth -> {root}")


def configure(root: Optional[Path] = None, say=print,
              config_path: Optional[Path] = None) -> Dict:
    """安装后配置：修隐形引用 + 把运行时位置写进配置 + 重建各声库的推理配置。

    `config_path` 只给测试用（避免测试把用户真实的 config.json 改掉）。
    """
    from .config import AppConfig, default_config_path
    from .voice import VoiceStore

    root = Path(root) if root else runtime_root()
    cfg_file = Path(config_path) if config_path else default_config_path()
    say(f"配置运行时：{root}")
    if not root.exists():
        say("  !! 目录不存在")
        return {"ok": False}

    say("  [1] 修 users.pth 的隐形引用")
    repair_users_pth(root, say)

    cfg = AppConfig.load(cfg_file)
    say("  [2] 把推理运行时位置写进 config.json 并重建各声库的 tts_infer.yaml")
    cfg.tts.runtime_root = str(root)
    try:
        cfg.save(cfg_file)
    except Exception as e:
        say(f"      !! 写配置失败：{e}")

    # 声库可能在两处：<程序目录>/voices 和运行时旁边的 voices。
    # 注意不能写成 `cfg_file.parent / "voices"` —— 那是 data/voices，永远不存在。
    # 每个声库都要重建推理配置，所以两处都得扫。
    seen = set()
    store = None
    for cand in cfg.voices_roots():
        try:
            if not (cand.is_dir() and any(cand.glob("*/voice.json"))):
                continue
        except Exception:
            continue
        if store is None:
            store = VoiceStore(cand)
        names = VoiceStore(cand).list_names()
        for name in names:
            key = (str(cand), name)
            if key in seen:
                continue
            seen.add(key)
            try:
                p = VoiceStore(cand).load(name)
                p.write_api_config(VoiceStore(cand).dir_of(name), runtime_root=root)
                say(f"      {name}: 推理配置已指向 {root}")
            except Exception as e:
                say(f"      !! {name} 写入失败：{e}")
    if store is None:
        store = VoiceStore(cfg.voices_root())
    if not seen:
        say("      （还没有声库；装好声库后跑 voice refresh 即可）")
    say("      注意：声库档案里的 train.gsv_root 保持不动（那是训练用的）")

    say("  [3] 校验关键文件")
    miss = check_files(root)
    if miss:
        for m in miss:
            say(f"      !! 缺少 {m}")
        return {"ok": False, "missing": miss}

    (root / MARKER_NAME).write_text(
        json.dumps({"version": "1", "installed_at": time.strftime("%Y-%m-%d %H:%M:%S")},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    say(f"  [4] 完成，运行时大小 {human(dir_size(root))}")
    say(f"      声库已重新指向新位置，重启界面 / 引擎后生效")
    return {"ok": True, "size": dir_size(root)}


# --------------------------------------------------------------------------- #
def check_writable(path: Path) -> Dict:
    """真正试着写一下，确认这个目录可用。

    为什么必须"真写"：安装要落 9 GB。如果目录不可写（权限不足、被安全软件拦、
    父目录属于别的用户），失败会发生在**解压到一半**的时候，用户看到的是一个
    莫名其妙的 `[WinError 5]`，而且磁盘上留下一堆残渣。
    提前 1 毫秒试一下，就能给出一句人话。
    """
    p = Path(path)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        return {"ok": False, "reason": f"没有权限创建目录：{e}", "kind": "permission"}
    except OSError as e:
        return {"ok": False, "reason": f"无法创建目录：{e}", "kind": "os"}
    probe = p / ".__write_test"
    try:
        probe.write_bytes(b"ok")
        probe.unlink()
    except PermissionError:
        return {"ok": False, "reason": f"目录存在但没有写入权限：{p}", "kind": "permission"}
    except OSError as e:
        return {"ok": False, "reason": f"目录不可写：{e}", "kind": "os"}
    finally:
        try:
            probe.unlink(missing_ok=True)
        except Exception:
            pass
    return {"ok": True, "reason": ""}


def free_space(path: Path) -> int:
    probe = Path(path)
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        return shutil.disk_usage(str(probe)).free
    except Exception:
        return -1


def is_admin() -> bool:
    if os.name != "nt":
        return os.geteuid() == 0 if hasattr(os, "geteuid") else False
    try:
        import ctypes

        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def classify_location(path: Path) -> str:
    """只按**路径本身**判断为什么不能写（不考虑进程完整性级别）。

    拆出来是为了可测：`describe_location` 会先看完整性级别，
    在沙箱里跑测试时那段总是先命中，就测不到这里的分支了。
    """
    p = Path(path)
    try:
        s = str(p.resolve()).lower()
    except Exception:
        s = str(p).lower()

    def under(env_name: str) -> bool:
        base = os.environ.get(env_name, "").strip().lower()
        return bool(base) and s.startswith(base)

    if under("ProgramFiles") or under("ProgramFiles(x86)") or under("ProgramW6432"):
        if is_admin():
            return "在 Program Files 目录下，需要管理员权限。（当前已是管理员，可重试）"
        return ("装在 Program Files 目录下，普通权限写不进去。\n"
                "  两个办法：① 换一个位置（推荐，右边有建议）；"
                "② 右键程序 → 以管理员身份运行。")
    if under("SystemRoot") or under("windir"):
        return "这是 Windows 系统目录，不能往这里写。请换一个位置。"
    try:
        # 注意：不能写 `p.parent == p.anchor`——那是拿 WindowsPath 和 str 比，
        # 永远为 False。用 parts 长度判断盘根最稳（'D:/' -> ('D:\\',)）。
        if p.anchor and len(p.parts) <= 1:
            return ("选的是磁盘根目录。Windows 通常不允许普通用户在根目录下新建文件夹。\n"
                    "  请在资源管理器里先建一个文件夹（比如 D:\\AutoScreenReader），再选它。")
    except Exception:
        pass
    if under("ProgramData"):
        return "ProgramData 下创建子目录通常需要管理员权限，换一个位置更省事。"
    return ("这个目录当前账户没有写入权限。\n"
            "  最简单的办法：在资源管理器里**自己新建一个文件夹**，然后选它——\n"
            "  你自己建的文件夹，权限一定是够的。或者直接点右边建议的位置。")


def describe_location(path: Path) -> str:
    """用**人话**解释某个位置为什么不能写，以及怎么办。

    只回一句 "拒绝访问" 用户根本不知道该做什么。
    **先查完整性级别**——那是独立于 ACL 的一层，也是最容易误判的一层。
    """
    if integrity_is_low():
        return ("本进程运行在**低完整性级别**下（沙箱/容器常见），\n"
                "  这种情况下写任何中完整性的目录都会被拒绝——跟文件夹权限无关。\n"
                "  解决：在资源管理器里直接双击运行本程序，而不是从沙箱/远程环境里启动。")
    return classify_location(path)


def suggest_dest(min_free_gb: float = 12.0) -> Path:
    """挑一个**确实能写、且空间够**的默认安装位置。

    顺序：程序目录 -> %LOCALAPPDATA% -> 用户目录 -> 各磁盘上的候选。
    同时要求剩余空间 >= min_free_gb（运行时解压后约 9 GB，还要留余量），
    否则用户会被"装到一半没空间"坑到。
    """
    import os
    import string

    need = int(min_free_gb * 1024 ** 3)
    candidates: List[Path] = [
        runtime_root(),
        Path(os.environ.get("LOCALAPPDATA", "")) / "AutoScreenReader" / "gsv",
        Path(os.environ.get("USERPROFILE", "")) / "AutoScreenReader" / "gsv",
    ]
    # 再把每个可用磁盘加进来（很多人 C 盘很紧，D 盘才是游戏盘）
    for letter in string.ascii_uppercase:
        d = Path(f"{letter}:/")
        if d.exists():
            candidates.append(d / "AutoScreenReader" / "gsv")

    writable = []
    for c in candidates:
        if not str(c).strip() or str(c) in (".", "\\"):
            continue
        if c.parent == c:      # 跳过盘根
            continue
        wr = check_writable(c)
        if wr["ok"]:
            free = free_space(c)
            writable.append((c, free))
            if free >= need:
                return c
    if writable:
        # 都不够大就挑最空的
        return max(writable, key=lambda x: x[1])[0]
    return runtime_root()


INTEGRITY_LEVELS = {
    0x0000: "Untrusted",
    0x1000: "Low",           # 4096
    0x2000: "Medium",        # 8192
    0x2100: "Medium Plus",   # 8448
    0x3000: "High",          # 12288
    0x4000: "System",        # 16384
}


def token_info() -> Dict:
    """看当前进程令牌的**完整性级别**和组。

    为什么需要——这是"明明能写却报拒绝访问"的元凶，
    而且它跟 DACL 权限是**两套独立机制**：

      Windows 强制完整性控制（MIC）：**低完整性进程不能写中完整性的目录**，
      哪怕 DACL 明明白白给了写权限。

    实测（本机）：同一个目录 `D:\\test`
      · PowerShell（Medium）        -> 能写
      · WMI 服务起的进程（Medium）  -> 能写
      · 沙箱里起的 Python（**Low**）-> 拒绝访问

    所以排障时**必须先看完整性级别**，光看 ACL 会得出错误结论。
    """
    info = {
        "integrity": "", "integrity_value": None, "user": "",
        "admin": is_admin(), "has_authenticated_users": None,
        "deny_only_count": 0, "groups": 0, "raw_ok": False,
    }
    if os.name != "nt":
        return info
    try:
        import re as _re

        from .proc import run_hidden

        r = run_hidden(["whoami", "/groups"], timeout=10)
        out = r.stdout or ""
        if not out.strip():
            info["error"] = (r.stderr or "whoami 无输出").strip()[:120]
            return info
        info["raw_ok"] = True
        # whoami /groups 头两行是标题，用户那行带 S-1-5-21-… 这种账户 SID
        for line in out.splitlines():
            if "S-1-5-21-" in line or "S-1-5-32-544" in line:
                info["user"] = line.strip().split("  ")[0].strip()
                break
        if not info["user"]:
            body = [l.strip() for l in out.splitlines()
                    if l.strip() and "GROUP INFORMATION" not in l and not set(l.strip()) <= {"-"}]
            info["user"] = body[0].split("  ")[0].strip() if body else ""
        info["groups"] = sum(1 for l in out.splitlines() if "S-1-5-" in l or "S-1-1-" in l)
        info["has_authenticated_users"] = "S-1-5-11" in out
        info["deny_only_count"] = len(_re.findall(r"deny only", out, _re.IGNORECASE))
        m = _re.search(r"S-1-16-(\d+)", out)
        if m:
            v = int(m.group(1))
            info["integrity_value"] = v
            info["integrity"] = INTEGRITY_LEVELS.get(v, f"未知({v})")
    except Exception as e:
        info["error"] = str(e)
    return info


def integrity_is_low(ti: Optional[Dict] = None) -> bool:
    ti = ti or token_info()
    v = ti.get("integrity_value")
    return v is not None and v <= 0x1000      # Untrusted / Low


def doctor(say=print) -> Dict:
    """打印一份可直接发出去的安装环境体检报告（目标机排障用，不需要 Python）。"""
    import platform

    from .config import _app_data_base

    say("=" * 78)
    say("安装环境体检")
    say("=" * 78)
    say(f"  程序目录   {_app_data_base()}")
    say(f"  系统       {platform.platform()}")
    say(f"  Python     {platform.python_version()}")

    ti = token_info()
    say("\n[进程令牌 / 完整性级别]")
    say(f"  用户       {ti.get('user') or '?'}")
    say(f"  管理员     {ti.get('admin')}")
    say(f"  完整性级别 {ti.get('integrity') or '无法判断'}")
    if ti.get("raw_ok"):
        say(f"  令牌组数   {ti.get('groups')}"
            f"    Authenticated Users: {'有' if ti.get('has_authenticated_users') else '无'}"
            f"    deny-only 条目: {ti.get('deny_only_count')}")
    else:
        say(f"  （拿不到令牌信息：{ti.get('error', '?')}）")

    if integrity_is_low(ti):
        say("  ★★ 低完整性进程！")
        say("     这是'明明有权限却报拒绝访问'的典型原因——")
        say("     Windows 强制完整性控制规定：**低完整性进程不能写中完整性的目录**，")
        say("     即使 DACL 明明白白给了写权限。")
        say("     常见于：在沙箱/容器里启动的进程、被 AppContainer 包裹的进程。")
        say("     **正常双击运行的程序是 Medium 完整性，不会有这个问题。**")
        say("     解决办法：在资源管理器里直接双击运行（而不是从沙箱/远程环境启动）。")
    elif ti.get("integrity"):
        say("  （中完整性及以上，正常）")
    else:
        say("  （完整性级别未知，按 DACL 权限判断）")

    say("\n[当前位置可用性]")
    st = status()
    say(f"  运行时目录 {st['root']}")
    say(f"    已安装 {st['installed']}  大小 {human(st['size'])}")
    wr = check_writable(st["root"])
    say(f"    可写   {'是' if wr['ok'] else '否 — ' + wr['reason']}")

    say("\n[候选安装位置]")
    import string

    cands: List[Path] = [runtime_root()]
    for env in ("LOCALAPPDATA", "USERPROFILE", "ProgramData"):
        v = os.environ.get(env, "").strip()
        if v:
            cands.append(Path(v) / "AutoScreenReader" / "gsv")
    sysroot = os.environ.get("SystemRoot", "")
    pf = os.environ.get("ProgramFiles", "")
    for letter in string.ascii_uppercase:
        d = Path(f"{letter}:/")
        if d.exists() and str(d) not in (sysroot, pf):
            cands.append(d / "AutoScreenReader" / "gsv")

    seen = set()
    for c in cands:
        key = str(c).lower()
        if key in seen:
            continue
        seen.add(key)
        wr = check_writable(c)
        free = free_space(c)
        free_s = f"{free / 1024 ** 3:.0f} GB" if free >= 0 else "?"
        mark = "可写 ✔" if wr["ok"] else "不可写 ✘"
        say(f"  [{mark}] {c}")
        say(f"            剩余 {free_s}")
        if not wr["ok"]:
            for line in describe_location(c).splitlines():
                say(f"            {line}")

    say("\n[结论]")
    alt = suggest_dest()
    if check_writable(alt)["ok"]:
        say(f"  建议装到：{alt}")
        say(f"  （该盘剩余 {human(free_space(alt))}）")
    else:
        say("  !! 没有找到可写的位置，请手动在资源管理器里新建一个文件夹再试")
    say("=" * 78)
    return {"token": ti, "suggest": str(alt)}


# --------------------------------------------------------------------------- #
# TTS 推理服务
#
# 用户实测反馈："点开始识别，卡片全是'失败'，估计是服务没起来"。
# 确实 —— 光装好运行时还不够，还得有个 `api_v2.py` 进程在跑。
# 让用户自己去开命令行是不现实的（目标机连 Python 都没有），
# 所以这里提供"检查 + 自动拉起"。
# --------------------------------------------------------------------------- #
def ensure_voice_configs(cfg, voices_root: Optional[Path] = None, say=print) -> int:
    """确保每个声库都有 `tts_infer.yaml`，并且指向**当前运行时**。

    这个必须做，原因很实在：声库包**不带** `tts_infer.yaml`
    （它是本机生成的，打进去没意义），所以刚 `voice install` 完的声库
    在目标机上**一个推理配置都没有** → 服务起不来 → 所有合成都失败。

    实测症状：装好声库包、点开始识别，卡片全是「失败」，状态栏说
    "还没有安装任何声库（voices/ 下没有 tts_infer.yaml）"。
    """
    from .voice import VoiceStore

    store = VoiceStore(voices_root or cfg.voices_root())
    root_s = (cfg.tts.runtime_root or "").strip()
    runtime_root = Path(root_s) if root_s and Path(root_s).exists() else None
    n = 0
    for name in store.list_names():
        try:
            p = store.load(name)
            p.write_api_config(store.dir_of(name), runtime_root=runtime_root)
            n += 1
        except Exception as e:
            say(f"  !! 声库 {name} 生成推理配置失败：{e}")
    if n:
        say(f"  已为 {n} 个声库生成推理配置（运行时：{runtime_root or '声库自带的 gsv_root'}）")
    return n


def service_port(cfg) -> int:
    url = (cfg.tts.api_url or "http://127.0.0.1:9880").rstrip("/")
    try:
        return int(url.rsplit(":", 1)[-1])
    except Exception:
        return 9880


def service_ready(port: int = 9880, timeout: float = 2.0) -> bool:
    """TTS 服务是否已经在监听。"""
    import socket

    try:
        with socket.create_connection(("127.0.0.1", port), timeout=timeout):
            return True
    except OSError:
        return False


def find_infer_config(cfg, voices_root: Optional[Path] = None) -> Optional[Path]:
    """找出当前声库的 tts_infer.yaml（没有就用任意一个声库的）。"""
    from .voice import VoiceStore

    store = VoiceStore(voices_root or cfg.voices_root())
    names = store.list_names()
    if not names:
        return None
    preferred = (cfg.tts.voice or "").strip()
    if preferred in names:
        names = [preferred] + [n for n in names if n != preferred]
    for n in names:
        p = store.dir_of(n) / "tts_infer.yaml"
        if p.exists():
            return p
    return None


def start_tts_service(cfg, gsv_root: Optional[Path] = None,
                      voices_root: Optional[Path] = None, say=print) -> Dict:
    """后台拉起 `api_v2.py`（**隐藏窗口**，否则 GUI 里会弹一个黑框）。

    启动一次要几十秒加载权重，所以这里只负责"发车"，不负责等。
    等就绪请用 `wait_service_ready()`。
    """
    from .proc import popen_hidden

    port = service_port(cfg)
    if service_ready(port):
        return {"ok": True, "already": True, "port": port}

    root = Path(gsv_root) if gsv_root else ((cfg.tts.runtime_root or "").strip() and Path(cfg.tts.runtime_root)) or runtime_root()
    py = Path(root) / "runtime" / "python.exe"
    if not py.exists():
        msg = f"找不到推理运行时：{py}"
        say(f"  !! {msg}")
        return {"ok": False, "message": msg}

    infer = find_infer_config(cfg, voices_root)
    if infer is None:
        msg = "还没有安装任何声库（voices/ 下没有 tts_infer.yaml）"
        say(f"  !! {msg}")
        return {"ok": False, "message": msg}

    logf = Path(root).parent / "data"
    try:
        logf.mkdir(parents=True, exist_ok=True)
    except Exception:
        logf = Path(root)
    logfile = logf / "tts_service.log"

    say(f"  启动推理服务：{py}")
    say(f"   配置 {infer}")
    try:
        fh = open(logfile, "w", encoding="utf-8", errors="replace")
        p = popen_hidden(
            [str(py), "-s", "api_v2.py", "-a", "127.0.0.1", "-p", str(port), "-c", str(infer)],
            cwd=str(root), stdout=fh, stderr=subprocess.STDOUT,
        )
    except Exception as e:
        msg = f"启动失败：{e}"
        say(f"  !! {msg}")
        return {"ok": False, "message": msg}
    say(f"    已启动 PID={p.pid}，日志 {logfile}")
    return {"ok": True, "pid": p.pid, "port": port, "log": str(logfile), "config": str(infer)}


def wait_service_ready(port: int = 9880, timeout_s: float = 120.0,
                       should_stop: Optional[StopFn] = None, say=print) -> bool:
    """等服务就绪。首次启动要加载全部权重，实测约 10~40 秒。"""
    t0 = time.time()
    last = 0.0
    while time.time() - t0 < timeout_s:
        if should_stop and should_stop():
            return False
        if service_ready(port, timeout=1.0):
            say(f"  推理服务已就绪（{time.time() - t0:.0f}s）")
            return True
        now = time.time()
        if now - last > 5:
            last = now
            say(f"    等待推理服务…（{now - t0:.0f}s）")
        time.sleep(1.5)
    say(f"  !! 等待 {timeout_s:.0f}s 仍未就绪，看日志：{runtime_root().parent / 'data' / 'tts_service.log'}")
    return False


def ensure_tts_service(cfg, say=print, timeout_s: float = 120.0) -> Dict:
    """检查服务；没起来就拉起来并等它就绪。"""
    port = service_port(cfg)
    if service_ready(port):
        return {"ok": True, "already": True, "port": port}

    # 先把声库的推理配置补齐。声库包**不带** tts_infer.yaml，
    # 缺了它服务起不来，表现是"装好了声库但每句都合成失败"。
    try:
        ensure_voice_configs(cfg, say=say)
    except Exception as e:
        say(f"  !! 生成声库推理配置失败：{e}")

    say(f"  TTS 服务（端口 {port}）没在跑，尝试自动启动…")
    res = start_tts_service(cfg, say=say)
    if not res.get("ok"):
        return res
    if wait_service_ready(port, timeout_s, say=say):
        res["ready"] = True
        return res
    res["ok"] = False
    res["message"] = f"服务在 {timeout_s:.0f} 秒内没起来，看日志 {res.get('log')}"
    return res


def _split_auth(url: str):
    """把 URL 里的 `user:pass@` 拆出来，返回 (干净 URL, Authorization 头或 None)。

    为什么需要：WebDAV / 需要认证的 S3 端点，直链是带账号密码的，
    例如 `https://user:pass@dav.example.cn/path/gsv.zip`。
    Python 的 urllib **不会**自动把 userinfo 变成 Basic 认证头，
    直接丢进去会得到 401。所以这里手动处理。
    """
    import base64
    import urllib.parse as up

    parts = up.urlsplit(url)
    if "@" not in (parts.netloc or ""):
        return url, None
    userinfo, host = parts.netloc.rsplit("@", 1)
    if ":" not in userinfo:
        return url, None
    user, pwd = userinfo.split(":", 1)
    clean = up.urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))
    token = base64.b64encode(f"{up.unquote(user)}:{up.unquote(pwd)}".encode("utf-8")).decode("ascii")
    return clean, f"Basic {token}"


def _open_url(url: str, extra_headers: Optional[Dict[str, str]] = None,
              timeout: int = 120, range_from: int = 0):
    """打开 URL，自动处理 URL 里的账号密码与断点续传范围。"""
    clean, auth = _split_auth(url)
    req = urllib.request.Request(clean)
    if auth:
        req.add_header("Authorization", auth)
    for k, v in (extra_headers or {}).items():
        req.add_header(k, v)
    if range_from:
        req.add_header("Range", f"bytes={range_from}-")
    return urllib.request.urlopen(req, timeout=timeout)


def download_file(url: str, dest: Path, expect_sha: str = "", expect_size: int = 0,
                  progress: Optional[ProgressFn] = None, say=print,
                  should_stop: Optional[StopFn] = None,
                  headers: Optional[Dict[str, str]] = None) -> bool:
    """带断点续传与 sha256 校验的下载。`should_stop()` 返回真则取消。

    支持 `https://user:pass@host/...` 形式的认证直链（WebDAV、
    需要认证的 S3 端点），也支持 S3 预签名 URL（query 里带签名参数）。
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    done = part.stat().st_size if part.exists() else 0

    if expect_size and done == expect_size:
        say(f"    已有完整文件，跳过下载：{dest.name}")
    else:
        if done:
            say(f"    断点续传 {dest.name}（已有 {human(done)}）")
        else:
            say(f"    下载 {dest.name}")
        try:
            with _open_url(url, headers, range_from=done) as r:
                # 关键：请求了 Range 但服务器返回 200，说明它**不支持断点续传**，
                # 回给的是完整文件。这时候必须清空重写——否则会追加到
                # 已有内容后面，得到一个长度翻倍、内容错乱的坏文件。
                status = getattr(r, "status", None) or r.getcode()
                if done and status != 206:
                    say("    （服务器不支持断点续传，从头开始下）")
                    done = 0
                    mode = "wb"
                else:
                    mode = "ab"

                total = expect_size or (done + int(r.headers.get("Content-Length", 0) or 0))
                t0 = time.time()
                last = 0.0
                with open(part, mode) as f:
                    while True:
                        if should_stop and should_stop():
                            raise InstallCancelled()
                        buf = r.read(1 << 20)
                        if not buf:
                            break
                        f.write(buf)
                        done += len(buf)
                        now = time.time()
                        if progress and now - last > 0.3:
                            last = now
                            progress(dest.name, done, total)
                        elif say and now - last > 3:
                            last = now
                            sp = done / max(0.1, now - t0)
                            say(f"      {human(done)}/{human(total)}  {human(sp)}/s")
        except InstallCancelled:
            say("    已取消（保留已下载的部分，下次可断点续传）")
            raise
        except urllib.error.HTTPError as e:
            hint = ""
            if e.code in (401, 403):
                hint = ("\n       （401/403：链接需要认证或被防盗链拦了。"
                        "如果是网盘分享页，请改用「本地文件」；"
                        "如果是 WebDAV，把账号密码写进 URL：https://user:pass@host/...）")
            elif e.code == 404:
                hint = "\n       （404：文件不存在，检查链接是否过期）"
            say(f"    !! 下载失败：HTTP {e.code} {e.reason}{hint}")
            return False
        except Exception as e:
            say(f"    !! 下载失败：{e}")
            return False
        part.replace(dest)

    if expect_sha:
        say("    校验 sha256 …")
        got = _sha256(dest)
        if got != expect_sha:
            say(f"    !! 校验不通过\n       期望 {expect_sha}\n       实际 {got}")
            dest.unlink(missing_ok=True)
            return False
        say("    sha256 正确 ✅")
    return True


def _long_path(p: Path) -> str:
    r"""Windows 长路径：加 `\\?\` 前缀绕过 260 字符限制。

    这里最长的一条是 168 字符，暂时没超；但换一台机器、或者安装目录更深
    （比如 `C:\Users\某某某\AppData\Local\AutoScreenReader\gsv`），
    `runtime/Lib/site-packages/transformers/models/deprecated/...` 这类路径
    很容易突破 260，报出来的是 `[WinError 3] 找不到路径`，非常难查。
    提前加上前缀，一劳永逸。
    """
    if os.name != "nt":
        return str(p)
    s = os.path.abspath(str(p))
    if s.startswith("\\\\?\\"):
        return s
    if s.startswith("\\\\"):          # UNC 路径
        return "\\\\?\\UNC\\" + s[2:]
    return "\\\\?\\" + s


def _safe_join(dest: Path, name: str) -> Optional[Path]:
    """把 zip 里的条目名安全地拼到目标目录下；非法条目返回 None。"""
    rel = Path(name.replace("\\", "/"))
    if rel.is_absolute() or ".." in rel.parts:
        return None
    parts = [p for p in rel.parts if p not in ("", ".")]
    if not parts:
        return None
    return dest.joinpath(*parts)


def extract_zip(zip_path: Path, dest: Path, say=print,
                should_stop: Optional[StopFn] = None,
                progress: Optional[ProgressFn] = None) -> bool:
    """解压运行时。

    自己实现解压循环而不是用 `ZipFile.extractall`，因为需要：
      * **进度反馈**——9 GB / 5.3 万个文件要跑好几分钟，
        没有进度用户会以为卡死然后强杀，留下一个半拉子的目录
      * 长路径支持（`\\?\` 前缀）
      * 跳过 `__pycache__`（几千个文件，纯浪费）
      * 单个文件失败时**不整个中断**（杀毒软件偶尔会锁文件）

    返回 False 表示结构性失败（比如一个文件都没解出来）。
    """
    say(f"    解压 {zip_path.name} -> {dest}")
    wr = check_writable(dest)
    if not wr["ok"]:
        say(f"    !! 解压目标不可写：{wr['reason']}")
        return False

    try:
        z = zipfile.ZipFile(zip_path)
    except Exception as e:
        say(f"    !! 打不开压缩包：{e}")
        return False

    with z:
        infos = z.infolist()
        total_bytes = sum(i.file_size for i in infos) or 1
        total_files = sum(1 for i in infos if not i.is_dir())
        say(f"      {total_files} 个文件，{human(total_bytes)}")

        done_bytes = 0
        done_files = 0
        failed: List[str] = []
        skipped = 0
        t0 = time.time()
        last = 0.0

        for i, info in enumerate(infos):
            if should_stop and should_stop():
                raise InstallCancelled()

            name = info.filename
            if "__pycache__" in name or name.endswith((".pyc", ".pyo")):
                skipped += 1
                done_bytes += info.file_size
                continue

            target = _safe_join(dest, name)
            if target is None:
                skipped += 1
                continue

            try:
                if info.is_dir():
                    os.makedirs(_long_path(target), exist_ok=True)
                    continue
                os.makedirs(_long_path(target.parent), exist_ok=True)
                with z.open(info) as srcf, open(_long_path(target), "wb") as out:
                    while True:
                        buf = srcf.read(1 << 20)
                        if not buf:
                            break
                        out.write(buf)
                        done_bytes += len(buf)
                        now = time.time()
                        if progress and now - last > 0.3:
                            last = now
                            progress(zip_path.name, done_bytes, total_bytes)
                done_files += 1
            except InstallCancelled:
                raise
            except Exception as e:
                # 单个文件失败（杀毒锁文件、路径异常）不要拖垮整个 9GB 安装
                failed.append(f"{name}: {e}")
                if len(failed) <= 3:
                    say(f"      !! 解压失败 {name[:70]}：{e}")

            now = time.time()
            if say and now - last > 3:
                last = now
                sp = done_bytes / max(0.1, now - t0)
                say(f"      {done_files}/{total_files} 个文件  "
                    f"{human(done_bytes)}/{human(total_bytes)}  {human(sp)}/s")

        if progress:
            progress(zip_path.name, total_bytes, total_bytes)

    if failed:
        say(f"    !! 有 {len(failed)} 个文件解压失败（其余已解出）")
        for f in failed[:5]:
            say(f"       {f}")
    if skipped:
        say(f"      （跳过 {skipped} 个 __pycache__/缓存条目）")

    # 结构性失败：一个文件都没解出来
    if done_files == 0:
        say("    !! 一个文件都没解出来，压缩包内容可能不对")
        return False
    return True


def _looks_like_manifest(src) -> bool:
    s = str(src).lower().split("?")[0]
    return s.endswith(".json")


def install(src, dest: Optional[Path] = None, progress: Optional[ProgressFn] = None,
            keep_parts: bool = False, say=print, should_stop: Optional[StopFn] = None) -> Dict:
    """统一的安装入口，**自动判断来源类型**：

      * `...json`（网址或本地文件）  -> 按清单分卷安装
      * 目录                        -> 离线安装（直接复制）
      * 其它（.zip / 任意直链）      -> 单包安装

    之所以要"自动判断"：发布者可能只把整个 gsv 目录压成一个 zip 传网盘，
    拿到的就是一条直链，没有清单。这种最朴素的做法也要能用。
    """
    s = str(src)
    if not s.startswith(("http://", "https://")):
        p = Path(s)
        if p.is_dir():
            return install_from_dir(p, dest, say=say, should_stop=should_stop)
        if _looks_like_manifest(p):
            return install_parts(p, dest, progress, keep_parts, say, should_stop)
        return install_single_zip(p, dest, progress, say, should_stop)

    # URL：看扩展名判断
    if _looks_like_manifest(s):
        return install_parts(s, dest, progress, keep_parts, say, should_stop)
    return install_single_zip(s, dest, progress, say, should_stop)


def _unnest(dest: Path, say=print) -> bool:
    """压缩包/文件夹里多套了一层目录时，把内容提上来。

    很多人压缩时会把 `gsv` 整个文件夹压进去，解压出来是 `<dest>/gsv/runtime/...`。
    """
    if (dest / "runtime" / "python.exe").exists():
        return True
    for sub in [d for d in dest.iterdir() if d.is_dir()]:
        if (sub / "runtime" / "python.exe").exists():
            say(f"    发现多套了一层目录（{sub.name}），内容提到上层")
            for item in list(sub.iterdir()):
                tgt = dest / item.name
                if tgt.exists():
                    shutil.rmtree(tgt, ignore_errors=True) if tgt.is_dir() else tgt.unlink()
                shutil.move(str(item), str(tgt))
            try:
                sub.rmdir()
            except OSError:
                pass
            return True
    return False


def install_single_zip(src, dest: Optional[Path] = None, progress: Optional[ProgressFn] = None,
                       say=print, should_stop: Optional[StopFn] = None) -> Dict:
    """单包安装：一条直链（或本地 zip）指向整个运行时。

    不需要 manifest。装完会把算出来的 sha256 写进安装标记，便于以后核对。
    下载支持断点续传（网盘/浏览器中断后重来不会从头开始）。
    """
    dest = Path(dest) if dest else runtime_root()
    s = str(src)
    is_url = s.startswith(("http://", "https://"))

    say("=" * 78)
    say(f"安装 TTS 运行时（单包） -> {dest}")
    say(f"  来源 {s}")
    say("=" * 78)

    # 先确认可写再干别的：check_writable 自己会建目录，
    # 所以这里**不能**先 mkdir——父路径是个文件时它会直接抛异常。
    wr = check_writable(dest)
    if not wr["ok"]:
        say(f"  !! 安装位置不可用：{wr['reason']}")
        say("     换一个位置再试（比如 D:\\ 下新建一个文件夹，或用户目录下）。")
        return {"ok": False, "failed": wr["reason"], "kind": "permission"}

    cache = dest / "_parts"
    cache.mkdir(parents=True, exist_ok=True)
    name = "gsv.zip"
    if not is_url:
        name = Path(s).name or name
    local = cache / name

    if is_url:
        try:
            if not download_file(s, local, progress=progress, say=say, should_stop=should_stop):
                return {"ok": False, "failed": name}
        except InstallCancelled:
            say("已取消下载（已下载的部分保留着，下次断点续传）")
            return {"ok": False, "cancelled": True}
    else:
        local = Path(s)
        if not local.exists():
            return {"ok": False, "failed": f"文件不存在：{local}"}
        if not local.is_file():
            return install_from_dir(local, dest, say=say, should_stop=should_stop)

    if not zipfile.is_zipfile(local):
        say(f"    !! 这不像一个 zip 文件：{local}")
        say("       （网盘分享链接通常会返回一个网页，程序下载到的是 HTML）")
        try:
            head = local.read_bytes()[:200]
            say(f"        文件开头：{head[:80]!r}")
        except Exception:
            pass
        return {"ok": False, "failed": "不是 zip（可能拿到的是网盘网页）"}

    say(f"    解压 {local.name}（{human(local.stat().st_size)}）…")
    # 空间预检：解压后要占 ~9GB，装到一半没空间会很难看
    need = int(local.stat().st_size * 1.1)
    free = free_space(dest)
    if free >= 0 and free < need:
        say(f"    !! 空间不足：剩余 {human(free)}，解压至少需要 {human(need)}")
        return {"ok": False, "failed": f"磁盘空间不足（剩余 {human(free)}）", "kind": "space"}
    try:
        if not extract_zip(local, dest, say, should_stop, progress):
            return {"ok": False, "failed": "解压失败"}
    except InstallCancelled:
        say("已取消解压")
        return {"ok": False, "cancelled": True}

    _unnest(dest, say)

    if not (dest / "runtime" / "python.exe").exists():
        say("    !! 解压后没找到 runtime\\python.exe，这不像运行时目录")
        return {"ok": False, "failed": "压缩包内容不对"}

    digest = _sha256(local)
    say(f"    sha256 {digest[:16]}…")
    res = configure(dest, say)
    if res.get("ok"):
        try:
            (dest / MARKER_NAME).write_text(
                json.dumps({
                    "version": "1",
                    "installed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "source": s[:300],
                    "sha256": digest,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
        # 单包装完，压缩包本身可以删掉腾空间
        if is_url:
            try:
                local.unlink()
            except Exception:
                pass
    try:
        if not any(cache.iterdir()):
            cache.rmdir()
    except OSError:
        pass
    return res


def install_parts(manifest_src, dest: Optional[Path] = None, progress: Optional[ProgressFn] = None,
                  keep_parts: bool = False, say=print, should_stop: Optional[StopFn] = None) -> Dict:
    """按 manifest 分卷下载并解压运行时。`should_stop()` 返回真则取消。"""
    dest = Path(dest) if dest else runtime_root()
    man = RuntimeManifest.load(manifest_src)
    say("=" * 78)
    say(f"安装 TTS 运行时（分卷） -> {dest}")
    say(f"  版本 {man.runtime_version}   共 {len(man.parts)} 个分卷"
        + (f"   合计 {human(man.total_size)}" if man.total_size else ""))
    if man.note:
        say(f"  说明：{man.note}")
    say("=" * 78)

    dest = Path(dest) if dest else runtime_root()
    wr = check_writable(dest)
    if not wr["ok"]:
        say(f"  !! 安装位置不可用：{wr['reason']}")
        say("     换一个位置再试（比如 D:\\ 下新建一个文件夹，或用户目录下）。")
        return {"ok": False, "failed": wr["reason"], "kind": "permission"}
    cache = dest / "_parts"
    cache.mkdir(parents=True, exist_ok=True)
    try:
        for p in man.parts:
            if should_stop and should_stop():
                raise InstallCancelled()
            url = man.resolve_url(p)
            f = cache / p.name
            if not download_file(url, f, p.sha256, p.size, progress, say, should_stop):
                return {"ok": False, "failed": p.name}
            if not extract_zip(f, dest, say, should_stop, progress):
                return {"ok": False, "failed": p.name}
            if not keep_parts:
                f.unlink(missing_ok=True)
    except InstallCancelled:
        say("已取消安装（下次会从断点继续）")
        return {"ok": False, "cancelled": True}
    finally:
        try:
            if not any(cache.iterdir()):
                cache.rmdir()
        except OSError:
            pass
    return configure(dest, say)


def install_from_dir(src: Path, dest: Optional[Path] = None, move: bool = False, say=print,
                     should_stop: Optional[StopFn] = None) -> Dict:
    """离线安装：src 是已经解压好的运行时目录（比如网盘下载后解压的）。"""
    src = Path(src)
    dest = Path(dest) if dest else runtime_root()
    say(f"从本地安装运行时：{src} -> {dest}")
    if not src.exists():
        say("  !! 源目录不存在")
        return {"ok": False}
    if src.is_file():
        # 传进来的是压缩包，走单包流程
        return install_single_zip(src, dest, say=say, should_stop=should_stop)

    # 允许用户把包解开后多套了一层目录
    probe = src
    for _ in range(3):
        if (probe / "api_v2.py").exists() or (probe / "runtime" / "python.exe").exists():
            break
        subs = [d for d in probe.iterdir() if d.is_dir()]
        if len(subs) == 1:
            probe = subs[0]
        else:
            break
    if not (probe / "runtime" / "python.exe").exists():
        say(f"  !! 在 {probe} 下找不到 runtime\\python.exe，这不是运行时目录")
        return {"ok": False}

    dest.mkdir(parents=True, exist_ok=True)
    total_files = sum(1 for f in probe.rglob("*") if f.is_file())
    done = 0
    t0 = time.time()
    last = t0
    for item in probe.iterdir():
        tgt = dest / item.name
        if tgt.exists():
            shutil.rmtree(tgt, ignore_errors=True) if tgt.is_dir() else tgt.unlink()
        if move:
            shutil.move(str(item), str(tgt))
        elif item.is_dir():
            for f in item.rglob("*"):
                if should_stop and should_stop():
                    raise InstallCancelled()
                rel = f.relative_to(item)
                out = tgt / rel
                if f.is_dir():
                    out.mkdir(parents=True, exist_ok=True)
                    continue
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, out)
                done += 1
                now = time.time()
                if now - last > 3:
                    last = now
                    say(f"      {done}/{total_files} 个文件 …")
        else:
            shutil.copy2(item, tgt)
    return configure(dest, say)


# --------------------------------------------------------------------------- #
def pack_single(src_root: Path, out_dir: Path, name: str = "gsv.zip", say=print) -> Path:
    """把整个运行时压成**一个** zip。

    适合"传网盘拿直链"这种最朴素的分发方式：一条链接指向这个 zip 就能装。
    缺点是单个 9GB 文件，网盘/浏览器上传容易中断，断了要重传整包；
    文件数多、体积大时更推荐 `pack()` 的分卷方式。
    """
    src_root = Path(src_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (src_root / "runtime" / "python.exe").exists():
        raise SystemExit(f"{src_root} 不像运行时目录（缺 runtime\\python.exe）")

    out = out_dir / name
    files = sorted(f for f in src_root.rglob("*") if f.is_file())
    total = sum(f.stat().st_size for f in files)
    say(f"打包成单个 zip：{out}")
    say(f"  {len(files)} 个文件，{human(total)}（压缩前）")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for i, f in enumerate(files):
            z.write(f, str(f.relative_to(src_root)))
            if i and i % 3000 == 0:
                say(f"    {i}/{len(files)} …")
    digest = _sha256(out)
    sidecar = out.with_suffix(out.suffix + ".sha256")
    sidecar.write_text(f"{digest}  {out.name}\n", encoding="utf-8")
    say("\n" + "=" * 78)
    say(f"完成：{out}  {human(out.stat().st_size)}")
    say(f"sha256：{digest}")
    say(f"（也写进了 {sidecar.name}）")
    say("=" * 78)
    say("\n把它传到网盘/服务器，拿到**直链**后，让使用方在首次启动的窗口里")
    say("选「在线下载」把直链填进去即可。（注意：网盘分享页面的网址不是直链）")
    return out


def pack(src_root: Path, out_dir: Path, base_url: str = "", part_mb: int = DEFAULT_PART_MB,
         say=print) -> Path:
    """把运行时目录分卷打包 + 生成 manifest.json。

    分卷而不是一个大包：9GB 一个文件在很多网盘/浏览器上传里都会出问题，
    而且分卷可以单独重传坏掉的那一卷。
    """
    src_root = Path(src_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (src_root / "runtime" / "python.exe").exists():
        raise SystemExit(f"{src_root} 不像运行时目录（缺 runtime\\python.exe）")

    files = sorted(f for f in src_root.rglob("*") if f.is_file())
    total = sum(f.stat().st_size for f in files)
    say(f"打包 {src_root}  共 {len(files)} 个文件，{human(total)}")

    parts: List[RuntimePart] = []
    idx = 0
    cur: List[Path] = []
    cur_size = 0
    limit = part_mb * 1024 * 1024

    def flush():
        nonlocal idx, cur, cur_size
        if not cur:
            return
        idx += 1
        name = f"gsv_part{idx:02d}.zip"
        out = out_dir / name
        say(f"  写 {name} （{len(cur)} 个文件，{human(cur_size)}）…")
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for f in cur:
                z.write(f, str(f.relative_to(src_root)))
        parts.append(RuntimePart(name=name, sha256=_sha256(out), size=out.stat().st_size))
        cur = []
        cur_size = 0

    for f in files:
        sz = f.stat().st_size
        if cur and cur_size + sz > limit:
            flush()
        cur.append(f)
        cur_size += sz
    flush()

    man = RuntimeManifest(
        runtime_version="1",
        base_url=base_url,
        parts=parts,
        total_size=sum(p.size for p in parts),
        note="自动生成的 TTS 推理运行时（由 tools/build_runtime.py 裁剪后打包）",
    )
    mpath = out_dir / MANIFEST_NAME
    man.save(mpath)
    say("\n" + "=" * 78)
    say(f"完成：{len(parts)} 个分卷，合计 {human(man.total_size)}（压缩后）")
    say(f"清单：{mpath}")
    say("=" * 78)
    say("\n上传到你的网盘/服务器后，把 manifest.json 里的 base_url 改成下载前缀，")
    say("然后让使用方执行：")
    say(f"  python -m screenreader runtime install --manifest <manifest.json 的网址>")
    return mpath
