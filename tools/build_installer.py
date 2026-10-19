"""打包主程序 → 组装安装目录 → （可选）编译 Inno Setup 安装器。

装出来的东西长这样：

    安装目录/
      读屏台.exe              主程序（界面 + OCR + 引擎）
      _internal/              PyInstaller 带的运行时与 RapidOCR 模型
      data/config.json        配置模板
      voices/                 声库目录（空的，用户往里放声库包）
      gsv/                    TTS 推理运行时（**安装时为空**，首次启动按需下载）
      start.bat               一键启动
      run_console.bat         带控制台启动（排障用）
      README.txt              给使用者的说明

用法：
    $env:PYTHONPATH='src'
    .\\.venv\\Scripts\\python tools\\build_installer.py                 # 只打主程序
    .\\.venv\\Scripts\\python tools\\build_installer.py --inno          # 顺便编译安装器
    .\\.venv\\Scripts\\python tools\\build_installer.py --skip-pyinstaller   # 复用已有 dist
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DIST = ROOT / "dist"
BUILD = ROOT / "build"
OUT = ROOT / "build" / "installer_tree"


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def dir_size(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


# --------------------------------------------------------------------------- #
START_BAT = r"""@echo off
setlocal
chcp 65001 >nul
REM ============================================================
REM  Auto Screen Reader - launcher
REM
REM  Starts the TTS API (if the runtime is installed) then the GUI.
REM
REM  IMPORTANT: this file is intentionally 100% pure ASCII.
REM    * Chinese text inside a .bat breaks cmd parsing (GBK trailing
REM      bytes collide with & | > etc. even after chcp 65001).
REM    * So we NEVER hardcode the exe name - it is localized. We find
REM      it with a wildcard instead. Hardcoding it once shipped a
REM      launcher that told the user the exe did not exist.
REM ============================================================
set "HERE=%~dp0"
cd /d "%HERE%"

REM --- locate the single .exe shipped next to this script ---
set "APP="
for %%f in ("%HERE%*.exe") do set "APP=%%~ff"
if "%APP%"=="" (
  echo [ERROR] No .exe found next to this script.
  echo         Expected the main program in: %HERE%
  pause
  exit /b 1
)

if not exist "%HERE%gsv\runtime\python.exe" goto noruntime

REM --- start TTS API in background if port 9880 is free ---
netstat -ano | findstr /r /c:"LISTENING.*:9880" >nul
if %errorlevel%==0 (
  echo [1/2] TTS service already running on 9880
  goto startgui
)

echo [1/2] Starting TTS service ...
set "CFG="
for %%f in ("%HERE%voices\*\tts_infer.yaml") do set "CFG=%%~ff"
if "%CFG%"=="" (
  echo    [!] no voice pack found under voices\ - skipping TTS service
  goto startgui
)
start "TTS" /min "%HERE%gsv\runtime\python.exe" -s api_v2.py -a 127.0.0.1 -p 9880 -c "%CFG%"
echo    waiting for the service ...
for /l %%i in (1,1,60) do (
  timeout /t 1 /nobreak >nul
  netstat -ano | findstr /r /c:"LISTENING.*:9880" >nul && goto startgui
)
echo    [!] service did not come up in 60s, starting the app anyway

:startgui
echo [2/2] Starting the app ...
start "" "%APP%"
exit /b 0

:noruntime
echo.
echo   TTS runtime is not installed yet.
echo   The app can still read text, but it cannot speak.
echo.
echo   Just let the app start - it will pop up an installer dialog where you can
echo   pick where to install and whether to download online or use a local file.
echo.
start "" "%APP%"
exit /b 0
"""

RUN_CONSOLE_BAT = r"""@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

REM Pure ASCII on purpose - see the note in start.bat.
set "APP="
for %%f in ("%~dp0*.exe") do set "APP=%%~ff"
if "%APP%"=="" (
  echo [ERROR] No .exe found next to this script.
  pause
  exit /b 1
)

echo Running with console output (for troubleshooting) ...
echo.
echo   This exe is also a command line tool - no Python required.
echo   Run it in this folder with any of these:
echo.
echo     --runtime status                      check if the runtime is installed
echo     --runtime verify                      check the runtime for missing files
echo     --runtime install --manifest ^<url^>     download and install it
echo     --runtime install --from ^<folder^>      install from a local folder/zip
echo     --voice list                          list installed voices
echo     --voice install ^<xxx.voicepack.zip^>    install a voice pack
echo.
echo   Output is also written to data\cli_output.txt
echo.
"%APP%" %*
pause
"""


def _ascii_bat(text: str, name: str) -> str:
    """确认 .bat 里没有非 ASCII 字符。

    血泪教训：曾经用 errors="replace" 写 .bat，把 `读屏台.exe` 变成了 `???.exe`，
    用户双击直接报「Windows 找不到文件 '???.exe'」。
    所以这里**不允许静默替换**——发现非 ASCII 就直接报错，逼着改成通配符写法。
    """
    bad = sorted({c for c in text if ord(c) > 127})
    if bad:
        raise SystemExit(
            f"!! {name} 里含有非 ASCII 字符 {bad!r}\n"
            "   .bat 一律用纯 ASCII，文件名用通配符找（%%~ff），不要硬编码中文。"
        )
    return text

README_TXT = """自动读屏台 —— 使用说明
==================================================

【这个程序不需要你装 Python，也不需要 CUDA 工具链，双击就能用。】


一、这是什么
    框选屏幕上的一块或多块区域，程序持续 OCR 识别里面的文字，
    按句列成卡片，点行首按钮就能用克隆音色朗读。
    文字从屏幕上消失后，对应的卡片和语音会自动销毁。


二、第一次使用（三步）

    第 1 步：装主程序
        双击安装包，装完约 325 MB。

    第 2 步：装"语音运行时"（约 9 GB，按需下载）
        直接双击 读屏台.exe 或 start.bat。程序发现没装运行时会**自动弹出安装窗口**，
        在里面可以：
          * 选择装到哪个盘（默认装在程序目录下的 gsv\\）
          * 选择"在线下载"（填一个清单网址）或"本地文件"（网盘下好的压缩包/文件夹）
          * 看着进度条下载，随时可以取消（下次会断点续传）
        装完会自动配置好，不用手工改任何东西。

        如果窗口里不知道该填什么网址，去找发布者要"运行时清单地址"。

    第 3 步：装声库（音色）
        把 xxx.voicepack.zip **解压到程序的 voices\\ 目录下**（会得到一个文件夹）。
        重开程序，工具栏右侧的「声库」下拉框里就能选到。
        程序会自动识别 voices\\ 下的所有声库，不需要注册。


三、日常使用
    1. 点「框选区域」，在屏幕上拖出要读的区域（可以框多个）
    2. 点「开始识别」
    3. 识别结果按句出现在右侧列表，点每句左边的按钮播放/停止
    4. 想不挡游戏画面就点「显示悬浮条」
    5. 「显示区域角标」可以在屏幕上常驻显示各区域边界（点击穿透，不挡操作）
    6. 工具栏右侧的「声库」下拉框可以随时换音色


四、给直播用
    1. 装一个虚拟声卡（如 VB-CABLE）
    2. 用记事本打开 data\\config.json，把 "device" 改成 "CABLE Input"
    3. 在 OBS 里把该虚拟声卡加为音频输入源


五、出问题了
    * 用 run_console.bat 启动，能看到详细日志
    * 日志文件：data\\screenreader.log
    * 启动失败会写：data\\crash.log

    这个程序自带命令行（**不需要装 Python**），在程序目录打开 cmd：

        读屏台.exe --runtime status      看运行时装好没有
        读屏台.exe --runtime verify      检查运行时有没有坏
        读屏台.exe --runtime install --manifest <清单网址>
        读屏台.exe --runtime install --from <解压好的运行时目录>
        读屏台.exe --voice list          看有哪些声库
        读屏台.exe --voice install <xxx.voicepack.zip>

    （这些命令的输出也会写进 data\\cli_output.txt）

    常见情况：
    - 只能识别文字但没声音  -> 语音运行时没装好，跑 --runtime status 看
    - 声库下拉框是空的      -> voices\\ 目录下没有声库，解压一个 voicepack 进去
    - 播放没声音但日志正常  -> 检查 data\\config.json 里的 audio.device
"""


def make_launchers(tree: Path) -> None:
    (tree / "start.bat").write_text(_ascii_bat(START_BAT, "start.bat"), encoding="ascii")
    (tree / "run_console.bat").write_text(
        _ascii_bat(RUN_CONSOLE_BAT, "run_console.bat"), encoding="ascii"
    )
    (tree / "README.txt").write_text(README_TXT, encoding="utf-8")


def verify_tree(tree: Path, say=print) -> bool:
    """组装完必须自检，否则这类问题会直接送到用户手上。

    检查项都是实际踩过的：
      * .bat 里有 `?`（中文被 ascii 替换掉的痕迹）
      * .bat 里硬编码了中文文件名
      * 安装目录里到底有没有 exe
      * 关键目录（data / voices / gsv）是否建了
      * **exe 有没有请求管理员权限**——请求了就会弹 UAC 的「未知发布者」
    """
    say("  自检：")
    ok = True

    exes = list(tree.glob("*.exe"))
    if not exes:
        say("    [FAIL] 安装目录里没有任何 .exe")
        ok = False
    else:
        exe = exes[0]
        say(f"    [OK] 主程序 {exe.name}  {human(exe.stat().st_size)}")
        # 检查嵌入的 manifest 有没有请求提权。请求了的话，每次启动都会弹
        # UAC 的「未知发布者」——而这完全没必要，我们的程序不需要管理员。
        try:
            raw = exe.read_bytes()
            if b"requireAdministrator" in raw:
                say("    [FAIL] 主程序嵌入了 requireAdministrator —— 每次启动都会弹 UAC")
                ok = False
            elif b"asInvoker" in raw:
                say("    [OK] 主程序是 asInvoker（不请求提权，不会弹 UAC）")
            else:
                say("    [??] 没在 exe 里找到 manifest 的执行级别声明")
        except Exception as e:
            say(f"    [!!] 检查 exe manifest 失败：{e}")

    for name in ("start.bat", "run_console.bat"):
        f = tree / name
        if not f.exists():
            say(f"    [FAIL] 缺少 {name}")
            ok = False
            continue
        try:
            txt = f.read_text(encoding="ascii")
        except UnicodeDecodeError as e:
            say(f"    [FAIL] {name} 不是纯 ASCII：{e}")
            ok = False
            continue

        # 只在**代码行**里找可疑的 '?'（注释里引用报错信息是合理的）
        code_lines = [
            l for l in txt.splitlines()
            if l.strip() and not l.strip().upper().startswith(("REM", "::"))
        ]
        suspects = [l.strip() for l in code_lines if "?" in l]
        if suspects:
            say(f"    [FAIL] {name} 的代码行里有 '?'（多半是中文被 ascii 替换了）：")
            for s in suspects[:3]:
                say(f"           {s}")
            ok = False
        else:
            say(f"    [OK] {name} 纯 ASCII，代码行无替换痕迹")

        # 更直接的检查：bat 里写死的 *.exe 必须真实存在
        import re as _re

        for m in _re.finditer(r'"([^"%]*\.exe)"', txt):
            ref = m.group(1)
            if "?" in ref:
                say(f"    [FAIL] {name} 引用了被替换成 '?' 的文件名：{ref}")
                ok = False
            else:
                cand = tree / Path(ref.replace("\\", os.sep)).name
                if not cand.exists():
                    say(f"    [FAIL] {name} 引用了不存在的 {ref}")
                    ok = False
        if name == "start.bat":
            if "%%~ff" in txt:
                say("    [OK] start.bat 用通配符定位 exe（不依赖文件名）")
            else:
                say("    [FAIL] start.bat 没有用通配符找 exe，可能硬编码了文件名")
                ok = False

    for d in ("data", "voices", "gsv"):
        if not (tree / d).is_dir():
            say(f"    [FAIL] 缺少目录 {d}\\")
            ok = False
    if (tree / "data" / "config.json").exists():
        say("    [OK] data\\config.json 已生成")
    else:
        say("    [FAIL] 缺少 data\\config.json")
        ok = False

    say("    " + ("全部通过 ✅" if ok else "存在问题 ❌"))
    return ok


def make_config_template(tree: Path) -> None:
    from screenreader.config import AppConfig

    d = tree / "data"
    d.mkdir(parents=True, exist_ok=True)
    cfg = AppConfig()
    cfg.tts.voice = ""
    cfg.tts.ref_audio = ""
    cfg.tts.prompt_text = ""
    cfg.tts.runtime_root = ""
    data = cfg.to_dict()
    (d / "config.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (tree / "voices").mkdir(exist_ok=True)
    (tree / "gsv").mkdir(exist_ok=True)
    (tree / "voices" / "把声库包解压到这里.txt").write_text(
        "把 xxx.voicepack.zip 解压到本目录下（会得到一个声库文件夹）。\n\n"
        "也可以在本目录打开 cmd 执行：\n"
        "    读屏台.exe --voice install xxx.voicepack.zip\n\n"
        "重开程序后，工具栏右侧的「声库」下拉框里就能选到。\n",
        encoding="utf-8",
    )
    (tree / "gsv" / "语音运行时会自动装到这里.txt").write_text(
        "语音（TTS）推理运行时约 9 GB，因为体积大、平时又用不到，没有放进安装包。\n\n"
        "直接双击 读屏台.exe 或 start.bat，程序发现没装会**自动弹出安装窗口**，\n"
        "在里面选择安装位置、在线下载或用本地文件即可。\n\n"
        "也可以在本目录打开 cmd 手动装（不需要装 Python）：\n"
        "    读屏台.exe --runtime install --manifest <清单网址>\n"
        "    读屏台.exe --runtime install --from <解压好的运行时目录>\n"
        "    读屏台.exe --runtime status\n",
        encoding="utf-8",
    )


def run_pyinstaller(say=print) -> bool:
    say("=" * 78)
    say("[1/4] PyInstaller 打包主程序")
    say("=" * 78)
    cmd = [sys.executable, "-m", "PyInstaller", str(ROOT / "packaging" / "screenreader.spec"),
           "--noconfirm", "--distpath", str(DIST), "--workpath", str(BUILD / "pyi")]
    say("  $ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        say(f"  !! PyInstaller 失败（退出码 {r.returncode}）")
        return False
    app = DIST / "screenreader"
    if not app.exists():
        say(f"  !! 没找到产物 {app}")
        return False
    say(f"  产物 {app}  {human(dir_size(app))}")
    return True


def assemble(say=print) -> Path:
    say("\n" + "=" * 78)
    say("[2/4] 组装安装目录")
    say("=" * 78)
    if OUT.exists():
        shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)

    src = DIST / "screenreader"
    for item in src.iterdir():
        tgt = OUT / item.name
        if item.is_dir():
            shutil.copytree(item, tgt)
        else:
            shutil.copy2(item, tgt)
    say(f"  主程序 -> {OUT}  {human(dir_size(OUT))}")

    say("  生成启动脚本与配置模板")
    make_launchers(OUT)
    make_config_template(OUT)
    say(f"  合计 {human(dir_size(OUT))}")
    return OUT


def make_zip(tree: Path, out_dir: Path, say=print) -> Path:
    """打成 ZIP 分发（**完全没有 UAC 提示**）。

    为什么要有这条路：安装器（哪怕设成按用户安装）在部分环境下仍会触发
    SmartScreen「未知发布者」。ZIP 解压即用，连安装步骤都省了。

    注意：从网上下载的 ZIP 解出来的 exe 仍然带 Mark-of-the-Web，
    首次运行可能被 SmartScreen 拦一次 —— 右键属性里勾「解除锁定」即可，
    或者由发布者做代码签名（见文档）。
    """
    import zipfile

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"AutoScreenReader-{_version()}-portable.zip"
    files = sorted(f for f in tree.rglob("*") if f.is_file())
    say(f"\n打包 ZIP：{out}  （{len(files)} 个文件）")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in files:
            z.write(f, str(f.relative_to(tree)))
    say(f"  {out}  {human(out.stat().st_size)}")

    # 校验：**必须能打开、条目数对得上、CRC 全过**。
    # 实测踩过 —— 进程中途被终止，zip 只写了 146KB（第一个文件写了一半），
    # 但脚本照样打出了"打包 ZIP：…（283 个文件）"，看着像成功。
    # 这种半截包发给用户就是"解压失败"。所以这里强制自检。
    try:
        with zipfile.ZipFile(out) as z:
            got = len(z.infolist())
            bad = z.testzip()
        if got != len(files):
            say(f"  !! ZIP 条目数不对：{got} != {len(files)}")
        elif bad is not None:
            say(f"  !! ZIP 损坏于 {bad}")
        elif out.stat().st_size < 1_000_000:
            say(f"  !! ZIP 太小（{out.stat().st_size} 字节），像是没写完")
        else:
            say(f"  [OK] ZIP 自检通过：{got} 个条目，CRC 全部有效")
    except Exception as e:
        say(f"  !! ZIP 打不开（{e}）—— 多半是写到一半被中断，重跑一次")
    return out


def _version() -> str:
    m = re.search(r'MyAppVersion "([^"]+)"', (ROOT / "packaging" / "installer.iss").read_text(encoding="utf-8"))
    return m.group(1) if m else "0.0.0"


def run_inno(say=print) -> bool:
    say("\n" + "=" * 78)
    say("[4/4] 编译 Inno Setup 安装器")
    say("=" * 78)
    iscc = None
    for base in (r"C:\Program Files (x86)\Inno Setup 6", r"C:\Program Files\Inno Setup 6",
                 r"C:\Program Files (x86)\Inno Setup 5", r"C:\Program Files\Inno Setup 5"):
        p = Path(base) / "ISCC.exe"
        if p.exists():
            iscc = p
            break
    if iscc is None:
        say("  !! 没找到 Inno Setup 的 ISCC.exe")
        say("     装一个：https://jrsoftware.org/isdl.php")
        say(f"     或者手工把 {ROOT / 'packaging' / 'installer.iss'} 拖进 Inno Setup 编译")
        return False
    cmd = [str(iscc), str(ROOT / "packaging" / "installer.iss")]
    say("  $ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        say(f"  !! 编译失败（退出码 {r.returncode}）")
        return False
    outs = sorted((ROOT / "build" / "installer").glob("*.exe")) if (ROOT / "build" / "installer").exists() else []
    for o in outs:
        say(f"  安装器 {o}  {human(o.stat().st_size)}")
    return True


def preflight(say=print) -> bool:
    """打包前的静态防线。

    教训：`runtime.py` 用了 `subprocess.STDOUT` 却从没 `import subprocess`。
    开发机上一直没暴露 —— 因为 TTS 服务都是手动起的，那条**自动启动**的代码
    从没执行过。直到全新机器上才炸，日志里只有一句
    `启动失败：name 'subprocess' is not defined`，
    用户看到的现象却是「声库没装上 / 整段模式不出声」，很容易查错方向。

    所以打包前先扫一遍：凡是"用了但没导入"的模块名，一律拦下。
    """
    say("=" * 78)
    say("[0/4] 打包前静态检查")
    say("=" * 78)
    checker = ROOT / "tools" / "check_imports.py"
    if not checker.exists():
        say("  !! 缺少 tools\\check_imports.py，跳过")
        return True
    r = subprocess.run([sys.executable, str(checker)], capture_output=True,
                       text=True, errors="replace")
    for line in (r.stdout or "").splitlines():
        if line.strip():
            say("  " + line)
    if r.returncode != 0:
        say("  !! 静态检查没过，**不打包**")
        return False
    say("  [OK] 没有「用了但没导入」的模块名")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="打包主程序与安装器")
    ap.add_argument("--skip-pyinstaller", action="store_true", help="复用已有的 dist\\screenreader")
    ap.add_argument("--inno", action="store_true", help="顺便编译 Inno Setup 安装器")
    ap.add_argument("--zip", action="store_true",
                    help="额外打一个免安装 ZIP（不触发 UAC / 不用装）")
    ap.add_argument("--skip-preflight", action="store_true", help="跳过静态检查")
    args = ap.parse_args()

    print("=" * 78)
    print("打包读屏台（主程序；TTS 运行时按需下载，不打包）")
    print("=" * 78)

    if not args.skip_preflight and not preflight():
        return 1

    if not args.skip_pyinstaller:
        if not run_pyinstaller():
            return 1
    else:
        print("\n[1/4] 跳过 PyInstaller")

    tree = assemble()
    ok = verify_tree(tree)

    print("\n" + "=" * 78)
    print("[3/4] 结果")
    print("=" * 78)
    print(f"  安装目录 {tree}")
    print(f"  大小     {human(dir_size(tree))}")
    print()
    print("  目录结构：")
    for p in sorted(tree.iterdir()):
        if p.is_dir():
            print(f"    {p.name + '/':<22} {human(dir_size(p))}")
        else:
            print(f"    {p.name:<22} {human(p.stat().st_size)}")
    print()
    print("  把这个目录整体打包发给别人，或者加 --inno 生成安装器。")
    print("  注意：gsv\\ 是空的——TTS 运行时由首次启动按需安装。")

    if not ok:
        print("\n  !! 自检没通过，先修上面标 FAIL 的项。")
        return 1

    if args.zip:
        make_zip(tree, ROOT / "build" / "installer")

    if args.inno:
        run_inno()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
