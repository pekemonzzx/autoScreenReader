"""冒烟测试 train/ 下面的流程脚本。

不真跑训练（太慢），只验证：
  * 每个脚本语法过、--help 能出
  * _common 的工具函数对
  * 00_check_source 能识别好坏素材（拿已知的 48kHz / 16kHz 语料验）
  * run.py 能串到 --until check 就停
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "train"
PY = ROOT / ".venv" / "Scripts" / "python.exe"

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'OK' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def run(args, timeout=180):
    return subprocess.run([str(PY)] + [str(a) for a in args],
                          capture_output=True, text=True, errors="replace",
                          cwd=str(ROOT), timeout=timeout)


def test_scripts_exist():
    print("\n[脚本齐全]")
    need = ["_common.py", "README.md", "00_check_source.py", "01_make_corpus.py",
            "02_filter_list.py", "03_annotate.py", "04_verify_list.py",
            "05_train_voice.py", "07_pack_voice.py", "run.py"]
    for n in need:
        check(f"{n}", (TRAIN / n).exists())


def test_help():
    print("\n[--help 都能出]")
    for n in ["00_check_source", "01_make_corpus", "02_filter_list", "03_annotate",
              "04_verify_list", "05_train_voice", "07_pack_voice", "run"]:
        r = run([TRAIN / f"{n}.py", "--help"], timeout=60)
        check(f"{n} --help", "usage:" in (r.stdout or ""),
              (r.stderr or "")[:60])


def test_common_helpers():
    print("\n[_common 工具]")
    sys.path.insert(0, str(TRAIN))
    import _common as C

    check("ROOT 指向项目根", C.ROOT.resolve() == ROOT.resolve(), str(C.ROOT))
    check("能找到整合包 Python", C.EMBED_PY.exists(), str(C.EMBED_PY))
    check("能找到 ffmpeg", Path(C.find_ffmpeg()).exists() or C.find_ffmpeg() == "ffmpeg",
          C.find_ffmpeg())
    check("human_mb 对", C.human_mb(1536) == "1.5KB", C.human_mb(1536))
    check("human_mb 大数", C.human_mb(150 * 1024 * 1024) == "150.0MB",
          C.human_mb(150 * 1024 * 1024))


def _sample():
    """找一份可用的源素材（**不在测试里写死具体文件名**）。

    以前这里硬编码 `data/BVxxxxxxxxxx_p1.f30280.m4a` —— 等于把训练语料的
    来源视频公开了，别人用自己的素材也跑不了。现在交给 `_common.sample_source()`
    按顺序找：环境变量 SCREENREADER_SAMPLE → data/sample_source.txt → data/ 下最新音频。
    """
    sys.path.insert(0, str(TRAIN))
    import _common as C

    return C.sample_source(ROOT)


def test_source_check():
    print("\n[素材体检：能区分好坏素材]")
    good = _sample()
    if good is None:
        check("跳过：没找到源素材（设置 SCREENREADER_SAMPLE，或往 data/ 放一个）", True)
        return

    # 坏素材**自己造**：把好素材降采样到 16kHz。
    # 不依赖仓库里有没有那份 16k 文件（它可能被清理掉），测试才自包含。
    import tempfile
    sys.path.insert(0, str(TRAIN))
    import _common as C

    tmpd = Path(tempfile.mkdtemp(prefix="sr_trainflow_"))
    bad = tmpd / "bad16k.wav"
    try:
        r = subprocess.run(
            [C.find_ffmpeg(), "-v", "error", "-y", "-t", "30", "-i", str(good),
             "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(bad)],
            capture_output=True, text=True, errors="replace")
        check("能造出 16kHz 坏素材", bad.exists(), (r.stderr or "")[:80])
        if bad.exists():
            r = run([TRAIN / "00_check_source.py", bad], timeout=180)
            out = (r.stdout or "") + (r.stderr or "")
            check("坏素材被判问题", r.returncode == 1, f"退出码 {r.returncode}")
            check("指出采样率太低", "太低" in out)
            check("指出高频缺失", "高频基本没了" in out)
    finally:
        import shutil
        shutil.rmtree(tmpd, ignore_errors=True)

    r = run([TRAIN / "00_check_source.py", good], timeout=180)
    out = (r.stdout or "") + (r.stderr or "")
    check("好素材通过", r.returncode == 0, f"退出码 {r.returncode}")
    check("判定高频健康", "高频健康" in out)
    check("大小显示正确（不是 B）", "MB" in out and "147.5MB" in out,
          [l for l in out.splitlines() if "大小" in l][:1])


def test_run_until_check():
    print("\n[run.py --until check]")
    good = _sample()
    if good is None:
        check("跳过：没找到源素材", True)
        return
    r = run([TRAIN / "run.py", "--input", good, "--work",
             str(ROOT / "_tmp_run_test"), "--name", "冒烟测试",
             "--until", "check"], timeout=180)
    out = (r.stdout or "") + (r.stderr or "")
    check("跑到 check 就停", "①" in out, f"退出码 {r.returncode}")
    check("没有继续造语料", "②" not in out)
    # 清理（万一建了目录）
    import shutil
    shutil.rmtree(ROOT / "_tmp_run_test", ignore_errors=True)


def main() -> int:
    for fn in (test_scripts_exist, test_help, test_common_helpers,
               test_source_check, test_run_until_check):
        try:
            fn()
        except Exception as e:
            check(f"{fn.__name__} 未抛异常", False, str(e))
    print("\n" + "=" * 60)
    print(f"通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
    for f in FAIL:
        print(f"  FAIL: {f}")
    print("=" * 60)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
