"""⑦ 试听 + 打包：合成样本自检 → 打包成可分发的 .voicepack.zip。

    python train\\07_pack_voice.py --name 我的声库

产物：

    voices\\_packs\\<名字>.voicepack.zip     ← 分发给别人用这个

别人拿到之后：

    读屏台.exe --voice install <名字>.voicepack.zip
    或者把 zip 解压到程序的 voices\\ 目录下，重开程序

---

## 声库包里有什么 / 没什么

**有**：模型权重（SoVITS + GPT）、`voice.json`（声库名/参考文本）、`ref.wav`

**没有**：`tts_infer.yaml` —— 它是**本机生成**的（里面写死了本机路径），
打进去没意义。安装时程序会自动重建。

> 所以如果你手动解压声库包，记得跑一次
> `读屏台.exe --voice refresh <名字>` 生成推理配置。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import ROOT, VENV_PY, info, ok, run, step, title, warn


def main() -> int:
    ap = argparse.ArgumentParser(description="试听 + 打包声库")
    ap.add_argument("--name", required=True, help="声库名")
    ap.add_argument("--skip-test", action="store_true", help="跳过试听合成")
    args = ap.parse_args()

    title(f"试听 + 打包「{args.name}」")
    env = {"PYTHONPATH": str(ROOT / "src")}
    name = args.name

    # ---- 1) 合成样本 ----
    if not args.skip_test:
        step("1/3", "合成几句样本做自检")
        info("  需要 TTS 服务在跑。没跑的话先开程序，或：")
        info(f'    读屏台.exe --voice use {name}')
        tester = ROOT / "tools" / "test_voice.py"
        if tester.exists():
            run([VENV_PY, "-u", tester, "--voice", name],
                env_extra=env, cwd=str(ROOT), check=False)
        else:
            warn(f"找不到 {tester}，跳过试听")
        print()
        info("  样本在 data\\voice_test\\")
        info("  **自己听一遍** —— 重点听擦音 ch/zh/sh/c/z/s 和句尾是否完整")

    # ---- 2) 打包 ----
    step("2/3", "打包成 voicepack")
    cmd = [VENV_PY, "-u", "-m", "screenreader", "voice", "pack", name]
    if run(cmd, env_extra=env, cwd=str(ROOT), check=False) != 0:
        warn("打包失败")
        return 1

    # ---- 3) 核对 ----
    step("3/3", "核对产物")
    packdir = ROOT / "voices" / "_packs"
    cand = sorted(packdir.glob(f"{name}*.zip"))
    if not cand:
        warn(f"没找到声库包（{packdir}）")
        return 1
    pack = cand[-1]
    size = pack.stat().st_size / 1024 / 1024
    ok(f"{pack.name}  {size:.1f} MB")

    # 包内清单
    try:
        import zipfile
        with zipfile.ZipFile(pack) as z:
            names = z.namelist()
        info(f"包内 {len(names)} 个条目：")
        for nn in sorted(names):
            info(f"    {nn}")
        if any("tts_infer.yaml" in nn for nn in names):
            warn("包里带了 tts_infer.yaml —— 它写死了本机路径，不该分发！")
        else:
            ok("没带 tts_infer.yaml（正确，安装时会自动生成）")
    except Exception as e:
        warn(f"读包失败：{e}")

    print()
    title("完成")
    info("分发：把这个 zip 给别人，他们")
    info(f'    读屏台.exe --voice install {pack.name}')
    info(" 或解压到程序的 voices\\ 目录下")
    print()
    info("本机启用这个声库：")
    info(f'    读屏台.exe --voice use {name}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
