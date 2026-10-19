"""`screenreader runtime` 子命令：TTS 推理运行时的打包 / 安装 / 配置 / 自检。

主程序只有 ~200MB，TTS 运行时 ~9GB。所以：
  * 安装包只带主程序；
  * 运行时首次启动按需下载，或者用离线包安装。

发布方（有完整整合包的机器）：
    python -m screenreader runtime pack --src build\\gsv-runtime2 --out D:\\build\\gsv
    # 上传 D:\\build\\gsv 到网盘，把 manifest.json 里的 base_url 改掉

使用方（目标机）：
    python -m screenreader runtime status
    python -m screenreader runtime install --manifest https://.../runtime_manifest.json
    python -m screenreader runtime install --from D:\\下载\\gsv
    python -m screenreader runtime configure
"""

from __future__ import annotations

import logging
from pathlib import Path

from . import runtime as R
from .config import _app_data_base

log = logging.getLogger(__name__)


def cmd_prefix() -> str:
    """给出当前环境下正确的命令前缀。

    打包后（目标机没有 Python）必须显示 `读屏台.exe --runtime ...`，
    不能显示 `python -m screenreader ...`——用户根本执行不了。
    """
    import sys

    if getattr(sys, "frozen", False):
        return f'"{Path(sys.executable).name}" --runtime'
    return "python -m screenreader runtime"


def cmd_status(args) -> int:
    st = R.status()
    print("=" * 78)
    print("TTS 推理运行时")
    print("=" * 78)
    print(f"  位置     {st['root']}")
    print(f"  已安装   {'是' if st['installed'] else '否'}")
    if st["installed"]:
        print(f"  版本     {st.get('version', '?')}")
        print(f"  大小     {R.human(st['size'])}")
        print(f"  安装时间 {st.get('installed_at', '?')}")
    else:
        if st.get("missing"):
            print(f"  缺失     {st['missing']}")
        c = cmd_prefix()
        print()
        print("  安装方式（三选一）：")
        print("    A. 直接开界面，程序会自动弹出安装窗口（推荐）")
        print(f"    B. 在线：{c} install --manifest <manifest.json 网址>")
        print(f"    C. 离线：{c} install --from <解压后的运行时目录>")
    print("=" * 78)
    return 0 if st["installed"] else 1


def cmd_pack(args) -> int:
    src = Path(args.src).resolve() if args.src else (Path(_app_data_base()) / "build" / "gsv-runtime2")
    if not src.exists():
        print(f"运行时目录不存在：{src}")
        print("先跑： python tools\\build_runtime.py --dest <目录> --voice <声库> --verify")
        return 2
    if args.single:
        R.pack_single(src, Path(args.out), say=print)
    else:
        R.pack(src, Path(args.out), base_url=args.base_url, part_mb=args.part_mb)
    return 0


def cmd_install(args) -> int:
    src = args.manifest or args.from_dir
    if not src:
        print("要指定来源，三选一：")
        print(f"  {cmd_prefix()} install --manifest <清单网址>        # 分卷（推荐）")
        print(f"  {cmd_prefix()} install --zip <单个压缩包直链>       # 一条直链")
        print(f"  {cmd_prefix()} install --from <本地目录或zip>      # 离线")
        return 2

    if args.from_dir:
        res = R.install_from_dir(Path(args.from_dir), move=args.move)
    else:
        # install() 会按来源自动区分：.json -> 分卷，其它 -> 单包
        res = R.install(src, keep_parts=args.keep_parts)

    print()
    if res.get("ok"):
        print("运行时安装完成 ✅")
        print("下一步：启动推理服务，或直接开界面（首次会自动检查）")
        return 0
    if res.get("cancelled"):
        print("已取消安装（已下载的部分保留着，下次断点续传）")
        return 1
    print("运行时安装失败 ❌")
    if res.get("missing"):
        print("  缺少文件：", res["missing"])
    if res.get("failed"):
        print("  失败：", res["failed"])
    return 1


def cmd_configure(args) -> int:
    res = R.configure()
    return 0 if res.get("ok") else 1


def cmd_doctor(args) -> int:
    """安装环境体检：跑一下就知道这台机器能不能装、该装哪。"""
    R.doctor()
    return 0


def cmd_verify(args) -> int:
    """轻量自检：文件齐不齐、users.pth 有没有指向自己。"""
    root = R.runtime_root()
    print("=" * 78)
    print(f"运行时自检：{root}")
    print("=" * 78)
    ok = True
    miss = R.check_files(root)
    if miss:
        ok = False
        for m in miss:
            print(f"  [FAIL] 缺少 {m}")
    else:
        print(f"  [OK] {len(R.REQUIRED)} 个关键文件齐全")

    pth = root / "runtime" / "Lib" / "site-packages" / "users.pth"
    if pth.exists():
        txt = pth.read_text(encoding="utf-8", errors="replace")
        bad = [l for l in txt.splitlines() if l.strip() and not l.strip().startswith(str(root))]
        if bad:
            ok = False
            print(f"  [FAIL] users.pth 里有 {len(bad)} 行指向别处：")
            for b in bad[:5]:
                print(f"         {b}")
        else:
            print("  [OK] users.pth 指向自己（没有隐形引用）")
    else:
        print("  [!!] 没有 users.pth")

    print(f"  [{'OK' if ok else 'FAIL'}] 大小 {R.human(R.dir_size(root))}")
    print("=" * 78)
    return 0 if ok else 1


def add_parser(sub) -> None:
    p = sub.add_parser("runtime", help="TTS 推理运行时（打包/安装/配置/自检）")
    rs = p.add_subparsers(dest="runtime_cmd", required=True)

    q = rs.add_parser("status", help="查看运行时是否已安装")
    q.set_defaults(func=cmd_status)

    q = rs.add_parser("pack", help="打包运行时（发布方用）")
    q.add_argument("--src", default=None, help="运行时目录（默认 build\\gsv-runtime2）")
    q.add_argument("--out", required=True, help="输出目录")
    q.add_argument("--base-url", default="", help="下载前缀，会写进 manifest（分卷模式）")
    q.add_argument("--part-mb", type=int, default=R.DEFAULT_PART_MB, help="分卷大小 MB")
    q.add_argument("--single", action="store_true",
                   help="压成单个 zip（适合传网盘拿直链；大文件容易传断）")
    q.set_defaults(func=cmd_pack)

    q = rs.add_parser("install", help="安装运行时（在线或离线）")
    q.add_argument("--manifest", default=None,
                   help="清单地址或单个压缩包直链（.json 走分卷，其它当单包）")
    q.add_argument("--zip", dest="manifest", default=None,
                   help="单个压缩包直链（与 --manifest 等价，写起来更直观）")
    q.add_argument("--from", dest="from_dir", default=None, help="离线：本地目录或 zip")
    q.add_argument("--move", action="store_true", help="离线安装时移动而不是复制")
    q.add_argument("--keep-parts", action="store_true", help="保留下载的分卷压缩包")
    q.set_defaults(func=cmd_install)

    q = rs.add_parser("configure", help="修隐形引用 + 把声库指向新位置")
    q.set_defaults(func=cmd_configure)

    q = rs.add_parser("verify", help="轻量自检")
    q.set_defaults(func=cmd_verify)

    q = rs.add_parser("doctor", help="安装环境体检（权限/令牌/候选位置，排障用）")
    q.set_defaults(func=cmd_doctor)
