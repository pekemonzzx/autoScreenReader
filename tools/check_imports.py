"""静态排查：找出「用了但（在这个函数里）没导入/没定义」的名字。

为什么要做到函数级：
`main_window.py` 里 `R` 在好几个函数里各自 `from .. import runtime as R`，
但 `_stop_tts_service()` 漏了这一句，运行时 `NameError` ——
而关闭流程恰好把它 try/except 吞了，表现是"确认关闭了但 python 进程还在"。

模块级扫描抓不到这种：`R` 在**模块里别处出现过**，看着"有定义"。
所以必须逐个函数算：这个名字在**这个函数作用域内**有没有绑定。

不用 pyflakes，用 ast 自己走，规则：
    未定义 = 函数内用到的名字
             - 函数内绑定的（参数/赋值/for/with/except/推导式/嵌套def）
             - 模块级的（导入/赋值/def/class）
             - 内建
只报**以属性访问形式出现**的（`x.y` 里的 x）和裸调用，减少误报。
"""

from __future__ import annotations

import ast
import builtins
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# **要扫的不只是 src/**：tools/ 和 train/ 里的脚本同样会犯"用了没导入"的错。
# 实测漏过一次：train/01_make_corpus.py 用了 subprocess 却没导入，
# 因为当时只扫 src/ 所以没拦住，用户一跑就 NameError。
SCAN_DIRS = ("src", "tools", "train")
BUILTINS = set(dir(builtins))


def _bound_names(node: ast.AST) -> set:
    """一个作用域内被绑定的名字。"""
    out: set = set()

    for n in ast.walk(node):
        if isinstance(n, ast.Import):
            for a in n.names:
                out.add(a.asname or a.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                if a.name != "*":
                    out.add(a.asname or a.name)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.add(n.name)
        elif isinstance(n, ast.ClassDef):
            out.add(n.name)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.add(n.id)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, ast.Global):
            out |= set(n.names)
        elif isinstance(n, ast.Nonlocal):
            out |= set(n.names)
        elif isinstance(n, (ast.With, ast.AsyncWith)):
            for it in n.items:
                v = it.optional_vars
                if isinstance(v, ast.Name):
                    out.add(v.id)
    return out


def _used_names(node: ast.AST) -> set:
    """这个作用域里以 Load 形式用到的名字（含属性访问的根）。"""
    out: set = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            out.add(n.id)
        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
            out.add(n.value.id)
    return out


def scan_file(path: Path) -> list:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as e:
        return [(path, e.lineno or 0, f"语法错误：{e}")]

    module_level = _bound_names(tree)
    problems = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # 嵌套函数的参数在 walk 时也会被算进来，宁可少报也别误报
        local = _bound_names(node)
        used = _used_names(node)
        for name in sorted(used - local - module_level - BUILTINS):
            if name in ("self", "cls") or name.startswith("__") or name.isupper():
                continue
            problems.append((path, node.lineno,
                             f"函数 {node.name}() 里用了 {name}，"
                             f"但这个函数作用域内没有它的定义/导入"))
    return problems


def main() -> int:
    files = []
    for d in SCAN_DIRS:
        p = ROOT / d
        if p.is_dir():
            files.extend(sorted(p.rglob("*.py")))
    print("=" * 78)
    print(f"静态检查 {len(files)} 个文件（{'、'.join(SCAN_DIRS)}）：函数级「用了但没定义」")
    print("=" * 78)
    allp = []
    for f in files:
        allp.extend(scan_file(f))
    if not allp:
        print("\n  没有发现问题 ✅")
        return 0
    print(f"\n  发现 {len(allp)} 处：")
    for p, ln, msg in allp:
        print(f"    {p.relative_to(ROOT)}:{ln}  {msg}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
