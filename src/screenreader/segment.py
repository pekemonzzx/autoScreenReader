"""分句。

方案 §3.5：把 OCR 出来的一整块文字切成**短句**，让 TTS 逐句合成、第一句一好就播。
这是纯 CPU / 游戏满载场景下把"感知延迟"从"全文合成时间"降到"首句合成时间"的关键。

背景（实测）：TTS 非流式整段 p50 1786ms，流式首包 667ms。
句子切得短，首句出声就早；但切太碎会让 TTS 反复做文本前端，反而更慢，
所以要合并过短的碎片。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

DEFAULT_SPLIT = "。！？!?；;…"
DEFAULT_SOFT = "，,、"


@dataclass
class SegmentOptions:
    split_chars: str = DEFAULT_SPLIT
    soft_split_chars: str = DEFAULT_SOFT
    min_chars: int = 4
    max_chars: int = 28


def _split_keep(text: str, chars: str) -> List[str]:
    """按 chars 中的任意字符切分，并把分隔符保留在前一段末尾。"""
    if not chars:
        return [text]
    pattern = "([" + re.escape(chars) + "]+)"
    parts = re.split(pattern, text)
    out: List[str] = []
    i = 0
    while i < len(parts):
        piece = parts[i]
        if i + 1 < len(parts) and re.fullmatch("[" + re.escape(chars) + "]+", parts[i + 1]):
            piece = piece + parts[i + 1]
            i += 2
        else:
            i += 1
        if piece:
            out.append(piece)
    return out


def _hard_split(text: str, max_chars: int) -> List[str]:
    """没有任何标点的长串，按长度硬切。"""
    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def normalize_text(text: str) -> str:
    """清洗 OCR 常见噪声：多余空白、全角空格、零宽字符。"""
    if not text:
        return ""
    text = text.replace("\u200b", "").replace("\ufeff", "").replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# 送进 TTS 之前的清洗
# --------------------------------------------------------------------------- #
# 带圈数字、罗马数字符号、项目符号这类"排版装饰"，念出来只会变成怪音
_DECOR = re.compile(
    "[\u2460-\u2473\u24ea\u24b6-\u24e9\u2776-\u277f\u2160-\u217f"
    "\u2022\u25cf\u25cb\u25aa\u25ab\u25a0\u25a1\u2043\u2219\u00b7\u30fb"
    "\u2190-\u21ff\u2600-\u26ff\u2b00-\u2bff]"
)
# 行首的列表编号：1 / 1. / 2) / 3、 / (4) / 【5】  —— 分隔符可省略（OCR 常把"1语音"粘在一起）
# 用 \d+ 而不是 \d{1,3}：不然 "2024年" 会被切成 "202" + "4年" 而误删。
_LEADING_NUM = re.compile(r"^\s*[\(\[【（]?\s*(\d+)\s*[\.\)、:：\]】）]?\s*")

# 数字后面接这些量词/单位时，数字是内容的一部分，不能删（"3分钟" 不是列表编号）
_UNIT_AFTER_NUMBER = (
    "分钟", "小时", "秒钟", "秒", "个", "次", "年", "月", "日", "号", "元", "岁",
    "级", "层", "人", "条", "块", "万", "亿", "千", "百", "点", "时", "天", "周",
    "米", "公里", "斤", "公斤", "度", "倍", "件", "张", "页", "行", "字", "名",
    "位", "只", "台", "部", "首", "集", "话", "楼", "区", "章", "节", "款", "版",
)


def _strip_leading_number(m) -> str:
    """行首数字是列表编号就删掉；是数量（"3分钟"）就保留。"""
    rest = m.string[m.end():]
    if any(rest.startswith(u) for u in _UNIT_AFTER_NUMBER):
        return m.group(0)
    return ""


def clean_for_tts(text: str, strip_leading_numbers: bool = False) -> str:
    """把"屏幕上看起来正常、念出来很怪"的文字整理成适合 TTS 的形式。

    **行首编号的过滤现在默认关闭**（用户要求）。

    当初加它是因为：用户框选聊天窗口时 OCR 出来的行首带着列表编号
    （`1语音输出前有奇怪的东西`），GPT-SoVITS 会把那个孤立的 `1` 念成一个
    单独的音。但那个问题的**真正原因**是当时示例音频/参考文本配串了 ——
    修好之后这个过滤就成了副作用：正文里本来就该念出来的数字被吃掉。

    现在改成可选（默认关），只有确实需要处理列表编号的场景才打开。

    规则：
      * 去掉带圈数字/项目符号等排版装饰
      * 行首列表编号：**默认保留**，`strip_leading_numbers=True` 时才去掉
      * 去掉行首残留的标点
    """
    if not text:
        return ""
    t = normalize_text(text)
    t = _DECOR.sub("", t)
    if strip_leading_numbers:
        t = _LEADING_NUM.sub(_strip_leading_number, t)
    # 行首可能还剩标点（含 > * # - 这类列表/引用标记）
    t = re.sub(r"^[\s\.\)、,，:：;；\]】）\-–—~>》»|*#=+]+", "", t)
    t = re.sub(r"\s+$", "", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def has_speakable(text: str) -> bool:
    """是否值得送去合成：至少要有 2 个汉字或字母，纯符号/纯数字就不念了。"""
    if not text:
        return False
    body = re.sub(r"[^\w\u4e00-\u9fff]", "", text)
    if len(body) < 2:
        return False
    # 纯数字（"12345"）跳过，念出来没意义还容易报错
    if body.isdigit():
        return False
    return True


# --------------------------------------------------------------------------- #
# 整段模式：把 OCR 的多行还原成"真正的句子"
# --------------------------------------------------------------------------- #
# 句末标点：出现这些才算一句结束
SENT_END = "。！？；!?;…"
# 中文省略号常被 OCR 成多个点，也算句末
SENT_END_TAIL = "。！？；!?;…."
# 句末标点后面允许跟的收尾符号（引号/括号），它们不算新句子
CLOSERS = "」』】》〉\"'”’）)】]｝}"


def _is_sentence_end(text: str, i: int) -> bool:
    """text[i] 是不是一个句子边界。

    中文标点直接算。英文句号要小心：
      "This is a test."   -> 是（后面是空白或结尾）
      "3.5"               -> 不是（后面是数字）
      "www.a.com"         -> 不是（后面是字母）
    """
    ch = text[i]
    if ch == ".":
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if nxt and (nxt.isdigit() or nxt.isalpha()):
            return False
        # 前一个字符是数字也要小心（"1." 后面没东西的情况很少见，保守放过）
        return True
    return ch in SENT_END


def split_sentences(text: str) -> List[str]:
    """按句末标点切句，标点**和紧跟的收尾引号/括号**都跟着前一句走。

    连续的句末标点（"！！！" "……"）算一个边界，不切成一堆空串。
    收尾符号（」』）" 等）如果另起一句，会被当成没内容的碎片丢掉，
    所以这里一并吃进来。
    """
    if not text:
        return []
    out: List[str] = []
    buf = ""
    i, n = 0, len(text)
    while i < n:
        buf += text[i]
        if _is_sentence_end(text, i):
            while i + 1 < n and (text[i + 1] in SENT_END or
                                 (text[i + 1] == "." and _is_sentence_end(text, i + 1))):
                i += 1
                buf += text[i]
            while i + 1 < n and text[i + 1] in CLOSERS:
                i += 1
                buf += text[i]
            out.append(buf)
            buf = ""
        i += 1
    if buf.strip():
        out.append(buf)
    return [s.strip() for s in out if s.strip()]


def _ends_sentence(line: str) -> bool:
    """这一行是不是以句末标点结尾（允许后面跟引号/括号）。"""
    s = line.rstrip()
    while s and s[-1] in CLOSERS:
        s = s[:-1]
    return bool(s) and s[-1] in SENT_END_TAIL


def _ends_hard(line: str) -> bool:
    """视觉上的硬换行：以冒号/破折号结尾，或者短到不像被折行。"""
    s = line.rstrip()
    return bool(s) and s[-1] in "：:—-–"


def order_lines(lines: List[str], spans: Optional[List] = None) -> List[str]:
    """按**图上的位置**从上到下、从左到右排序。

    用户明确要求：先按位置排好序，再考虑合并。
    不排的话，合并出来的顺序就是 OCR 返回的顺序 —— 多栏排版、浮动元素、
    检测框交错时那个顺序不一定是阅读顺序。

    spans 是 [(top, left), ...]，与 lines 一一对应；不给就原样返回
    （OCR 层其实已经排过一遍，这里是第二道保险，也方便单独测试）。
    """
    if not spans or len(spans) != len(lines):
        return list(lines)

    def key(i: int):
        s = spans[i]
        try:
            return (float(s[0]), float(s[1]))
        except Exception:
            return (0.0, 0.0)

    return [lines[i] for i in sorted(range(len(lines)), key=key)]


def split_by_priority(text: str, max_chars: int = 80) -> List[str]:
    """按**优先级**断句：先句号，再逗号，**绝不在句子中间切**。

    用户的要求：
      * 最优先按句号（。！？；…）切
      * 其次按逗号（，、）切
      * **不要在句子中间切分**

    所以超过 max_chars 的处理只有两级：句末标点 -> 逗号。逗号也切不动
    （整段一个逗号都没有）就**保持整句不切** —— 宁可让 TTS 念长一点，
    也不要念到一半断掉，那样听起来是硬伤。
    """
    if not text:
        return []

    out: List[str] = []
    for sent in (split_sentences(text) or [text]):
        if len(sent) <= max_chars:
            out.append(sent)
            continue
        # 太长 -> 退而按逗号/顿号切
        buf = ""
        for ch in sent:
            buf += ch
            if len(buf) >= max_chars and ch in "，,、":
                out.append(buf)
                buf = ""
        if buf.strip():
            # 剩下的可能是没逗号的长句 —— **不硬切**，原样保留
            out.append(buf)
    return [s for s in out if s.strip()]


def merge_lines_to_sentences(lines: List[str], max_chars: int = 80,
                             min_chars: int = 2, spans: Optional[List] = None) -> List[str]:
    """把 OCR 的多个行还原成句子列表 —— 整段模式的核心。

    **处理顺序**（用户明确要求的顺序）：

      1. **按图上的位置排序**（从上到下、从左到右）—— 见 `order_lines`
      2. **合并折行**：行尾没有句末标点 -> 是自动折行，和下一行拼起来
      3. **按优先级断句**：先句号，再逗号，**绝不在句子中间切**

    **要解决的问题**：OCR 是按视觉行给出的，一段话被折成好几行时，
    每行都是一句"半截话"。逐行送去合成，听起来就是一句一顿、语义破碎。

    合并规则：
      * 行尾**没有**句末标点 → 自动折行，和下一行**合并**
      * 行尾**有**句末标点（。！？；…）→ 真正的句子边界
      * 行尾是逗号/顿号 → 也是同一句的中间，合并

    例：
        ["他那个写了很多规则，", "主要是为了让玩家不要乱想，", "不要去浪费时间。"]
        -> ["他那个写了很多规则，主要是为了让玩家不要乱想，不要去浪费时间。"]

    `min_chars` 默认 2：太短的碎片不念（"、"），但"在的"这种短句要保留。
    """
    if not lines:
        return []

    # 1) 先按位置排序 —— 顺序错了后面全错
    lines = order_lines(lines, spans)

    # 2) 按"行尾有没有句末标点"把行合并成句
    units: List[str] = []
    buf = ""
    for raw in lines:
        line = normalize_text(raw or "")
        if not line:
            continue
        # 中英交界处补空格（"GPT-SoVITS" + "做的" -> "GPT-SoVITS 做的"）
        if buf and _needs_space(buf[-1], line[0]):
            buf += " "
        buf += line
        if _ends_sentence(buf) or _ends_hard(buf):
            units.append(buf)
            buf = ""
    if buf.strip():
        units.append(buf)

    # 3) 按优先级断句：句号优先，逗号其次，不硬切
    final: List[str] = []
    for u in units:
        final.extend(split_by_priority(u, max_chars))

    # 4) 过滤掉不值得念的碎片
    return [s for s in final if len(s) >= min_chars and has_speakable(s)]


def _needs_space(prev: str, nxt: str) -> bool:
    """中英/数字交界处补空格：'SoVITS' + '做的' -> 'SoVITS 做的'。"""
    lat = lambda c: bool(re.match(r"[A-Za-z0-9]", c))
    cjk = lambda c: bool(re.match(r"[\u4e00-\u9fff]", c))
    return (lat(prev) and cjk(nxt)) or (cjk(prev) and lat(nxt))


def _split_long(text: str, max_chars: int) -> List[str]:
    """超长句子在逗号/顿号处再切，尽量不让单句过长。"""
    if len(text) <= max_chars:
        return [text]
    out, buf = [], ""
    for ch in text:
        buf += ch
        if len(buf) >= max_chars and ch in "，,、":
            out.append(buf)
            buf = ""
    if buf.strip():
        out.append(buf)
    # 还是太长（整段没逗号）就硬切
    res: List[str] = []
    for s in out:
        while len(s) > max_chars:
            res.append(s[:max_chars])
            s = s[max_chars:]
        if s.strip():
            res.append(s)
    return res


def paragraph_key(text: str) -> str:
    """整段内容的指纹：去标点空白后的小写串，用来判断"内容变了没有"。"""
    return re.sub(r"[^\w\u4e00-\u9fff]", "", text or "").lower()


def split_text(text: str, options: SegmentOptions = None) -> List[str]:
    """把一段文本切成适合逐句合成的短句列表。

    规则：
      1. 先按换行分段（屏幕上的不同行通常是不同语义单元）
      2. 段内按强标点切句
      3. 仍过长的句子用弱标点再切，还长就硬切
      4. 过短的碎片并入前一句（避免"嗯。""啊。"这种独立成句）
    """
    opt = options or SegmentOptions()
    text = normalize_text(text)
    if not text:
        return []

    result: List[str] = []

    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        pieces: List[str] = []
        for sent in _split_keep(line, opt.split_chars):
            if len(sent) <= opt.max_chars:
                pieces.append(sent)
                continue
            # 太长了，先用弱标点切
            subs = _split_keep(sent, opt.soft_split_chars)
            for s in subs:
                if len(s) <= opt.max_chars:
                    pieces.append(s)
                elif s.strip():
                    pieces.extend(_hard_split(s, opt.max_chars))

        # 合并碎片
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            # 纯标点的碎片直接丢弃
            if not re.search(r"[\w\u4e00-\u9fff]", piece):
                continue
            if (
                result
                and len(piece) < opt.min_chars
                and len(result[-1]) + len(piece) <= opt.max_chars + opt.min_chars
            ):
                result[-1] = result[-1] + piece
            else:
                result.append(piece)

    return [s for s in result if s]


def join_for_display(segments: Sequence[str]) -> str:
    return " / ".join(segments)


def iter_segments(text: str, options: SegmentOptions = None) -> Iterable[str]:
    yield from split_text(text, options)
