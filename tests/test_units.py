"""单元测试（纯逻辑，不需要 GPU / TTS 服务 / 音频设备）。

运行：
    .\\.venv\\Scripts\\python tests\\test_units.py
"""

from __future__ import annotations

import io
import struct
import sys
import tempfile
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from screenreader.cache import AudioCache, make_key, wav_from_pcm, wav_info  # noqa: E402
from screenreader.config import AppConfig  # noqa: E402
from screenreader.segment import SegmentOptions, normalize_text, split_text  # noqa: E402
from screenreader.tts.gsv_api import parse_wav_header  # noqa: E402
from screenreader.watchdog import GpuInfo, decide_tier  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'OK' if cond else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))


def expect_eq(name, got, want):
    check(name, got == want, f"得到 {got!r}，期望 {want!r}")


# --------------------------------------------------------------------------- #
def test_paragraph_merge():
    """整段模式：折行的多行要合并回句子（用户明确要求）。"""
    from screenreader.segment import merge_lines_to_sentences as M, split_sentences as S

    print("\n[整段模式：折行合并 + 按标点分句]")

    # 核心场景：一段话被 OCR 折成 4 行，行尾没有句末标点
    got = M(["他那个写了很多规则，", "主要是为了让玩家不要乱想，",
             "不要去浪费时间，", "做一些不可能的假设之类的。"])
    check("折行合并成 1 句", len(got) == 1, str(got))
    check("合并后内容完整", got and "做一些不可能的假设" in got[0], got[0] if got else "")
    check("没有丢字", got and got[0].startswith("他那个写了很多规则"), got[0] if got else "")

    # 行尾本来就有标点 -> 是真正的句子边界，不能合并
    got = M(["今天直播到此结束。", "谢谢大家陪我玩剧本杀。"])
    check("行尾有标点则各成一句", len(got) == 2, str(got))

    # 一行里写了多句
    got = M(["你好。今天天气不错。我们开始吧。"])
    check("一行多句要切开", len(got) == 3, str(got))
    check("短句不被吞掉", got and got[0] == "你好。", str(got))

    # 逗号结尾的行也要合并（不是句子边界）
    got = M(["所以你已经有一年出生于朵朵，", "不工作了吗？", "主要是感觉每个人读剧本的时候，", "不能有任何干扰。"])
    check("逗号结尾合并、问号处断开", len(got) == 2, str(got))

    # 中英混排补空格
    got = M(["这个是 GPT-SoVITS", "做的语音克隆。", "效果还行。"])
    check("中英之间补空格", len(got) == 2 and "GPT-SoVITS 做的" in got[0], str(got))

    # 收尾引号跟着前一句
    got = S("他说：「这样做不对。应该那样。」")
    check("收尾引号不丢", len(got) == 2 and got[1].endswith("」"), str(got))

    # 英文句号：句子结尾算，小数点/域名不算
    got = S("This is a test. And another.")
    check("英文句号算句末", len(got) == 2, str(got))
    got = M(["价格是 3.5 元，", "网址 www.a.com 也行。"])
    check("小数点/域名不误切", len(got) == 1, str(got))

    # 超长但没有任何标点 -> **不切**（用户明确要求：不要切在句子中间）
    got = M(["啊" * 200])
    check("无标点的超长句保持完整（不硬切）",
          len(got) == 1 and got[0] == "啊" * 200, f"{len(got)} 句")

    # 有逗号才切
    got = M(["啊" * 100 + "，" + "啊" * 100 + "。"])
    check("有逗号才切", len(got) >= 2, f"{len(got)} 句")

    # 空白行忽略
    got = M(["", "第一句。", "  ", "第二句。", ""])
    check("空行被忽略", len(got) == 2, str(got))


def test_paragraph_ordering():
    """整段模式：卡片要按出现顺序朗读，不能按时间戳/字典序。"""
    from screenreader.board import BoardConfig, CardStore

    print("\n[整段模式：朗读顺序]")
    st = CardStore(BoardConfig())
    r1 = st.add_region("单行区", (0, 0, 10, 10))
    r2 = st.add_region("整段区", (0, 20, 10, 10))
    st.update_region(r2.id, mode="paragraph")

    # 故意让文本的字典序和出现顺序相反
    out = st.observe(r2.id, ["z 开头的句子。", "a 开头的句子。", "m 开头的句子。"])
    check("整段区建了 3 张卡", len(out.added) == 3)
    check("整段区的卡标记了自动朗读", all(c.auto_play for c in out.added))
    seqs = [c.seq for c in out.added]
    check("seq 单调递增", seqs == sorted(seqs) and len(set(seqs)) == 3, str(seqs))

    cards = [c for c in st.sorted_cards() if c.region_id == r2.id]
    texts = [c.text for c in cards]
    check("按出现顺序而非字典序", texts == ["z 开头的句子。", "a 开头的句子。", "m 开头的句子。"], str(texts))

    # 现在是**手动**排队：不点播放按钮就不会出队（滚动时自动播会念碎片）
    check("没点播放时队列为空", st.next_auto_card() is None)
    st.queue_region(r2.id)
    nxt = st.next_auto_card()
    check("点播放后从最先出现的那句开始",
          nxt is not None and nxt.text.startswith("z"), nxt.text if nxt else "None")

    # 单行区的卡不该被整段队列带走
    out2 = st.observe(r1.id, ["单行 A", "单行 B"])
    check("单行区的卡不自动朗读", not any(c.auto_play for c in out2.added))
    check("排队只作用于指定区域",
          all(c.region_id == r2.id for c in [nxt] if c), "ok")

    # 换模式时要能清干净（否则新旧句子混读）
    n = st.clear_region(r2.id)
    check("clear_region 清掉该区卡片", n == 3, str(n))
    check("清完后自动队列为空", st.next_auto_card() is None)
    check("别的区域不受影响", len([c for c in st.sorted_cards() if c.region_id == r1.id]) == 2)


def test_region_persist():
    """识别区域（含单行/整段模式）必须能存下来 —— 否则重启每次都要重设。"""
    import json
    import tempfile

    from screenreader.board import BoardConfig, CardStore

    print("\n[识别区域持久化]")
    cfg = AppConfig()
    st = CardStore(BoardConfig())
    st.add_region("聊天区", (0, 0, 300, 200))
    st.add_region("正文区", (0, 300, 600, 400), mode="paragraph")

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "c.json"
        # 必须显式传路径：不传就会写到真的 data/config.json，
        # 把声库/参考音频冲掉（这个坑真踩过）
        cfg.save_regions(st.regions, p)
        check("save_regions 只写指定文件", p.exists())

        # 只改 regions，不能碰别的键
        p.write_text(json.dumps({"tts": {"voice": "不该被动"}, "log_level": "DEBUG"},
                                ensure_ascii=False), encoding="utf-8")
        cfg.save_regions(st.regions, p)
        kept = json.loads(p.read_text(encoding="utf-8"))
        check("save_regions 不覆盖其它配置项",
              kept.get("tts", {}).get("voice") == "不该被动", str(kept.get("tts")))
        check("save_regions 保留了 log_level", kept.get("log_level") == "DEBUG")
        check("regions 写进去了", isinstance(kept.get("regions"), list))

        cfg.save(p)
        raw = json.loads(p.read_text(encoding="utf-8"))
        check("config.json 里有 regions", isinstance(raw.get("regions"), list))
        check("存了 mode 字段", "mode" in raw["regions"][1], str(raw["regions"][1]))

        back = AppConfig.load(p)
        items = back.load_regions()
        check("读回 2 个区域", len(items) == 2, str(len(items)))
        check("模式读回正确", [i["mode"] for i in items] == ["line", "paragraph"], str(items))
        check("矩形读回正确", tuple(items[1]["rect"]) == (0, 300, 600, 400), str(items[1]["rect"]))

        st2 = CardStore(BoardConfig())
        n = st2.restore_regions(items)
        check("能恢复到 store", n == 2, str(n))
        check("区域 id 保持不变（角标/卡片才对得上）",
              [r.id for r in st2.regions] == [r.id for r in st.regions])
        check("恢复后整段模式仍在", st2.regions[1].mode == "paragraph")

    # 老配置没有 mode -> 默认单行，保持原有行为
    old = AppConfig()
    old.regions = [{"id": "x1", "name": "旧区", "rect": [1, 2, 3, 4]}]
    got = old.load_regions()
    check("老配置默认单行", got and got[0]["mode"] == "line", str(got))

    # 脏数据要扛得住，不能因为配置坏了就打不开程序
    bad = AppConfig()
    bad.regions = [{"no": "rect"}, "garbage", {"rect": [1, 2, 3]},
                   {"rect": [5, 6, 7, 8], "mode": "bogus"}]
    got = bad.load_regions()
    check("脏数据被过滤", len(got) == 1, str(got))
    check("非法 mode 归为单行", got and got[0]["mode"] == "line", str(got))

    # 环境变量覆盖：测试必须靠这个才不会写到真配置里
    import os

    from screenreader.config import default_config_path

    old = os.environ.get("SCREENREADER_CONFIG")
    try:
        os.environ["SCREENREADER_CONFIG"] = r"C:\tmp\某个临时配置.json"
        check("SCREENREADER_CONFIG 能覆盖配置路径",
              str(default_config_path()) == r"C:\tmp\某个临时配置.json",
              str(default_config_path()))
    finally:
        if old is None:
            os.environ.pop("SCREENREADER_CONFIG", None)
        else:
            os.environ["SCREENREADER_CONFIG"] = old
    check("去掉环境变量后回到默认路径",
          "screenshot2Speach" in str(default_config_path()) or
          "AppData" in str(default_config_path()),
          str(default_config_path()))


def test_paragraph_settle():
    """滚动/翻页时内容一直在变 —— 绝不能每次都建卡合成。

    用户实测：快速滑动时会不停播报"刚出现的前几个字"，
    等页面停下来想完整读，卡片已经被这些碎片搅乱、没法顺序读。
    """
    import time

    from screenreader.board import BoardConfig, CardStore, Region
    from screenreader.worker import BoardEngine, EngineConfig

    print("\n[整段模式：内容稳定才建卡]")

    class Dummy:
        def __init__(self, settle=0.35):
            self.ecfg = EngineConfig(paragraph_settle_s=settle)
            self._para_state = {}
            self._playing_regions = set()
            self.events = []
            self.store = CardStore(BoardConfig())

        def _emit(self, kind, **kw):
            self.events.append(kind)

        def _drop_pending(self, rid):
            return 0

    d = Dummy()
    r = Region.create("t", (0, 0, 100, 100), mode="paragraph")
    settled = BoardEngine._paragraph_settled

    # 内容一直在变 -> 一次都不许建卡
    results = []
    for i in range(5):
        results.append(settled(d, r, [f"第 {i} 次滚动中的内容。"]))
        time.sleep(0.08)
    check("滚动中一次都不建卡", not any(results), str(results))
    check("每次都发了 settling 事件",
          d.events.count("paragraph_settling") == 5, str(d.events))

    # 停下来：立刻问还是不行
    check("刚停下还不行", settled(d, r, ["稳定下来的这一段。"]) is False)
    time.sleep(0.4)
    check("稳够时间后可以建卡", settled(d, r, ["稳定下来的这一段。"]) is True)
    check("建过之后不重复建", settled(d, r, ["稳定下来的这一段。"]) is False)
    check("发了 settled 事件", "paragraph_settled" in d.events)

    # 又变了 -> 重新计时
    check("内容再变时挡住", settled(d, r, ["又变了。"]) is False)
    time.sleep(0.4)
    check("再稳够可以再建", settled(d, r, ["又变了。"]) is True)

    # 空白内容不该建卡
    d2 = Dummy()
    check("空白内容不建卡", settled(d2, r, []) is False)


def test_paragraph_manual_queue():
    """整段朗读必须由用户手动触发，不能自动开播。"""
    from screenreader.board import BoardConfig, CardStore

    print("\n[整段模式：手动排队朗读]")
    st = CardStore(BoardConfig())
    r = st.add_region("整段区", (0, 0, 10, 10))
    st.update_region(r.id, mode="paragraph")

    out = st.observe(r.id, ["第一句。", "第二句。", "第三句。"])
    check("建了 3 张卡", len(out.added) == 3, str(len(out.added)))

    # 关键：建卡之后**不能**自动进播放队列
    check("新建的卡不会自动播放", st.next_auto_card() is None)

    n = st.queue_region(r.id)
    check("手动排队排入 3 句", n == 3, str(n))
    first = st.next_auto_card()
    check("队列从头开始", first is not None and first.text == "第一句。",
          first.text if first else "None")

    # 按顺序取
    order = []
    while True:
        c = st.next_auto_card()
        if c is None:
            break
        order.append(c.text)
        c.auto_done = True
        c.queued = False
    check("严格按顺序出队", order == ["第一句。", "第二句。", "第三句。"], str(order))

    # 取消队列
    st.queue_region(r.id)
    check("重新排队后又有内容", st.next_auto_card() is not None)
    m = st.unqueue_region(r.id)
    check("取消队列清掉 3 张", m == 3, str(m))
    check("取消后队列为空", st.next_auto_card() is None)

    # 单行模式的卡不该被整段队列带走
    r2 = st.add_region("单行区", (0, 20, 10, 10))
    st.observe(r2.id, ["单行 A", "单行 B"])
    n = st.queue_region(r.id)
    check("排队只针对指定区域", n == 3, str(n))


def test_paragraph_cleanup_and_append():
    """内容一变立刻清卡；播放中新句子自动接进队列。"""
    import time

    from screenreader.board import BoardConfig, CardStore, Region
    from screenreader.worker import BoardEngine, EngineConfig

    print("\n[整段模式：变化即清卡 / 播放中追加]")

    class Dummy:
        def __init__(self):
            self.ecfg = EngineConfig(paragraph_settle_s=0.05)
            self._para_state = {}
            self._playing_regions = set()
            self.events = []

        def _emit(self, kind, **kw):
            self.events.append((kind, kw))

        def _drop_pending(self, rid):
            return 0

    d = Dummy()
    d.store = CardStore(BoardConfig())
    r = d.store.add_region("t", (0, 0, 100, 100))
    d.store.update_region(r.id, mode="paragraph")

    settled = BoardEngine._paragraph_settled

    # 先建立一段内容
    check("首次可建卡", settled(d, r, ["第一段内容。"]) is False)  # 第一次是"变化"
    time.sleep(0.08)
    check("稳够后可建卡", settled(d, r, ["第一段内容。"]) is True)
    out = d.store.observe(r.id, ["第一段内容。"])
    check("建了卡", len(out.added) == 1, str(len(out.added)))

    # 内容一变 -> 立刻清卡
    settled(d, r, ["完全不同的新内容。"])
    left = len([c for c in d.store.sorted_cards() if c.region_id == r.id])
    check("内容一变旧卡立刻清掉", left == 0, f"还剩 {left} 张")
    check("发了 cleared 事件", any(k == "paragraph_cleared" for k, _ in d.events),
          str([k for k, _ in d.events]))

    # 播放中：新句子要自动接进队列
    time.sleep(0.08)
    settled(d, r, ["稳定内容 A。", "稳定内容 B。"])
    out = d.store.observe(r.id, ["稳定内容 A。", "稳定内容 B。"])
    d._playing_regions.add(r.id)
    d.store.queue_region(r.id)
    first = d.store.next_auto_card()
    check("队列开头是 A", first is not None and first.text == "稳定内容 A。",
          first.text if first else "None")

    # 模拟"又识别出第三句" —— 正在播的区域，新卡应自动排进队列
    out2 = d.store.observe(r.id, ["稳定内容 A。", "稳定内容 B。", "稳定内容 C。"])
    check("新增了 C", any(c.text == "稳定内容 C。" for c in out2.added))
    for c in out2.added:
        if r.id in d._playing_regions:
            c.queued = True
    c_new = [c for c in d.store.sorted_cards() if c.text == "稳定内容 C。"]
    check("播放中新句子自动进队列", c_new and c_new[0].queued, str(c_new[0].queued if c_new else None))

    # 播放中内容又变了 -> 停止该区域并清卡
    settled(d, r, ["又换了。"])
    check("变化时退出播放状态", r.id not in d._playing_regions)


def test_paragraph_punctuation_priority():
    """断句：先句号，再逗号，**绝不在句子中间切**（用户明确要求）。"""
    from screenreader.segment import merge_lines_to_sentences as M
    from screenreader.segment import order_lines, split_by_priority

    print("\n[整段模式：断句优先级 + 按位置排序]")

    # 按图上位置排序
    got = order_lines(["第三", "第一", "第二"], [(30, 0), (10, 0), (20, 0)])
    check("按 top 从上到下排", got == ["第一", "第二", "第三"], str(got))
    got = order_lines(["右", "左"], [(10, 200), (10, 50)])
    check("同一行按 left 从左到右", got == ["左", "右"], str(got))
    check("没给位置就原样返回", order_lines(["a", "b"]) == ["a", "b"])

    # 句号优先
    got = split_by_priority("第一句。第二句。", 80)
    check("句号处切开", got == ["第一句。", "第二句。"], str(got))

    # 超长 -> 退到逗号
    long_sent = "这是一句很长的话，中间有逗号，所以可以在这里切，但是整句还是有点长啊。"
    got = split_by_priority(long_sent, 20)
    check("超长时按逗号切", len(got) >= 2, str(got))
    check("每段都以逗号或句号收尾",
          all(s.endswith(("，", "。", "、")) for s in got), str(got))

    # **关键**：没逗号的长句绝不硬切
    no_comma = "这句话有三十多个字但是一个逗号都没有所以就绝对不能硬切因为它是一个完整的句子啊"
    got = split_by_priority(no_comma, 20)
    check("没逗号的长句不硬切", len(got) == 1 and got[0] == no_comma,
          f"{len(got)} 段")

    # 合并 + 排序 + 断句的组合
    got = M(["第二段文字到此结束。", "第一段文字，", "还没说完。"],
            spans=[(30, 0), (10, 0), (20, 0)])
    check("先排序再合并",
          got == ["第一段文字，还没说完。", "第二段文字到此结束。"], str(got))


def test_calm_score():
    """参考音频的「平和度」评分：平稳的要比激动的分高。"""
    import math
    import struct
    import tempfile
    import wave

    import numpy as np

    from screenreader.calm import calm_score, pitch_track

    print("\n[参考音频：叙述型/平和度评分]")
    sr = 32000

    def write(path, f0_track, amp_track, dur=4.0):
        """按给定的音高/振幅轨迹合成一个假的"说话"音频。"""
        t = np.arange(int(sr * dur)) / sr
        x = np.zeros_like(t)
        n = len(f0_track)
        for i in range(n):
            a, b = int(len(t) * i / n), int(len(t) * (i + 1) / n)
            if b <= a:
                continue
            seg = t[a:b]
            x[a:b] = amp_track[i] * np.sin(2 * np.pi * f0_track[i] * seg)
        x += np.random.RandomState(0).normal(0, 60, len(x))
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(np.clip(x, -32000, 32000).astype(np.int16).tobytes())

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        calm = d / "calm.wav"
        excited = d / "excited.wav"
        # 平稳：音高和音量都基本不动
        write(calm, [200] * 20, [6000] * 20)
        # 激动：音高上下跳、音量忽大忽小
        write(excited, [150 + 180 * (i % 2) for i in range(20)],
              [3000 + 9000 * (i % 2) for i in range(20)])

        s_calm, d_calm = calm_score(calm, "这种情况一般要等到下周才会有新的消息。")
        s_exc, d_exc = calm_score(excited, "太太好玩了！哈哈哈！")
        print(f"    平稳样本 {s_calm:.1f}  f0波动={d_calm.get('f0_cv')} "
              f"能量波动={d_calm.get('env_cv')}")
        print(f"    激动样本 {s_exc:.1f}  f0波动={d_exc.get('f0_cv')} "
              f"能量波动={d_exc.get('env_cv')}")
        check("平稳的得分更高", s_calm > s_exc, f"{s_calm:.1f} vs {s_exc:.1f}")
        # 注意不能用 `x or 9` 兜底：f0_cv 正常值可能就是 0.0，被 or 吃掉了
        cv_calm = d_calm.get("f0_cv")
        cv_exc = d_exc.get("f0_cv")
        check("识别出音高波动差异",
              cv_calm is not None and cv_exc is not None and cv_calm < cv_exc,
              f"{cv_calm} vs {cv_exc}")

        # 感叹号要扣分
        s_plain, _ = calm_score(calm, "这种情况一般要等到下周才会有新的消息。")
        s_bang, _ = calm_score(calm, "太好了！！！")
        check("感叹号扣分", s_bang < s_plain, f"{s_bang:.1f} vs {s_plain:.1f}")

        check("读不了的给极低分", calm_score(d / "不存在.wav", "x")[0] < -1e8)


def test_paragraph_clear_emits_died():
    """清卡必须逐张发 card_died —— 否则界面列表会卡住旧卡片。"""
    import time

    from screenreader.board import BoardConfig, CardStore, Region
    from screenreader.worker import BoardEngine, EngineConfig

    print("\n[整段模式：清卡要通知界面]")

    class Dummy:
        def __init__(self):
            self.ecfg = EngineConfig(paragraph_settle_s=0.05)
            self._para_state = {}
            self._playing_regions = set()
            self.store = CardStore(BoardConfig())
            self.events = []

        def _emit(self, kind, **kw):
            self.events.append((kind, kw))

        def _drop_pending(self, rid):
            return 0

    d = Dummy()
    r = d.store.add_region("t", (0, 0, 10, 10))
    settled = BoardEngine._paragraph_settled

    settled(d, r, ["第一段。"])
    time.sleep(0.08)
    settled(d, r, ["第一段。"])
    out = d.store.observe(r.id, ["第一段。", "第二段。"])
    d.events.clear()

    # 内容一变 -> 每张卡都要有 card_died
    settled(d, r, ["完全不同的内容。"])
    died = [kw for k, kw in d.events if k == "card_died"]
    check("每张卡都发了 card_died", len(died) == len(out.added),
          f"{len(died)} vs {len(out.added)}")
    check("card_died 带 region_id", all("region_id" in x for x in died))
    check("发了 paragraph_cleared",
          any(k == "paragraph_cleared" for k, _ in d.events))
    check("store 里确实清空了",
          not [c for c in d.store.sorted_cards() if c.region_id == r.id])


def test_volume():
    """播放音量：0-300，100 = 原始输出，>100 放大且不硬削波。"""
    import numpy as np

    from screenreader.audio import Player
    from screenreader.config import AppConfig

    print("\n[播放音量]")

    cfg = AppConfig()
    check("配置里有 volume", hasattr(cfg.audio, "volume"),
          str(getattr(cfg.audio, "volume", None)))
    check("默认是 100%（= 原始音量）", cfg.audio.volume == 100, str(cfg.audio.volume))

    p = Player(volume=100)
    a = np.array([1000, -2000, 30000, -32768], dtype=np.int16)

    out = p._apply_volume(a)  # noqa: SLF001
    check("100% 原样输出", np.array_equal(out, a), str(out.tolist()))

    p.volume = 50
    out = p._apply_volume(a)  # noqa: SLF001
    check("50% 减半", out.tolist() == [500, -1000, 15000, -16384], str(out.tolist()))
    check("输出仍是 int16", out.dtype == np.int16, str(out.dtype))

    p.volume = 0
    check("0% 静音", not p._apply_volume(a).any())  # noqa: SLF001

    # ---- 放大（用户反馈默认偏小）----
    sine = (np.sin(np.linspace(0, 40, 4000)) * 16000).astype(np.int16)
    base = float(np.sqrt((sine.astype(float) ** 2).mean()))

    def rms_at(v):
        pp = Player(volume=v)
        o = pp._apply_volume(sine)  # noqa: SLF001
        return float(np.sqrt((o.astype(float) ** 2).mean()))

    check("150% 比 100% 响", rms_at(150) > base * 1.3, f"{rms_at(150) / base:.2f}x")
    check("200% 约 2 倍", 1.7 < rms_at(200) / base < 2.2, f"{rms_at(200) / base:.2f}x")
    check("300% 比 200% 更响（软限幅后仍在升）",
          rms_at(300) >= rms_at(200), f"{rms_at(300) / base:.2f}x vs {rms_at(200) / base:.2f}x")

    # 软限幅：不能超出 int16 范围（硬削波会破音）
    for v in (200, 300):
        o = Player(volume=v)._apply_volume(a)  # noqa: SLF001
        check(f"{v}% 不越界", o.min() >= -32768 and o.max() <= 32767,
              f"[{o.min()}, {o.max()}]")

    # 越界截断（构造函数也要走校验，不能绕过）
    check("构造时 999 截断到 300", Player(volume=999).volume == 300,
          str(Player(volume=999).volume))
    p = Player(volume=100)
    p.volume = 999
    check("设置时 999 截断到 300", p.volume == 300, str(p.volume))
    p.volume = -50
    check("低于 0 截断到 0", p.volume == 0, str(p.volume))
    p.volume = 65
    check("播放中途能改", p.volume == 65, str(p.volume))
    p.volume = "不是数字"
    check("非法值回落 100", p.volume == 100, str(p.volume))


def test_playing_card_not_destroyed():
    """正在播放的卡片不能被销毁，否则音频会被掐断（"最后几个字没了"）。

    单行模式下卡片在文字离开屏幕约 1.2 秒后就销毁
    （destroy_after_misses=3 × 轮询 0.4s），而一句话要念 4~5 秒。
    卡片先死 -> 朗读循环提前认为播完了 -> 播下一句时 sd.stop() 掐断上一句。
    """
    from screenreader.board import BoardConfig, CardStore

    print("\n[播放中的卡片不被销毁]")
    st = CardStore(BoardConfig())
    r = st.add_region("单行区", (0, 0, 10, 10))
    out = st.observe(r.id, ["这一句有二十来个字所以念起来要四五秒钟。"])
    card = out.added[0]
    cid = card.id

    st.mark_playing(cid)
    check("标记为播放中", st.playing_id == cid)

    # 自动路径（force=False）：文字走了要销毁它
    removed = st._destroy(card)  # noqa: SLF001
    check("正在播的卡没被真删", removed is False, f"返回 {removed}")
    check("卡片还在", st.card(cid) is not None)
    check("打了 doomed 标记", getattr(card, "doomed", False) is True)
    check("playing_id 没被清（否则会被误判成播完）", st.playing_id == cid)

    # 播完 -> 这时才收
    st.mark_playing(None)
    check("播完后才真删", st.card(cid) is None)
    check("播放状态清空", st.playing_id is None)

    # 用户主动清空 -> 立即删，不等播放
    out2 = st.observe(r.id, ["第二句也够长足够念好几秒的。"])
    c2 = out2.added[0]
    st.mark_playing(c2.id)
    n = st.clear_region(r.id)
    check("clear_region 立即清掉（用户主动）", n == 1 and st.card(c2.id) is None, str(n))
    check("播放状态被清", st.playing_id is None)

    # clear() 同理
    out3 = st.observe(r.id, ["第三句。"])
    c3 = out3.added[0]
    st.mark_playing(c3.id)
    st.clear()
    check("clear() 立即清掉", st.card(c3.id) is None)

    # 超限淘汰也不能动正在播的
    st2 = CardStore(BoardConfig())
    r2 = st2.add_region("t", (0, 0, 10, 10))
    st2.cfg.max_cards = 2
    a = st2.observe(r2.id, ["A 句。"]).added[0]
    st2.mark_playing(a.id)
    for i in range(5):
        st2.observe(r2.id, [f"第{i}句够长的一句话用来把容量撑满。"])
    check("淘汰时跳过了正在播的卡", st2.card(a.id) is not None)


def test_frame_gate():
    """画面没变就跳过 OCR：判据要准、开销要低。

    这是省 CPU 的核心优化 —— OCR 单次几百毫秒，而签名只要 0.03ms。
    """
    import numpy as np

    from screenreader.worker import BoardEngine, EngineConfig

    print("\n[帧变化判据（省 CPU 的核心）]")

    check("默认开启", EngineConfig().skip_unchanged_frame is True)

    def sig(x):
        return BoardEngine._frame_signature(x)

    def fresh():
        e = BoardEngine.__new__(BoardEngine)
        e._frame_sig = {}
        return e

    def frame(n=1, level=255):
        f = np.zeros((200, 300, 3), dtype=np.uint8)
        for i in range(n):
            f[50 + i * 50: 60 + i * 50, 20:200] = level
        return f

    e = fresh()
    check("没有基线时算变了", e._frame_unchanged("r", sig(frame())) is False)
    check("同一帧算没变", e._frame_unchanged("r", sig(frame())) is True)

    e = fresh()
    e._frame_unchanged("r", sig(frame(1, 255)))
    check("亮度微动被忽略（255->250）",
          e._frame_unchanged("r", sig(frame(1, 250))) is True)

    e = fresh()
    e._frame_unchanged("r", sig(frame(1)))
    check("多出一整行算变了", e._frame_unchanged("r", sig(frame(2))) is False)

    e = fresh()
    e._frame_unchanged("r", sig(frame(1)))
    f = frame(1).copy()
    f[50:52, 20:30] = 0        # 光标闪烁级别的小变化
    check("极小变化被忽略", e._frame_unchanged("r", sig(f)) is True)

    e = fresh()
    e._frame_unchanged("r", sig(frame(1)))
    check("区域尺寸变了要重算",
          e._frame_unchanged("r", sig(np.zeros((90, 400, 3), np.uint8))) is False)

    import time as _t
    big = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    t0 = _t.perf_counter()
    for _ in range(100):
        sig(big)
    per_ms = (_t.perf_counter() - t0) / 100 * 1000
    check("签名开销 < 1ms（比 OCR 便宜几个数量级）", per_ms < 1.0,
          f"{per_ms:.3f} ms/次")


def test_touch_region():
    """跳过 OCR 时必须给卡片续命，否则会被当成"文字消失"删掉。"""
    from screenreader.board import BoardConfig, CardStore

    print("\n[跳过 OCR 时给卡片续命]")

    st = CardStore(BoardConfig())
    r = st.add_region("t", (0, 0, 10, 10))
    out = st.observe(r.id, ["第一句要留着。", "第二句也要留着。"])
    check("先建出两张卡", len(out.added) == 2, str(len(out.added)))

    c = out.added[0]
    c.misses = 99
    n = st.touch_region(r.id)
    check("续命覆盖了本区域全部卡片", n == 2, str(n))
    check("misses 被清零", c.misses == 0, str(c.misses))

    for _ in range(10):
        st.touch_region(r.id)
    check("反复续命后卡片还在", len(st.cards_of(r.id)) == 2,
          str(len(st.cards_of(r.id))))

    st2 = CardStore(BoardConfig())
    r2 = st2.add_region("t2", (0, 0, 10, 10))
    st2.observe(r2.id, ["这句会被删掉。"])
    for _ in range(5):
        st2.observe(r2.id, [""])
    check("不续命时会被删（对照组）", len(st2.cards_of(r2.id)) == 0,
          str(len(st2.cards_of(r2.id))))


def _mini_engine():
    """构造一个只够测逻辑的最小引擎（不碰 OCR / 播放 / 真 TTS）。"""
    import queue
    import threading

    from screenreader.config import AppConfig
    from screenreader.worker import BoardEngine, EngineConfig

    class FakeB:
        name = "fake"
        speed_factor = 1.0

        def synthesize(self, text, **kw):
            return b"RIFF" + b"\x00" * 40

    e = BoardEngine(AppConfig(), backend=FakeB(), engine_cfg=EngineConfig())
    e._emit = lambda *a, **k: None      # noqa: SLF001
    return e


def test_speed_setting():
    """语速 0.5~2.0，步进由界面控制；引擎侧要钳制并同步到后端。"""
    print("\n[语速设置]")
    e = _mini_engine()

    check("默认 1.0", float(e.cfg.tts.speed_factor) == 1.0, str(e.cfg.tts.speed_factor))
    check("设 1.5 生效", e.set_speed(1.5) == 1.5)
    check("后端也同步了", e.backend.speed_factor == 1.5, str(e.backend.speed_factor))
    check("配置也同步了", e.cfg.tts.speed_factor == 1.5, str(e.cfg.tts.speed_factor))
    check("超过 2.0 被钳住", e.set_speed(9.0) == 2.0)
    check("低于 0.5 被钳住", e.set_speed(0.1) == 0.5)
    check("边界 0.5 可用", e.set_speed(0.5) == 0.5)
    check("边界 2.0 可用", e.set_speed(2.0) == 2.0)
    check("非法值不炸", isinstance(e.set_speed("乱码"), float))


def test_retry_synth():
    """「重新生成」要重置状态、绕过缓存、重新排队。"""
    print("\n[卡片重新生成]")
    e = _mini_engine()
    r = e.store.add_region("t", (0, 0, 10, 10))
    card = e.store.observe(r.id, ["这句要重新生成。"]).added[0]

    card.audio_state = "ready"
    card.audio = b"old-audio"
    card.auto_done = True

    check("重试返回成功", e.retry_synth(card.id) is True)
    check("状态回到 pending", card.audio_state == "pending", card.audio_state)
    check("旧音频被清掉", card.audio is None)
    check("auto_done 复位（允许自动朗读再播）", card.auto_done is False)
    check("标记了绕过缓存", card.id in e._force_synth)  # noqa: SLF001
    check("已排进合成队列", e._synth_q.qsize() >= 1)  # noqa: SLF001
    check("不存在的卡片返回 False", e.retry_synth("nope") is False)


def test_cache_ttl():
    """缓存 TTL：超过 ttl 的条目被 sweep 掉；ttl=0 不过期；clear 清空。"""
    import tempfile
    import time as _t

    from screenreader.cache import AudioCache

    print("\n[缓存自动清理]")

    td = Path(tempfile.mkdtemp())
    c = AudioCache(td, enabled=True, ttl_s=0.25)
    c.put("k1", b"RIFF" + b"\x00" * 40)
    check("刚写入能读到", c.get("k1") is not None)
    _t.sleep(0.4)
    check("过期后被 sweep 掉", c.sweep() == 1)
    check("过期后读不到了", c.get("k1") is None)

    c0 = AudioCache(Path(tempfile.mkdtemp()), enabled=True, ttl_s=0)
    c0.put("k2", b"x" * 100)
    _t.sleep(0.15)
    check("ttl_s=0 时什么都不清", c0.sweep() == 0)

    c3 = AudioCache(Path(tempfile.mkdtemp()), enabled=True, ttl_s=30)
    for i in range(3):
        c3.put(f"kk{i}", b"y" * 50)
    check("clear 返回清掉的条数", c3.clear() == 3)
    check("clear 之后为空", c3.stats().get("entries") == 0, str(c3.stats()))


def test_auto_read():
    """自动朗读：勾选后新卡片自动排队；不勾选不排。"""
    print("\n[自动朗读]")
    e = _mini_engine()
    r = e.store.add_region("单行区", (0, 0, 10, 10))

    # 默认关
    check("默认关闭", e.ecfg.auto_read_all is False)
    out = e.store.observe(r.id, ["第一句不会被自动排队。"])
    e._after_observe(r, out)   # noqa: SLF001
    check("关闭时不排队", not out.added[0].queued)

    # 打开
    e.ecfg.auto_read_all = True
    out2 = e.store.observe(r.id, ["第二句应该自动排队。"])
    e._after_observe(r, out2)  # noqa: SLF001
    check("开启后新卡自动排队", bool(out2.added[0].queued))

    # 自动朗读对**单行模式**也要生效（用户明确要求）
    check("区域是单行模式", e.store.region(r.id).mode == "line",
          e.store.region(r.id).mode)

    # 只能出队一次（播过就不再排） —— 靠 auto_done
    c = e.store.next_auto_card()
    check("能取到排队中的卡片", c is not None)
    if c is not None:
        c.auto_done = True
        c.queued = False
        check("播过之后不再出队", e.store.next_auto_card() is None)


def test_segment():
    print("\n[segment 分句]")

    got = split_text("大家好，欢迎来到直播间。今天我们继续看这个游戏！")
    expect_eq("按强标点切句", got, ["大家好，欢迎来到直播间。", "今天我们继续看这个游戏！"])

    got = split_text("哇塞")
    expect_eq("短句保留", got, ["哇塞"])

    # 碎片并入前一句
    opt = SegmentOptions(min_chars=4)
    got = split_text("这是一句完整的话。啊。", opt)
    check("短碎片并入前句", len(got) == 1 and got[0].endswith("啊。"), f"得到 {got}")

    # 过长句用弱标点再切
    long_line = "这是一个非常非常长的句子" * 5
    got = split_text(long_line, SegmentOptions(max_chars=20))
    check("超长句被切开", all(len(s) <= 40 for s in got) and len(got) > 1, f"{len(got)} 段")

    # 无标点长串硬切
    got = split_text("啊" * 100, SegmentOptions(max_chars=28))
    check("无标点硬切", len(got) >= 4 and all(len(s) <= 28 for s in got), f"{len(got)} 段")

    # 纯标点/空白丢弃
    got = split_text("。。。\n\n   \n")
    expect_eq("纯标点丢弃", got, [])

    expect_eq("空串", split_text(""), [])
    expect_eq("归一化零宽", normalize_text("a\u200bb\ufeffc"), "abc")
    expect_eq("归一化全角空格", normalize_text("a\u3000b"), "a b")

    # 换行分段
    got = split_text("第一行。\n第二行。")
    expect_eq("按换行分段", got, ["第一行。", "第二行。"])


def test_cache_key():
    print("\n[cache 键]")
    a = make_key("你好", ref_audio="r.wav", prompt_text="p", speed_factor=1.0)
    b = make_key("你好", ref_audio="r.wav", prompt_text="p", speed_factor=1.0)
    c = make_key("你好", ref_audio="r.wav", prompt_text="p", speed_factor=1.1)
    d = make_key("你好啊", ref_audio="r.wav", prompt_text="p", speed_factor=1.0)
    check("同参数同键", a == b)
    check("语速不同键不同", a != c)
    check("文本不同键不同", a != d)


def test_cache_io():
    print("\n[cache 读写与淘汰]")
    with tempfile.TemporaryDirectory() as td:
        c = AudioCache(Path(td), max_entries=3, max_mb=1)
        data = wav_from_pcm(b"\x00\x00" * 3200, 32000)
        for i in range(5):
            c.put(make_key(f"句子{i}"), data)
        st = c.stats()
        check("超出 max_entries 会淘汰", st["entries"] == 3, f"entries={st['entries']}")

        k = make_key("句子4")
        check("最近写入的还在", c.get(k) is not None)
        check("命中计数", c.stats()["hits"] == 1)

        # 重新打开，索引要能恢复
        c2 = AudioCache(Path(td), max_entries=3, max_mb=1)
        check("索引可恢复", c2.stats()["entries"] == 3, f"entries={c2.stats()['entries']}")

        n = c2.clear()
        check("clear 生效", n == 3 and c2.stats()["entries"] == 0)

        # 禁用时全部 no-op
        c3 = AudioCache(Path(td), enabled=False)
        c3.put("x", data)
        check("禁用时不写入", c3.get("x") is None and c3.stats()["entries"] == 0)


def test_wav_helpers():
    print("\n[wav 工具]")
    # 32000 Hz、16bit、1 声道 -> 1 秒 = 32000 个样本 = 64000 字节
    pcm = b"\x01\x02" * 32000
    data = wav_from_pcm(pcm, 32000, channels=1, sampwidth=2)
    sr, ch, dur = wav_info(data)
    check("wav_from_pcm 头正确", (sr, ch) == (32000, 1), f"{sr} {ch}")
    check("时长约 1 秒", abs(dur - 1.0) < 0.01, f"{dur:.3f}s")

    with wave.open(io.BytesIO(data), "rb") as w:
        check("可被 wave 打开", w.getnframes() == 32000, f"{w.getnframes()} 帧")


def test_wav_header_parse():
    print("\n[流式 WAV 头解析]")
    pcm = b"\x00\x00" * 800
    good = bytearray(wav_from_pcm(pcm, 32000))
    parsed = parse_wav_header(bytes(good))
    check("普通 WAV 可解析", parsed is not None)
    if parsed:
        fmt, off = parsed
        expect_eq("解析出采样率", fmt[0], 32000)
        expect_eq("data 偏移正确", len(good) - off, len(pcm))

    # 模拟流式：把 RIFF size 和 data size 都写成 0
    streamed = bytearray(good)
    struct.pack_into("<I", streamed, 4, 0)
    idx = streamed.find(b"data")
    struct.pack_into("<I", streamed, idx + 4, 0)
    parsed2 = parse_wav_header(bytes(streamed))
    check("size=0 的流式 WAV 仍可解析", parsed2 is not None)
    if parsed2:
        expect_eq("流式采样率", parsed2[0][0], 32000)

    check("垃圾数据返回 None", parse_wav_header(b"not a wav file at all") is None)
    check("半截头返回 None", parse_wav_header(bytes(streamed)[:16]) is None)


def test_config():
    print("\n[config]")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "c.json"
        cfg = AppConfig()
        cfg.audio.device = "CABLE Input"
        cfg.ocr.max_lines = 12
        cfg.capture.region = [0, 0, 900, 300]
        cfg.save(p)
        back = AppConfig.load(p)
        expect_eq("设备回读", back.audio.device, "CABLE Input")
        expect_eq("行数上限回读", back.ocr.max_lines, 12)
        expect_eq("区域回读", back.capture.region, [0, 0, 900, 300])

        # 未知键忽略而不是崩
        p.write_text('{"ocr": {"max_lines": 5, "不存在": 1}, "nope": 2}', encoding="utf-8")
        cfg2 = AppConfig.load(p)
        expect_eq("未知键被忽略", cfg2.ocr.max_lines, 5)

        check("缺文件用默认值", AppConfig.load(Path(td) / "nope.json").ocr.max_lines == 25)


def test_tier():
    print("\n[显存分档]（阈值来自实测 TTS 占用 2.8GB + 768MB 安全余量）")
    # A: budget = free - 768 >= 3300  ->  free >= 4068
    expect_eq("空闲 14.4G -> A", decide_tier([GpuInfo(0, "4080S", 16376, 2000, 14376, 5)]).key, "A")
    expect_eq("空闲 4.4G  -> A", decide_tier([GpuInfo(0, "4080S", 16376, 12000, 4376, 90)]).key, "A")
    # B: 3068 <= free < 4068
    expect_eq("空闲 3.5G  -> B", decide_tier([GpuInfo(0, "4080S", 16376, 12900, 3476, 95)]).key, "B")
    # C: 2168 <= free < 3068
    expect_eq("空闲 2.6G  -> C", decide_tier([GpuInfo(0, "4060", 8188, 5600, 2588, 99)]).key, "C")
    # D: free < 2168
    expect_eq("空闲 1.2G  -> D", decide_tier([GpuInfo(0, "4060", 8188, 7000, 1188, 99)]).key, "D")
    expect_eq("空闲 0.09G -> D", decide_tier([GpuInfo(0, "4060", 8188, 8100, 88, 100)]).key, "D")
    expect_eq("无 N 卡 -> D", decide_tier([]).key, "D")

    # 安全余量可调：同样的空闲显存，余量给大就降档
    g = [GpuInfo(0, "4080S", 16376, 12000, 4376, 90)]
    expect_eq("余量 768  -> A", decide_tier(g, reserve_mb=768).key, "A")
    expect_eq("余量 2500 -> C", decide_tier(g, reserve_mb=2500).key, "C")   # 4376-2500=1876 -> C
    expect_eq("余量 4000 -> D", decide_tier(g, reserve_mb=4000).key, "D")   # 4376-4000=376  -> D


def test_clean_for_tts():
    print("\n[送 TTS 前的文字清洗]")
    from screenreader.segment import clean_for_tts, has_speakable

    # **行首编号过滤现在默认关闭**（用户要求）。
    # 当初加它是因为 OCR 行首的列表编号会被念成一个孤立怪音，
    # 但那个问题的真正原因是当时参考音频配串了；修好之后这个过滤
    # 反而把正文里本来就该念的数字吃掉了。
    expect_eq("默认保留行首数字（紧贴）", clean_for_tts("1语音输出前有奇怪的东西"),
              "1语音输出前有奇怪的东西")
    expect_eq("默认保留行首数字（带点）", clean_for_tts("2. 开始识别后"), "2. 开始识别后")
    expect_eq("默认保留行首数字（括号）", clean_for_tts("(3) 希望将按钮"), "(3) 希望将按钮")
    expect_eq("默认保留行首数字（顿号）", clean_for_tts("4、框选范围"), "4、框选范围")

    # 需要时仍可打开（保留原有能力，只是不再默认启用）
    expect_eq("打开开关才去编号（紧贴）",
              clean_for_tts("1语音输出前有奇怪的东西", strip_leading_numbers=True),
              "语音输出前有奇怪的东西")
    expect_eq("打开开关才去编号（带点）",
              clean_for_tts("2. 开始识别后", strip_leading_numbers=True), "开始识别后")
    expect_eq("打开开关才去编号（括号）",
              clean_for_tts("(3) 希望将按钮", strip_leading_numbers=True), "希望将按钮")
    expect_eq("打开开关才去编号（顿号）",
              clean_for_tts("4、框选范围", strip_leading_numbers=True), "框选范围")

    # "数字+量词"是内容，开关打开时也不能削
    expect_eq("保留数量词-分钟", clean_for_tts("3分钟", strip_leading_numbers=True), "3分钟")
    expect_eq("保留数量词-个", clean_for_tts("5个人", strip_leading_numbers=True), "5个人")
    expect_eq("保留数量词-年", clean_for_tts("2024年", strip_leading_numbers=True), "2024年")
    expect_eq("句中的数字不动", clean_for_tts("金币 12345 个"), "金币 12345 个")

    # 排版装饰
    expect_eq("去掉带圈数字", clean_for_tts("④ 新会话"), "新会话")
    expect_eq("去掉项目符号", clean_for_tts("• 第一项"), "第一项")
    expect_eq("去掉箭头", clean_for_tts(">. 运行命令"), "运行命令")

    check("纯符号不可念", not has_speakable("。。。"))
    check("纯数字不可念", not has_speakable("12345"))
    check("单个字不可念", not has_speakable("啊"))
    check("正常句子可念", has_speakable("你好啊"))
    check("清洗后变空的不可念", not has_speakable(clean_for_tts("1。")))


# --------------------------------------------------------------------------- #
def main() -> int:
    for fn in (
        test_segment,
        test_paragraph_merge,
        test_paragraph_ordering,
        test_paragraph_settle,
        test_paragraph_manual_queue,
        test_paragraph_cleanup_and_append,
        test_paragraph_punctuation_priority,
        test_paragraph_clear_emits_died,
        test_calm_score,
        test_volume,
        test_playing_card_not_destroyed,
        test_frame_gate,
        test_touch_region,
        test_speed_setting,
        test_retry_synth,
        test_cache_ttl,
        test_auto_read,
        test_region_persist,
        test_clean_for_tts,
        test_cache_key,
        test_cache_io,
        test_wav_helpers,
        test_wav_header_parse,
        test_config,
        test_tier,
    ):
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
