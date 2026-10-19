# TTS 训练与延迟报告

> 日期：2026-10-14
> 机器：i7-14700KF + RTX 4080 SUPER 16GB
> 整合包：`D:\GPT-SoVITS\GPT-SoVITS-v2pro-20250604`（内嵌 Python 3.9.13 / torch 2.0.0+cu118）
> 版本：**GPT-SoVITS v2ProPlus**（选它是因为只有 v2 / v2ProPlus 有 ONNX 推理路径，见方案 §3.2）

---

## 1. 结论速览

| 项 | 结果 |
| :--- | :--- |
| 语料来源 | `P1_0-示例素材_..._480p.mp4` 的 **0:00–5:07** 与 **18:08–25:47** 两段（合计 12:46） |
| 人声分离 | UVR5 `HP5_only_main_vocal`，GPU 推理，两段均成功 |
| 切片 + ASR | 119 条 → 机械清洗 **93 条 / 6.1 分钟**（1938 字） |
| **SoVITS 训练** | ✅ 成功，**151 秒** → `voice01_e4_s124.pth` / `voice01_e8_s248.pth`（各 165MB） |
| **GPT 训练** | ✅ 成功，**61 秒** → `voice01-e4/e8/e12.ckpt`（各 148MB） |
| **推理验证** | ✅ 合成出 32kHz / 3.76 秒语音 |
| **非流式延迟** | ⚠️ **p50 1786 ms** |
| **流式首包延迟** | ⚠️ **p50 667 ms**（min 471ms），比非流式快 2.7× |

**一句话：训练链路完全跑通且很快（总 3.5 分钟）；但推理延迟比方案预估的 120–300ms 高出 5–10 倍，必须用流式 + 分句 + 缓存来补。**

---

## 2. 数据准备结果

```
00_raw          2 个文件   23.4 MB   16kHz 单声道（按时间段截取）
01_vocal        2 个文件  128.9 MB   UVR5 分离出的人声
01_instrumental 2 个文件  128.9 MB   伴奏（不需要，占地方可删）
02_sliced     119 个文件   30.5 MB   切片
03_list         1 个 .list            FunASR 自动标注
```

清洗规则（`tools/filter_list.py`）与结果：

| 剔除原因 | 条数 |
| :--- | :--- |
| 太短（<4 字） | 7 |
| 太长（>25 字） | 18 |
| 中文占比 <50% | 1 |
| **保留** | **93**（78%） |

保留后：平均 16 字/条、单条 4.2 秒、总时长 6.1 分钟。

### 2.1 初版：未做人工校对

> **更新：已人工校对并重训，见 §3.1。**

ASR 输出抽样里可见的错误：

- `来来了。`（应为"来了来了"）
- `车要吃饭的`（多了个"车"）
- `该续味了四有时候是反应慢`（明显串词）
- `嗯，我啊啊hello.`（中英混杂噪声）
- `谢谢莫飞的薛定二的SC`（SC=SuperChat，人名识别错）

**这些错误会直接进入训练集，拉低音色与韵律质量。** 机械清洗只解决了长度和语种问题，同音字错误必须人工听才能改。当前 93 条是"能训出可用效果"的水平，但离"好"还有距离。

### 2.2 人工校对后重训（第 2 版）

用整合包自带的 **SubFix 校对工具**（`0d` 页签 / 端口 9871）逐条听音频改文本，实际修正的例子：

| 校对前 | 校对后 |
| :--- | :--- |
| 来来了。 | **来了来了。** |
| 这里挺好看的 | **这集**挺好看的 |
| 感觉昨天听完演唱会之后，**在**听这一段 | **再**听这一段 |
| 就一两句话说好了**哈。** | 就一两句话说好了**，哈哈哈。** |
| **该续味了四**有时候是反应慢 | **该续费了，**有时候是反应慢 |
| 有的时候是**冲**在心中 | 有的时候是**痛**在心中 |

重训结果（同样 93 条，只改了文本）：

| | 第 1 版（未校对） | **第 2 版（已校对）** |
| :--- | :--- | :--- |
| SoVITS 训练 | 151 s | **131 s** |
| GPT 训练 | 61 s | **42 s** |
| 总字数 | 1938 | 1371（更凝练、去掉了口水话） |
| 平均字数/条 | 16.3 | 14.7 |
| 非流式推理 p50 | 1786 ms | 1830 ms（噪声范围内） |

产物路径不变，旧版权重已备份到 `D:\GPT-SoVITS\backup\voice01_preproofread\`。

**A/B 试听**：
- 第 1 版：`data\voice01_test.wav`
- 第 2 版：`data\voice01_test_v2.wav`

（同一句文本、同一段参考音频，可直接对比音色与韵律。）

### 2.3 校对工具踩的坑：Gradio 403

SubFix 用 Gradio 4 提供试听，而 **Gradio 默认只服务自身工作目录下的文件**，其它一律返回 403。

我们的切片在 `D:\GPT-SoVITS\work\voice01\02_sliced`，SubFix 的工作目录是整合包根目录 → **在允许列表之外**。表现就是：音频加载不出来，点「Submit Text」重新渲染全部音频组件时每条都 403 → 界面报错。

实测对比：

| 环境 | `/file=<音频路径>` |
| :--- | :--- |
| 默认 | **403 已禁止** |
| 设 `GRADIO_ALLOWED_PATHS=切片目录` | **200 OK, 80044 bytes** |

解决：`GRADIO_ALLOWED_PATHS`（Gradio 4 官方支持的环境变量，逗号分隔绝对路径）。
已封装成 `tools/subfix_launch.py`（**从 .list 自动推导放行目录**）+ `scripts/start_subfix.bat` 一键启动。

另外两个 SubFix 使用要点：
1. **改完必须点「Submit Text」才写回文件**，直接翻页会回滚。
2. 保存是就地覆盖，文本末尾会多一个空格（`new_text.strip() + " "`），训练无影响；`tts_train.py` 运行前会顺手去掉。

---

## 3. 训练结果

配置（`tools/tts_train.py`）：

| 参数 | SoVITS | GPT |
| :--- | :--- | :--- |
| batch_size | 6 | 8 |
| epochs | 8 | 15 |
| 精度 | fp16 | fp16 |
| 耗时（第 1 版） | 151 s | 61 s |
| 耗时（第 2 版·已校对） | **131 s** | **42 s** |

产物（`D:\GPT-SoVITS\GPT-SoVITS-v2pro-20250604\`）：

```
SoVITS_weights_v2ProPlus\voice01_e4_s124.pth   165 MB
SoVITS_weights_v2ProPlus\voice01_e8_s248.pth   165 MB   <- 推荐用 e8
GPT_weights_v2ProPlus\voice01-e4.ckpt          148 MB
GPT_weights_v2ProPlus\voice01-e8.ckpt          148 MB
GPT_weights_v2ProPlus\voice01-e12.ckpt         148 MB   <- 推荐用 e12
```

推理时加载权重会打印 `_IncompatibleKeys(missing_keys=['enc_q.*'])` —— **这是正常的**，`enc_q` 是训练用的后验编码器，推理不需要。

---

## 4. 推理延迟实测（最重要的一节）

### 4.1 实测数据

文本：`大家好，欢迎来到直播间，今天我们继续来看这个游戏。`（25 字）
参考音频：4.8 秒切片；输出：32kHz 单声道 3.76 秒

| 模式 | n | p50 | min | max |
| :--- | :--- | :--- | :--- | :--- |
| 预热（首次，含参考特征加载） | 1 | 1688 ms | — | — |
| **非流式（整段生成完再返回）** | 3 | **1786 ms** | 1749 | 1870 |
| **流式首包（`streaming_mode=true`）** | 3 | **667 ms** | **471** | 836 |
| 流式整段 | 3 | 1585 ms | 1576 | 1914 |

### 4.2 与方案预估的差距

| | 方案 §5 预估 | 实测 | 差距 |
| :--- | :--- | :--- | :--- |
| TTS 首包（目标机空闲） | 120–300 ms | **667 ms**（流式） | **2–5×** |
| TTS 首包（本机，4080S） | — | 1786 ms（非流式） | 6–15× |

**方案里的 TTS 延迟预估必须上调。** 原因：官方 README 的 RTF 0.014–0.028 是"纯模型前向"的数字，不含文本前端（BERT/G2PW）、参考音频特征提取、采样与调度开销。

### 4.3 流式返回的真实格式（实现要点）

`streaming_mode=true` 时返回的仍是 RIFF/WAV 头，但 **RIFF size 字段是 0**（流式标准做法），`wave.open` 会认为时长 0。
后面的负载是 **32kHz 单声道 int16 PCM**，分块下发。

→ 运行时实现要点：**不要等文件落盘再用 wave 读**，应该把每个 chunk 直接推进音频设备的播放缓冲，这才是流式真正的价值。

### 4.4 可继续优化的方向（按预计收益排序）

1. **结果缓存**（收益最高，且与场景强相关）：读屏内容重复率高（同一句弹幕、固定 UI 文案），命中即 0 延迟。
2. **`sample_steps` 32 → 16/8**：直接减少采样计算量，代价是音质略降。
3. **缩短参考音频到 3 秒 + 用 `/set_refer_audio` 固定**：避免每次请求重算 CNHuBERT 特征。
4. **参考文本与参考音频固定**：让 API 侧的 `self.cache` 命中。
5. **分句更短**：把一句话拆成两个短句并行合成，首句先播（方案 §3.5 的流水线）。

---

## 5. 踩过的坑（全部已固化进脚本，勿回退）

写 `tools/tts_prepare.py` / `tools/tts_train.py` 时在整合包里踩了一串坑，都是"不看源码绝对想不到"的类型：

| # | 坑 | 现象 | 处理 |
| :--- | :--- | :--- | :--- |
| 1 | 集成包（v3lora）**不支持 v2ProPlus** | `version` 只认 v1/v2/v3 | 改用 `GPT-SoVITS-v2pro-20250604.7z` |
| 2 | `runtime/Lib/site-packages/users.pth` **硬编码打包者路径** | `ModuleNotFoundError: No module named 'text'`，且 PYTHONPATH 无效（有 `python39._pth`） | 按实际解压路径重写 users.pth（已自动化） |
| 3 | `tools/uvr5/webui.py` **模块级直接 `app.queue().launch()`** | import 就拉起 Gradio 服务并阻塞 | import 前把 `gr.Blocks.launch` 换成空操作 |
| 4 | `tools/uvr5/webui.py` **模块级读 `sys.argv[1..4]`** | 被自己的命令行参数污染 → SyntaxError | import 前临时替换 `sys.argv` |
| 5 | `tools/slice_audio.py` **末行裸调 `print(slice(*sys.argv[1:]))`** | 同上 | 换成"不存在的输入路径 + 合法参数"使其空转 |
| 6 | `uvr()` 是**生成器** | 不迭代消费就什么都不做 | 必须 for 循环消费 |
| 7 | `uvr5/webui.py` 用 `os.system("ffmpeg ...")` 调**裸命令名** | 找不到 ffmpeg | 集成包根目录前置到 PATH |
| 8 | `slice_audio.py` 用 `from slicer2 import ...`（裸模块名） | ImportError | `tools/` 与 `tools/uvr5/` 都要进 sys.path |
| 9 | v2Pro/v2ProPlus **多一步 `2-get-sv.py`** | `assert os.path.exists(self.path7)` 失败（缺 `7-sv_cn/`） | 1B 之后补跑 SV 步骤，env 传 `sv_path` |
| 10 | 1a/1c 产物是**分片的**（`2-name2text-0.txt`） | 校验找不到正式文件 | 按 WebUI 逻辑合并分片 |
| 11 | 实验目录必须是 `<GSV_ROOT>/logs/<exp>` | `FileNotFoundError: logs_s2_v2ProPlus/G_xxx.pth`（跑完一轮才失败） | 用 WebUI 的 `exp_root` 约定并预建子目录 |
| 12 | 沙箱下 venv 的 Python **只能写工作区** | `PermissionError` 写 `D:\GPT-SoVITS\...` | 需写外部的脚本改用集成包内嵌 Python，或输出到工作区 |
| 13 | API 端口复用 | `[Errno 10048]` 绑定失败 | 启动前先清理 9880 上的残留进程 |
| 14 | PowerShell 发 JSON 时中文乱码 | `/tts` 返回 400 | 用 UTF-8 文件 + `-InFile`，或 `charset=utf-8` |

---

## 6. 复现命令

```powershell
# 1) 录像 -> 标注（按时间段 + 人声分离）
& 'D:\GPT-SoVITS\GPT-SoVITS-v2pro-20250604\runtime\python.exe' -s tools\tts_prepare.py `
  --input "<录像路径>" --work "D:\GPT-SoVITS\work\voice01" `
  --segments "0:00-5:07,18:08-25:47"

# 2) 机械清洗（venv 的 Python，输出必须落在工作区内）
.\.venv\Scripts\python.exe tools\filter_list.py `
  --list "D:\GPT-SoVITS\work\voice01\03_list\02_sliced.list" `
  --wav-dir "D:\GPT-SoVITS\work\voice01\02_sliced" `
  --out "D:\work\autoScreenReader\data\voice01_train_clean.list"

# 3) 训练（会自动修正 users.pth）
& 'D:\GPT-SoVITS\GPT-SoVITS-v2pro-20250604\runtime\python.exe' -s tools\tts_train.py `
  --list "D:\work\autoScreenReader\data\voice01_train_clean.list" `
  --wav-dir "D:\GPT-SoVITS\work\voice01\02_sliced" --exp voice01

# 4) 启动推理 API
& 'D:\GPT-SoVITS\GPT-SoVITS-v2pro-20250604\runtime\python.exe' -s api_v2.py `
  -a 127.0.0.1 -p 9880 -c D:\work\autoScreenReader\data\tts_infer_voice01.yaml

# 5) 延迟探针
.\.venv\Scripts\python.exe tools\probe_tts.py --n 5
```

试听样本：`data\voice01_test.wav`（3.76 秒，32kHz），流式样本 `data\tts_probe\stream*.wav`。

---

## 7. 对方案的影响

1. **§5 延迟表必须改**：TTS 首包从 "120–300ms" 改为 **"流式 670ms（min 470ms）"**。
   叠加 OCR（80–200ms）+ 截屏（6ms），端到端首句出声 ≈ **750–900ms**，不是原来说的 250–600ms。
2. **§3.5 分句流水线从"优化项"升级为"必需项"**：只有它能让长文本的首句先出声。
3. **§3.6 的音频 LRU 缓存从"高收益"升级为"必须有"**：弹幕场景重复率高，这是把感知延迟拉回 500ms 以内的主要手段。
4. **P3（ONNX 导出）的意义更大了**：目前是 torch + CUDA 运行时（约 3GB）；导出 ONNX 后目标机包体可降到 700MB 左右，且能用 DirectML 跑在任意显卡上。
5. **人工校对必须补**：当前 93 条含明显 ASR 错误，建议至少抽检一遍再重训（重训只要 3.5 分钟，成本很低）。
