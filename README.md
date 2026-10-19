# 自动读屏台（screenshot2Speach）

截屏 → OCR → 分句 → TTS → 送进 OBS 当麦克风源的 Windows 桌面工具。

> **为什么 OCR 刻意走 CPU**：这个程序的目标场景是"游戏直播机"——读屏服务要和游戏、
> OBS 编码抢同一块显卡。把 OCR 放在 CPU 上，显存和算力全留给游戏和 TTS。

---

## 它做什么

| 功能 | 说明 |
| :--- | :--- |
| **多区域持续识别** | 框选屏幕上的文字区域（可多个），后台按 0.3~3 秒间隔持续 OCR |
| **单行 / 整段两种模式** | 单行=每行一句手动点播；整段=合并折行、按标点断句、顺序连读 |
| **悬浮字幕条** | 置顶可拖动，显示所有区域的卡片，可调透明度、可直接开始/停止识别 |
| **截图朗读** | 冻结一帧画面 → 框一段 → 立刻读一遍 → 读完自动清场（适合位置不固定的段落） |
| **全局快捷键** | 键盘组合键 / 鼠标侧键都能绑，游戏全屏时也生效（走 Windows 低级钩子） |
| **卡片重新生成** | 某句合成不完整时可以单独重跑（绕过缓存） |
| **声库可插拔** | 一个 `.voicepack.zip` 就是一个音色，装/换/删都不碰代码 |

---

## 环境要求

- **Windows 10 / 11**
- **NVIDIA 显卡**（TTS 推理走 CUDA；显存 8 GB 起，16 GB 更稳）
- **Python 3.11**（开发/打包用；运行发行版不需要）
- 磁盘：运行时约 **9 GB**

---

## 快速开始

### 1. 装推理运行时（约 9 GB，不在仓库里）

```bat
读屏台.exe --runtime install
```

或离线包：

```bat
读屏台.exe --runtime install --from D:\下载\gsv
```

### 2. 装声库

```bat
读屏台.exe --voice install 你的声库.voicepack.zip
```

也可以把 zip 直接解压到程序的 `voices\` 目录下，程序会自动识别。

### 3. 用

```bat
读屏台.exe
```

工具栏「▭ 框选区域」框一块屏幕区域 →「▶ 开始识别」。

---

## 仓库里有什么 / 没有什么

| | 内容 | 说明 |
| :--- | :--- | :--- |
| ✅ **有** | `src/` 源码、`tests/` 测试（860+ 项）、`train/` 训练流程、`tools/` 工具、`docs/` 文档、`packaging/` 打包脚本 | 全部可复现 |
| ❌ **没有** | TTS 推理运行时（9 GB） | 第三方内容，用 `--runtime install` 按需下载 |
| ❌ **没有** | 微调的声库权重（`.pth` / `.ckpt`） | 见下 |
| ❌ **没有** | 训练语料、日志、构建产物 | 见 `.gitignore` |

### 为什么不带声库权重

两个原因：

1. **体积**：单个 `.pth` 165 MB / `.ckpt` 148 MB，都超过 GitHub 的
   **单文件 100 MB 硬限制**，普通 git 推不上去。
2. **隐私**：微调权重是「某个真人声音」的克隆结果，公开分发需要本人同意。

需要分发时请用 **GitHub Releases**（单文件上限 2 GB，不占仓库体积）
或网盘，不要塞进仓库。

---

## 目录结构

```
src/screenreader/        主程序（OCR / 合成 / 播放 / 界面）
  ui/                    界面（主窗、悬浮条、框选层、设置对话框）
  tts/                   TTS 后端（GPT-SoVITS HTTP 客户端）
train/                   训练流程（体检 → 造语料 → 标注 → 训练 → 打包）
tools/                   构建、打包、清理、基准等工具
tests/                   测试（不需要 GPU / 音频设备也能跑大部分）
docs/                    文档
packaging/               Inno Setup 打包配置
```

---

## 文档

| 文档 | 内容 |
| :--- | :--- |
| [docs/使用说明.md](docs/使用说明.md) | 界面操作、快捷键、常见问题 |
| [train/README.md](train/README.md) | **训练新声库的完整流程**（每步一个脚本） |
| [docs/声库与训练.md](docs/声库与训练.md) | 声库档案结构、训练配置、踩过的坑 |
| [docs/实施方案.md](docs/实施方案.md) | 整体设计与资源共存策略 |
| [docs/打包与部署.md](docs/打包与部署.md) | 运行时裁剪、安装包结构、目标机部署 |
| [docs/P1基准报告.md](docs/P1基准报告.md) | 截屏 + OCR 延迟基准 |
| [docs/TTS延迟报告.md](docs/TTS延迟报告.md) | 训练全流程、推理延迟、整合包陷阱 |
| [docs/P4集成说明.md](docs/P4集成说明.md) | 自动循环模式、命令行 |

---

## 开发

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements\dev.txt
.\.venv\Scripts\python -m screenreader          # 跑界面
```

### 跑测试

```powershell
.\.venv\Scripts\python tests\test_units.py       # 纯逻辑，不需要 GPU
.\.venv\Scripts\python tests\test_ui_smoke.py    # 界面（offscreen 也能跑）
```

大部分测试**不需要 GPU、不需要 TTS 服务、不需要音频设备**。

### 训练自己的声库

```powershell
python train\run.py --input "<素材>" --work "<工作目录>" --name <声库名>
```

会在「人工标注」那步停下（必须人来做），把后续命令打给你。
详见 [train/README.md](train/README.md)。

---

## 已知限制

- **只在 Windows 上跑**（全局快捷键依赖 Windows 低级钩子，截屏依赖 DXGI）
- OCR 是 CPU 密集的：整屏 2560×1440 单次约 825 ms，**框小一点会快很多**
- TTS 服务首次启动要加载模型，约 **28 秒**

---

## 许可

[MIT](LICENSE)
