# voices/ —— 声库（音色）目录（不在仓库里，需自行安装或训练）

声库**故意不放进仓库**，原因有两个：

1. **体积**：单个权重文件 148~165 MB，超过 GitHub 的 100 MB 硬限制；
2. **隐私**：微调权重是「某个真人声音」的克隆结果，公开分发需要本人同意。

## 每个声库的结构

```
voices\
└─ voice01\                  ← 目录名 = 声库名（建议 ASCII）
   ├─ voice.json             ★ 元数据（必填）
   ├─ ref.wav                ★ 参考音频（必填，3~10 秒干净人声）
   ├─ weights\               ★ 微调权重（必填）
   │   ├─ voice01-e12.ckpt        GPT 权重
   │   └─ voice01_e8_s152.pth     SoVITS 权重
   ├─ train.yaml             训练配置（自动生成）
   ├─ train.list             训练标注
   ├─ tts_infer.yaml         推理配置（程序自动生成，别手改）
   └─ README.md              备注
```

`voice.json` 里 **`prompt_text` 必须和 `ref.wav` 实际说的话一字不差**，
否则音色会跑偏。

## 三种获取方式

```bat
:: A. 装别人打包好的声库包（最快）
读屏台.exe --voice install voice01.voicepack.zip
:: 也可以直接把 zip 解压到这个 voices\ 目录下，会自动识别

:: B. 自己训练（见 train/README.md）
python train\run.py --input "<录音>" --work "<工作目录>" --name voice01

:: C. 只拿到权重：建档案 + 放权重 + 指定参考音频
读屏台.exe --voice new voice01 --list train.list --wav-dir <切片目录>
读屏台.exe --voice set-ref voice01 --wav ref.wav --text "参考音频里说的话"
读屏台.exe --voice refresh voice01
```

## 装完验证

```bat
读屏台.exe --voice list
读屏台.exe --voice scan          :: 校验手工放进来的声库
读屏台.exe --voice verify voice01
```

## 详细说明

见 [../docs/运行时与声库.md](../docs/运行时与声库.md) 和
[../docs/声库与训练.md](../docs/声库与训练.md)。
