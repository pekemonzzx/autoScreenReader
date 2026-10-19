# gsv/ —— TTS 推理运行时（不在仓库里，需自行安装）

这个目录**故意不放进仓库**：它约 **9 GB**，而且是 GPT-SoVITS 整合包里的第三方内容。

## 放什么

```
gsv\
├─ api_v2.py                 推理服务入口
├─ config.py                 GPT-SoVITS 配置
├─ .runtime_installed.json   安装标记（少了它程序会认为没装）
├─ GPT_SoVITS\               模型代码 + 预训练模型（约 2.5 GB）
├─ runtime\                  内嵌 Python 3.9 + torch（约 6.4 GB）
│   └─ python.exe            ← 程序用它启动服务
├─ tools\                    GPT-SoVITS 自带工具（uvr5 / asr）
├─ GPT_weights_v2ProPlus\    训练产出的 GPT 权重（可先空着）
└─ SoVITS_weights_v2ProPlus\ 训练产出的 SoVITS 权重（可先空着）
```

## 怎么装

```bat
:: 在线清单
读屏台.exe --runtime install --manifest https://<地址>/manifest.json

:: 本地文件夹 / zip（最省事）
读屏台.exe --runtime install --from D:\下载\gsv
```

程序首次启动发现没装运行时会**自动弹出安装窗口**，跟着走也行。

## 装完验证

```bat
读屏台.exe --runtime status
读屏台.exe --runtime verify
```

## 详细说明

见 [../docs/运行时与声库.md](../docs/运行时与声库.md)。
