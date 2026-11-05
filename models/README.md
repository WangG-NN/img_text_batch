# LaMa 模型

`lama.py` 需要 ONNX 格式的 LaMa 模型才能使用「高质量抹除」；缺失时会自动回退到 OpenCV 快速抹除，程序仍可正常运行。

## 下载

将模型放在本目录下，文件名二选一：

- `lama_fp32.onnx`
- `big-lama.onnx`

下载地址（约 208 MB）：

```bash
curl -L --retry 3 -o models/lama_fp32.onnx "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"
```

国内网络可用镜像：

```bash
curl -L --retry 3 -o models/lama_fp32.onnx "https://hf-mirror.com/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"
```

> 模型文件较大，未纳入 Git 仓库，请在打包或首次使用时下载。
