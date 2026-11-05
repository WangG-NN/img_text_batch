# 图片文字批量替换工具

一个基于 Tkinter 的桌面小工具：把一批模板图片上的旧文字抹掉，再批量填上新的文字。适合做海报、活动图、证书等需要批量换文案的场景。

## 功能特性

- **两种文字来源**：上传图片用 OCR 识别文字，或直接手动输入纯文字。
- **两种抹字模式**
  - 固定区域：按设置的百分比区域抹除旧文字；
  - OCR 检测：自动找出模板上的文字再抹除，可只抹文字块附近或抹全图。
- **两种抹除引擎**
  - 高质量（LaMa）：深度修复，效果自然，需要 ONNX 模型；
  - 快速（OpenCV）：TELEA 局部修复，无需模型，速度更快。
- **灵活排版**：文字块位置、对齐、字号、行距、颜色均可调；行尾加 `（40）` 可单独设置该行字号；空行保留为间隔；内容超高时整体自动缩放。
- **批量生成**：每个文字源对应输出目录下的一个子文件夹，模板 × 文字源一次跑完。
- **配置记忆**：文件夹、字体、格式设置自动保存到 `config.json`。

## 环境要求

- Python 3.9 及以上（开发时使用 3.12）
- Windows / macOS / Linux 均可运行（界面使用 Tkinter，需系统自带或安装）

## 安装

```bash
git clone <本仓库地址>
cd img_text_batch
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate    # macOS / Linux
pip install -r requirements.txt
```

首次使用高质量抹除时，按 [models/README.md](models/README.md) 下载 LaMa 模型；不下载也能用 OpenCV 快速抹除。

## 使用

```bash
python app.py
```

1. 选择「模板文件夹」（必填）、「干净底图文件夹」（可选）、「输出文件夹」（必填）和字体文件（可留空，自动使用系统中文字体）。
2. 在左侧「图片文字源」里添加图片并点「识别选中」，或在「文字文字源」里新建文本手动输入。
3. 调整格式设置（模式、文字块、字号、行距、颜色、抹除引擎）。
4. 点「开始批量生成」，结果输出到 `输出文件夹/文字源名/001.png ...`。

### 配置说明

程序启动时读取同目录的 `config.json`，不存在则使用内置默认值。仓库提供了 [config.example.json](config.example.json) 作为模板，可复制为 `config.json` 后修改。`config.json` 含个人本地路径，不纳入版本管理。

| 字段 | 说明 |
| --- | --- |
| `mode` | `fixed` 固定区域抹字；`ocr` 自动检测文字抹除 |
| `text_block` | 文字块位置，`x/y/w/h` 为占图片宽高的比例（0~1），`align` 为 `center/left/right` |
| `font_size` | 默认字号（px） |
| `color` | 文字颜色，十六进制 |
| `line_spacing` | 行距倍数 |
| `erase` | 是否抹除旧文字 |
| `erase_engine` | `lama` 高质量 / `opencv` 快速 |
| `ocr_erase_all` | OCR 模式下是否抹除全图文字（默认只抹文字块附近） |

## 打包成可执行文件

### Windows

```bat
pip install -r requirements-dev.txt
build_windows.bat
```

产物在 `dist\图片文字批量替换.exe`，首次运行前把 `models\lama_fp32.onnx` 放到程序同目录的 `models` 文件夹（或打包完成后手动拷贝）。

### macOS

```bash
bash build_mac.sh
```

脚本会自动创建虚拟环境、安装依赖、下载模型并打包，产物在 `dist/图片文字批量替换`。首次打开若被系统拦截（未认证开发者），执行一次：

```bash
xattr -cr "dist/图片文字批量替换"
```

## 目录结构

```
img_text_batch/
├── app.py            # GUI 入口
├── core.py           # 抹字 + 渲染核心
├── ocr.py            # RapidOCR 封装
├── lama.py           # LaMa ONNX 抹除封装
├── config.example.json
├── requirements.txt
├── requirements-dev.txt
├── build_windows.bat
├── build_mac.sh
├── build_mac.command
├── 图片文字批量替换.spec
└── models/
    └── README.md     # 模型下载说明
```

## 常见问题

- **找不到中文字体**：在界面里手动选择字体文件（如 `C:\Windows\Fonts\msyh.ttc`）。
- **LaMa 首次运行很慢**：程序会在后台预热模型，且每块推理约 10~20 秒；着急可切换「快速(OpenCV)」。
- **抹除效果不理想**：调整文字块范围，或改用 OCR 模式只抹检测到的文字。
