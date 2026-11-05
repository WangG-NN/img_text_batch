#!/bin/bash
# 在 macOS 上一键打包"图片文字批量替换"
# 用法：克隆本仓库到 Mac，在项目根目录执行：bash build_mac.sh
set -e
cd "$(dirname "$0")"

echo "[1/4] 创建虚拟环境并安装依赖..."
python3 -m venv .venv-mac
source .venv-mac/bin/activate
pip install --upgrade pip -q
pip install -q pillow numpy opencv-python rapidocr-onnxruntime onnxruntime pyinstaller

echo "[2/4] 准备 LaMa 模型（约 208MB，已有则跳过）..."
mkdir -p models
if [ ! -f models/lama_fp32.onnx ]; then
  curl -L --retry 3 -o models/lama_fp32.onnx "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx" \
    || curl -L --retry 3 -o models/lama_fp32.onnx "https://hf-mirror.com/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"
fi

echo "[3/4] 打包..."
python -m PyInstaller --noconfirm --clean --onefile --windowed \
  --name "图片文字批量替换" \
  --collect-all rapidocr_onnxruntime \
  --hidden-import lama \
  app.py

echo "[4/4] 整理输出..."
mkdir -p dist/models
cp models/lama_fp32.onnx dist/models/

echo ""
echo "完成！可执行文件：dist/图片文字批量替换"
echo "分发时把 dist 里的程序和 models 文件夹一起拷贝。"
echo "首次打开若被系统拦截（未认证开发者），在终端执行一次："
echo "  xattr -cr dist/图片文字批量替换"
