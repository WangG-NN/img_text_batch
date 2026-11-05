@echo off
REM 在 Windows 上一键打包"图片文字批量替换"
REM 用法：双击本文件，或在项目根目录执行：build_windows.bat
setlocal
cd /d "%~dp0"

echo [1/3] 安装依赖...
python -m pip install --upgrade pip -q
python -m pip install -q -r requirements.txt -r requirements-dev.txt

echo [2/3] 准备 LaMa 模型（约 208MB，已有则跳过）...
if not exist "models\lama_fp32.onnx" (
  curl -L --retry 3 -o "models\lama_fp32.onnx" "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx"
)

echo [3/3] 打包...
python -m PyInstaller --noconfirm --clean --onefile --windowed --name "图片文字批量替换" --collect-all rapidocr_onnxruntime --hidden-import lama app.py

echo.
echo 完成！可执行文件：dist\图片文字批量替换.exe
echo 首次运行前把 models\lama_fp32.onnx 拷贝到 dist\models\ 目录。
pause
