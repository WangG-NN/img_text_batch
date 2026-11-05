#!/bin/bash
# 双击即可在 Mac 上自动打包（若提示无法打开，右键→打开）
cd "$(dirname "$0")"
bash build_mac.sh
echo ""
read -p "打包结束，按回车关闭窗口" _
