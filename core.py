# -*- coding: utf-8 -*-
"""核心处理：抹除模板旧文字 + 渲染新文字行"""
import json
import os
import re

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_CONFIG = {
    # 模式: fixed=按固定文字块抹字+画字; ocr=自动检测模板上的文字再抹除
    "mode": "ocr",
    # 文字块位置（占图片宽高的比例，0~1）
    "text_block": {"x": 0.05, "y": 0.84, "w": 0.90, "h": 0.14, "align": "center"},
    "font_size": 30,
    "color": "#FFFFFF",
    "line_spacing": 1.5,
    # 是否抹除旧文字（模板本身是干净底图时可关掉）
    "erase": True,
    # 抹除引擎: lama=高质量深度修复（需 models/big-lama.onnx，缺模型自动回退 opencv）; opencv=快速
    "erase_engine": "lama",
    # OCR 模式下：False=只抹文字块附近的文字（推荐）；True=抹全图检测到的所有文字
    "ocr_erase_all": False,
    # 上次选择的文件夹与字体（自动记忆）
    "paths": {"tpl": "", "src": "", "out": "", "font": ""},
}

# 没选字体时按顺序尝试的系统字体（Windows / macOS / Linux）
_FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyh.ttf",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
]


def find_default_font():
    """找一个可用的系统中文字体，找不到返回 None"""
    for p in _FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def load_config(path):
    """读取 config.json，缺失字段用默认值补齐"""
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                user = json.load(f)
            for k, v in user.items():
                if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    cfg[k].update(v)
                else:
                    cfg[k] = v
        except Exception:
            pass
    return cfg


def save_config(path, cfg):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def imread(path):
    """支持中文路径的图片读取"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite(path, img_bgr):
    """支持中文路径的图片保存"""
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img_bgr)
    if not ok:
        raise RuntimeError("图片编码失败: " + path)
    buf.tofile(path)


def _hex_to_rgb(s):
    s = str(s).lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def _block_rect_px(img_shape, block):
    h, w = img_shape[:2]
    x = int(block["x"] * w)
    y = int(block["y"] * h)
    bw = int(block["w"] * w)
    bh = int(block["h"] * h)
    return x, y, max(1, bw), max(1, bh)


def _text_mask_in_roi(roi_gray):
    """两段式找文字像素：
    1) 与局部背景差分很大的（亮字/深字）直接算文字；
    2) 比背景暗的弱差异像素只有紧挨着文字（±10px 内）才算投影阴影，
    远处的云雾纹理不受影响，避免误抹出斑点。返回 None 表示基本没有文字。"""
    if roi_gray.size == 0:
        return None
    bg = cv2.medianBlur(roi_gray, 21)
    diff = cv2.absdiff(roi_gray, bg)
    _, strong = cv2.threshold(diff, 60, 255, cv2.THRESH_BINARY)
    if float(strong.mean()) / 255.0 < 0.004:
        return None  # 没有明显文字
    dark = cv2.subtract(bg, roi_gray)
    _, weak_dark = cv2.threshold(dark, 22, 255, cv2.THRESH_BINARY)
    near = cv2.dilate(strong, np.ones((21, 21), np.uint8))
    m = cv2.bitwise_or(strong, cv2.bitwise_and(weak_dark, near))
    if float(m.mean()) / 255.0 > 0.45:
        return None  # 异常：几乎整块都被标记，宁可不抹
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    m = cv2.dilate(m, np.ones((5, 5), np.uint8), iterations=1)
    return m


def _inpaint_feather(img_bgr, mask):
    """TELEA 修复 + 1px 边缘羽化（只平滑边界跳变，不把原字混回）"""
    if not mask.any():
        return img_bgr
    out = cv2.inpaint(img_bgr, mask, 5, cv2.INPAINT_TELEA)
    alpha = cv2.GaussianBlur(mask, (0, 0), 1).astype(np.float32) / 255.0
    alpha = alpha[..., None]
    blend = img_bgr.astype(np.float32) * (1 - alpha) + out.astype(np.float32) * alpha
    return np.clip(blend, 0, 255).astype(np.uint8)


def _do_inpaint(img_bgr, mask, engine="opencv"):
    """按引擎抹除 mask 区域：lama 深度修复（缺模型/失败自动回退），否则快速羽化修复"""
    if mask is None or not mask.any():
        return img_bgr
    if engine == "lama":
        try:
            import lama
            if lama.available():
                return lama.lama_inpaint(img_bgr, mask)
        except Exception:
            pass
    return _inpaint_feather(img_bgr, mask)


def erase_boxes(img_bgr, boxes, engine="opencv"):
    """OCR 框内只抹文字像素（不抹框内背景），再局部修复，避免整框涂抹感。
    需要限定范围时，先在外层按区域过滤 boxes。"""
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((h, w), np.uint8)
    for (bx, by, bw, bh) in boxes:
        # 文字阴影通常偏下，下方多留一些余量
        x0 = max(0, int(bx) - 5)
        y0 = max(0, int(by) - 4)
        x1 = min(w, int(bx + bw) + 5)
        y1 = min(h, int(by + bh) + 10)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        m = _text_mask_in_roi(gray[y0:y1, x0:x1])
        if m is not None:
            mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], m)
    return _do_inpaint(img_bgr, mask, engine)


def erase_block_auto(img_bgr, block, engine="opencv"):
    """固定区域模式：只抹文字块里"像文字"的像素（与背景分离出来的少数派），
    背景像素完全不动。"""
    x, y, bw, bh = _block_rect_px(img_bgr.shape, block)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    m = _text_mask_in_roi(gray[y:y + bh, x:x + bw])
    if m is None:
        return img_bgr
    mask = np.zeros_like(gray)
    mask[y:y + bh, x:x + bw] = m
    return _do_inpaint(img_bgr, mask, engine)


def _wrap(text, draw, font, max_w):
    """按字符折行（适合中文），返回视觉行列表"""
    rows = []
    cur = ""
    for ch in text:
        if draw.textlength(cur + ch, font=font) <= max_w:
            cur += ch
        else:
            rows.append(cur)
            cur = ch
    if cur:
        rows.append(cur)
    return rows or [""]


_SIZE_MARK = re.compile(r"\s*[（(\[]\s*(\d{1,3})\s*[）)\]]\s*$")


def _parse_line(text):
    """解析行尾的（40）/(40)/[40] 标记为该行的单独字号，返回 (文字, 字号或None)"""
    m = _SIZE_MARK.search(text)
    if m:
        return text[:m.start()].rstrip(), int(m.group(1))
    return text, None


def render_lines(img_bgr, lines, cfg, font_path):
    """在文字块内绘制新文字。
    - 行尾加（数字）单独设置该行字号，如：最高保额300万（40）
    - 空行保留，用作行间间隔
    - 总高度超出文字块时，所有行按比例整体缩小"""
    parsed = [_parse_line(str(t)) for t in lines]
    while parsed and parsed[-1][0].strip() == "":
        parsed.pop()
    if not parsed:
        return img_bgr
    block = cfg["text_block"]
    x, y, bw, bh = _block_rect_px(img_bgr.shape, block)
    color = _hex_to_rgb(cfg.get("color", "#FFFFFF"))
    spacing = float(cfg.get("line_spacing", 1.5))
    align = str(block.get("align", "center")).lower()

    img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    font_cache = {}

    def get_font(size):
        size = max(8, size)
        if size not in font_cache:
            font_cache[size] = ImageFont.truetype(font_path, size)
        return font_cache[size]

    default_size = max(10, int(cfg.get("font_size", 30)))
    scale = 1.0
    while True:
        plan = []  # 每个源行 -> (视觉行列表, 字体, 行高)
        total_h = 0
        for text, size in parsed:
            si = int((size or default_size) * scale)
            font = get_font(si)
            line_h = max(si + 2, int(si * spacing))
            rows = [""] if text.strip() == "" else _wrap(text, draw, font, bw)
            plan.append((rows, font, line_h))
            total_h += line_h * len(rows)
        if scale <= 0.3 or total_h <= bh:
            break
        scale -= 0.05

    cy = y + max(0, (bh - total_h) // 2)
    for rows, font, line_h in plan:
        for t in rows:
            tw = draw.textlength(t, font=font)
            if align == "left":
                tx = x
            elif align == "right":
                tx = x + bw - tw
            else:
                tx = x + (bw - tw) / 2
            draw.text((tx, cy), t, font=font, fill=color)
            cy += line_h
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def prepare_template(template_path, cfg, ocr_detect=None):
    """读取模板并抹除旧文字，返回抹好的图像。
    同一模板配多个文字源时，只需抹一次，结果可复用（省时间也省 CPU）。"""
    img = imread(template_path)
    if img is None:
        raise RuntimeError("无法读取图片: " + template_path)
    if cfg.get("erase", True):
        if cfg.get("mode") == "ocr" and ocr_detect is not None:
            boxes = ocr_detect(img)
            region = None if cfg.get("ocr_erase_all") else _block_rect_px(img.shape, cfg["text_block"])
            boxes = [b for b in boxes if region is None or (
                b[0] + b[2] >= region[0] and b[0] <= region[0] + region[2]
                and b[1] + b[3] >= region[1] and b[1] <= region[1] + region[3])]
            if boxes:
                img = erase_boxes(img, boxes, engine=cfg.get("erase_engine", "opencv"))
        else:
            img = erase_block_auto(img, cfg["text_block"], engine=cfg.get("erase_engine", "opencv"))
    return img


def render_to_file(erased_img, lines, cfg, font_path, out_path):
    """在已抹除旧字的模板上绘制新文字并保存"""
    img = render_lines(erased_img, lines, cfg, font_path)
    imwrite(out_path, img)
    return out_path


def process_one(template_path, lines, cfg, font_path, out_path, ocr_detect=None):
    """处理单张模板：抹旧字 -> 画新字 -> 保存"""
    img = prepare_template(template_path, cfg, ocr_detect=ocr_detect)
    return render_to_file(img, lines, cfg, font_path, out_path)
