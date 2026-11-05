# -*- coding: utf-8 -*-
"""RapidOCR 封装：识别图片文字行、检测模板上的文字框"""
import threading

_ocr = None
_lock = threading.Lock()


def get_ocr():
    """懒加载 OCR 引擎（首次调用会加载模型，稍慢）"""
    global _ocr
    with _lock:
        if _ocr is None:
            from rapidocr_onnxruntime import RapidOCR
            # 限制推理线程数，避免吃满全部 CPU 导致电脑卡顿
            _ocr = RapidOCR(intra_op_num_threads=2)
    return _ocr


def _boxes_of(result):
    out = []
    for box, text, conf in result:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        out.append({
            "x": min(xs), "y": min(ys),
            "w": max(xs) - min(xs), "h": max(ys) - min(ys),
            "cx": sum(xs) / len(xs), "cy": sum(ys) / len(ys),
            "text": str(text),
        })
    return out


def recognize_lines(img_bgr):
    """识别整张图的所有文字，按行从上到下返回文本列表"""
    result, _ = get_ocr()(img_bgr)
    if not result:
        return []
    items = _boxes_of(result)
    items.sort(key=lambda t: t["cy"])
    # 按行中心 y 聚类：与当前行中心差小于平均字高即视为同一行
    lines = []
    cur = []
    cur_y = None
    avg_h = sum(t["h"] for t in items) / len(items)
    for t in items:
        if cur_y is None or abs(t["cy"] - cur_y) < max(8, avg_h * 0.6):
            cur.append(t)
            cur_y = t["cy"] if cur_y is None else (cur_y * (len(cur) - 1) + t["cy"]) / len(cur)
        else:
            lines.append(cur)
            cur = [t]
            cur_y = t["cy"]
    if cur:
        lines.append(cur)
    return ["".join(t["text"] for t in sorted(g, key=lambda t: t["cx"])) for g in lines]


def detect_boxes(img_bgr):
    """检测图片上的文字框，返回 [(x, y, w, h), ...]，用于抹除模板旧文字"""
    result, _ = get_ocr()(img_bgr)
    if not result:
        return []
    return [(t["x"], t["y"], t["w"], t["h"]) for t in _boxes_of(result)]
