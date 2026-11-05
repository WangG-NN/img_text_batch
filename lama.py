# -*- coding: utf-8 -*-
"""LaMa 高质量文字抹除（ONNX 版，CPU 推理）。
模型文件 lama_fp32.onnx 放在 exe 旁边的 models 目录；找不到时自动回退 OpenCV。
模型固定 512x512 输入，对抹除区域分块推理再拼回。"""
import os
import threading

import cv2
import numpy as np

_session = None
_lock = threading.Lock()
_TILE = 512
_OVERLAP = 32
_progress = None


def set_progress(fn):
    """设置分块进度回调 fn(当前块, 总块数)，供界面显示"""
    global _progress
    _progress = fn


def warmup():
    """后台预热：提前加载模型会话，第一次抹字不用再等十几秒"""
    try:
        if available():
            _get_session()
    except Exception:
        pass


def _base_dir():
    import sys
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def model_path():
    names = ("lama_fp32.onnx", "big-lama.onnx")
    dirs = [os.path.join(_base_dir(), "models"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")]
    for d in dirs:
        for n in names:
            p = os.path.join(d, n)
            if os.path.isfile(p):
                return p
    return None


def available():
    return model_path() is not None


def _get_session():
    global _session
    with _lock:
        if _session is None:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = 2
            so.inter_op_num_threads = 1
            _session = ort.InferenceSession(model_path(), so,
                                            providers=["CPUExecutionProvider"])
    return _session


def _starts(a, b):
    """把 [a,b) 切成 512 宽、带重叠的窗口起点"""
    if b - a <= _TILE:
        return [a]
    step = _TILE - _OVERLAP
    out = list(range(a, b - _TILE, step))
    out.append(b - _TILE)
    return out


def lama_inpaint(img_bgr, mask):
    """只修复 mask 覆盖的像素，区域外原图不动。
    img_bgr: uint8 BGR; mask: uint8 0/255。"""
    if mask is None or not mask.any():
        return img_bgr
    sess = _get_session()
    h, w = img_bgr.shape[:2]
    ys, xs = np.where(mask > 0)
    pad = 64
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(w, int(xs.max()) + 1 + pad)
    y1 = min(h, int(ys.max()) + 1 + pad)

    result = img_bgr.copy()
    # LaMa 生成的背景与原图有细微差异，mask 需额外膨胀盖住笔画边缘/抗锯齿，避免拼接浮雕感
    mask_wide = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=1)
    xs_list = _starts(x0, x1)
    ys_list = _starts(y0, y1)
    total = len(xs_list) * len(ys_list)
    idx = 0
    for ty in ys_list:
        for tx in xs_list:
            idx += 1
            if _progress:
                try:
                    _progress(idx, total)
                except Exception:
                    pass
            ch = min(_TILE, h - ty)
            cw = min(_TILE, w - tx)
            crop = result[ty:ty + ch, tx:tx + cw]
            m = mask_wide[ty:ty + ch, tx:tx + cw]
            if not m.any():
                continue
            crop8 = crop if (ch == _TILE and cw == _TILE) else cv2.copyMakeBorder(
                crop, 0, _TILE - ch, 0, _TILE - cw, cv2.BORDER_REFLECT)
            m8 = m if (ch == _TILE and cw == _TILE) else cv2.copyMakeBorder(
                m, 0, _TILE - ch, 0, _TILE - cw, cv2.BORDER_CONSTANT, value=0)
            rgb = cv2.cvtColor(crop8, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            x = np.transpose(rgb, (2, 0, 1))[None]
            mm = (m8 > 127).astype(np.float32)[None, None]
            out = sess.run(None, {"image": x, "mask": mm})[0][0]
            out = np.clip(np.transpose(out, (1, 2, 0)), 0, 255).astype(np.uint8)
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)[:ch, :cw]
            region = result[ty:ty + ch, tx:tx + cw]
            result[ty:ty + ch, tx:tx + cw] = np.where((m > 0)[..., None], out, region)
    return result
