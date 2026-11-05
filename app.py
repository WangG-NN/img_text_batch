# -*- coding: utf-8 -*-
"""图片文字批量替换工具（GUI 入口）

用法：选择 模板文件夹 / 输出文件夹，
文字源通过"添加图片识别"或"新建文本手输"两种方式创建，
批量生成到 输出/文字源名/001.png ...
"""
import os
import queue
import sys
import threading
import traceback

import tkinter as tk
from tkinter import colorchooser, filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

import core
import ocr

# 限制 OpenCV 线程数，避免批量处理时吃满全部 CPU 导致电脑卡顿
cv2.setNumThreads(2)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def app_dir():
    """exe 或脚本所在目录（config.json 存放位置）"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def list_images(folder):
    if not folder or not os.path.isdir(folder):
        return []
    return sorted(
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    )


class App:
    def __init__(self, root):
        self.root = root
        root.title("图片文字批量替换工具")
        root.geometry("960x700")
        root.minsize(820, 620)

        self.cfg_path = os.path.join(app_dir(), "config.json")
        self.cfg = core.load_config(self.cfg_path)

        self.source_texts = {}   # 文字源文件名 -> 文字内容
        self.source_paths = {}   # 文字源文件名 -> 图片完整路径
        self.selected_source = None
        self.preview_photo = None
        self.msg_queue = queue.Queue()
        self.running = False

        self._build_ui()
        self._load_cfg_to_ui()
        self._restore_paths()
        if self.cfg.get("erase_engine", "lama") == "lama":
            try:
                import lama
                if not lama.available():
                    self.var_status.set("未找到 LaMa 模型（models/lama_fp32.onnx），将自动使用快速引擎")
                else:
                    # 后台预热模型，第一次点生成不用等加载
                    threading.Thread(target=lama.warmup, daemon=True).start()
            except Exception:
                pass
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(120, self._poll_queue)

    # ---------- 界面 ----------
    def _build_ui(self):
        pad = dict(padx=6, pady=4)
        top = ttk.Frame(self.root)
        top.pack(fill="x", **pad)

        self.var_tpl = tk.StringVar()
        self.var_clean = tk.StringVar()
        self.var_out = tk.StringVar()
        self.var_font = tk.StringVar()
        rows = [
            ("模板文件夹（图1）", self.var_tpl, self._pick_tpl),
            ("干净底图文件夹（可选）", self.var_clean, self._pick_clean),
            ("输出文件夹", self.var_out, self._pick_out),
            ("字体文件（可留空用默认）", self.var_font, self._pick_font),
        ]
        for i, (label, var, cmd) in enumerate(rows):
            ttk.Label(top, text=label).grid(row=i, column=0, sticky="e", **pad)
            ttk.Entry(top, textvariable=var, width=70).grid(row=i, column=1, sticky="we", **pad)
            ttk.Button(top, text="浏览…", command=cmd, width=8).grid(row=i, column=2, **pad)
        top.columnconfigure(1, weight=1)

        # 格式设置
        gf = ttk.LabelFrame(self.root, text="格式设置（保存在 config.json）")
        gf.pack(fill="x", **pad)
        self.var_mode = tk.StringVar(value=self.cfg["mode"])
        ttk.Radiobutton(gf, text="固定区域", variable=self.var_mode, value="fixed").grid(row=0, column=0, padx=6, pady=4)
        ttk.Radiobutton(gf, text="OCR检测模板文字", variable=self.var_mode, value="ocr").grid(row=0, column=1, padx=6)
        self.var_erase = tk.BooleanVar(value=True)
        ttk.Checkbutton(gf, text="抹除旧文字", variable=self.var_erase).grid(row=0, column=2, padx=6)
        self.var_erase_all = tk.BooleanVar(value=False)
        ttk.Checkbutton(gf, text="抹除全图文字(OCR模式)", variable=self.var_erase_all).grid(row=0, column=3, columnspan=2, padx=6)

        self.vars_blk = {}
        for j, key in enumerate(["x", "y", "w", "h"]):
            ttk.Label(gf, text=f"文字块{key}(%)").grid(row=1, column=j * 2, sticky="e", padx=(10, 2))
            v = tk.StringVar()
            self.vars_blk[key] = v
            ttk.Entry(gf, textvariable=v, width=7).grid(row=1, column=j * 2 + 1, sticky="w")
        ttk.Label(gf, text="对齐").grid(row=1, column=8, sticky="e", padx=(10, 2))
        self.var_align = tk.StringVar()
        ttk.Combobox(gf, textvariable=self.var_align, values=["center", "left", "right"], width=7, state="readonly").grid(row=1, column=9)

        ttk.Label(gf, text="字号").grid(row=2, column=0, sticky="e", padx=(10, 2))
        self.var_size = tk.StringVar()
        ttk.Spinbox(gf, from_=8, to=300, textvariable=self.var_size, width=6).grid(row=2, column=1, sticky="w")
        ttk.Label(gf, text="行距").grid(row=2, column=2, sticky="e")
        self.var_lsp = tk.StringVar()
        ttk.Entry(gf, textvariable=self.var_lsp, width=6).grid(row=2, column=3, sticky="w")
        ttk.Label(gf, text="颜色").grid(row=2, column=4, sticky="e", padx=(10, 2))
        self.btn_color = tk.Button(gf, text="#FFFFFF", width=9, command=self._pick_color)
        self.btn_color.grid(row=2, column=5)
        ttk.Label(gf, text="抹除引擎").grid(row=2, column=6, sticky="e", padx=(10, 2))
        self.var_engine = tk.StringVar()
        ttk.Combobox(gf, textvariable=self.var_engine, values=["高质量(LaMa)", "快速(OpenCV)"],
                     width=12, state="readonly").grid(row=2, column=7, sticky="w")
        ttk.Button(gf, text="保存设置", command=self._save_cfg_ui, width=10).grid(row=2, column=9, pady=4)

        # 底部执行条：先以 side=bottom 打包，保证窗口再小也始终可见
        bottom = ttk.Frame(self.root)
        bottom.pack(side="bottom", fill="x", **pad)
        self.btn_run = ttk.Button(bottom, text="开始批量生成", command=self._start_run, width=16)
        self.btn_run.pack(side="left", padx=6)
        ttk.Button(bottom, text="打开输出文件夹", command=self._open_out, width=14).pack(side="left", padx=6)
        self.progress = ttk.Progressbar(bottom, length=260, mode="determinate")
        self.progress.pack(side="left", padx=10)
        self.var_status = tk.StringVar(value="就绪")
        ttk.Label(bottom, textvariable=self.var_status).pack(side="left", padx=6)

        # 中部：左文字源两组列表，右预览+文字编辑
        mid = ttk.PanedWindow(self.root, orient="horizontal")
        mid.pack(fill="both", expand=True, **pad)

        left = ttk.Frame(mid)
        mid.add(left, weight=1)

        def make_list(parent, height):
            frame = ttk.Frame(parent)
            frame.pack(fill="both", expand=True, padx=4, pady=(4, 0))
            lb = tk.Listbox(frame, height=height)
            sb = ttk.Scrollbar(frame, command=lb.yview)
            lb.config(yscrollcommand=sb.set)
            lb.pack(side="left", fill="both", expand=True)
            sb.pack(side="right", fill="y")
            return frame, lb

        # 第一组：图片识别
        g1 = ttk.LabelFrame(left, text="图片文字源（添加图片 → 识别）")
        g1.pack(fill="both", expand=True, padx=4, pady=4)
        _, self.lb_img = make_list(g1, 5)
        self.lb_img.bind("<<ListboxSelect>>", self._on_img_select)
        b1 = ttk.Frame(g1)
        b1.pack(fill="x", padx=4, pady=3)
        ttk.Button(b1, text="添加图片", command=self._add_source_images).pack(side="left", expand=True, fill="x", padx=1)
        ttk.Button(b1, text="识别选中", command=self._recognize).pack(side="left", expand=True, fill="x", padx=1)
        ttk.Button(b1, text="删除选中", command=self._del_img).pack(side="left", expand=True, fill="x", padx=1)

        # 第二组：手动文本
        g2 = ttk.LabelFrame(left, text="文字文字源（新建 → 手动输入）")
        g2.pack(fill="both", expand=True, padx=4, pady=4)
        _, self.lb_txt = make_list(g2, 5)
        self.lb_txt.bind("<<ListboxSelect>>", self._on_txt_select)
        b2 = ttk.Frame(g2)
        b2.pack(fill="x", padx=4, pady=3)
        ttk.Button(b2, text="新建文本", command=self._add_text_source).pack(side="left", expand=True, fill="x", padx=1)
        ttk.Button(b2, text="删除选中", command=self._del_txt).pack(side="left", expand=True, fill="x", padx=1)
        ttk.Label(left, text="识别结果可在右侧继续修改；每个文字源对应输出里的一个子文件夹。",
                  foreground="#888").pack(anchor="w", padx=8, pady=(0, 2))

        right = ttk.LabelFrame(mid, text="预览与文字")
        mid.add(right, weight=2)
        self.canvas = tk.Canvas(right, height=240, background="#202020", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=6, pady=6)
        tip = tk.Frame(right, background="#FFF7DA")
        tip.pack(fill="x", padx=6, pady=(0, 2))
        tk.Label(tip, text="★ 每行字号怎么设？", font=("Microsoft YaHei", 10, "bold"),
                 background="#FFF7DA", foreground="#B45309").pack(anchor="w", padx=8, pady=(4, 0))
        tk.Label(tip, text="在这一行文字末尾加（数字）即可，如：最高保额300万（40）→ 该行用 40 号字；不加则用上方格式设置里的字号。空行会保留作间隔。",
                 font=("Microsoft YaHei", 9), background="#FFF7DA", foreground="#92400E",
                 wraplength=560, justify="left").pack(anchor="w", padx=8, pady=(0, 4))
        self.txt = tk.Text(right, height=7, wrap="char", font=("Microsoft YaHei", 11))
        self.txt.pack(fill="x", padx=6, pady=(0, 2))
        self.txt.bind("<<Modified>>", self._on_text_edit)

    # ---------- 配置 ----------
    def _load_cfg_to_ui(self):
        tb = self.cfg["text_block"]
        self.var_mode.set(self.cfg.get("mode", "fixed"))
        self.var_erase.set(bool(self.cfg.get("erase", True)))
        self.var_erase_all.set(bool(self.cfg.get("ocr_erase_all", False)))
        self.var_engine.set("高质量(LaMa)" if self.cfg.get("erase_engine", "lama") == "lama" else "快速(OpenCV)")
        for k in ["x", "y", "w", "h"]:
            self.vars_blk[k].set(str(round(tb[k] * 100, 2)))
        self.var_align.set(tb.get("align", "center"))
        self.var_size.set(str(self.cfg.get("font_size", 30)))
        self.var_lsp.set(str(self.cfg.get("line_spacing", 1.5)))
        self._set_color_btn(self.cfg.get("color", "#FFFFFF"))

    def _set_color_btn(self, hexv):
        self._color = hexv
        self.btn_color.config(text=hexv, bg=hexv,
                              fg="#000000" if hexv.lower() in ("#ffffff", "#fff", "#ffffe0") else "#FFFFFF")

    def _pick_color(self):
        c = colorchooser.askcolor(self._color)
        if c and c[1]:
            self._set_color_btn(c[1])

    def _ui_to_cfg(self):
        cfg = core.load_config(self.cfg_path)
        cfg["mode"] = self.var_mode.get()
        cfg["erase"] = bool(self.var_erase.get())
        cfg["ocr_erase_all"] = bool(self.var_erase_all.get())
        cfg["erase_engine"] = "lama" if "LaMa" in self.var_engine.get() else "opencv"
        tb = cfg["text_block"]
        for k in ["x", "y", "w", "h"]:
            tb[k] = float(self.vars_blk[k].get()) / 100.0
        tb["align"] = self.var_align.get()
        cfg["font_size"] = int(float(self.var_size.get()))
        cfg["line_spacing"] = float(self.var_lsp.get())
        cfg["color"] = self._color
        return cfg

    def _save_cfg_ui(self):
        try:
            self._save_state()
        except ValueError:
            messagebox.showerror("格式有误", "请检查文字块坐标/字号/行距是否为数字")
            return
        messagebox.showinfo("已保存", "设置已保存到\n" + self.cfg_path)

    def _save_state(self):
        """把格式设置 + 上次选择的文件夹/字体写入 config.json"""
        cfg = self._ui_to_cfg()
        cfg["paths"] = {
            "tpl": self.var_tpl.get(),
            "clean": self.var_clean.get(),
            "out": self.var_out.get(),
            "font": self.var_font.get(),
        }
        core.save_config(self.cfg_path, cfg)
        self.cfg = cfg

    def _restore_paths(self):
        p = self.cfg.get("paths") or {}
        self.var_tpl.set(p.get("tpl", ""))
        self.var_clean.set(p.get("clean", ""))
        self.var_out.set(p.get("out", ""))
        self.var_font.set(p.get("font", ""))

    def _on_close(self):
        try:
            self._save_state()
        except Exception:
            pass
        self.root.destroy()

    # ---------- 文件夹选择 ----------
    def _pick_folder(self, var, after=None):
        p = filedialog.askdirectory()
        if p:
            var.set(os.path.normpath(p))
            self._save_state_safe()
            if after:
                after()

    def _save_state_safe(self):
        try:
            self._save_state()
        except Exception:
            pass

    def _pick_tpl(self):
        self._pick_folder(self.var_tpl)

    def _pick_clean(self):
        self._pick_folder(self.var_clean)

    def _pick_out(self):
        self._pick_folder(self.var_out)

    def _pick_font(self):
        p = filedialog.askopenfilename(
            title="选择字体文件",
            filetypes=[("字体文件", "*.ttf *.otf *.ttc"), ("所有文件", "*.*")])
        if p:
            self.var_font.set(os.path.normpath(p))
            self._save_state_safe()

    # ---------- 文字源 ----------
    def _add_source_images(self):
        """第一组：选图片加入列表，之后可识别文字"""
        paths = filedialog.askopenfilenames(
            title="添加文字源图片",
            filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.webp"), ("所有文件", "*.*")])
        for p in paths:
            name = os.path.basename(p)
            if name not in self.source_paths:
                self.lb_img.insert("end", name)
                self.source_texts[name] = ""
            self.source_paths[name] = os.path.normpath(p)

    def _add_text_source(self):
        """第二组：新建一个不依赖图片的纯文字源"""
        i = 1
        while ("自定义文字%d" % i) in self.source_paths:
            i += 1
        name = "自定义文字%d" % i
        self.source_paths[name] = None
        self.source_texts[name] = ""
        self.lb_txt.insert("end", name)
        idx = self.lb_txt.size() - 1
        self.lb_txt.selection_clear(0, "end")
        self.lb_txt.selection_set(idx)
        self.lb_txt.see(idx)
        self._select_source(name)
        self.txt.focus_set()

    def _del_img(self):
        self._delete_selected(self.lb_img)

    def _del_txt(self):
        self._delete_selected(self.lb_txt)

    def _delete_selected(self, lb):
        sel = lb.curselection()
        if not sel:
            return
        name = lb.get(sel[0])
        lb.delete(sel[0])
        self.source_paths.pop(name, None)
        self.source_texts.pop(name, None)
        if self.selected_source == name:
            self.selected_source = None
            self.txt.delete("1.0", "end")
            self.txt.edit_modified(False)
            self._show_preview(None)

    def _on_img_select(self, _evt):
        sel = self.lb_img.curselection()
        if not sel:
            return
        self.lb_txt.selection_clear(0, "end")
        self._select_source(self.lb_img.get(sel[0]))

    def _on_txt_select(self, _evt):
        sel = self.lb_txt.curselection()
        if not sel:
            return
        self.lb_img.selection_clear(0, "end")
        self._select_source(self.lb_txt.get(sel[0]))

    def _select_source(self, name):
        self._flush_text_buffer()
        self.selected_source = name
        self._show_preview(self.source_paths.get(name))
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", self.source_texts.get(name, ""))
        self.txt.edit_modified(False)

    def _on_text_edit(self, _evt):
        if self.txt.edit_modified() and self.selected_source:
            self.source_texts[self.selected_source] = self.txt.get("1.0", "end-1c")

    def _flush_text_buffer(self):
        if self.selected_source:
            self.source_texts[self.selected_source] = self.txt.get("1.0", "end-1c")

    def _show_preview(self, path):
        self.canvas.delete("all")
        self.preview_photo = None
        if not path or not os.path.isfile(path):
            return
        img = core.imread(path)
        if img is None:
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img)
        cw = max(200, self.canvas.winfo_width())
        ch = max(120, self.canvas.winfo_height())
        if ch < 100:
            ch = 240
        pil.thumbnail((cw, ch))
        self.preview_photo = ImageTk.PhotoImage(pil)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self.preview_photo)

    def _recognize(self):
        if not self.selected_source:
            messagebox.showinfo("提示", "请先在左侧选择一张文字源图片")
            return
        name = self.selected_source
        path = self.source_paths.get(name)
        if not path or not os.path.isfile(path):
            messagebox.showinfo("提示", "这是纯文字源，不需要识别，直接在右侧输入文字即可")
            return
        self.var_status.set("正在识别文字（首次加载模型稍慢）…")
        self.root.config(cursor="watch")

        def work():
            try:
                img = core.imread(path)
                lines = ocr.recognize_lines(img)
                self.msg_queue.put(("ocr_done", name, "\n".join(lines)))
            except Exception:
                self.msg_queue.put(("error", traceback.format_exc()))

        threading.Thread(target=work, daemon=True).start()

    # ---------- 批量执行 ----------
    def _start_run(self):
        if self.running:
            return
        try:
            cfg = self._ui_to_cfg()
        except ValueError:
            messagebox.showerror("格式有误", "请检查文字块坐标/字号/行距是否为数字")
            return
        self._flush_text_buffer()
        tpl_dir = self.var_tpl.get()
        clean_dir = self.var_clean.get()
        out_dir = self.var_out.get()
        font_path = self.var_font.get()
        templates = list_images(tpl_dir)
        sources = {k: v for k, v in self.source_texts.items() if v.strip()}
        if not templates:
            messagebox.showerror("缺少模板", "模板文件夹里没有图片")
            return
        if not sources:
            messagebox.showerror("缺少文字", "请先选择文字源图片并填写/识别文字")
            return
        if not font_path or not os.path.isfile(font_path):
            font_path = core.find_default_font()   # 没选字体就用系统默认中文字体
            if not font_path:
                messagebox.showerror("缺少字体", "未选择字体，且系统中找不到可用中文字体")
                return
            self.var_status.set("未选字体，已自动使用系统默认字体")
        if not out_dir:
            messagebox.showerror("缺少输出", "请选择输出文件夹")
            return
        os.makedirs(out_dir, exist_ok=True)

        # 每个文字源 -> (输出子文件夹, 文字行)
        src_jobs = []
        for src_name, text in sorted(sources.items()):
            lines = text.splitlines()   # 保留空行，由 core 处理
            sub = os.path.join(out_dir, os.path.splitext(src_name)[0])
            os.makedirs(sub, exist_ok=True)
            src_jobs.append((sub, lines))
        self.running = True
        self.btn_run.config(state="disabled")
        self.progress.config(maximum=len(templates) + len(templates) * len(src_jobs), value=0)
        self.var_status.set("开始处理：模板 %d 张 × 文字源 %d 个" % (len(templates), len(src_jobs)))

        def work():
            detect = ocr.detect_boxes if cfg.get("mode") == "ocr" else None
            if cfg.get("erase_engine") == "lama":
                try:
                    import lama
                    lama.set_progress(lambda i, n: self.msg_queue.put(
                        ("status", "LaMa 高质量抹除中：第 %d/%d 块（每块约 10~20 秒）" % (i, n))))
                except Exception:
                    pass
            ok, fail = 0, 0
            # 第一阶段：每张模板准备一次（优先用同名干净底图，否则抹旧字），结果缓存复用
            prepared = []
            for i, tpl in enumerate(templates, 1):
                try:
                    self.msg_queue.put(("status", "准备模板 %d/%d：%s" % (i, len(templates), tpl)))
                    img = None
                    if clean_dir:
                        stem = os.path.splitext(tpl)[0]
                        for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                            cp = os.path.join(clean_dir, stem + ext)
                            if os.path.isfile(cp):
                                img = core.imread(cp)
                                break
                    if img is None:
                        img = core.prepare_template(os.path.join(tpl_dir, tpl), cfg, ocr_detect=detect)
                    else:
                        # 底图与模板尺寸不一致时对齐到模板尺寸
                        timg = core.imread(os.path.join(tpl_dir, tpl))
                        if timg is not None and img.shape[:2] != timg.shape[:2]:
                            img = cv2.resize(img, (timg.shape[1], timg.shape[0]))
                    prepared.append((tpl, img))
                except Exception:
                    fail += 1
                self.msg_queue.put(("progress", i, len(templates) + len(templates) * len(src_jobs),
                                    "准备模板 %d/%d" % (i, len(templates))))
            # 第二阶段：在抹好的模板上画新文字并保存
            total = len(prepared) * len(src_jobs)
            done = len(templates)
            for sub, lines in src_jobs:
                for i, (tpl, img) in enumerate(prepared, 1):
                    done += 1
                    try:
                        core.render_to_file(img, lines, cfg, font_path,
                                            os.path.join(sub, "%03d.png" % i))
                        ok += 1
                    except Exception:
                        fail += 1
                    self.msg_queue.put(("progress", done, len(templates) + total,
                                        "生成 %s %d/%d" % (os.path.basename(sub), i, len(prepared))))
            self.msg_queue.put(("done", ok, fail))

        threading.Thread(target=work, daemon=True).start()

    def _poll_queue(self):
        try:
            while True:
                msg = self.msg_queue.get_nowait()
                kind = msg[0]
                if kind == "ocr_done":
                    _, name, text = msg
                    self.source_texts[name] = text
                    if self.selected_source == name:
                        self.txt.delete("1.0", "end")
                        self.txt.insert("1.0", text)
                        self.txt.edit_modified(False)
                    self.var_status.set("识别完成")
                    self.root.config(cursor="")
                elif kind == "progress":
                    self.progress.config(value=msg[1])
                    self.var_status.set("%s（总进度 %d/%d）" % (msg[3], msg[1], msg[2]))
                elif kind == "status":
                    self.var_status.set(msg[1])
                elif kind == "done":
                    self.running = False
                    self.btn_run.config(state="normal")
                    self.var_status.set("完成：成功 %d，失败 %d" % (msg[1], msg[2]))
                elif kind == "error":
                    self.root.config(cursor="")
                    self.var_status.set("出错")
                    messagebox.showerror("出错", msg[1])
        except queue.Empty:
            pass
        self.root.after(120, self._poll_queue)

    def _open_out(self):
        p = self.var_out.get()
        if p and os.path.isdir(p):
            import subprocess
            import sys
            try:
                if sys.platform == "win32":
                    os.startfile(p)
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", p])
                else:
                    subprocess.Popen(["xdg-open", p])
            except Exception:
                messagebox.showinfo("提示", "请手动打开：%s" % p)
        else:
            messagebox.showinfo("提示", "输出文件夹还不存在")


def main():
    root = tk.Tk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)  # 高分屏防模糊
    except Exception:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
