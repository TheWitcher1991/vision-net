import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import torch
from PIL import Image, ImageTk

from ml.inference import Inference
from ml.model import load_model
from utils.config_manager import ConfigManager
from utils.image_utils import create_overlay


CLASS_COLORS = [
    "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
    "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
    "#F8B500", "#00CED1", "#FF69B4", "#32CD32", "#FF4500",
]


class InferencePanel(ttk.Frame):
    def __init__(self, parent, main_window):
        super().__init__(parent)
        self.main_window = main_window
        self.model = None
        self.config = None
        self.inference_engine = None

        self.current_original = None
        self.current_mask = None
        self.current_overlay = None
        self.photo_original = None
        self.photo_overlay = None
        self.log_callback = None

        self._setup_ui()

    def _setup_ui(self):
        title_label = ttk.Label(self, text="Распознавание", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)

        control_frame = ttk.Frame(self)
        control_frame.pack(pady=5)

        ttk.Button(
            control_frame, text="Загрузить модель", command=self.load_model
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            control_frame, text="Выберите изображение", command=self.select_image
        ).pack(side=tk.LEFT, padx=5)

        self.model_label = ttk.Label(self, text="Модель: Не загружено")
        self.model_label.pack(pady=5)

        classes_frame = ttk.LabelFrame(self, text="Распознанные классы")
        classes_frame.pack(fill=tk.BOTH, expand=False, padx=20, pady=10)

        self.classes_canvas = tk.Canvas(classes_frame, height=120, bg="#2b2b2b", highlightthickness=0)
        self.classes_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.classes_scroll = ttk.Scrollbar(classes_frame, orient="vertical", command=self.classes_canvas.yview)
        self.classes_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.classes_canvas.configure(yscrollcommand=self.classes_scroll.set)

        self.classes_inner = ttk.Frame(self.classes_canvas)
        self.classes_canvas.create_window((0, 0), window=self.classes_inner, anchor=tk.NW)
        self.classes_inner.bind("<Configure>", lambda e: self.classes_canvas.configure(scrollregion=self.classes_canvas.bbox("all")))

        self.detected_items = []

        images_container = ttk.Frame(self)
        images_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        images_container.columnconfigure(0, weight=1)
        images_container.columnconfigure(1, weight=1)

        left_frame = ttk.Frame(images_container)
        left_frame.grid(row=0, column=0, padx=10, sticky="nsew")
        right_frame = ttk.Frame(images_container)
        right_frame.grid(row=0, column=1, padx=10, sticky="nsew")

        ttk.Label(left_frame, text="Оригинальное изображение", font=("Arial", 12, "bold")).pack(pady=5)
        self.canvas_original = tk.Canvas(left_frame, width=400, height=400, bg="#1a1a1a")
        self.canvas_original.pack(fill=tk.BOTH, expand=True)

        ttk.Label(right_frame, text="Выделение объектов", font=("Arial", 12, "bold")).pack(pady=5)
        self.canvas_overlay = tk.Canvas(right_frame, width=400, height=400, bg="#1a1a1a")
        self.canvas_overlay.pack(fill=tk.BOTH, expand=True)

    def set_log_callback(self, callback):
        self.log_callback = callback

    def load_model(self):
        model_file = filedialog.askopenfilename(
            title="Выберите model.pth",
            filetypes=[("PyTorch models", "*.pth"), ("All files", "*.*")],
        )

        if not model_file:
            return

        config_file = filedialog.askopenfilename(
            title="Выберите model_config.json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )

        if not config_file:
            return

        try:
            config = ConfigManager.load_model_config(config_file)

            device = (
                "cuda"
                if torch.cuda.is_available() and config.get("device") == "cuda"
                else "cpu"
            )

            self.model, loaded_config = load_model(model_file, device=device)

            self.config = {
                "num_classes": loaded_config.get(
                    "num_classes", config.get("num_classes")
                ),
                "classes": loaded_config.get("classes", config.get("classes", [])),
                "image_size": loaded_config.get(
                    "image_size", config.get("image_size", 512)
                ),
                "in_channels": loaded_config.get(
                    "in_channels", config.get("in_channels", 3)
                ),
            }

            self.inference_engine = Inference(
                model=self.model,
                device=device,
                classes=self.config["classes"],
                image_size=self.config["image_size"],
            )

            self.model_label.config(
                text=f"Модель: {loaded_config.get('model_name', 'VisionNetAdapter')} - {self.config['num_classes']} классы"
            )

            if self.log_callback:
                self.log_callback(f"[INFO] Модель загружена: {model_file}")

            messagebox.showinfo("Успех", "Модель успешно загружена!")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить модель: {str(e)}")

    def select_image(self):
        if not self.inference_engine:
            messagebox.showwarning(
                "Предупреждение", "Пожалуйста, сначала загрузите модель"
            )
            return

        file_path = filedialog.askopenfilename(
            title="Выберите изображение для вывода",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*"),
            ],
        )

        if not file_path:
            return

        try:
            result = self.inference_engine.infer(file_path)

            original_img = result["original_image"]
            mask_img = result["mask"]
            detected_classes = result["detected_classes"]

            print(f"[GUI] Mask image mode: {mask_img.mode}, size: {mask_img.size}")
            mask_arr = np.array(mask_img)
            print(
                f"[GUI] Mask array shape: {mask_arr.shape}, unique values: {np.unique(mask_arr)}"
            )

            overlay_img = create_overlay(original_img, mask_img, alpha=0.5)

            self._display_images(original_img, mask_img, overlay_img)
            self._display_results(detected_classes)

            mask_stats = np.unique(mask_arr, return_counts=True)
            stats_str = ", ".join([f"class {v}: {c} px" for v, c in zip(*mask_stats)])

            if self.log_callback:
                self.log_callback(
                    f"[INFO] Вывод завершен для {os.path.basename(file_path)}"
                )
                self.log_callback(f"[INFO] Статистика маски: {stats_str}")
                for cls, conf in detected_classes:
                    self.log_callback(f"[INFO] Detected: {cls} = {conf:.4f}")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Вывод не удался: {str(e)}")

    def _display_images(self, original, mask, overlay):
        max_w, max_h = 400, 400

        orig_w, orig_h = original.size
        scale = min(max_w / orig_w, max_h / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)

        original_resized = original.resize((new_w, new_h), Image.Resampling.BILINEAR)
        overlay_resized = overlay.resize((new_w, new_h), Image.Resampling.BILINEAR)

        self.photo_original = ImageTk.PhotoImage(original_resized)
        self.photo_overlay = ImageTk.PhotoImage(overlay_resized)

        canvas_w = self.canvas_original.winfo_width() or max_w
        canvas_h = self.canvas_original.winfo_height() or max_h

        self.canvas_original.delete("all")
        self.canvas_original.create_image(
            canvas_w // 2, canvas_h // 2, image=self.photo_original, anchor=tk.CENTER
        )

        self.canvas_overlay.delete("all")
        self.canvas_overlay.create_image(
            canvas_w // 2, canvas_h // 2, image=self.photo_overlay, anchor=tk.CENTER
        )

    def _display_results(self, detected_classes):
        for widget in self.classes_inner.winfo_children():
            widget.destroy()

        if not detected_classes:
            no_result = ttk.Label(
                self.classes_inner,
                text="Объекты не найдены",
                font=("Arial", 11),
                foreground="#888888"
            )
            no_result.pack(pady=10, padx=10, anchor=tk.W)
            return

        for idx, (class_name, confidence) in enumerate(detected_classes):
            color = CLASS_COLORS[idx % len(CLASS_COLORS)]
            conf_percent = confidence * 100

            item_frame = ttk.Frame(self.classes_inner, style="Custom.TFrame")
            item_frame.pack(fill=tk.X, padx=5, pady=3)

            color_indicator = tk.Canvas(item_frame, width=6, height=30, bg=color, highlightthickness=0)
            color_indicator.pack(side=tk.LEFT, padx=(0, 8))

            info_frame = ttk.Frame(item_frame)
            info_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

            class_label = ttk.Label(
                info_frame,
                text=class_name,
                font=("Arial", 11, "bold"),
                foreground=color
            )
            class_label.pack(anchor=tk.W)

            conf_label = ttk.Label(
                info_frame,
                text=f"Уверенность: {conf_percent:.1f}%",
                font=("Arial", 9),
                foreground="#aaaaaa"
            )
            conf_label.pack(anchor=tk.W)

            bar_frame = ttk.Frame(item_frame, style="Custom.TFrame")
            bar_frame.pack(side=tk.LEFT, padx=(10, 0))

            bar_bg = tk.Canvas(bar_frame, width=100, height=8, bg="#3a3a3a", highlightthickness=0)
            bar_bg.pack()

            bar_fill = tk.Canvas(bar_frame, width=int(conf_percent), height=8, bg=color, highlightthickness=0)
            bar_fill.place(x=0, y=0)
