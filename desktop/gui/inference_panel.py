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
        self.photo_mask = None
        self.photo_overlay = None

        self._setup_ui()

    def _setup_ui(self):
        title_label = ttk.Label(self, text="Распознование", font=("Arial", 16, "bold"))
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

        display_frame = ttk.Frame(self)
        display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        display_frame.columnconfigure(0, weight=1)
        display_frame.columnconfigure(1, weight=1)
        display_frame.columnconfigure(2, weight=1)

        ttk.Label(
            display_frame, text="Исходное изображение", font=("Arial", 12, "bold")
        ).grid(row=0, column=0, pady=5)
        ttk.Label(
            display_frame, text="Маска сегментации", font=("Arial", 12, "bold")
        ).grid(row=0, column=1, pady=5)
        ttk.Label(display_frame, text="Наложение", font=("Arial", 12, "bold")).grid(
            row=0, column=2, pady=5
        )

        self.canvas_original = tk.Canvas(
            display_frame, width=300, height=300, bg="gray"
        )
        self.canvas_original.grid(row=1, column=0, padx=5, pady=5)

        self.canvas_mask = tk.Canvas(display_frame, width=300, height=300, bg="gray")
        self.canvas_mask.grid(row=1, column=1, padx=5, pady=5)

        self.canvas_overlay = tk.Canvas(display_frame, width=300, height=300, bg="gray")
        self.canvas_overlay.grid(row=1, column=2, padx=5, pady=5)

        results_frame = ttk.LabelFrame(self, text="Результаты распознования")
        results_frame.pack(fill=tk.X, padx=20, pady=10)

        self.results_text = tk.Text(results_frame, height=8, state="disabled")
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

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
        max_size = 300

        orig_w, orig_h = original.size
        scale = min(max_size / orig_w, max_size / orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)

        original_resized = original.resize((new_w, new_h), Image.Resampling.BILINEAR)
        mask_resized = mask.resize((new_w, new_h), Image.Resampling.NEAREST)
        overlay_resized = overlay.resize((new_w, new_h), Image.Resampling.BILINEAR)

        self.photo_original = ImageTk.PhotoImage(original_resized)
        self.photo_mask = ImageTk.PhotoImage(mask_resized)
        self.photo_overlay = ImageTk.PhotoImage(overlay_resized)

        self.canvas_original.delete("all")
        self.canvas_original.create_image(
            max_size // 2, max_size // 2, image=self.photo_original, anchor=tk.CENTER
        )

        self.canvas_mask.delete("all")
        self.canvas_mask.create_image(
            max_size // 2, max_size // 2, image=self.photo_mask, anchor=tk.CENTER
        )

        self.canvas_overlay.delete("all")
        self.canvas_overlay.create_image(
            max_size // 2, max_size // 2, image=self.photo_overlay, anchor=tk.CENTER
        )

    def _display_results(self, detected_classes):
        self.results_text.config(state="normal")
        self.results_text.delete("1.0", tk.END)

        self.results_text.insert("1.0", "Обнаруженные классы:\n\n")

        if not detected_classes:
            self.results_text.insert(tk.END, "Объект не найден на изображении\n")

        for class_name, confidence in detected_classes:
            self.results_text.insert(tk.END, f"{class_name} — {confidence:.2f}\n")

        self.results_text.config(state="disabled")
