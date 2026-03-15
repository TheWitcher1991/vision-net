import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from utils.coco_parser import COCOParser


class DatasetPanel(ttk.Frame):
    def __init__(self, parent, main_window):
        super().__init__(parent)
        self.main_window = main_window
        self.dataset_path = None
        self.annotations_path = None
        self.coco_parser = None
        self.dataset_info = None
        self._setup_ui()

    def _setup_ui(self):
        title_label = ttk.Label(self, text="Управление наборами данных", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)

        control_frame = ttk.Frame(self)
        control_frame.pack(pady=10)

        ttk.Button(control_frame, text="Выберите папку набора данных", command=self.select_dataset).pack(
            side=tk.LEFT, padx=5
        )

        self.info_frame = ttk.LabelFrame(self, text="Информация о наборе данных")
        self.info_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.info_text = tk.Text(self.info_frame, height=15, state="disabled")
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def select_dataset(self):
        folder_selected = filedialog.askdirectory(title="Выберите папку набора данных")
        if not folder_selected:
            return

        annotations_file = os.path.join(folder_selected, "annotations.json")

        if not os.path.exists(annotations_file):
            messagebox.showerror("Error", "annotations.json не найден в выбранной папке")
            return

        try:
            self.dataset_path = folder_selected
            self.annotations_path = annotations_file

            self.coco_parser = COCOParser(annotations_file)
            self.coco_parser.load()

            stats = self.coco_parser.get_stats()
            categories = self.coco_parser.get_category_names()

            self.dataset_info = {
                "path": self.dataset_path,
                "num_images": stats["num_images"],
                "num_annotations": stats["num_annotations"],
                "num_classes": stats["num_categories"],
                "classes": categories,
            }

            self._display_info(stats, categories)

            if hasattr(self.main_window, "dataset_callback"):
                self.main_window.dataset_callback(self.dataset_info)

            self.main_window.log_message("[INFO] Набор данных загружен")
            messagebox.showinfo("Success", "Набор данных успешно загружен!")

        except Exception as e:
            messagebox.showerror("Error", f"Не удалось загрузить набор данных: {str(e)}")

    def _display_info(self, stats, categories):
        self.info_text.config(state="normal")
        self.info_text.delete("1.0", tk.END)

        info_lines = [
            f"Number of images: {stats['num_images']}",
            f"Number of annotations: {stats['num_annotations']}",
            f"Number of classes: {stats['num_categories']}",
            "",
            "Classes:",
        ]

        for cat in categories:
            info_lines.append(f"  - {cat}")

        self.info_text.insert("1.0", "\n".join(info_lines))
        self.info_text.config(state="disabled")

    def get_dataset_info(self):
        return self.dataset_info
