import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import torch
from torch.utils.data import DataLoader, random_split

from ml.dataset import COCOSegmentationDataset
from ml.model import create_model, save_model
from ml.trainer import Trainer
from utils.config_manager import ConfigManager


class TrainingPanel(ttk.Frame):
    def __init__(self, parent, main_window):
        super().__init__(parent)
        self.main_window = main_window
        self.dataset_info = None
        self.model = None
        self.trainer = None
        self.device = "cpu"
        self.current_classes = None

        self._setup_ui()

    def _setup_ui(self):
        title_label = ttk.Label(
            self, text="Модельное обучение", font=("Arial", 16, "bold")
        )
        title_label.pack(pady=10)

        settings_frame = ttk.LabelFrame(self, text="Настройки обучения")
        settings_frame.pack(fill=tk.X, padx=20, pady=10)

        self._create_settings(settings_frame)

        buttons_frame = ttk.Frame(self)
        buttons_frame.pack(pady=10)

        self.start_button = ttk.Button(
            buttons_frame, text="Начать обучение", command=self.start_training
        )
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(
            buttons_frame,
            text="Остановить тренировку",
            command=self.stop_training,
            state="disabled",
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)

        self.save_button = ttk.Button(
            buttons_frame,
            text="Сохранить модель",
            command=self.save_model,
            state="disabled",
        )
        self.save_button.pack(side=tk.LEFT, padx=5)

        progress_frame = ttk.LabelFrame(self, text="Прогресс обучения")
        progress_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.progress_label = ttk.Label(progress_frame, text="Готов тренироваться")
        self.progress_label.pack(pady=10)

        self.progress_bar = ttk.Progressbar(progress_frame, mode="determinate")
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)

    def _create_settings(self, parent):
        row = 0

        ttk.Label(parent, text="Эпохи:").grid(
            row=row, column=0, sticky=tk.W, padx=10, pady=5
        )
        self.epochs_var = tk.IntVar(value=10)
        ttk.Spinbox(
            parent, from_=1, to=1000, textvariable=self.epochs_var, width=13
        ).grid(row=row, column=1, sticky=tk.W, padx=10, pady=5)

        row += 1
        ttk.Label(parent, text="Размер батча:").grid(
            row=row, column=0, sticky=tk.W, padx=10, pady=5
        )
        self.batch_size_var = tk.IntVar(value=4)
        ttk.Spinbox(
            parent, from_=1, to=64, textvariable=self.batch_size_var, width=13
        ).grid(row=row, column=1, sticky=tk.W, padx=10, pady=5)

        row += 1
        ttk.Label(parent, text="Скорость обучения:").grid(
            row=row, column=0, sticky=tk.W, padx=10, pady=5
        )
        self.lr_var = tk.DoubleVar(value=0.001)
        lr_spinbox = ttk.Spinbox(
            parent,
            from_=0.00001,
            to=1.0,
            increment=0.0001,
            textvariable=self.lr_var,
            width=13,
            format="%.5f",
        )
        lr_spinbox.grid(row=row, column=1, sticky=tk.W, padx=10, pady=5)

        row += 1
        ttk.Label(parent, text="Размер изображения:").grid(
            row=row, column=0, sticky=tk.W, padx=10, pady=5
        )
        self.image_size_var = tk.IntVar(value=512)
        ttk.Spinbox(
            parent,
            from_=64,
            to=1024,
            increment=64,
            textvariable=self.image_size_var,
            width=13,
        ).grid(row=row, column=1, sticky=tk.W, padx=10, pady=5)

        row += 1
        ttk.Label(parent, text="Устройство:").grid(
            row=row, column=0, sticky=tk.W, padx=10, pady=5
        )
        self.device_var = tk.StringVar(value="cpu")
        device_combo = ttk.Combobox(
            parent, textvariable=self.device_var, values=["cpu", "cuda"], width=13
        )
        device_combo.grid(row=row, column=1, sticky=tk.W, padx=10, pady=5)

        if torch.cuda.is_available():
            self.device_var.set("cuda")

    def set_dataset_callback(self, callback):
        self.dataset_callback = callback

    def update_dataset_info(self, dataset_info):
        self.dataset_info = dataset_info
        if dataset_info:
            self.main_window.log_message(
                f"[INFO] Набор данных готов: {dataset_info['num_images']} images, {dataset_info['num_classes']} classes"
            )

    def start_training(self):
        print("=" * 50)
        print("CUDA Available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU Device:", torch.cuda.get_device_name(0))
            print("Current device:", torch.cuda.current_device())
            print("Memory allocated:", torch.cuda.memory_allocated(0) / 1024**2, "MB")
        print("=" * 50)

        if not self.dataset_info:
            messagebox.showwarning(
                "Предупреждение", "Пожалуйста, сначала загрузите набор данных"
            )
            return

        try:
            epochs = self.epochs_var.get()
            batch_size = self.batch_size_var.get()
            lr = self.lr_var.get()
            image_size = self.image_size_var.get()
            device = self.device_var.get()

            if device == "cuda" and not torch.cuda.is_available():
                messagebox.showwarning(
                    "Предупреждение", "CUDA недоступна, используется ЦП"
                )
                device = "cpu"

            self.device = device

            dataset_path = self.dataset_info["path"]
            annotations_path = os.path.join(dataset_path, "annotations.json")

            dataset = COCOSegmentationDataset(
                root=dataset_path,
                annotations_file=annotations_path,
                image_size=image_size,
            )

            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size

            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size_var.get(),
                shuffle=True,
                num_workers=4,
                pin_memory=True,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size_var.get(),
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )

            classes = dataset.get_classes()
            num_classes = len(classes)
            self.current_classes = classes

            self.model = create_model(num_classes=num_classes)

            dataset_classes = (
                self.dataset_info.get("classes", classes[1:])
                if self.dataset_info
                else classes[1:]
            )

            self.progress_label.config(text=f"Обучение на {device}...")
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")

            self.trainer = Trainer(
                model=self.model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                learning_rate=lr,
                on_log=self.main_window.log_message,
                on_epoch_end=self._on_epoch_end,
            )

            self.trainer.train(epochs)

        except Exception as e:
            messagebox.showerror("Error", f"Training failed: {str(e)}")
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")

    def _on_epoch_end(self, epoch, loss):
        epochs = self.epochs_var.get()
        self.progress_bar["value"] = (epoch / epochs) * 100
        self.progress_label.config(text=f"'Эхоха' {epoch}/{epochs} - Loss: {loss:.4f}")

        if epoch >= epochs:
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.save_button.config(state="normal")
            self.main_window.log_message("[INFO] Обучение завершено!")

    def stop_training(self):
        if self.trainer:
            self.trainer.stop()
            self.main_window.log_message("[INFO] Обучение остановлено")

        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")

    def save_model(self):
        if not self.model:
            messagebox.showwarning("Warning", "No trained model to save")
            return

        folder_selected = filedialog.askdirectory(
            title="Выберите папку для сохранения модели"
        )
        if not folder_selected:
            return

        try:
            model_path = os.path.join(folder_selected, "model.pth")
            config_path = os.path.join(folder_selected, "model_config.json")

            config = ConfigManager.create_model_config(
                model_name="VisionNetAdapter",
                in_channels=3,
                num_classes=self.model.num_classes,
                image_size=self.image_size_var.get(),
                classes=self.current_classes,
            )

            save_model(self.model, model_path, config)
            ConfigManager.save_model_config(config, config_path)

            self.main_window.log_message(f"[INFO] Модель сохранена в {model_path}")
            messagebox.showinfo("Успех", "Модель успешно сохранена!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save model: {str(e)}")
