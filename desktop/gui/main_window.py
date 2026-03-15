import tkinter as tk
from tkinter import ttk

from gui.dataset_panel import DatasetPanel
from gui.inference_panel import InferencePanel
from gui.logs_panel import LogsPanel
from gui.training_panel import TrainingPanel


class MainWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("VisionNet Studio")
        self.root.geometry("1000x700")

        self.dataset_panel = None
        self.training_panel = None
        self.inference_panel = None
        self.logs_panel = None

        self._setup_ui()

    def _setup_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        self.dataset_panel = DatasetPanel(self.notebook, self)
        self.training_panel = TrainingPanel(self.notebook, self)
        self.inference_panel = InferencePanel(self.notebook, self)
        self.logs_panel = LogsPanel(self.notebook, self)

        self.notebook.add(self.dataset_panel, text="Датасет")
        self.notebook.add(self.training_panel, text="Обучение")
        self.notebook.add(self.inference_panel, text="Распознование")
        self.notebook.add(self.logs_panel, text="Логи")

        self.training_panel.set_dataset_callback(self._on_dataset_loaded)
        self.inference_panel.set_log_callback(self.log_message)

        self.dataset_callback = self._on_dataset_loaded

    def _on_dataset_loaded(self, dataset_info):
        self.training_panel.update_dataset_info(dataset_info)

    def log_message(self, message):
        if self.logs_panel:
            self.logs_panel.log(message)

    def run(self):
        self.root.mainloop()
