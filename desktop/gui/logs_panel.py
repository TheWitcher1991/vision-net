import datetime
import tkinter as tk
from tkinter import ttk


class LogsPanel(ttk.Frame):
    def __init__(self, parent, main_window):
        super().__init__(parent)
        self.main_window = main_window
        self._setup_ui()

    def _setup_ui(self):
        title_label = ttk.Label(self, text="Логи", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)

        control_frame = ttk.Frame(self)
        control_frame.pack(pady=5)

        ttk.Button(control_frame, text="Очистить логи", command=self.clear_logs).pack(side=tk.LEFT, padx=5)

        self.logs_text = tk.Text(self, height=30, state="disabled")
        self.logs_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        scrollbar = ttk.Scrollbar(self.logs_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.logs_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.logs_text.yview)

    def log(self, message):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"

        self.logs_text.config(state="normal")
        self.logs_text.insert(tk.END, log_entry)
        self.logs_text.see(tk.END)
        self.logs_text.config(state="disabled")

    def clear_logs(self):
        self.logs_text.config(state="normal")
        self.logs_text.delete("1.0", tk.END)
        self.logs_text.config(state="disabled")
