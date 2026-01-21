import tkinter as tk
from tkinter import font


class LogPanel(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)

        log_font = font.Font(family="Segoe UI", size=8)

        self.text = tk.Text(
            self,
            height=15,
            wrap="word",
            state=tk.DISABLED,
            font=log_font
        )
        self.text.pack(fill=tk.BOTH, expand=True)

    def log(self, message):
        self.text.config(state=tk.NORMAL)
        self.text.insert(tk.END, message + "\n")
        self.text.see(tk.END)
        self.text.config(state=tk.DISABLED)


