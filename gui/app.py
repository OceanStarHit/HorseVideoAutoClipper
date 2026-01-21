import tkinter as tk
from tkinter import filedialog, ttk
from gui.widgets import LogPanel
from engine.pipeline import Pipeline
import threading
from pathlib import Path


class HorseAutoClipperApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Horse Video Auto Clipper")
        self.geometry("600x400")

        self.input_path_var = tk.StringVar(value=str(Path.cwd()))
        self.output_path_var = tk.StringVar(value=str(Path.cwd() / "output"))

        self.worker_thread = None

        self._build_ui()

        self.after(0, self.center_window)
        self.after(100, self.validate_paths)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        container = ttk.Frame(self, padding=10)
        container.pack(fill=tk.X)

        # Input folder
        ttk.Label(container, text="Input folder containing horse videos:").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Entry(
            container,
            textvariable=self.input_path_var,
            width=70,
            state="readonly",
        ).grid(row=0, column=1, sticky="we", padx=5)
        ttk.Button(container, text="Browse", command=self.select_input_folder).grid(
            row=0, column=2
        )

        # Output folder
        ttk.Label(container, text="Output folder to save video clips:").grid(
            row=1, column=0, sticky="w", pady=(5, 0)
        )
        ttk.Entry(
            container,
            textvariable=self.output_path_var,
            width=70,
            state="readonly",
        ).grid(row=1, column=1, sticky="we", padx=5, pady=(5, 0))
        ttk.Button(container, text="Browse", command=self.select_output_folder).grid(
            row=1, column=2, pady=(5, 0)
        )

        container.columnconfigure(1, weight=1)

        # Bottom
        bottom = ttk.Frame(self)
        bottom.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # LEFT: Start button
        btn_panel = ttk.Frame(bottom)
        btn_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        self.start_btn = ttk.Button(
            btn_panel, text="Start Processing", command=self.start_processing
        )
        self.start_btn.pack(fill=tk.X)

        # RIGHT: Progress + Logs
        right_panel = ttk.Frame(bottom)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.progress = ttk.Progressbar(right_panel)
        self.progress.pack(fill=tk.X, pady=(0, 5))

        self.log_panel = LogPanel(right_panel)
        self.log_panel.pack(fill=tk.BOTH, expand=True)

    # ------------------------------------------------------------------
    # Window positioning
    # ------------------------------------------------------------------
    def center_window(self):
        self.update_idletasks()
        w, h = self.winfo_width(), self.winfo_height()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw - w)//2}+{(sh - h)//2}")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate_paths(self):
        input_dir = Path(self.input_path_var.get())
        output_dir = Path(self.output_path_var.get())

        self.start_btn.config(state=tk.DISABLED)

        if not input_dir.exists():
            self.log_panel.log("⚠ Input folder does not exist")
            return

        videos = list(input_dir.glob("*.mp4"))
        if not videos:
            self.log_panel.log("⚠ No MP4 videos found")
            return

        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
            self.log_panel.log(f"📁 Created output folder: {output_dir}")

        self.log_panel.log(f"✔ Ready ({len(videos)} video(s))")
        self.start_btn.config(state=tk.NORMAL)

    # ------------------------------------------------------------------
    # Folder selection
    # ------------------------------------------------------------------
    def select_input_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.input_path_var.set(path)
            self.log_panel.log(f"Input folder set: {path}")
            self.validate_paths()

    def select_output_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.output_path_var.set(path)
            self.log_panel.log(f"Output folder set: {path}")
            self.validate_paths()

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------
    def start_processing(self):
        input_dir = Path(self.input_path_var.get())
        output_dir = Path(self.output_path_var.get())

        videos = sorted(input_dir.glob("*.mp4"))
        if not videos:
            self.log_panel.log("❌ No videos to process")
            return

        self.start_btn.config(state=tk.DISABLED)
        self.progress["value"] = 0

        pipeline = Pipeline(
            video_paths=[str(v) for v in videos],
            output_dir=str(output_dir),
            log_panel=self.log_panel,
            progress_bar=self.progress,
        )

        self.worker_thread = threading.Thread(
            target=pipeline.run,
            daemon=True
        )
        self.worker_thread.start()
