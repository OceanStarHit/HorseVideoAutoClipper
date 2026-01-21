import threading
from datetime import datetime

class AppLogger:
    def __init__(self, gui_log_callback=None):
        self.gui_log_callback = gui_log_callback
        self.lock = threading.Lock()

    def log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{timestamp}] {message}"

        with self.lock:
            print(line)
            if self.gui_log_callback:
                self.gui_log_callback(line)
