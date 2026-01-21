# Detection / tracking
DETECTION_INTERVAL = 3          # Run YOLO every N frames
MIN_TRACK_SECONDS = 3.0         # Minimum valid tracking time
DIRECTION_CONFIDENCE = 0.8      # 80% frames moving right

# Perpendicular detection
ASPECT_RATIO_SMOOTHING = 11     # Frames for rolling mean
MIN_STABLE_FRAMES = 15

# Clip extraction
CLIP_SECONDS_BEFORE = 2.0
CLIP_SECONDS_AFTER = 2.0

# Video
SUPPORTED_EXTENSIONS = (".mp4",)
