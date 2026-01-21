import os
from utils.config import SUPPORTED_EXTENSIONS

def collect_video_files(paths):
    """
    Accepts a list of files and/or folders.
    Returns a flat list of video file paths.
    """
    videos = []

    for path in paths:
        if os.path.isfile(path) and path.lower().endswith(SUPPORTED_EXTENSIONS):
            videos.append(path)

        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                for f in files:
                    if f.lower().endswith(SUPPORTED_EXTENSIONS):
                        videos.append(os.path.join(root, f))

    return sorted(videos)
