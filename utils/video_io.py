import os
import cv2
import subprocess
from pathlib import Path


class VideoInfo:
    def __init__(self, path, fps, frame_count, width, height, duration):
        self.path = path
        self.fps = fps
        self.frame_count = frame_count
        self.width = width
        self.height = height
        self.duration = duration


def load_video_info(video_path: str) -> VideoInfo:
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    if fps <= 0 or frame_count <= 0:
        raise ValueError("Invalid video metadata")

    duration = frame_count / fps

    return VideoInfo(
        path=video_path,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration=duration
    )



def export_clip(video_path, clip_info, output_dir):
    """
    Uses ffmpeg to export a frame-accurate clip.

    clip_info:
        {
          "center_frame": int,
          "start_frame": int,
          "end_frame": int
        }
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    os.makedirs(output_dir, exist_ok=True)

    input_path = Path(video_path)
    output_name = input_path.stem + "_clip.mp4"
    output_path = Path(output_dir) / output_name

    start_frame = clip_info["start_frame"]
    end_frame = clip_info["end_frame"]

    if end_frame <= start_frame:
        raise ValueError("Invalid clip frame range")

    frame_count = end_frame - start_frame

    # ffmpeg command:
    # - Accurate frame seeking using select filter
    # - Re-encode for reliability (copy is NOT frame-accurate)
    cmd = [
        "ffmpeg",
        "-y",                       # overwrite output
        "-i", str(input_path),
        "-vf", f"select=between(n\\,{start_frame}\\,{end_frame}),setpts=PTS-STARTPTS",
        "-vsync", "vfr",
        "-an",                      # drop audio
        str(output_path)
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg failed:\n{result.stderr}"
        )

    return str(output_path)


import cv2
import os
from pathlib import Path


def export_clip_cv2(video_path, clip_info, output_dir, clip_index=None):
    """
    Export a video clip using OpenCV (frame-accurate, no audio).

    clip_info:
        {
            "start_frame": int,
            "end_frame": int,
            "center_frame": int (optional)
        }
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 30.0

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    input_path = Path(video_path)

    # -------------------------------------------------
    # Output filename (multi-clip safe)
    # -------------------------------------------------
    center = clip_info.get("center_frame", "X")

    if clip_index is not None:
        output_name = f"{input_path.stem}_clip_{clip_index:02d}_f{center}.mp4"
    else:
        output_name = f"{input_path.stem}_clip_f{center}.mp4"

    output_path = Path(output_dir) / output_name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    start = clip_info["start_frame"]
    end   = clip_info["end_frame"]

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    frame_idx = start
    while frame_idx < end:
        ret, frame = cap.read()
        if not ret:
            break

        writer.write(frame)
        frame_idx += 1

    writer.release()
    cap.release()

    return str(output_path)
