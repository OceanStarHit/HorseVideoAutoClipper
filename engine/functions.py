# engine/utils.py
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

###################################################################
# 1. Foreground Objects Detection (YOLO + ByteTrack)
###################################################################

model_path = "models/yolov8m.pt"
TRACKER_CFG = Path("configs/bytetrack.yaml")

print("🔧 Loading YOLO model...")
_YOLO_MODEL = YOLO(model_path)
print("✅ YOLO model loaded")

def detect_foreground_objects(
    frame_idx,
    frame,
    imgsz: int = 320,
    conf: float = 0.25,
):

    results = _YOLO_MODEL.track(
        frame,
        tracker=TRACKER_CFG,
        persist=True,
        imgsz=imgsz,
        classes=[0, 17],
        conf=conf,
        verbose=False,
    )

    horse_bboxes = []
    person_bboxes = []

    boxes = results[0].boxes
    if boxes is None:
        print("⚠️ No detections")
        return horse_bboxes, person_bboxes

    for cls, box in zip(boxes.cls.tolist(), boxes.xyxy.tolist()):
        cls = int(cls)
        bbox = tuple(map(int, box))

        if cls == 17:
            horse_bboxes.append(bbox)
        elif cls == 0:
            person_bboxes.append(bbox)

    print(f"\n🔍 Frame {frame_idx}: 🐎 horses={len(horse_bboxes)} | 🧍 persons={len(person_bboxes)}")
    return horse_bboxes, person_bboxes

###################################################################
# 2. Valid Horse Condition
###################################################################

MIN_HORSE_AREA_RATIO = 0.05
MIN_HW_RATIO = 0.5
MAX_HW_RATIO = 1.0

def is_horse_bboxes_valid(horse_bboxes, fw, fh):
    if len(horse_bboxes) != 1:
        print(
            f"❌ Horse rejected (num)"
        )
        return False, None

    x1, y1, x2, y2 = horse_bboxes[0]
    w = x2 - x1
    h = y2 - y1

    area = w * h
    min_area = MIN_HORSE_AREA_RATIO * fw * fh

    if area < min_area:
        print(
            f"❌ Horse rejected (area): {area:.0f} < {min_area:.0f}"
        )
        return False, None

    hw = h / max(1, w)
    if hw < MIN_HW_RATIO or hw > MAX_HW_RATIO:
        print(
            f"❌ Horse rejected (aspect): h/w={hw:.2f}"
        )
        return False, None

    print(
        f"✅ Horse valid: area={area:.0f}, h/w={hw:.2f}"
    )
    return True, w / max(1, h)

###################################################################
# 3. Define background mask
###################################################################

def define_bg_mask(
    frame_shape,
    horse_bbox,
    person_bboxes,
    scale: float = 0.5,
):
    fh, fw = frame_shape[:2]
    mh = int(fh * scale)
    mw = int(fw * scale)

    bg_mask = np.ones((mh, mw), dtype=np.uint8) * 255

    if horse_bbox is None:
        print("⚠️ No horse bbox → using full-frame BG mask")
        return bg_mask

    def _exclude_bbox(bbox):
        x1, y1, x2, y2 = bbox
        sx1 = max(0, int(x1 * scale))
        sy1 = max(0, int(y1 * scale))
        sx2 = min(mw, int(x2 * scale))
        sy2 = min(mh, int(y2 * scale))
        bg_mask[sy1:sy2, sx1:sx2] = 0

    print(f"🐎 Horse bbox: {horse_bbox}")

    # 1️⃣ exclude full horse bbox
    _exclude_bbox(horse_bbox)

    # 2️⃣ exclude persons
    for b in person_bboxes:
        _exclude_bbox(b)

    # 3️⃣ exclude region ABOVE horse vertical center (global)
    _, y1, _, y2 = horse_bbox
    y_center = int(((y1 + y2) * 0.5) * scale)
    y_center = np.clip(y_center, 0, mh)
    y_min = min(y1, y2)

    # bg_mask[0:y_center, :] = 0
    bg_mask[0:y_min, :] = 0

    print(
        f"🧱 BG mask built | horses=1 persons={len(person_bboxes)} "
        f"| excluded above y={y_center}"
    )

    return bg_mask


###################################################################
# 4. Estimate background motion
###################################################################

def estimate_bg_motion(
    prev_gray,
    curr_gray,
    bg_mask,
    max_features=600,
    min_inliers=20,
    min_inlier_ratio=0.5,
    max_scale_dev=0.08,
):
    print("📐 Estimating BG motion")

    result = {
        "valid": False,
        "motion_dx": 0.0,
        "motion_dy": 0.0,
        "inlier_ratio": 0.0,
        "inliers": None,
        "pts_next": None,
    }

    pts_prev = cv2.goodFeaturesToTrack(
        prev_gray,
        max_features,
        qualityLevel=0.01,
        minDistance=8,
        mask=bg_mask,
    )

    if pts_prev is None:
        print("❌ No background features")
        return result

    print(f"🔹 BG features detected: {len(pts_prev)}")

    pts_next, st, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, pts_prev, None
    )

    good_prev = pts_prev[st.flatten() == 1]
    good_next = pts_next[st.flatten() == 1]

    if len(good_prev) < min_inliers:
        print(f"❌ Too few tracked points: {len(good_prev)}")
        return result

    M, inliers = cv2.estimateAffinePartial2D(
        good_prev,
        good_next,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )

    if M is None or inliers is None:
        print("❌ RANSAC failed")
        return result

    inlier_count = int(inliers.sum())
    inlier_ratio = inlier_count / len(good_prev)

    print(
        f"🔸 RANSAC inliers: {inlier_count}/{len(good_prev)} "
        f"({inlier_ratio:.2f})"
    )

    if inlier_count < min_inliers or inlier_ratio < min_inlier_ratio:
        print("❌ Inlier threshold failed")
        return result

    scale_x = np.hypot(M[0, 0], M[0, 1])
    scale_y = np.hypot(M[1, 0], M[1, 1])

    if (
        abs(scale_x - 1.0) > max_scale_dev or
        abs(scale_y - 1.0) > max_scale_dev
    ):
        print(
            f"❌ Scale drift rejected: sx={scale_x:.3f} sy={scale_y:.3f}"
        )
        return result

    result.update({
        "valid": True,
        "motion_dx": float(M[0, 2]),
        "motion_dy": float(M[1, 2]),
        "inlier_ratio": inlier_ratio,
        "inliers": inliers,
        "pts_next": good_next,
    })

    print(
        f"✅ BG motion dx={M[0,2]:.3f} dy={M[1,2]:.3f}"
    )

    return result


###################################################################
# 5. Normalize background motion
###################################################################

def normalize_bg_motion(
    motion_dx,
    motion_dy,
    fps,
    horse_bbox,
    frame_width,
):
    x1, _, x2, _ = horse_bbox
    horse_width = x2 - x1

    if horse_width <= 1 or frame_width <= 1 or fps <= 0:
        print("❌ Normalization invalid")
        return 0.0, 0.0

    dx_norm = (motion_dx * fps) / horse_width
    dy_norm = (motion_dy * fps) / horse_width

    print(
        f"📏 Normalized motion dx_n={dx_norm:.3f} dy_n={dy_norm:.3f}"
    )

    return dx_norm, dy_norm

###################################################################
# 6. Decide walking direction
###################################################################

DX_NORM_THRESHOLD = 0.1

def decide_direction(
    dx_norm,
    dy_norm,
    min_horiz_ratio=1.5,
):
    abs_dx = abs(dx_norm)
    abs_dy = abs(dy_norm)

    if abs_dx < DX_NORM_THRESHOLD:
        print("❌ Direction rejected: weak horizontal motion")
        return None, 0.0

    if abs_dx < abs_dy * min_horiz_ratio:
        print("❌ Direction rejected: vertical dominance")
        return None, 0.0

    direction = "L2R" if dx_norm < 0 else "R2L"
    confidence = abs_dx / (abs_dx + abs_dy + 1e-6)
    confidence = float(np.clip(confidence, 0.0, 1.0))

    print(
        f"➡️ Direction={direction} confidence={confidence:.2f}"
    )

    return direction, confidence


###################################################################
# 7. Preview helper
###################################################################

def resize_for_preview(img, max_w, max_h):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    return cv2.resize(
        img,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA,
    )


def draw_motion_arrow(
    frame,
    horse_bbox,
    dx_norm,
    direction,
    max_length=200,
    alpha=0.5,
    color=(0, 255, 255),
    thickness=10,
):
    """
    Draw semi-transparent arrow showing motion direction & magnitude.

    dx_norm: horse-widths / second
    """

    if horse_bbox is None or direction is None:
        return

    x1, y1, x2, y2 = horse_bbox
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    # Normalize arrow length
    length = int(min(abs(dx_norm) * 200, max_length))
    if length < 10:
        return

    # Direction
    if direction == "L2R":
        dx = length
    elif direction == "R2L":
        dx = -length
    else:
        return

    start = (cx, cy)
    end = (cx + dx, cy)

    # Draw on overlay
    overlay = frame.copy()
    cv2.arrowedLine(
        overlay,
        start,
        end,
        color,
        thickness=thickness,
        tipLength=0.3,
    )

    # Blend overlay
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_on_frame(
    frame,
    total_frames,
    frame_idx,
    horse_bboxes,
    person_bboxes,
    active_horse_bbox=None,
    inlier_points=None,
    direction=None,
    dx_norm=0.0,
    confidence=0.0,
    motion_valid=False,
):
    """
    Draw all preview elements on the frame.

    Draws:
      - All horse bboxes (thin green)
      - Active horse bbox (thick green)
      - Person bboxes (blue)
      - BG inlier points (red dots)
      - Motion label text
      - Motion arrow (optional)

    Args:
        frame:
            Original BGR frame (modified in-place)
        horse_bboxes:
            List of horse bboxes [(x1,y1,x2,y2), ...]
        person_bboxes:
            List of person bboxes
        active_horse_bbox:
            The single valid horse bbox (or None)
        inlier_points:
            Nx2 array of inlier points in ORIGINAL frame coords
        direction:
            "L2R", "R2L", or None
        dx_norm:
            Normalized motion in x
        confidence:
            Confidence score
        valid:
            Whether motion is valid
        draw_arrow_fn:
            Function to draw motion arrow (optional)
    """

    # ----------------------------
    # Horses (all)
    # ----------------------------
    for b in horse_bboxes:
        cv2.rectangle(frame, b[:2], b[2:], (0, 180, 0), 1)

    # Active horse (thick)
    if active_horse_bbox is not None:
        cv2.rectangle(
            frame,
            active_horse_bbox[:2],
            active_horse_bbox[2:],
            (0, 255, 0),
            3,
        )

    # ----------------------------
    # Persons
    # ----------------------------
    for b in person_bboxes:
        cv2.rectangle(frame, b[:2], b[2:], (255, 0, 0), 2)

    # ----------------------------
    # Inlier points
    # ----------------------------
    if inlier_points is not None:
        for (x, y) in inlier_points:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)

    # ----------------------------
    # Frame idx label
    # ----------------------------
    label = f"Frame {frame_idx} / {total_frames}"
    color = (255, 0, 0)
    cv2.putText(
        frame,
        label,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2,
    )

    # ----------------------------
    # Motion label
    # ----------------------------
    if motion_valid:
        label = f"{direction} dx_n={dx_norm:.2f} c={confidence:.2f}"
        color = (0, 255, 0)
    else:
        label = "Motion: None"
        color = (0, 0, 255)

    cv2.putText(
        frame,
        label,
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color,
        2,
    )

    # ----------------------------
    # Motion arrow
    # ----------------------------
    if motion_valid and active_horse_bbox is not None:
        draw_motion_arrow(
            frame,
            active_horse_bbox,
            dx_norm,
            direction,
        )


###################################################################
# 8. Determine Clips (ROBUST)
###################################################################

import numpy as np

###################################################################
# 8. Determine Clips (PER-SEGMENT + ROBUST + FIXED LENGTH)
###################################################################

import numpy as np

MIN_SEGMENT_LEN = 15
MAX_FRAME_GAP = 5
DEFAULT_CLIP_SEC = 2.0


def determine_clips(track, video_info):
    """
    Determine one fixed-length clip per valid motion segment.

    Returns:
        List[dict] or None
    """

    print("    📐 Analyzing motion & perpendicularity...")

    if not track or len(track) < MIN_SEGMENT_LEN:
        print(
            f"     ❌ Track too short "
            f"({0 if not track else len(track)} frames)"
        )
        return None

    # ---------------------------------------------------
    # 1️⃣ Select candidate L→R frames
    # ---------------------------------------------------
    candidates = []

    for p in track:
        if not p.get("motion_valid"):
            continue
        if p.get("direction") != "L2R":
            continue
        if p.get("frame_idx") is None:
            continue

        candidates.append(p)

    print(
        f"     ➤ Candidate L→R frames: "
        f"{len(candidates)} / {len(track)}"
    )

    if len(candidates) < MIN_SEGMENT_LEN:
        print("     ❌ Not enough directional frames")
        return None

    # ---------------------------------------------------
    # 2️⃣ Group into motion segments
    # ---------------------------------------------------
    segments = []
    current = [candidates[0]]

    for prev, curr in zip(candidates, candidates[1:]):
        gap = curr["frame_idx"] - prev["frame_idx"]
        if gap <= MAX_FRAME_GAP:
            current.append(curr)
        else:
            if len(current) >= MIN_SEGMENT_LEN:
                segments.append(current)
            current = [curr]

    if len(current) >= MIN_SEGMENT_LEN:
        segments.append(current)

    print(f"     ➤ Motion segments found: {len(segments)}")

    if not segments:
        return None

    # ---------------------------------------------------
    # 3️⃣ Build one clip per segment
    # ---------------------------------------------------
    fps = video_info.fps
    half_window = int(DEFAULT_CLIP_SEC * fps)
    clip_len = 2 * half_window

    clips = []

    for i, seg in enumerate(segments):
        frames = [p["frame_idx"] for p in seg]

        wh_ratios = [
            p.get("wh_ratio")
            for p in seg
            if p.get("wh_ratio") is not None
        ]

        if len(wh_ratios) < MIN_SEGMENT_LEN // 2:
            print(
                f"       ⚠ Segment {i}: "
                f"insufficient WH ratios ({len(wh_ratios)})"
            )
            continue

        center_idx = int(np.argmax(wh_ratios))
        center_frame = seg[center_idx]["frame_idx"]

        seg_start = frames[0]
        seg_end = frames[-1]

        start_frame = center_frame - half_window
        end_frame = center_frame + half_window

        # Shift right
        if start_frame < seg_start:
            shift = seg_start - start_frame
            start_frame += shift
            end_frame += shift

        # Shift left
        if end_frame > seg_end:
            shift = end_frame - seg_end
            start_frame -= shift
            end_frame -= shift

        # Validate
        if (
            start_frame < seg_start
            or end_frame > seg_end
            or end_frame - start_frame != clip_len
        ):
            print(
                f"       ❌ Segment {i}: "
                f"cannot fit fixed clip "
                f"({seg_start}→{seg_end})"
            )
            continue

        print(
            f"       ✂ Segment {i}: "
            f"clip={start_frame}→{end_frame} "
            f"(center={center_frame})"
        )

        clips.append({
            "direction": "L2R",
            "segment_index": i,
            "center_frame": center_frame,
            "start_frame": start_frame,
            "end_frame": end_frame,
        })

    if not clips:
        print("     ❌ No valid clips produced")
        return None

    print(f"     ✅ Total clips produced: {len(clips)}")
    return clips
