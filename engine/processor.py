import time
import cv2

from engine.functions import (
    detect_foreground_objects,
    is_horse_bboxes_valid,
    define_bg_mask,
    estimate_bg_motion,
    normalize_bg_motion,
    decide_direction,
    resize_for_preview,
    draw_on_frame,
    determine_clips,
)

from utils.video_io import (
    load_video_info,
    export_clip,
    export_clip_cv2,
)
# ---------------------------------
# Preview settings
# ---------------------------------
MAX_PREVIEW_WIDTH = 960
MAX_PREVIEW_HEIGHT = 720
SHOW_INLIERS = True


class Processor:
    def __init__(
        self,
        output_dir,
        logger=None,
        imgsz=320,
        scale=0.5,
        preview=True,
    ):
        self.output_dir = output_dir
        self.imgsz = imgsz
        self.scale = scale
        self.preview = preview
        self.log = logger or (lambda *_: None)

        self.log("🤖 Processor initialized")
        self.log(f"📐 imgsz={imgsz}, scale={scale}")

    # ---------------------------------------------------------
    def run(self, video_path):
        # ---------------------------------------------------
        # Load metadata
        # ---------------------------------------------------
        video_info = load_video_info(video_path)
        self.log(
            f"       ℹ FPS={video_info.fps:.2f}, "
            f"Frames={video_info.frame_count}, "
            f"Duration={video_info.duration:.2f}s"
        )

                
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.log("❌ Cannot open video")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1:
            fps = 30.0

        # ⏭ Skip every other frame if FPS is high
        skip_every_other = fps > 30
        effective_fps = fps / 2 if skip_every_other else fps

        prev_gray = None
        track = []
        frame_idx = 0

        valid_frames = 0
        ltr_frames = 0
        success = False
        start = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # ⏭ Skip odd frames if FPS > 30
            if skip_every_other and (frame_idx % 2 == 1):
                continue

            fh, fw = frame.shape[:2]

            # ---------------------------------
            # Defaults
            # ---------------------------------
            motion_dx = 0.0
            motion_dy = 0.0
            dx_norm = 0.0
            dy_norm = 0.0
            direction = None
            confidence = 0.0
            motion_valid = False
            wh_ratio = None

            # ---------------------------------
            # 1️⃣ Detection
            # ---------------------------------
            horse_bboxes, person_bboxes = detect_foreground_objects(
                frame_idx,
                frame,
                imgsz=self.imgsz,
            )

            # BG motion only if EXACTLY one valid horse
            horse_valid, wh_ratio = is_horse_bboxes_valid(horse_bboxes, fw, fh)
            horse_bbox = (horse_bboxes[0] if horse_valid else None)

            # ---------------------------------
            # 2️⃣ Prepare frames
            # ---------------------------------
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_s = cv2.resize(gray, None, fx=self.scale, fy=self.scale)

            # ---------------------------------
            # 3️⃣ Background motion
            # ---------------------------------
            if prev_gray is not None and horse_bbox is not None:
                bg_mask = define_bg_mask(
                    frame.shape,
                    horse_bbox,
                    person_bboxes,
                    scale=self.scale,
                )

                motion = estimate_bg_motion(
                    prev_gray,
                    gray_s,
                    bg_mask,
                )

                if motion["valid"]:
                    motion_dx = motion["motion_dx"]
                    motion_dy = motion["motion_dy"]

                    dx_norm, dy_norm = normalize_bg_motion(
                        motion_dx,
                        motion_dy,
                        effective_fps,
                        horse_bbox,
                        fw,
                    )

                    direction, confidence = decide_direction(
                        dx_norm,
                        dy_norm,
                    )

                    if direction is not None:
                        motion_valid = True
                        valid_frames += 1
                        if direction == "L2R":
                            ltr_frames += 1

            # Update previous frame ONLY for processed frames
            prev_gray = gray_s

            # ---------------------------------
            # 4️⃣ Store
            # ---------------------------------
            track.append({
                "frame_idx": frame_idx,
                "motion_valid": motion_valid,
                "wh_ratio": wh_ratio,
                "direction": direction,
                "confidence": confidence,
            })

            self.log(
                f"🔄 {frame_idx:05d} | "
                f"dx_n={dx_norm:.3f} | dir={direction} | c={confidence:.2f}"
            )

            # ---------------------------------
            # 5️⃣ Preview
            # ---------------------------------
            if self.preview:
                inlier_pts = None
                if (
                    horse_valid
                    and SHOW_INLIERS
                    and motion.get("inliers") is not None
                ):
                    mask = motion["inliers"].flatten() == 1
                    pts = motion["pts_next"][mask]
                    inlier_pts = [
                        (p[0] / self.scale, p[1] / self.scale)
                        for p in pts.reshape(-1, 2)
                    ]

                draw_on_frame(
                    frame=frame,
                    horse_bboxes=horse_bboxes,
                    person_bboxes=person_bboxes,
                    active_horse_bbox=horse_bbox,
                    inlier_points=inlier_pts,
                    direction=direction,
                    dx_norm=dx_norm,
                    confidence=confidence,
                    motion_valid=motion_valid,
                )

                cv2.imshow(
                    "BG Motion Estimation",
                    resize_for_preview(
                        frame,
                        MAX_PREVIEW_WIDTH,
                        MAX_PREVIEW_HEIGHT,
                    ),
                )

                if cv2.waitKey(1) & 0xFF == 27:
                    break


        cap.release()
        cv2.destroyAllWindows()

        self.log(
            f"⏱ {time.time() - start:.1f}s | "
            f"{valid_frames}/{frame_idx} valid | "
            f"{ltr_frames} L→R"
        )


        # ---------------------------------------------------
        # Motion analysis
        # ---------------------------------------------------
        clips = determine_clips(track, video_info)

        if not clips:
            self.log(
                "     ❌ Rejected: no valid perpendicular moment"
            )
            return False

        self.log(f"     🎯 {len(clips)} clip(s) selected")

        for i, clip_info in enumerate(clips):
            self.log(
                f"       ▶ Clip {i}: "
                f"center={clip_info['center_frame']} | "
                f"{clip_info['start_frame']} → {clip_info['end_frame']}"
            )

            export_clip_cv2(
                video_path,
                clip_info,
                self.output_dir,
                clip_index=i+1,
            )

            self.log(
                f"       ✅ Clip {i+1} exported successfully"
            )

        success = True

        return success
