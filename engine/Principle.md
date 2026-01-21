# Principle of Analysis

## Requirements

Horse Video Auto Clipper App

- Select only the frames where the horse is clearly walking left-to-right.
- From those frames, detect the moment where the horse is most perpendicular to the camera.
- Extract a standardised segment around this point (e.g., ±75 frames or ±2 seconds depending on FPS).
- Output the clipped video using the original file naming structure.


## Observation

The video was captured with panning camera that tracks horse, so the horse stays roughly centered in the frames and the background moves instead.


## Methods

- Using object detection, we can determine an horse's existance and location in the image.
- Horse's left-to-right walking can be recognized via the bounding box aspect (W/H) ratio and background motion (optical flow) estimation.
- The moment where the horse is most perpendicular to the camera is when the bounding box aspect (W/H) ratio is maximum.


## Approach

- Using object detection (Yolo), detect horse and person(s) in the frame of the video.

- If there is only one horse and the horse is in the effective central region(excluding 5% edge) in the frame image and the bounding box aspect ratio is in the proper range(h/w=1.2~2.0), 
we estimate the background (the image region excluding horse mask) motion via optical flow estimation so that we can determine if the horse walks from left to right.

- We can get effective video segments (a sequence of enough-length consecutive left-to-right walking frames).

- In each effective video segments, we determine the frame where the bounding box aspect (W/H) ratio is maximum and extract a standardised segment around this frame (±2 seconds).


## Algorithm

Determine whether a horse walks **Left → Right** or **Right → Left** in a **panning camera video**, using **background motion estimation** rather than horse displacement.

---

### 1️⃣ Foreground Object Detection (YOLO)

For each video frame, detect foreground objects using YOLO:

* **Horse** → class `17`
* **Person** → class `0`

These objects are treated as **foreground movers** and must be excluded from background motion estimation.

### 2️⃣ Valid Horse Condition

Background motion estimation is performed **only if all conditions below are met**:

* **Exactly one horse** is detected in the frame
* The horse bounding box satisfies:

  * **Area ≥ 5%** of the frame area
  * **Aspect ratio**:
    [
    h / w \in [0.5, 1.0]
    ]
  * Located inside the **central 90%** of the frame (i.e., a 5% margin is excluded on all sides)


### 3️⃣ Background Mask & Feature Detection

#### Background Mask Definition

The background region is defined as:

```
BG region = frame − (horse bounding box ∪ all person bounding boxes)
```

the lower half region 

Pixels belonging to horses or persons **must not contribute** to background motion.

---

#### Feature Detection (Shi–Tomasi)

* Algorithm: `cv2.goodFeaturesToTrack`
* Applied **only inside the BG region**
* Maximum number of features: **600**

This ensures that detected features represent **static background structure**, not moving objects.

---

### 4️⃣ Optical Flow & Motion Model

#### Optical Flow Tracking

* Track BG features between consecutive frames using:

  * **Lucas–Kanade (LK) optical flow**

#### Motion Estimation

* Estimate dominant background motion using:

  * **Affine transform with RANSAC**
  * `cv2.estimateAffinePartial2D`

From the estimated affine matrix `M`, extract translation:

```
motion_dx = M[0, 2]
motion_dy = M[1, 2]
```

These values represent **background displacement per frame**.

---

### 5️⃣ Motion Validation & Normalization

#### Motion Rejection Criteria

Reject background motion estimation if any of the following holds:

* Number of inliers < **20**
* Inlier ratio < **0.5**
* Estimated scale deviates significantly from **1.0** (to avoid zoom / parallax corruption)

#### Motion Normalization

Normalize background motion by **frame rate** and **horse scale**:

```
dx_norm = (motion_dx * fps) / horse_width
dy_norm = (motion_dy * fps) / horse_width
```

This yields **horse-widths per second**, making motion magnitude:

* resolution-independent
* frame-rate–independent
* comparable across videos

---

### 6️⃣ Direction Decision

Determine walking direction from the **orientation and magnitude** of the normalized background motion vector:

* Dominant **horizontal motion** is used
* Sign convention:

  * Negative `dx_norm` → **Horse walking Left → Right**
  * Positive `dx_norm` → **Horse walking Right → Left**

Vertical motion (`dy_norm`) is used only for **sanity checks** or rejection.

---

#### 🖥️ Visualization & Debug Preview

For each frame:

* **Horse bounding box** → green (thick)
* **Person bounding boxes** → blue
* **Background inlier features** → red dots



Expected behavior:

* ❌ No optical-flow points inside horse bbox
* ❌ No optical-flow points inside person bboxes
* ✅ Inliers should lie on static background regions only

If violated, the BG mask is incorrect.

---

### ✅ Final Rule

> **Background motion is estimated exclusively from pixels belonging to neither horses nor people.**


