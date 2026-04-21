# IBB_POSE — ComfyUI Pose Estimation Node

A ComfyUI custom node for robust human pose estimation.  
Inspired by [SDPose-OOD](https://github.com/judian17/ComfyUI-SDPose-OOD).

**Compatibility**: Python 3.13 · torch 2.10+ · CUDA 12/13 · ComfyUI Portable

---

## Features

| Feature | Details |
|---|---|
| **Skeleton types** | Body (17 kpts COCO), OpenPose (18 kpts), WholeBody (133 kpts) |
| **Detection modes** | Single person · Multi-person |
| **Output** | Pose image · OpenPose JSON · Both |
| **Auto-download** | Models fetched automatically on first use |
| **Compatibility** | Python 3.13, torch 2.x, CUDA 12/13, CPU |

---

## Installation

1. Clone into your `ComfyUI/custom_nodes/` directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/IBB666/IBB_POSE
   ```
2. Install dependencies:
   ```bash
   cd IBB_POSE
   pip install -r requirements.txt
   ```
3. Start ComfyUI – models download automatically on first use.

### Manual model placement

| Mode | Model | Destination |
|---|---|---|
| Body / OpenPose | `yolo11n-pose.pt` (small), `yolo11m-pose.pt` (medium), `yolo11x-pose.pt` (large) | `ComfyUI/models/IBB_POSE/` |
| WholeBody | `dw-ll_ucoco_384.onnx` + `yolox_l.onnx` | `ComfyUI/models/IBB_POSE/` |

WholeBody models: [yzd-v/DWPose](https://huggingface.co/yzd-v/DWPose)

---

## Nodes

### `IBB Pose — Load Model`

Loads and caches the pose estimation model.

| Parameter | Options | Description |
|---|---|---|
| `skeleton_type` | Body / WholeBody / OpenPose | Pose skeleton format |
| `model_size` | small / medium / large | Model size / accuracy tradeoff |
| `device` | auto / cuda / cpu | Inference device |
| `precision` | fp32 / fp16 / bf16 | Numeric precision (fp32 for CPU) |
| `auto_download` | True / False | Fetch model files automatically |

**Output**: `IBB_POSE_MODEL`

---

### `IBB Pose — Run Estimation`

Runs pose estimation on one or more images.

| Parameter | Options | Description |
|---|---|---|
| `ibb_pose_model` | IBB_POSE_MODEL | Model from Load node |
| `image` | IMAGE | Input image(s) |
| `detection_mode` | single / multi | Person detection strategy |
| `output_type` | PoseImage / JSON / Both | What to compute |
| `score_threshold` | 0.05–0.95 | Minimum keypoint confidence |
| `overlay_alpha` | 0.0–1.0 | 0=original, 1=pure pose map |
| `pose_scale` | 0.1–5.0 | Line / dot thickness multiplier |
| `keep_face` | bool | Include face keypoints (WholeBody) |
| `keep_hands` | bool | Include hand keypoints (WholeBody) |
| `keep_feet` | bool | Include foot keypoints (WholeBody) |
| `save_json` | bool | Save JSON to `ComfyUI/output/` |
| `filename_prefix` | string | Output JSON file prefix |

**Outputs**: `pose_image` (IMAGE) · `pose_json` (STRING)

---

### `IBB Pose — JSON to Image`

Renders an OpenPose JSON string to a pose visualisation image.

---

## JSON Format

Output follows the standard OpenPose JSON format:

```json
{
  "people": [
    {
      "pose_keypoints_2d":       [x, y, score, ...],
      "face_keypoints_2d":       [...],
      "hand_left_keypoints_2d":  [...],
      "hand_right_keypoints_2d": [...],
      "foot_keypoints_2d":       [...]
    }
  ],
  "canvas_width":  512,
  "canvas_height": 768
}
```

Compatible with [ComfyUI-OpenPose-Editor](https://github.com/judian17/ComfyUI-OpenPose-Editor-jd).

---

## Skeleton Keypoint Layout

### Body / OpenPose (18 pts)
OpenPose-18: nose · neck · shoulders · elbows · wrists · hips · knees · ankles · eyes · ears

### WholeBody (133 pts, DWPose)
- 0–16: Body (COCO-17)
- 17–22: Feet (6)
- 23–90: Face (68)
- 91–111: Left hand (21)
- 112–132: Right hand (21)

---

## Recommended Workflow

```
Load Image → IBB Pose Load Model → IBB Pose Run Estimation → Preview Image
                                                            └→ Save Text (JSON)
```
