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
2. Install base dependencies:
   ```bash
   cd IBB_POSE
   pip install -r requirements.txt
   ```
3. Install only the backend you plan to use:
   ```bash
   # Choose ONE backend path based on your workflow / hardware:

   # Body / OpenPose
   pip install ultralytics

   # WholeBody (CPU)
   pip install onnxruntime

   # WholeBody (GPU, optional replacement for CPU package)
   # python -m pip uninstall -y onnxruntime
   # pip install onnxruntime-gpu
   ```
4. Start ComfyUI – models download automatically on first use.

### Windows / ComfyUI install note

- `IBB_POSE` no longer requires `groundingdino-py`, `chumpy` or `huggingface_hub` during node installation.
- If your embedded ComfyUI Python had previous failed installs cached, remove those partial packages before retrying, for example. Only do this if you know other nodes do not rely on those packages:
  ```bash
  python -m pip uninstall -y groundingdino-py chumpy huggingface_hub
  python -m pip cache purge
  ```
- The node now keeps heavy pose backends optional and only imports them when the selected mode needs them.
- The ComfyUI runtime no longer needs the OpenMMLab stack just to load the SDPose decoder. `mmpose` / `mmcv` / `mmengine` are only needed for the standalone evaluation scripts under `/eval`.

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

## 模型目录

- SDPose 模型：`ComfyUI/models/IBB_POSE/`
- YOLO 模型：`ComfyUI/models/yolo/`
- GroundingDINO 模型：`ComfyUI/models/grounding-dino/`

首次运行时会自动下载以下资源：

- `teemosliang/SDPose-Body`
- `teemosliang/SDPose-Wholebody`
- GroundingDINO 配置和权重

仓库已内置 `empty_text_encoder/empty_embedding.safetensors`，也保留了 `generate_empty_embedding.py` 以便重新生成。

## 工作流

示例工作流位于：

- `/workflow/ibb_pose.json`

## 兼容说明

- 依赖 ComfyUI 的 `folder_paths` / `model_management`
- Florence2 检测为可选能力，需要额外安装对应 ComfyUI 节点
- OpenPose 编辑 JSON 可与 [ComfyUI-OpenPose-Editor-jd](https://github.com/judian17/ComfyUI-OpenPose-Editor-jd) 配合使用

## 致谢

- 上游实现：`judian17/ComfyUI-SDPose-OOD`
- 原始项目：`T-S-Liang/SDPose-OOD`
