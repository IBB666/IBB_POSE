"""
IBB_POSE - ComfyUI Pose Estimation Node
Supports: Body (17 kpts) / WholeBody (133 kpts) / OpenPose (18 kpts)
Single-person & multi-person detection, pose image + JSON output
Compatible with Python 3.13, torch 2.10+, CUDA 12/13
"""

import warnings
import logging
import os
import gc
import sys
import json
import math
import colorsys
import urllib.request
import traceback
import time
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import torch
    TORCH_AVAILABLE = True
except Exception as _e:
    TORCH_AVAILABLE = False

    class _TorchCudaStub:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def empty_cache():
            return None

    class _TorchDeviceStub:
        def __init__(self, device_type):
            self.type = str(device_type)

        def __str__(self):
            return self.type

    class _TorchStub:
        Tensor = object
        float32 = "float32"
        float16 = "float16"
        bfloat16 = "bfloat16"
        cuda = _TorchCudaStub()

        @staticmethod
        def device(device_type):
            return _TorchDeviceStub(device_type)

    torch = _TorchStub()
    print(f"IBB_POSE: torch not available ({_e}). Importing in limited mode.")

import numpy as np
import cv2
from PIL import Image

# ComfyUI integration (graceful fallback for standalone testing)
try:
    import folder_paths
    IS_COMFYUI = True
except ImportError:
    class _FolderPaths:
        models_dir = os.path.join(os.path.expanduser("~"), "ComfyUI", "models")
        output_dir = os.path.join(os.path.expanduser("~"), "ComfyUI", "output")
        supported_pt_extensions = {".pt", ".pth", ".ckpt", ".safetensors"}
        folder_names_and_paths: dict = {}

        @staticmethod
        def get_filename_list(folder_name):
            return []

        @staticmethod
        def get_full_path(folder_name, filename):
            return None

        @staticmethod
        def add_model_folder_path(folder_name, full_folder_path, is_default=True):
            pass

        @staticmethod
        def get_output_directory():
            return _FolderPaths.output_dir

    folder_paths = _FolderPaths()
    IS_COMFYUI = False

try:
    import model_management
except ImportError:
    class _ModelManagement:
        @staticmethod
        def get_torch_device():
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

        @staticmethod
        def soft_empty_cache():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    model_management = _ModelManagement()

# Optional dependencies
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from ultralytics import YOLO as _YOLO
    ULTRALYTICS_AVAILABLE = True
except Exception as _e:
    ULTRALYTICS_AVAILABLE = False
    print(f"IBB_POSE: ultralytics not available ({_e}). Body/OpenPose mode disabled.")

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except Exception as _e:
    ONNXRUNTIME_AVAILABLE = False
    print(f"IBB_POSE: onnxruntime not available ({_e}). WholeBody mode disabled.")

# Model directory
IBB_POSE_MODEL_DIR = os.path.join(folder_paths.models_dir, "IBB_POSE")
os.makedirs(IBB_POSE_MODEL_DIR, exist_ok=True)

try:
    folder_paths.add_model_folder_path("IBB_POSE", IBB_POSE_MODEL_DIR)
except Exception:
    pass

# Model download URLs
YOLO_POSE_URLS = {
    "small":  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt",
    "medium": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-pose.pt",
    "large":  "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt",
}
YOLO_POSE_FNAMES = {
    "small":  "yolo11n-pose.pt",
    "medium": "yolo11m-pose.pt",
    "large":  "yolo11x-pose.pt",
}
YOLO_DET_URL   = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
DWPOSE_HF_REPO   = "yzd-v/DWPose"
DWPOSE_POSE_FILE = "dw-ll_ucoco_384.onnx"
DWPOSE_DET_FILE  = "yolox_l.onnx"
DWPOSE_MODEL_URLS = {
    DWPOSE_POSE_FILE: f"https://huggingface.co/{DWPOSE_HF_REPO}/resolve/main/{DWPOSE_POSE_FILE}?download=true",
    DWPOSE_DET_FILE:  f"https://huggingface.co/{DWPOSE_HF_REPO}/resolve/main/{DWPOSE_DET_FILE}?download=true",
}

# Keypoint constants
# COCO-17 -> OpenPose-18 index map (-1 = computed Neck)
COCO_TO_OP = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]

# OpenPose-18 limb pairs (1-indexed)
OPENPOSE_LIMB_SEQ = [
    [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8],
    [2, 9], [9, 10], [10, 11], [2, 12], [12, 13], [13, 14],
    [2, 1], [1, 15], [15, 17], [1, 16], [16, 18],
]

BODY_COLORS = [
    [255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
    [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
    [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
    [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
    [255, 0, 170], [255, 0, 85],
]

HAND_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
]


def _download_file(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"IBB_POSE: Downloading {os.path.basename(dest)} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"IBB_POSE: Saved -> {dest}")
    except Exception as exc:
        raise RuntimeError(f"IBB_POSE: Failed to download {url}: {exc}") from exc


def _ensure_yolo_pose_model(model_size: str, auto_download: bool) -> str:
    fname = YOLO_POSE_FNAMES[model_size]
    local = os.path.join(IBB_POSE_MODEL_DIR, fname)
    if not os.path.exists(local):
        if not auto_download:
            raise FileNotFoundError(
                f"IBB_POSE: YOLO pose model not found at {local}. "
                "Enable auto_download or place it manually."
            )
        _download_file(YOLO_POSE_URLS[model_size], local)
    return local


def _ensure_yolo_det_model(auto_download: bool) -> str:
    fname = "yolo11n.pt"
    local = os.path.join(IBB_POSE_MODEL_DIR, fname)
    if not os.path.exists(local):
        if not auto_download:
            raise FileNotFoundError(
                f"IBB_POSE: YOLO detection model not found at {local}."
            )
        _download_file(YOLO_DET_URL, local)
    return local


def _ensure_dwpose_models(auto_download: bool) -> tuple:
    det_path  = os.path.join(IBB_POSE_MODEL_DIR, DWPOSE_DET_FILE)
    pose_path = os.path.join(IBB_POSE_MODEL_DIR, DWPOSE_POSE_FILE)

    for fpath, fname in [(det_path, DWPOSE_DET_FILE), (pose_path, DWPOSE_POSE_FILE)]:
        if not os.path.exists(fpath):
            if not auto_download:
                raise FileNotFoundError(
                    f"IBB_POSE: DWPose model {fname} not found at {fpath}. "
                    "Enable auto_download or place manually."
                )
            try:
                _download_file(DWPOSE_MODEL_URLS[fname], fpath)
            except Exception as exc:
                raise RuntimeError(
                    f"IBB_POSE: Could not download {fname} from HuggingFace: {exc}"
                ) from exc

    return det_path, pose_path


def _tensor_to_bgr(tensor: torch.Tensor) -> np.ndarray:
    img = tensor[0].cpu().numpy()
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _bgr_to_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img_rgb.astype(np.float32) / 255.0).unsqueeze(0)


def _coco17_to_openpose18(kpts: np.ndarray, scores: np.ndarray) -> tuple:
    op_kpts   = np.zeros((18, 2), dtype=np.float32)
    op_scores = np.zeros(18, dtype=np.float32)

    if len(kpts) >= 7:
        neck       = (kpts[5] + kpts[6]) / 2.0
        neck_score = float(min(scores[5], scores[6]))
    else:
        neck       = np.zeros(2, dtype=np.float32)
        neck_score = 0.0

    for i_op, i_coco in enumerate(COCO_TO_OP):
        if i_op == 1:
            op_kpts[i_op]   = neck
            op_scores[i_op] = neck_score
        elif 0 <= i_coco < len(kpts):
            op_kpts[i_op]   = kpts[i_coco]
            op_scores[i_op] = float(scores[i_coco])

    return op_kpts, op_scores


def _coco_wholebody_reorder(kpts: np.ndarray, scores: np.ndarray) -> tuple:
    n = len(kpts)
    out_kpts   = np.zeros((134, 2), dtype=np.float32)
    out_scores = np.zeros(134, dtype=np.float32)

    if n < 17:
        return out_kpts, out_scores

    body_k, body_s = _coco17_to_openpose18(kpts[:17], scores[:17])
    out_kpts[0:18]   = body_k
    out_scores[0:18] = body_s

    if n >= 23:
        out_kpts[18:24]   = kpts[17:23]
        out_scores[18:24] = scores[17:23]

    if n >= 91:
        out_kpts[24:92]   = kpts[23:91]
        out_scores[24:92] = scores[23:91]

    if n >= 112:
        out_kpts[92:113]   = kpts[91:112]
        out_scores[92:113] = scores[91:112]

    if n >= 133:
        out_kpts[113:134]   = kpts[112:133]
        out_scores[113:134] = scores[112:133]

    return out_kpts, out_scores


def _draw_body(canvas, op_kpts, op_scores, threshold=0.3, pose_scale=1.0):
    H, W = canvas.shape[:2]
    avg  = (H + W) / 2.0
    sw   = max(1, int((avg / 256.0) * pose_scale))
    cr   = max(1, int((avg / 192.0) * pose_scale))

    for i, (a1, b1) in enumerate(OPENPOSE_LIMB_SEQ):
        a, b = a1 - 1, b1 - 1
        if a >= len(op_scores) or b >= len(op_scores):
            continue
        if op_scores[a] < threshold or op_scores[b] < threshold:
            continue
        ax, ay = float(op_kpts[a, 0]), float(op_kpts[a, 1])
        bx, by = float(op_kpts[b, 0]), float(op_kpts[b, 1])
        mX     = (ax + bx) / 2.0
        mY     = (ay + by) / 2.0
        length = math.hypot(ax - bx, ay - by)
        if length < 1.0:
            continue
        angle   = math.degrees(math.atan2(ax - bx, ay - by))
        half_l  = max(1, int(length / 2))
        polygon = cv2.ellipse2Poly(
            (int(mX), int(mY)), (half_l, sw), int(angle), 0, 360, 1
        )
        cv2.fillConvexPoly(canvas, polygon, BODY_COLORS[i % len(BODY_COLORS)])

    for i in range(min(18, len(op_kpts))):
        if op_scores[i] < threshold:
            continue
        x, y = int(op_kpts[i, 0]), int(op_kpts[i, 1])
        if 0 <= x < W and 0 <= y < H:
            cv2.circle(canvas, (x, y), cr, BODY_COLORS[i % len(BODY_COLORS)], -1)

    return canvas


def _draw_wholebody(canvas, kpts, scores, threshold=0.3,
                    keep_face=True, keep_hands=True, keep_feet=True, pose_scale=1.0):
    H, W = canvas.shape[:2]
    avg  = (H + W) / 2.0
    sw   = max(1, int(4 * pose_scale))
    cr   = max(1, int(4 * pose_scale))
    f_r  = max(1, int(3 * pose_scale))
    f_lw = max(1, int(2 * pose_scale))

    for i, (a1, b1) in enumerate(OPENPOSE_LIMB_SEQ):
        a, b = a1 - 1, b1 - 1
        if a >= len(scores) or b >= len(scores):
            continue
        if scores[a] < threshold or scores[b] < threshold:
            continue
        ax, ay = float(kpts[a, 0]), float(kpts[a, 1])
        bx, by = float(kpts[b, 0]), float(kpts[b, 1])
        mX     = (ax + bx) / 2.0
        mY     = (ay + by) / 2.0
        length = math.hypot(ax - bx, ay - by)
        if length < 1.0:
            continue
        angle   = math.degrees(math.atan2(ax - bx, ay - by))
        polygon = cv2.ellipse2Poly(
            (int(mX), int(mY)), (max(1, int(length / 2)), sw),
            int(angle), 0, 360, 1
        )
        cv2.fillConvexPoly(canvas, polygon, BODY_COLORS[i % len(BODY_COLORS)])

    for i in range(min(18, len(kpts))):
        if i >= len(scores) or scores[i] < threshold:
            continue
        x, y = int(kpts[i, 0]), int(kpts[i, 1])
        if 0 <= x < W and 0 <= y < H:
            cv2.circle(canvas, (x, y), cr, BODY_COLORS[i % len(BODY_COLORS)], -1)

    if keep_feet and len(kpts) >= 24:
        for i in range(18, 24):
            if i >= len(scores) or scores[i] < threshold:
                continue
            x, y = int(kpts[i, 0]), int(kpts[i, 1])
            if 0 <= x < W and 0 <= y < H:
                cv2.circle(canvas, (x, y), cr, BODY_COLORS[i % len(BODY_COLORS)], -1)

    if keep_face and len(kpts) >= 92:
        for i in range(24, 92):
            if i >= len(scores) or scores[i] < threshold:
                continue
            x, y = int(kpts[i, 0]), int(kpts[i, 1])
            if x > 1 and y > 1 and 0 <= x < W and 0 <= y < H:
                cv2.circle(canvas, (x, y), f_r, (255, 255, 255), -1)

    def _draw_hand(start_idx):
        if len(kpts) < start_idx + 21:
            return
        for ie, (ea, eb) in enumerate(HAND_EDGES):
            a, b = start_idx + ea, start_idx + eb
            if a >= len(scores) or b >= len(scores):
                continue
            if scores[a] < threshold or scores[b] < threshold:
                continue
            x1, y1 = int(kpts[a, 0]), int(kpts[a, 1])
            x2, y2 = int(kpts[b, 0]), int(kpts[b, 1])
            if (x1 > 1 and y1 > 1 and x2 > 1 and y2 > 1
                    and 0 <= x1 < W and 0 <= y1 < H
                    and 0 <= x2 < W and 0 <= y2 < H):
                red, green, blue = colorsys.hsv_to_rgb(ie / len(HAND_EDGES), 1.0, 1.0)
                color = (int(blue * 255), int(green * 255), int(red * 255))
                cv2.line(canvas, (x1, y1), (x2, y2), color, f_lw)
        for i in range(start_idx, start_idx + 21):
            if i >= len(scores) or scores[i] < threshold:
                continue
            x, y = int(kpts[i, 0]), int(kpts[i, 1])
            if x > 1 and y > 1 and 0 <= x < W and 0 <= y < H:
                cv2.circle(canvas, (x, y), f_r, (0, 0, 255), -1)

    if keep_hands:
        _draw_hand(92)
        _draw_hand(113)

    return canvas


def _fmt_kpts(kpts, scores, start, end, threshold, filter_low):
    out = []
    for i in range(start, end):
        if i >= len(kpts):
            out.extend([0.0, 0.0, 0.0])
        else:
            s = float(scores[i]) if i < len(scores) else 0.0
            if filter_low and s < threshold:
                out.extend([0.0, 0.0, 0.0])
            else:
                out.extend([float(kpts[i, 0]), float(kpts[i, 1]), s])
    return out


def _build_openpose_json(all_kpts, all_scores, image_width, image_height,
                          skeleton_type="body", threshold=0.3, filter_low=True):
    if image_width == 0 or image_height == 0:
        return json.dumps({"people": [], "canvas_width": image_width,
                           "canvas_height": image_height})

    people = []
    for kpts, scores in zip(all_kpts, all_scores):
        if kpts is None or len(kpts) == 0:
            continue
        person = {}

        if skeleton_type in ("body", "openpose"):
            if kpts.shape[0] == 17:
                op_k, op_s = _coco17_to_openpose18(kpts, scores)
            else:
                op_k, op_s = kpts[:18], scores[:18]
            person["pose_keypoints_2d"]       = _fmt_kpts(op_k, op_s, 0, 18, threshold, filter_low)
            person["face_keypoints_2d"]       = []
            person["hand_left_keypoints_2d"]  = []
            person["hand_right_keypoints_2d"] = []
            person["foot_keypoints_2d"]       = []
        else:
            person["pose_keypoints_2d"]       = _fmt_kpts(kpts, scores,   0,  18, threshold, filter_low)
            person["foot_keypoints_2d"]       = _fmt_kpts(kpts, scores,  18,  24, threshold, filter_low)
            person["face_keypoints_2d"]       = _fmt_kpts(kpts, scores,  24,  92, threshold, filter_low)
            person["hand_left_keypoints_2d"]  = _fmt_kpts(kpts, scores,  92, 113, threshold, filter_low)
            person["hand_right_keypoints_2d"] = _fmt_kpts(kpts, scores, 113, 134, threshold, filter_low)

        people.append(person)

    return json.dumps({"people": people,
                       "canvas_width": int(image_width),
                       "canvas_height": int(image_height)}, indent=2)


def _preprocess_for_dwpose(img_bgr, input_wh=(288, 384)):
    W_in, H_in  = input_wh
    img_rgb     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (W_in, H_in)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_resized - mean) / std
    return img_norm.transpose(2, 0, 1)[np.newaxis].astype(np.float32)


def _decode_simcc(simcc_x, simcc_y, split_ratio=2.0):
    kpts_x  = np.argmax(simcc_x[0], axis=1).astype(np.float32) / split_ratio
    kpts_y  = np.argmax(simcc_y[0], axis=1).astype(np.float32) / split_ratio
    score_x = np.max(simcc_x[0], axis=1)
    score_y = np.max(simcc_y[0], axis=1)
    scores  = (score_x + score_y) / 2.0
    kpts    = np.stack([kpts_x, kpts_y], axis=1)
    return kpts, scores


def _run_dwpose_crop(pose_session, img_bgr, bbox, input_wh=(288, 384)):
    H_img, W_img = img_bgr.shape[:2]

    if bbox is not None:
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(W_img, int(bbox[2]))
        y2 = min(H_img, int(bbox[3]))
        if x2 <= x1 or y2 <= y1:
            return None, None
        crop     = img_bgr[y1:y2, x1:x2]
        crop_w   = x2 - x1
        crop_h   = y2 - y1
        offset_x = x1
        offset_y = y1
    else:
        crop     = img_bgr
        crop_w   = W_img
        crop_h   = H_img
        offset_x = 0
        offset_y = 0

    W_in, H_in = input_wh
    inp = _preprocess_for_dwpose(crop, input_wh)

    try:
        iname  = pose_session.get_inputs()[0].name
        output = pose_session.run(None, {iname: inp})
    except Exception as exc:
        print(f"IBB_POSE: DWPose inference error: {exc}")
        return None, None

    simcc_x, simcc_y = output[0], output[1]
    kpts_crop, scores = _decode_simcc(simcc_x, simcc_y)

    kpts_crop[:, 0] = kpts_crop[:, 0] * (crop_w / W_in) + offset_x
    kpts_crop[:, 1] = kpts_crop[:, 1] * (crop_h / H_in) + offset_y

    kpts_out, scores_out = _coco_wholebody_reorder(kpts_crop, scores)
    return kpts_out, scores_out


def _letterbox_yolox(img, new_shape=(640, 640)):
    h0, w0 = img.shape[:2]
    r  = min(new_shape[0] / h0, new_shape[1] / w0)
    h1 = int(round(h0 * r))
    w1 = int(round(w0 * r))
    dh = (new_shape[0] - h1) / 2.0
    dw = (new_shape[1] - w1) / 2.0
    img_r  = cv2.resize(img, (w1, h1))
    top    = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left   = int(round(dw - 0.1))
    right  = int(round(dw + 0.1))
    padded = cv2.copyMakeBorder(
        img_r, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return padded, r, (dw, dh)


def _nms(boxes, scores, iou_thresh=0.45):
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs):
        i = int(idxs[0])
        keep.append(i)
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        xi1  = np.maximum(boxes[i, 0], boxes[rest, 0])
        yi1  = np.maximum(boxes[i, 1], boxes[rest, 1])
        xi2  = np.minimum(boxes[i, 2], boxes[rest, 2])
        yi2  = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(xi2 - xi1, 0) * np.maximum(yi2 - yi1, 0)
        ai    = (boxes[i,    2] - boxes[i,    0]) * (boxes[i,    3] - boxes[i,    1])
        ar    = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        union = ai + ar - inter
        iou   = np.where(union > 0, inter / union, 0.0)
        idxs  = rest[iou < iou_thresh]
    return keep


def _detect_persons_yolox(det_session, img_bgr, conf_thresh=0.45):
    H_orig, W_orig = img_bgr.shape[:2]
    fallback = [[0, 0, W_orig, H_orig]]

    padded, scale, (dw, dh) = _letterbox_yolox(img_bgr, (640, 640))
    padded_rgb = padded[:, :, ::-1].astype(np.float32)
    inp = padded_rgb.transpose(2, 0, 1)[np.newaxis]

    try:
        iname  = det_session.get_inputs()[0].name
        preds  = det_session.run(None, {iname: inp})[0]
        preds  = preds[0]
    except Exception as exc:
        print(f"IBB_POSE: YOLOX detection error: {exc}")
        return fallback

    if preds.ndim != 2 or preds.shape[1] < 6:
        return fallback

    obj_conf  = preds[:, 4]
    cls_score = preds[:, 5:]
    person_s  = cls_score[:, 0] * obj_conf
    mask      = person_s > conf_thresh

    if not np.any(mask):
        return fallback

    boxes_cxcywh = preds[mask, :4]
    scores_f     = person_s[mask]

    boxes_xyxy = np.zeros_like(boxes_cxcywh)
    boxes_xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2

    keep       = _nms(boxes_xyxy, scores_f)
    boxes_xyxy = boxes_xyxy[keep]

    result = []
    for box in boxes_xyxy:
        x1 = max(0,      int((box[0] - dw) / scale))
        y1 = max(0,      int((box[1] - dh) / scale))
        x2 = min(W_orig, int((box[2] - dw) / scale))
        y2 = min(H_orig, int((box[3] - dh) / scale))
        if x2 > x1 and y2 > y1:
            result.append([x1, y1, x2, y2])

    return result if result else fallback


class IBBLoadPoseModel:
    """
    IBB_POSE — Load Model

    Body / OpenPose: ultralytics YOLO pose (17 COCO keypoints -> OpenPose-18).
    WholeBody:       DWPose ONNX (133 keypoints: body + face + hands + feet).
    auto_download:   Fetch models on first use; disable for air-gapped systems.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "skeleton_type": (["Body", "WholeBody", "OpenPose"],),
                "model_size":    (["small", "medium", "large"],),
                "device":        (["auto", "cuda", "cpu"],),
                "precision":     (["fp32", "fp16", "bf16"],),
                "auto_download": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES  = ("IBB_POSE_MODEL",)
    RETURN_NAMES  = ("ibb_pose_model",)
    FUNCTION      = "load_model"
    CATEGORY      = "IBB_POSE"
    OUTPUT_NODE   = False

    def load_model(self, skeleton_type, model_size, device, precision, auto_download):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "IBB_POSE: torch is required to run the node. "
                "Please launch it inside ComfyUI or install torch first."
            )

        if device == "auto":
            dev = model_management.get_torch_device()
        else:
            dev = torch.device(device)

        dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        dtype = dtype_map[precision]
        if dev.type == "cpu" and dtype != torch.float32:
            print("IBB_POSE: CPU only supports fp32 – switching to fp32.")
            dtype = torch.float32

        model_info = {
            "skeleton_type": skeleton_type.lower(),
            "device":        dev,
            "dtype":         dtype,
            "auto_download": auto_download,
        }

        if skeleton_type in ("Body", "OpenPose"):
            if not ULTRALYTICS_AVAILABLE:
                raise ImportError(
                    "IBB_POSE: ultralytics required for Body/OpenPose.\n"
                    "  Install: pip install ultralytics"
                )
            model_path = _ensure_yolo_pose_model(model_size, auto_download)
            print(f"IBB_POSE: Loading YOLO pose model -> {model_path}")
            yolo = _YOLO(model_path)
            model_info["backend"]   = "yolo_pose"
            model_info["yolo_pose"] = yolo
            try:
                det_path = _ensure_yolo_det_model(auto_download)
                model_info["yolo_det"] = _YOLO(det_path)
            except Exception as exc:
                print(f"IBB_POSE: YOLO detector load failed ({exc}). Multi-person via pose model.")
                model_info["yolo_det"] = None

        else:  # WholeBody
            if not ONNXRUNTIME_AVAILABLE:
                raise ImportError(
                    "IBB_POSE: onnxruntime required for WholeBody.\n"
                    "  Install: pip install onnxruntime  (or onnxruntime-gpu)"
                )
            det_path, pose_path = _ensure_dwpose_models(auto_download)

            providers = ["CPUExecutionProvider"]
            if dev.type != "cpu":
                cuda_prov = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                try:
                    _t = ort.InferenceSession(pose_path, providers=cuda_prov)
                    providers = cuda_prov
                    del _t
                except Exception:
                    print("IBB_POSE: ONNX CUDAExecutionProvider unavailable – using CPU.")

            print(f"IBB_POSE: Loading DWPose ONNX | providers: {providers}")
            det_session  = ort.InferenceSession(det_path,  providers=providers)
            pose_session = ort.InferenceSession(pose_path, providers=providers)

            model_info["backend"]      = "dwpose_onnx"
            model_info["det_session"]  = det_session
            model_info["pose_session"] = pose_session

            if ULTRALYTICS_AVAILABLE:
                try:
                    det_yolo_path = _ensure_yolo_det_model(auto_download)
                    model_info["yolo_det"] = _YOLO(det_yolo_path)
                except Exception as exc:
                    print(f"IBB_POSE: YOLO det unavailable ({exc}). Falling back to YOLOX.")
                    model_info["yolo_det"] = None
            else:
                model_info["yolo_det"] = None

        print(f"IBB_POSE: Model ready. skeleton={skeleton_type} device={dev} dtype={dtype}")
        return (model_info,)


class IBBRunPoseEstimation:
    """
    IBB_POSE — Run Estimation

    Detects persons (single or multi) and estimates poses.
    Outputs: visualisation image AND/OR OpenPose JSON string.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ibb_pose_model":  ("IBB_POSE_MODEL",),
                "image":           ("IMAGE",),
                "detection_mode":  (["single", "multi"],),
                "output_type":     (["PoseImage", "JSON", "Both"],),
                "score_threshold": ("FLOAT",  {"default": 0.3, "min": 0.05, "max": 0.95, "step": 0.05}),
                "overlay_alpha":   ("FLOAT",  {"default": 1.0, "min": 0.0,  "max": 1.0,  "step": 0.05}),
                "pose_scale":      ("FLOAT",  {"default": 1.0, "min": 0.1,  "max": 5.0,  "step": 0.1}),
            },
            "optional": {
                "keep_face":       ("BOOLEAN", {"default": True,  "label_on": "Keep Face",  "label_off": "No Face"}),
                "keep_hands":      ("BOOLEAN", {"default": True,  "label_on": "Keep Hands", "label_off": "No Hands"}),
                "keep_feet":       ("BOOLEAN", {"default": True,  "label_on": "Keep Feet",  "label_off": "No Feet"}),
                "save_json":       ("BOOLEAN", {"default": False}),
                "filename_prefix": ("STRING",  {"default": "poses/ibb_pose"}),
            },
        }

    RETURN_TYPES  = ("IMAGE", "STRING")
    RETURN_NAMES  = ("pose_image", "pose_json")
    FUNCTION      = "estimate"
    CATEGORY      = "IBB_POSE"
    OUTPUT_NODE   = True

    @staticmethod
    def _detect_yolo(yolo_det, img_bgr, conf=0.45):
        H, W = img_bgr.shape[:2]
        try:
            results = yolo_det(img_bgr, classes=[0], verbose=False)
            boxes   = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                bbs   = boxes.xyxy.cpu().numpy().tolist()
                confs = boxes.conf.cpu().numpy()
                out   = [b for b, c in zip(bbs, confs) if c >= conf]
                if out:
                    return out
        except Exception as exc:
            print(f"IBB_POSE: YOLO detection error: {exc}")
        return [[0, 0, W, H]]

    @staticmethod
    def _run_yolo_pose(yolo_pose, img_bgr, detection_mode, yolo_det, threshold):
        H, W = img_bgr.shape[:2]

        if detection_mode == "single":
            try:
                results  = yolo_pose(img_bgr, verbose=False)
                kps_data = results[0].keypoints
                if kps_data is not None and len(kps_data) > 0:
                    kpts_arr = kps_data.xy.cpu().numpy()
                    conf_arr = kps_data.conf.cpu().numpy()
                    if len(kpts_arr) > 0:
                        best = int(np.argmax(conf_arr.sum(axis=1)))
                        return ([kpts_arr[best].astype(np.float32)],
                                [conf_arr[best].astype(np.float32)])
            except Exception as exc:
                print(f"IBB_POSE: YOLO pose (single) error: {exc}")
            return [], []

        # multi
        all_kpts, all_scores = [], []
        try:
            results  = yolo_pose(img_bgr, verbose=False)
            kps_data = results[0].keypoints
            if kps_data is not None and len(kps_data) > 0:
                kpts_arr = kps_data.xy.cpu().numpy()
                conf_arr = kps_data.conf.cpu().numpy()
                for k, s in zip(kpts_arr, conf_arr):
                    all_kpts.append(k.astype(np.float32))
                    all_scores.append(s.astype(np.float32))
        except Exception as exc:
            print(f"IBB_POSE: YOLO pose (multi) error: {exc}")
        return all_kpts, all_scores

    @staticmethod
    def _run_wholebody(model_info, img_bgr, detection_mode, conf):
        H, W       = img_bgr.shape[:2]
        pose_sess  = model_info["pose_session"]
        yolo_det   = model_info.get("yolo_det")
        det_sess   = model_info.get("det_session")

        if detection_mode == "single":
            kpts, scores = _run_dwpose_crop(pose_sess, img_bgr, None)
            if kpts is None:
                return [], []
            return [kpts], [scores]

        bboxes = []
        if yolo_det is not None:
            try:
                results = yolo_det(img_bgr, classes=[0], verbose=False)
                boxes   = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    bbs   = boxes.xyxy.cpu().numpy().tolist()
                    confs = boxes.conf.cpu().numpy()
                    bboxes = [b for b, c in zip(bbs, confs) if c >= conf]
            except Exception as exc:
                print(f"IBB_POSE: YOLO det for wholebody failed: {exc}")

        if not bboxes and det_sess is not None:
            bboxes = _detect_persons_yolox(det_sess, img_bgr, conf)

        if not bboxes:
            bboxes = [[0, 0, W, H]]

        all_kpts, all_scores = [], []
        for bb in bboxes:
            k, s = _run_dwpose_crop(pose_sess, img_bgr, bb)
            if k is not None:
                all_kpts.append(k)
                all_scores.append(s)

        return all_kpts, all_scores

    def estimate(self, ibb_pose_model, image, detection_mode, output_type,
                 score_threshold, overlay_alpha, pose_scale,
                 keep_face=True, keep_hands=True, keep_feet=True,
                 save_json=False, filename_prefix="poses/ibb_pose"):

        backend   = ibb_pose_model.get("backend", "yolo_pose")
        skel_type = ibb_pose_model.get("skeleton_type", "body")
        threshold = score_threshold

        out_frames   = []
        json_results = []

        for b_idx in range(image.shape[0]):
            frame_tensor = image[b_idx:b_idx+1]
            img_bgr      = _tensor_to_bgr(frame_tensor)
            H, W         = img_bgr.shape[:2]

            all_kpts: list   = []
            all_scores: list = []

            try:
                if backend == "yolo_pose":
                    yolo_pose = ibb_pose_model["yolo_pose"]
                    yolo_det  = ibb_pose_model.get("yolo_det")
                    all_kpts, all_scores = self._run_yolo_pose(
                        yolo_pose, img_bgr, detection_mode, yolo_det, threshold
                    )
                elif backend == "dwpose_onnx":
                    all_kpts, all_scores = self._run_wholebody(
                        ibb_pose_model, img_bgr, detection_mode, threshold
                    )
                else:
                    print(f"IBB_POSE: Unknown backend '{backend}'")
            except Exception as exc:
                print(f"IBB_POSE: Estimation error frame {b_idx}: {exc}")
                traceback.print_exc()

            # Build pose image
            if output_type in ("PoseImage", "Both"):
                if overlay_alpha >= 1.0:
                    canvas = np.zeros_like(img_bgr)
                else:
                    canvas = (img_bgr * (1.0 - overlay_alpha)).astype(np.uint8)

                if skel_type in ("body", "openpose"):
                    for kpts, scores in zip(all_kpts, all_scores):
                        if kpts.shape[0] == 17:
                            op_k, op_s = _coco17_to_openpose18(kpts, scores)
                        else:
                            op_k, op_s = kpts[:18], scores[:18]
                        canvas = _draw_body(canvas, op_k, op_s, threshold, pose_scale)
                else:
                    for kpts, scores in zip(all_kpts, all_scores):
                        if kpts is not None:
                            canvas = _draw_wholebody(
                                canvas, kpts, scores, threshold,
                                keep_face, keep_hands, keep_feet, pose_scale
                            )

                out_frames.append(_bgr_to_tensor(canvas))
            else:
                out_frames.append(frame_tensor)

            # Build JSON
            json_str = _build_openpose_json(
                all_kpts, all_scores, W, H,
                skeleton_type=skel_type,
                threshold=threshold,
                filter_low=True,
            )
            json_results.append(json_str)

        # Save JSON
        if save_json:
            try:
                if hasattr(folder_paths, "get_output_directory"):
                    base_out = folder_paths.get_output_directory()
                else:
                    base_out = folder_paths.output_dir
                prefix_dir = os.path.dirname(filename_prefix)
                out_dir    = os.path.join(base_out, prefix_dir) if prefix_dir else base_out
                os.makedirs(out_dir, exist_ok=True)
                base_name = os.path.basename(filename_prefix)
                fname = f"{base_name}_{int(time.time())}.json"
                fpath = os.path.join(out_dir, fname)
                combined = (
                    json_results[0] if len(json_results) == 1
                    else json.dumps(
                        {"frames": [json.loads(j) for j in json_results]}, indent=2
                    )
                )
                with open(fpath, "w", encoding="utf-8") as fh:
                    fh.write(combined)
                print(f"IBB_POSE: JSON saved -> {fpath}")
            except Exception as exc:
                print(f"IBB_POSE: Could not save JSON: {exc}")

        pose_image_batch = torch.cat(out_frames, dim=0)
        final_json = (
            json_results[-1] if json_results
            else json.dumps({"people": [], "canvas_width": 0, "canvas_height": 0})
        )
        return (pose_image_batch, final_json)


class IBBPoseJsonToImage:
    """
    IBB_POSE — JSON to Image

    Renders an OpenPose JSON string back to a pose visualisation image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_json":  ("STRING",  {"forceInput": True}),
                "width":      ("INT",     {"default": 512, "min": 64, "max": 8192}),
                "height":     ("INT",     {"default": 768, "min": 64, "max": 8192}),
                "pose_scale": ("FLOAT",   {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "threshold":  ("FLOAT",   {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
                "keep_face":  ("BOOLEAN", {"default": True}),
                "keep_hands": ("BOOLEAN", {"default": True}),
                "keep_feet":  ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("pose_image",)
    FUNCTION     = "json_to_image"
    CATEGORY     = "IBB_POSE"

    def json_to_image(self, pose_json, width, height, pose_scale, threshold,
                      keep_face, keep_hands, keep_feet):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        try:
            data    = json.loads(pose_json)
            people  = data.get("people", [])
            cw      = data.get("canvas_width",  width)
            ch      = data.get("canvas_height", height)
            scale_x = width  / max(cw, 1)
            scale_y = height / max(ch, 1)

            for person in people:
                raw = person.get("pose_keypoints_2d", [])
                if raw:
                    n_k  = len(raw) // 3
                    op_k = np.zeros((n_k, 2), dtype=np.float32)
                    op_s = np.zeros(n_k,      dtype=np.float32)
                    for i in range(n_k):
                        op_k[i, 0] = raw[i * 3]     * scale_x
                        op_k[i, 1] = raw[i * 3 + 1] * scale_y
                        op_s[i]    = raw[i * 3 + 2]
                    canvas = _draw_body(canvas, op_k, op_s, threshold, pose_scale)

                if keep_face:
                    raw_f = person.get("face_keypoints_2d", [])
                    if raw_f:
                        n_f = len(raw_f) // 3
                        f_r = max(1, int(3 * pose_scale))
                        for i in range(n_f):
                            s = raw_f[i * 3 + 2]
                            if s < threshold:
                                continue
                            x = int(raw_f[i * 3]     * scale_x)
                            y = int(raw_f[i * 3 + 1] * scale_y)
                            if 0 < x < width and 0 < y < height:
                                cv2.circle(canvas, (x, y), f_r, (255, 255, 255), -1)

                if keep_hands:
                    f_lw = max(1, int(2 * pose_scale))
                    f_r  = max(1, int(3 * pose_scale))
                    for hand_key in ("hand_left_keypoints_2d", "hand_right_keypoints_2d"):
                        raw_h = person.get(hand_key, [])
                        if not raw_h:
                            continue
                        n_h = len(raw_h) // 3
                        hk  = np.zeros((n_h, 2), dtype=np.float32)
                        hs  = np.zeros(n_h,      dtype=np.float32)
                        for i in range(n_h):
                            hk[i, 0] = raw_h[i * 3]     * scale_x
                            hk[i, 1] = raw_h[i * 3 + 1] * scale_y
                            hs[i]    = raw_h[i * 3 + 2]
                        for ie, (ea, eb) in enumerate(HAND_EDGES):
                            if ea >= n_h or eb >= n_h:
                                continue
                            if hs[ea] < threshold or hs[eb] < threshold:
                                continue
                            x1, y1 = int(hk[ea, 0]), int(hk[ea, 1])
                            x2, y2 = int(hk[eb, 0]), int(hk[eb, 1])
                            if (0 < x1 < width and 0 < y1 < height
                                    and 0 < x2 < width and 0 < y2 < height):
                                red, green, blue = colorsys.hsv_to_rgb(ie / len(HAND_EDGES), 1.0, 1.0)
                                cv2.line(canvas, (x1, y1), (x2, y2),
                                         (int(blue * 255), int(green * 255), int(red * 255)), f_lw)
                        for i in range(n_h):
                            if hs[i] < threshold:
                                continue
                            x, y = int(hk[i, 0]), int(hk[i, 1])
                            if 0 < x < width and 0 < y < height:
                                cv2.circle(canvas, (x, y), f_r, (0, 0, 255), -1)

                if keep_feet:
                    raw_ft = person.get("foot_keypoints_2d", [])
                    if raw_ft:
                        n_ft = len(raw_ft) // 3
                        cr   = max(1, int(4 * pose_scale))
                        for i in range(n_ft):
                            s = raw_ft[i * 3 + 2]
                            if s < threshold:
                                continue
                            x = int(raw_ft[i * 3]     * scale_x)
                            y = int(raw_ft[i * 3 + 1] * scale_y)
                            if 0 <= x < width and 0 <= y < height:
                                cv2.circle(canvas, (x, y), cr,
                                           BODY_COLORS[i % len(BODY_COLORS)], -1)

        except Exception as exc:
            print(f"IBB_POSE: json_to_image error: {exc}")
            traceback.print_exc()

        return (_bgr_to_tensor(canvas),)


NODE_CLASS_MAPPINGS = {
    "IBBLoadPoseModel":     IBBLoadPoseModel,
    "IBBRunPoseEstimation": IBBRunPoseEstimation,
    "IBBPoseJsonToImage":   IBBPoseJsonToImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IBBLoadPoseModel":     "IBB Pose — Load Model",
    "IBBRunPoseEstimation": "IBB Pose — Run Estimation",
    "IBBPoseJsonToImage":   "IBB Pose — JSON to Image",
}
