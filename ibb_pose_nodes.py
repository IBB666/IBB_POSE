import warnings
import logging
import colorsys
import os
import urllib.request

# Suppress common warnings to reduce noise
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Redirects are currently not supported.*")

# Set logging level to reduce verbose output
logging.getLogger("torch.distributed").setLevel(logging.ERROR)
logging.getLogger("deepspeed").setLevel(logging.ERROR)

def _exc_summary(exc: Exception) -> str:
    return f"{exc.__class__.__name__}: {exc}"


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
    print(
        f"IBB_POSE: torch not available ({_exc_summary(_e)}). "
        "Importing in limited mode only; runtime execution is disabled until torch is installed."
    )

import numpy as np
from PIL import Image
import json
try:
    import folder_paths
except ImportError:
    # If not available, create a mock object for testing
    class MockFolderPaths:
        models_dir = os.path.expanduser("~/models")
        output_dir = os.path.expanduser("~/output")
        supported_pt_extensions = {".pt", ".pth", ".ckpt", ".safetensors"}
        folder_names_and_paths = {}

        @classmethod
        def add_model_folder_path(cls, folder_name, full_folder_path, is_default=True):
            cls.folder_names_and_paths.setdefault(folder_name, [])
            if full_folder_path not in cls.folder_names_and_paths[folder_name]:
                cls.folder_names_and_paths[folder_name].append(full_folder_path)

        @classmethod
        def get_folder_paths(cls, folder_name):
            return cls.folder_names_and_paths.get(folder_name, [])

        @classmethod
        def get_filename_list(cls, folder_name):
            names = []
            for base in cls.get_folder_paths(folder_name):
                if not os.path.isdir(base):
                    continue
                for entry in sorted(os.listdir(base)):
                    path = os.path.join(base, entry)
                    if os.path.isfile(path):
                        names.append(entry)
            return names

        @classmethod
        def get_full_path(cls, folder_name, filename):
            for base in cls.get_folder_paths(folder_name):
                path = os.path.join(base, filename)
                if os.path.exists(path):
                    return path
            return None

        @classmethod
        def get_output_directory(cls):
            os.makedirs(cls.output_dir, exist_ok=True)
            return cls.output_dir

        @classmethod
        def get_save_image_path(cls, filename_prefix, output_dir, width, height):
            subfolder = os.path.dirname(filename_prefix)
            base_filename = os.path.basename(filename_prefix)
            full_output_folder = os.path.join(output_dir, subfolder) if subfolder else output_dir
            os.makedirs(full_output_folder, exist_ok=True)
            return full_output_folder, base_filename, 0, subfolder, filename_prefix
    folder_paths = MockFolderPaths()
import sys
from pathlib import Path
import glob
import cv2
import math
import tempfile
try:
    import model_management
except ImportError:
    # If not available, create a mock object for testing
    class MockModelManagement:
        @staticmethod
        def get_torch_device():
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        @staticmethod
        def soft_empty_cache():
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    model_management = MockModelManagement()


# Optional dependencies
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from ultralytics import YOLO as _YOLO
    ULTRALYTICS_AVAILABLE = True
    YOLO = _YOLO
except Exception as _e:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None
    print(
        f"IBB_POSE: ultralytics not available ({_exc_summary(_e)}). "
        "Body/OpenPose mode disabled."
    )
YOLO_AVAILABLE = ULTRALYTICS_AVAILABLE

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except Exception as _e:
    ONNXRUNTIME_AVAILABLE = False
    print(
        f"IBB_POSE: onnxruntime not available ({_exc_summary(_e)}). "
        "WholeBody mode disabled."
    )

# --- Add custom folder paths to ComfyUI ---
EMPTY_EMBED_DIR = str(Path(__file__).resolve().parent / "empty_text_encoder")
SDPOSE_MODEL_DIR = os.path.join(folder_paths.models_dir, "IBB_POSE")
YOLO_MODEL_DIR = os.path.join(folder_paths.models_dir, "yolo")

try:
    folder_paths.add_model_folder_path("IBB_POSE", SDPOSE_MODEL_DIR, is_default=True)
    folder_paths.add_model_folder_path("yolo", YOLO_MODEL_DIR, is_default=False)
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
YOLO_DET_FNAME = "yolo11n.pt"
YOLO_DET_URL   = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{YOLO_DET_FNAME}"
DWPOSE_HF_REPO   = "yzd-v/DWPose"
DWPOSE_POSE_FILE = "dw-ll_ucoco_384.onnx"
DWPOSE_DET_FILE  = "yolox_l.onnx"
# Direct resolve URLs with HuggingFace's `?download=true` parameter keep
# install-time dependencies minimal by avoiding the `huggingface_hub` SDK. If the
# upstream Hub layout changes, users can still place the same filenames into
# IBB_POSE_MODEL_DIR.
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


def _hand_edge_color(edge_index: int) -> tuple[int, int, int]:
    color = colorsys.hsv_to_rgb(edge_index / float(len(HAND_EDGES)), 1.0, 1.0)
    return tuple(int(channel * 255) for channel in color)


def _download_file(url: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"IBB_POSE: Downloading {os.path.basename(dest)} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        if not os.path.exists(dest) or os.path.getsize(dest) <= 0:
            raise RuntimeError("download produced an empty file")
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
    local = os.path.join(IBB_POSE_MODEL_DIR, YOLO_DET_FNAME)
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

def draw_wholebody_keypoints_openpose_style(canvas, keypoints, scores=None, threshold=0.3, overlay_mode=False, overlay_alpha=0.6, scale_for_xinsr=False, pose_scale=1.0):
    H, W, C = canvas.shape
    
    # --- 动态计算粗细 (基于 pose_scale) ---
    base_stickwidth = int(4 * pose_scale) 
    base_radius = int(4 * pose_scale)
    
    face_hand_radius = max(1, int(3 * pose_scale))
    face_hand_line_width = max(1, int(2 * pose_scale))

    stickwidth = base_stickwidth

    # --- Xinsr Logic (Fixed Multiplier) ---
    if scale_for_xinsr:
        xinsr_fixed_multiplier = 2
        stickwidth = int(base_stickwidth * xinsr_fixed_multiplier)

    body_limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], [1, 16], [16, 18]]
    hand_edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    
    # Body Limbs
    if len(keypoints) >= 18:
        for i, limb in enumerate(body_limbSeq):
            idx1, idx2 = limb[0] - 1, limb[1] - 1
            if idx1 >= 18 or idx2 >= 18: continue
            if scores is not None and (scores[idx1] < threshold or scores[idx2] < threshold): continue
            Y = np.array([keypoints[idx1][0], keypoints[idx2][0]]); X = np.array([keypoints[idx1][1], keypoints[idx2][1]]); mX = np.mean(X); mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            if length < 1: continue
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, colors[i % len(colors)])
        for i in range(18):
            if scores is not None and scores[i] < threshold: continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if 0 <= x < W and 0 <= y < H: cv2.circle(canvas, (x, y), base_radius, colors[i % len(colors)], thickness=-1)
    
    # Feet
    if len(keypoints) >= 24:
        for i in range(18, 24):
            if scores is not None and scores[i] < threshold: continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if 0 <= x < W and 0 <= y < H: cv2.circle(canvas, (x, y), base_radius, colors[i % len(colors)], thickness=-1)
    
    # Hands
    if len(keypoints) >= 113:
        for ie, edge in enumerate(hand_edges):
            idx1, idx2 = 92 + edge[0], 92 + edge[1]
            if scores is not None and (scores[idx1] < threshold or scores[idx2] < threshold): continue
            x1, y1 = int(keypoints[idx1][0]), int(keypoints[idx1][1]); x2, y2 = int(keypoints[idx2][0]), int(keypoints[idx2][1])
            if x1 > 0.01 and y1 > 0.01 and x2 > 0.01 and y2 > 0.01 and 0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H:
                cv2.line(canvas, (x1, y1), (x2, y2), _hand_edge_color(ie), thickness=face_hand_line_width)
        for i in range(92, 113):
            if scores is not None and scores[i] < threshold: continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if x > 0.01 and y > 0.01 and 0 <= x < W and 0 <= y < H: cv2.circle(canvas, (x, y), face_hand_radius, (0, 0, 255), thickness=-1)
    if len(keypoints) >= 134:
        for ie, edge in enumerate(hand_edges):
            idx1, idx2 = 113 + edge[0], 113 + edge[1]
            if scores is not None and (scores[idx1] < threshold or scores[idx2] < threshold): continue
            x1, y1 = int(keypoints[idx1][0]), int(keypoints[idx1][1]); x2, y2 = int(keypoints[idx2][0]), int(keypoints[idx2][1])
            if x1 > 0.01 and y1 > 0.01 and x2 > 0.01 and y2 > 0.01 and 0 <= x1 < W and 0 <= y1 < H and 0 <= x2 < W and 0 <= y2 < H:
                cv2.line(canvas, (x1, y1), (x2, y2), _hand_edge_color(ie), thickness=face_hand_line_width)
        for i in range(113, 134):
            if scores is not None and i < len(scores) and scores[i] < threshold: continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if x > 0.01 and y > 0.01 and 0 <= x < W and 0 <= y < H: cv2.circle(canvas, (x, y), face_hand_radius, (0, 0, 255), thickness=-1)
    
    # Face
    if len(keypoints) >= 92:
        for i in range(24, 92):
            if scores is not None and scores[i] < threshold: continue
            x, y = int(keypoints[i][0]), int(keypoints[i][1])
            if x > 0.01 and y > 0.01 and 0 <= x < W and 0 <= y < H: cv2.circle(canvas, (x, y), face_hand_radius, (255, 255, 255), thickness=-1)
    return canvas

# --- GroundingDINO Prediction Functions (Adapted from SAM2 Node) ---
# (modified for SDPose context)
def load_dino_image(image_pil):
    # Need import from local_groundingdino here
    try:
        from groundingdino.datasets import transforms as T
    except ImportError:
        raise ImportError("IBB_POSE: Failed to import 'groundingdino'. Please install it via pip: 'pip install groundingdino-py'")

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image

# (modified for SDPose context)
def groundingdino_predict(dino_model_wrapper, image_pil, prompt, threshold):
    dino_model = dino_model_wrapper["model"] # Extract the actual model
    
    # (nested function)
    def get_grounding_output(model, image, caption, box_threshold, device):
        caption = caption.lower().strip()
        if not caption.endswith("."):
            caption = caption + "."
            
        # Move model to device just before inference
        model.to(device)
        image = image.to(device)
        
        boxes_filt = torch.tensor([]) # Initialize empty tensor
        try:
            with torch.no_grad():
                outputs = model(image[None], captions=[caption])
            
            logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
            boxes = outputs["pred_boxes"][0]  # (nq, 4) in cxcywh format
            
            # Filter output based on threshold
            filt_mask = logits.max(dim=1)[0] > box_threshold
            logits_filt = logits[filt_mask]  # num_filt, 256
            boxes_filt = boxes[filt_mask]  # num_filt, 4
            
            # Convert boxes from cxcywh to xyxy format
            boxes_filt = box_cxcywh_to_xyxy(boxes_filt)

        except Exception as e:
             print(f"IBB_POSE: GroundingDINO inference failed: {e}")
        finally:
             # Move model back to CPU after inference to save VRAM
             model.to("cpu")
             model_management.soft_empty_cache() # Clean VRAM
             
        return boxes_filt.cpu() # Return results on CPU

    # Helper function to convert box format
    def box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)
        
    # --- Main prediction logic ---
    dino_image = load_dino_image(image_pil.convert("RGB"))
    
    # Use ComfyUI's device management
    device = model_management.get_torch_device()
    
    boxes_filt_norm = get_grounding_output(dino_model, dino_image, prompt, threshold, device)
    
    # If no boxes found, return empty list
    if boxes_filt_norm.shape[0] == 0:
        return []
        
    # Scale boxes to original image size
    H, W = image_pil.size[1], image_pil.size[0]
    boxes_filt_abs = boxes_filt_norm * torch.Tensor([W, H, W, H])
    
    # Convert to list of lists format [[x1, y1, x2, y2], ...]
    return boxes_filt_abs.tolist()

# --- Processing functions (adapted from SDPose_gradio.py) ---
# (Functions detect_person_yolo, preprocess_image_for_sdpose, restore_keypoints_to_original, convert_to_openpose_json are omitted for brevity but would be pasted here)
def detect_person_yolo(image, yolo_model_path, confidence_threshold=0.5):
    if not YOLO_AVAILABLE: return [[0, 0, image.shape[1], image.shape[0]]], False
    try:
        model = YOLO(yolo_model_path)
        results = model(image, verbose=False)
        person_bboxes = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    if int(box.cls[0]) == 0 and float(box.conf[0]) > confidence_threshold:
                        person_bboxes.append(box.xyxy[0].cpu().numpy().tolist())
        if person_bboxes: return person_bboxes, True
        else: return [[0, 0, image.shape[1], image.shape[0]]], False
    except Exception as e:
        print(f"IBB_POSE: YOLO detection failed: {e}")
        return [[0, 0, image.shape[1], image.shape[0]]], False

def preprocess_image_for_sdpose(image, bbox=None, input_size=(768, 1024)):
    from torchvision import transforms

    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else: pil_image = image
    original_size = pil_image.size
    crop_info = (0, 0, pil_image.width, pil_image.height)
    if bbox is not None:
        x1, y1, x2, y2 = map(int, bbox)
        if x2 > x1 and y2 > y1:
            pil_image = pil_image.crop((x1, y1, x2, y2))
            crop_info = (x1, y1, x2 - x1, y2 - y1)
    
    transform = transforms.Compose([
        transforms.Resize((input_size[1], input_size[0]), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    input_tensor = transform(pil_image).unsqueeze(0)
    return input_tensor, original_size, crop_info

def restore_keypoints_to_original(keypoints, crop_info, input_size, original_size):
    x1, y1, crop_w, crop_h = crop_info
    scale_x = crop_w / input_size[0]; scale_y = crop_h / input_size[1]
    keypoints_restored = keypoints.copy()
    keypoints_restored[:, 0] = keypoints[:, 0] * scale_x + x1
    keypoints_restored[:, 1] = keypoints[:, 1] * scale_y + y1
    return keypoints_restored

def convert_to_loader_json(all_keypoints, all_scores, image_width, image_height, keypoint_scheme="body", threshold=0.3, enable_filter=True):

    """
    将关键点转换为编辑器可读的 "Loader" JSON 格式 (OpenPose-18 body only)。
    无论输入是 Body (17点) 还是 WholeBody (134点)，都统一输出为 OpenPose 18点格式。
    """
    people = []
    
    # 确保输入是列表
    if not isinstance(all_keypoints, list):
        return {"width": int(image_width), "height": int(image_height), "people": []}

    for keypoints, scores in zip(all_keypoints, all_scores):
        person_data = {}
        pose_kpts_18 = []
        
        # 确保 keypoints 是 numpy 数组
        if not isinstance(keypoints, np.ndarray):
            keypoints = np.array(keypoints)
        if not isinstance(scores, np.ndarray):
            scores = np.array(scores)

        if keypoint_scheme == "body":
            # --- Body (COCO-17) -> 需要计算 Neck 转换为 OpenPose-18 ---
            if len(keypoints) < 17 or len(scores) < 17:
                continue 

            # 1. 计算 Neck (Shoulder 中点)
            neck = (keypoints[5] + keypoints[6]) / 2
            neck_score = min(scores[5], scores[6])

            # 2. 映射关系
            op_keypoints = np.zeros((18, 2))
            op_scores = np.zeros(18)
            
            coco_to_op_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
            
            for i_op in range(18):
                if i_op == 1: # Neck
                    op_keypoints[i_op] = neck
                    op_scores[i_op] = neck_score
                else:
                    i_coco = coco_to_op_map[i_op]
                    if i_coco < len(keypoints):
                        op_keypoints[i_op] = keypoints[i_coco]
                        op_scores[i_op] = scores[i_coco]
            
            # 3. 扁平化 [x, y, score]
            for i in range(18):
                score = float(op_scores[i])
                if enable_filter and score < threshold:
                    pose_kpts_18.extend([0.0, 0.0, 0.0])
                else:
                    pose_kpts_18.extend([float(op_keypoints[i, 0]), float(op_keypoints[i, 1]), score])

        else: 
            # --- WholeBody (Already OpenPose format) ---
            # 这里的 keypoints 已经在 process_sequence 中被重组为 OpenPose 格式
            # 所以前 18 个点就是我们需要的 Body 18
            if len(keypoints) < 18 or len(scores) < 18:
                continue
                
            for i in range(18):
                score = float(scores[i])
                if enable_filter and score < threshold:
                    pose_kpts_18.extend([0.0, 0.0, 0.0])
                else:
                    pose_kpts_18.extend([float(keypoints[i, 0]), float(keypoints[i, 1]), score])

        person_data["pose_keypoints_2d"] = pose_kpts_18
        people.append(person_data)

    return {"width": int(image_width), "height": int(image_height), "people": people}

def convert_to_openpose_json(all_keypoints, all_scores, image_width, image_height, keypoint_scheme="body"):
    people = []
    
    # 防止除以零错误
    if image_width == 0 or image_height == 0:
        return {"people": [], "canvas_width": int(image_width), "canvas_height": int(image_height)}

    for person_idx, (keypoints, scores) in enumerate(zip(all_keypoints, all_scores)):
        person_data = {}
        
        # --- 修复核心：恢复归一化 (Normalized Coordinates) ---
        # OpenPose 预览工具 (util.py) 期望输入是 0.0-1.0，它会在绘图时自己乘以 W/H。
        def format_kpt(x, y, s):
            return [
                float(x), 
                float(y), 
                float(s)
            ]

        if keypoint_scheme == "body":
            # --- Body Only (17 -> 18) ---
            if len(keypoints) < 17: continue 

            # 计算颈部 (Neck)
            neck = (keypoints[5] + keypoints[6]) / 2 
            neck_score = min(scores[5], scores[6])

            # 映射关系 COCO -> OpenPose Body 18
            coco_to_op_map = [0, -1, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
            
            pose_kpts_18 = []
            for i_op in range(18):
                if i_op == 1: # Neck
                    pose_kpts_18.extend(format_kpt(neck[0], neck[1], neck_score))
                else:
                    i_coco = coco_to_op_map[i_op]
                    if i_coco < len(keypoints):
                        pose_kpts_18.extend(format_kpt(keypoints[i_coco, 0], keypoints[i_coco, 1], scores[i_coco]))
                    else:
                        pose_kpts_18.extend([0.0, 0.0, 0.0])

            person_data["pose_keypoints_2d"] = pose_kpts_18
            person_data["face_keypoints_2d"] = []
            person_data["hand_left_keypoints_2d"] = []
            person_data["hand_right_keypoints_2d"] = []
            person_data["foot_keypoints_2d"] = []

        else: 
            # --- WholeBody (134 Points) ---
            # 这里的 keypoints 是 process_sequence 处理后的：
            # 0-17: Body (含插入的 Neck)
            # 18-23: Foot
            # 24-91: Face (68点)
            # 92-112: Left Hand (21点)
            # 113-133: Right Hand (21点)

            # 1. Body (0-18)
            pose_kpts = []
            for i in range(18):
                pose_kpts.extend(format_kpt(keypoints[i, 0], keypoints[i, 1], scores[i]))
            
            # 2. Foot (18-24)
            foot_kpts = []
            if len(keypoints) >= 24:
                for i in range(18, 24):
                    foot_kpts.extend(format_kpt(keypoints[i, 0], keypoints[i, 1], scores[i]))

            # 3. Face (24-92) -> 70 点 (68 面部标志 + 双眼)
            face_kpts = []
            if len(keypoints) >= 92:
                # 68 面部标志点
                for i in range(24, 92):
                    face_kpts.extend(format_kpt(keypoints[i, 0], keypoints[i, 1], scores[i]))
                # 添加右眼 (body[14]) 和左眼 (body[15])
                face_kpts.extend(format_kpt(keypoints[14, 0], keypoints[14, 1], scores[14]))
                face_kpts.extend(format_kpt(keypoints[15, 0], keypoints[15, 1], scores[15]))

            # 4. Right Hand (92-113) - 官方规范：92-112 = 右手
            right_hand_kpts = []
            if len(keypoints) >= 113:
                for i in range(92, 113):
                    right_hand_kpts.extend(format_kpt(keypoints[i, 0], keypoints[i, 1], scores[i]))

            # 5. Left Hand (113-134) - 官方规范：113-133 = 左手
            left_hand_kpts = []
            if len(keypoints) >= 134:
                for i in range(113, 134):
                    left_hand_kpts.extend(format_kpt(keypoints[i, 0], keypoints[i, 1], scores[i]))

            person_data["pose_keypoints_2d"] = pose_kpts
            person_data["face_keypoints_2d"] = face_kpts
            person_data["hand_right_keypoints_2d"] = right_hand_kpts
            person_data["hand_left_keypoints_2d"] = left_hand_kpts
            person_data["foot_keypoints_2d"] = foot_kpts

        people.append(person_data)
        
    return {"people": people, "canvas_width": int(image_width), "canvas_height": int(image_height)}

def _combine_frame_jsons(frame_jsons):
    """合并所有帧的JSON数据"""
    combined_data = {
        "frame_count": len(frame_jsons),
        "frames": []
    }
    
    for i, frame_json_str in enumerate(frame_jsons):
        try:
            # 检查 frame_json_str 是否已经是字典
            if isinstance(frame_json_str, dict):
                frame_data = frame_json_str
            else:
                frame_data = json.loads(frame_json_str)
            
            frame_data["frame_index"] = i
            combined_data["frames"].append(frame_data)
        except json.JSONDecodeError as e:
            print(f"IBB_POSE: Warning - Failed to parse JSON for frame {i}: {e}")
            combined_data["frames"].append({"frame_index": i, "error": "JSON parsing failed"})
    
    return json.dumps(combined_data, indent=2)

class IBBYOLOModelLoader:
    """ComfyUI node to load a YOLO model for object detection."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("yolo"),),
            }
        }

    RETURN_TYPES = ("YOLO_MODEL",)
    FUNCTION = "load_yolo_model"
    CATEGORY = "IBB_POSE"

    def load_yolo_model(self, model_name):
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics library is not available. Please install it to use YOLO models.")
        
        model_path = folder_paths.get_full_path("yolo", model_name)
        if not model_path:
            raise FileNotFoundError(f"YOLO model not found: {model_name}")
            
        print(f"IBB_POSE: Loading YOLO model from {model_path}")
        model = YOLO(model_path)
        return (model,)


class IBBPoseModelLoader:
    """ComfyUI node to load SDPose models and dependencies."""
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_type": (["Body", "WholeBody"],),
                "unet_precision": (["fp32", "fp16", "bf16"],),
                "device": (["auto", "cuda", "cpu"],),
                "unload_on_finish": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IBB_POSE_MODEL",)
    FUNCTION = "load_pose_model"
    CATEGORY = "IBB_POSE"

    def get_model_path(self, repo_name):
        model_pathes = folder_paths.get_folder_paths("IBB_POSE")
        for path in model_pathes:
            model_path = os.path.join(path, repo_name)
            if os.path.exists(os.path.join(model_path, "unet")):
                return model_path

        return os.path.join(SDPOSE_MODEL_DIR, repo_name)

    def load_pose_model(self, model_type, unet_precision, device, unload_on_finish):
        from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file

        from .models.HeatmapHead import get_heatmap_head
        from .models.ModifiedUNet import Modified_forward

        repo_id = {
            "Body": "teemosliang/SDPose-Body",
            "WholeBody": "teemosliang/SDPose-Wholebody"
        }[model_type]
        
        keypoint_scheme = model_type.lower()
        model_path = self.get_model_path(repo_id.split('/')[-1])
        if not os.path.exists(os.path.join(model_path, "unet")):
            print(f"IBB_POSE: Downloading model from {repo_id} to {model_path}")
            snapshot_download(repo_id=repo_id, local_dir=model_path, local_dir_use_symlinks=False)
        if not TORCH_AVAILABLE:
            raise ImportError(
                "IBB_POSE: torch is required to run the node. "
                "Please launch it inside ComfyUI or install torch first."
            )

        if device == "auto":
            device = model_management.get_torch_device()
        else:
            device = torch.device(device)

        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[unet_precision]
        if device.type == "cpu" and dtype != torch.float32:
            print("IBB_POSE Warning: FP16/BF16 is not supported on CPU. Falling back to FP32.")
            dtype = torch.float32

        print(f"IBB_POSE: Loading model on device: {device} with UNet precision: {unet_precision}")

        embed_path = os.path.join(EMPTY_EMBED_DIR, "empty_embedding.safetensors")
        if not os.path.exists(embed_path):
            raise FileNotFoundError(
                f"Empty embedding not found at '{embed_path}'. Please run 'generate_empty_embedding.py' first."
            )
        empty_text_embed = load_file(embed_path)["empty_text_embed"].to(device)

        unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet", torch_dtype=dtype).to(device)
        unet = Modified_forward(unet, keypoint_scheme=keypoint_scheme)
        vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae").to(device)

        dec_path = os.path.join(model_path, "decoder", "decoder.safetensors")
        hm_decoder = get_heatmap_head(mode=keypoint_scheme).to(device)
        if os.path.exists(dec_path):
            hm_decoder.load_state_dict(load_file(dec_path, device=str(device)), strict=True)
        hm_decoder = hm_decoder.to(dtype)

        noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")

        sdpose_model = {
            "unet": unet,
            "vae": vae,
            "empty_text_embed": empty_text_embed,
            "decoder": hm_decoder,
            "scheduler": noise_scheduler,
            "keypoint_scheme": keypoint_scheme,
            "device": device,
            "unload_on_finish": unload_on_finish,
        }
        return (sdpose_model,)


# --- GroundingDINO Model Loader (Adapted from SAM2 Node) ---
groundingdino_model_dir_name = "grounding-dino"
groundingdino_model_list = {
    "GroundingDINO_SwinT_OGC (694MB)": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py",
        "model_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
    },
    "GroundingDINO_SwinB (938MB)": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinB.cfg.py",
        "model_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth",
    },
}


def list_groundingdino_model():
    return list(groundingdino_model_list.keys())


def get_local_filepath(url, folder_name):
    folder = os.path.join(folder_paths.models_dir, folder_name)
    os.makedirs(folder, exist_ok=True)
    filename = os.path.basename(url.split("?", 1)[0])
    local_path = os.path.join(folder, filename)
    if not os.path.exists(local_path):
        _download_file(url, local_path)
    return local_path


def load_groundingdino_model(model_name):
    try:
        from groundingdino.models import build_model
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.util.utils import clean_state_dict
    except ImportError as exc:
        raise ImportError(
            "IBB_POSE: Failed to import 'groundingdino'. Please install it via pip: 'pip install groundingdino-py'"
        ) from exc

    if model_name not in groundingdino_model_list:
        raise ValueError(f"IBB_POSE: Unknown GroundingDINO model: {model_name}")

    config_path = get_local_filepath(groundingdino_model_list[model_name]["config_url"], groundingdino_model_dir_name)
    model_path = get_local_filepath(groundingdino_model_list[model_name]["model_url"], groundingdino_model_dir_name)

    dino_model_args = SLConfig.fromfile(config_path)
    dino = build_model(dino_model_args)
    checkpoint = torch.load(model_path, map_location="cpu")
    dino.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    dino.eval()
    return dino

# (modified for SDPose context)
class IBBGroundingDinoModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list_groundingdino_model(),),
            }
        }

    CATEGORY = "IBB_POSE" # Match SDPose category
    FUNCTION = "load_groundingdino_model"
    RETURN_TYPES = ("GROUNDING_DINO_MODEL",)

    def load_groundingdino_model(self, model_name):
        dino_model = load_groundingdino_model(model_name)
        # Wrap in a dictionary for potential future extensions
        gd_model_wrapper = {
            "model": dino_model,
            "model_name": model_name,
        }
        return (gd_model_wrapper,)

class IBBPoseProcessor:
    """
    ComfyUI node to run SDPose inference using true parallel batching.
    Implements a 3-stage process:
    1. Collect: Detect all persons in all frames using YOLO.
    2. Batch Inference: Run SDPose model in batches on detected persons.
    3. Reconstruct: Draw poses back onto their original frames.
    """
    
    # 一个简单的数据类，用于跟踪检测到的人
    class DetectionJob:
        """Track one detected person crop through preprocessing, inference, and restore."""

        def __init__(self, frame_idx, person_idx, input_tensor, crop_info):
            self.frame_idx = frame_idx
            self.person_idx = person_idx
            self.input_tensor = input_tensor
            self.crop_info = crop_info
            
            # 这些字段将在推理后填充
            self.kpts = None
            self.scores = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ibb_pose_model": ("IBB_POSE_MODEL",),
                "images": ("IMAGE",),
                "score_threshold": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 0.9, "step": 0.05}),
                "overlay_alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
            },
            "optional": {
                "data_from_florence2": ("JSON",),
                "grounding_dino_model": ("GROUNDING_DINO_MODEL",),
                "prompt": ("STRING", {"default": "person .", "multiline": False}),
                "gd_threshold": ("FLOAT", {"default": 0.3, "min": 0, "max": 1.0, "step": 0.01}),
                "yolo_model": ("YOLO_MODEL",),
                "save_for_editor": ("BOOLEAN", {"default": False}),
                "filename_prefix_edit": ("STRING", {"default": "poses/ibb_pose"}),
                # --- 编辑选项 ---
                "keep_face": ("BOOLEAN", {"default": True, "label_on": "Keep Face", "label_off": "Remove Face"}),
                "keep_hands": ("BOOLEAN", {"default": True, "label_on": "Keep Hands", "label_off": "Remove Hands"}),
                "keep_feet": ("BOOLEAN", {"default": True, "label_on": "Keep Feet", "label_off": "Remove Feet"}),
                "scale_for_xinsr": ("BOOLEAN", {"default": False, "label_on": "Xinsr CN Scale", "label_off": "Default Scale"}),
                # --- 核心修复：找回 Pose Scale 参数 ---
                "pose_scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1, "label": "Pose Size Scale"}),
                "enable_confidence_filter": ("BOOLEAN", {
                    "default": True,
                    "label_on": "Filter Low Confidence",
                    "label_off": "Keep All Keypoints"
                }),
            }
            
        }

    RETURN_TYPES = ("IMAGE", "POSE_KEYPOINT")
    RETURN_NAMES = ("images", "pose_keypoint")
    FUNCTION = "process_sequence"
    CATEGORY = "IBB_POSE"

    def process_sequence(
        self,
        ibb_pose_model,
        images,
        score_threshold,
        overlay_alpha,
        batch_size=8,
        grounding_dino_model=None,
        prompt="person .",
        gd_threshold=0.3,
        yolo_model=None,
        save_for_editor=False,
        filename_prefix_edit="poses/ibb_pose",
        data_from_florence2=None,
        keep_face=True,
        keep_hands=True,
        keep_feet=True,
        scale_for_xinsr=False,
        pose_scale_factor=1.0, # <--- 必须包含此参数
        enable_confidence_filter=True
    ):

        import gc 

        device = ibb_pose_model["device"]
        keypoint_scheme = ibb_pose_model["keypoint_scheme"]
        input_size = (768, 1024)
        
        B, H, W, C = images.shape
        images_np = images.cpu().numpy()
        print(f"IBB_POSE: Received {B} frames. Starting 3-stage process with batch_size={batch_size}.")

        # 检查模型设备
        try:
            for key in ["unet", "vae", "decoder"]:
                if key in ibb_pose_model:
                    current_device = next(ibb_pose_model[key].parameters()).device
                    if current_device != device:
                        ibb_pose_model[key].to(device)
        except Exception as e:
            print(f"IBB_POSE: CRITICAL ERROR moving models to device: {e}. Aborting.")
            raise e
            
        class MockPipeline:
            def __init__(self, model_dict):
                self.unet = model_dict["unet"]
                self.vae = model_dict["vae"]
                self.decoder = model_dict["decoder"]
                self.scheduler = model_dict["scheduler"]
                self.empty_text_embed = model_dict["empty_text_embed"]
                self.device = model_dict["device"]

            @torch.no_grad()
            def __call__(self, rgb_in, timesteps, test_cfg, **kwargs):
                unet_dtype = self.unet.dtype
                bsz = rgb_in.shape[0]
                
                rgb_latent = self.vae.encode(rgb_in).latent_dist.sample() * 0.18215
                rgb_latent = rgb_latent.to(dtype=unet_dtype)
                t = torch.tensor(timesteps, device=self.device).long()
                text_embed = self.empty_text_embed.repeat((bsz, 1, 1)).to(dtype=unet_dtype)
                task_emb_anno = torch.tensor([1, 0]).float().unsqueeze(0).to(self.device)
                task_emb_anno = torch.cat([torch.sin(task_emb_anno), torch.cos(task_emb_anno)], dim=-1).repeat(bsz, 1)
                task_emb_anno = task_emb_anno.to(dtype=unet_dtype)
                
                feat = self.unet(rgb_latent, t, text_embed, class_labels=task_emb_anno, return_dict=False, return_decoder_feats=True)
                return self.decoder.predict((feat,), None, test_cfg=test_cfg)

        pipeline = MockPipeline(ibb_pose_model)
        
        # --- 步骤 1: 收集 ---
        all_jobs = [] 
        
        # Florence2 logic
        input_bboxes_per_frame_f2 = [None] * B 
        used_florence2 = False
        if data_from_florence2 is not None:
            if isinstance(data_from_florence2, list) and len(data_from_florence2) == B:
                valid_f2_data = True
                parsed_bboxes_list = []
                for i, frame_data in enumerate(data_from_florence2):
                    if isinstance(frame_data, dict) and 'bboxes' in frame_data and isinstance(frame_data['bboxes'], list):
                        bboxes_for_frame = [list(map(float, box)) for box in frame_data['bboxes'] if len(box) == 4]
                        parsed_bboxes_list.append(bboxes_for_frame)
                    else:
                        if frame_data is None or (isinstance(frame_data, dict) and not frame_data.get('bboxes')):
                            parsed_bboxes_list.append([]) 
                        else:
                            valid_f2_data = False
                            break 
                if valid_f2_data:
                        input_bboxes_per_frame_f2 = parsed_bboxes_list
                        used_florence2 = True

        for frame_idx in range(B):
            original_image_rgb = (images_np[frame_idx] * 255).astype(np.uint8)
            original_image_bgr = cv2.cvtColor(original_image_rgb, cv2.COLOR_RGB2BGR)
            H_curr, W_curr = original_image_rgb.shape[:2]

            bboxes = [] 
            detection_source = "None"

            if used_florence2 and input_bboxes_per_frame_f2[frame_idx] is not None:
                bboxes = input_bboxes_per_frame_f2[frame_idx]
                detection_source = "Florence2"
            
            if not bboxes and grounding_dino_model is not None:
                if prompt and prompt.strip():
                    try:
                        pil_image = Image.fromarray(original_image_rgb)
                        gd_bboxes = groundingdino_predict(grounding_dino_model, pil_image, prompt, gd_threshold)
                        if gd_bboxes:
                            bboxes = gd_bboxes
                            detection_source = "GroundingDINO"
                    except Exception: pass

            if not bboxes and yolo_model is not None and YOLO_AVAILABLE:
                try:
                    results = yolo_model(original_image_bgr, verbose=False)
                    yolo_bboxes = []
                    for result in results:
                        if result.boxes is not None:
                            for box in result.boxes:
                                if int(box.cls[0]) == 0 and float(box.conf[0]) > 0.5:
                                    yolo_bboxes.append(box.xyxy[0].cpu().numpy().tolist())
                    if yolo_bboxes:
                         bboxes = yolo_bboxes
                         detection_source = "YOLO"
                except Exception: pass

            if not bboxes:
                bboxes = [[0, 0, W_curr, H_curr]]
                detection_source = "Full Image"

            for person_idx, bbox in enumerate(bboxes):
                 if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4): continue
                 x1, y1, x2, y2 = bbox
                 if not (x2 > x1 and y2 > y1): continue
                 input_tensor, _, crop_info = preprocess_image_for_sdpose(original_image_bgr, bbox, input_size)
                 all_jobs.append(self.DetectionJob(frame_idx, person_idx, input_tensor, crop_info))

        total_detections = len(all_jobs)
        print(f"IBB_POSE: Stage 1 complete. Found {total_detections} persons.")

        # --- 步骤 2: 并行推理 (批量处理) - 内存优化版 ---
        model_management.soft_empty_cache()
        
        for i in range(0, total_detections, batch_size):
            batch_jobs = all_jobs[i : i + batch_size]
            current_batch_size = len(batch_jobs)
            
            batch_tensors = torch.cat([job.input_tensor for job in batch_jobs], dim=0).to(device)
            
            with torch.no_grad():
                out = pipeline(batch_tensors, timesteps=[999], test_cfg={'flip_test': False}, show_progress_bar=False, mode="inference")
            
            for j in range(current_batch_size):
                kpts_raw = out[j].keypoints[0]
                if hasattr(kpts_raw, 'cpu'):
                    batch_jobs[j].kpts = kpts_raw.cpu().numpy()
                else:
                    batch_jobs[j].kpts = kpts_raw
                
                scores_raw = out[j].keypoint_scores[0]
                if hasattr(scores_raw, 'cpu'):
                    batch_jobs[j].scores = scores_raw.cpu().numpy()
                else:
                    batch_jobs[j].scores = scores_raw
                
                batch_jobs[j].input_tensor = None

            del batch_tensors
            del out

        model_management.soft_empty_cache()

        # --- 步骤 3: 重组 (绘图和JSON) ---
        print("IBB_POSE: Stage 3/3 - Reconstructing frames...")
        
        frame_data = []
        for i in range(B):
            frame_data.append({
                "canvas": np.zeros_like((images_np[i] * 255).astype(np.uint8), order='C'),
                "all_keypoints": [],
                "all_scores": [],
            })
            
        for job in all_jobs:
            frame_idx = job.frame_idx
            
            kpts_original = restore_keypoints_to_original(job.kpts, job.crop_info, input_size, (W, H))
            scores = job.scores
            
            if keypoint_scheme == "body":
                kpts_final = kpts_original
                scores_final = scores
                frame_data[frame_idx]["canvas"] = draw_body17_keypoints_openpose_style(
                    frame_data[frame_idx]["canvas"], kpts_final, scores_final, 
                    threshold=score_threshold,
                    scale_for_xinsr=scale_for_xinsr,
                    pose_scale=pose_scale_factor # <--- 传递参数
                )
            
            else: # wholebody
                kpts_final = kpts_original.copy()
                scores_final = scores.copy()
                if len(kpts_original) >= 17: 
                    neck = (kpts_original[5] + kpts_original[6]) / 2
                    neck_score = min(scores[5], scores[6]) if scores[5] > 0.3 and scores[6] > 0.3 else 0
                    kpts_final = np.insert(kpts_original, 17, neck, axis=0)
                    scores_final = np.insert(scores, 17, neck_score)
                    
                    mmpose_idx = np.array([17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3])
                    openpose_idx = np.array([1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17])
                    temp_kpts = kpts_final.copy(); temp_scores = scores_final.copy()
                    temp_kpts[openpose_idx] = kpts_final[mmpose_idx]
                    temp_scores[openpose_idx] = scores_final[mmpose_idx]
                    kpts_final = temp_kpts; scores_final = temp_scores

                if len(kpts_final) >= 134:
                    if not keep_face:
                        kpts_final[24:92] = 0.0; scores_final[24:92] = 0.0
                    if not keep_hands:
                        kpts_final[92:134] = 0.0; scores_final[92:134] = 0.0
                    if not keep_feet:
                        kpts_final[18:24] = 0.0; scores_final[18:24] = 0.0

                frame_data[frame_idx]["canvas"] = draw_wholebody_keypoints_openpose_style(
                    frame_data[frame_idx]["canvas"], kpts_final, scores_final, 
                    threshold=score_threshold,
                    scale_for_xinsr=scale_for_xinsr,
                    pose_scale=pose_scale_factor # <--- 传递参数
                )
            
            frame_data[frame_idx]["all_keypoints"].append(kpts_final)
            frame_data[frame_idx]["all_scores"].append(scores_final)
        
        result_images = []
        all_frames_json_data = []

        for frame_idx in range(B):
            original_image_rgb = (images_np[frame_idx] * 255).astype(np.uint8)
            pose_canvas = frame_data[frame_idx]["canvas"]
            result_image = cv2.addWeighted(original_image_rgb, 1.0 - overlay_alpha, pose_canvas, overlay_alpha, 0)
            result_images.append(result_image)
            
            frame_json = convert_to_openpose_json(
                frame_data[frame_idx]["all_keypoints"],
                frame_data[frame_idx]["all_scores"],
                W, H, keypoint_scheme
            )
            all_frames_json_data.append(frame_json)
            
            # --- 修复后的保存编辑器 JSON 逻辑 ---
            if save_for_editor:
                try:
                    data_to_save = convert_to_loader_json(
                        frame_data[frame_idx]["all_keypoints"],
                        frame_data[frame_idx]["all_scores"],
                        W, H, keypoint_scheme, score_threshold, enable_confidence_filter
                    )
                    
                    output_dir = folder_paths.get_output_directory()
                    is_batch = B > 1 

                    if is_batch:
                        filename_to_use = f"{filename_prefix_edit}_frame{frame_idx:06d}"
                        full_output_folder, base_filename, _, _, _ = folder_paths.get_save_image_path(filename_to_use, output_dir, W, H)
                        os.makedirs(full_output_folder, exist_ok=True)
                        final_filename = f"{base_filename}.json"
                        file_path = os.path.join(full_output_folder, final_filename)
                        
                    else:
                        full_output_folder, base_filename, _, subfolder, _ = folder_paths.get_save_image_path(filename_prefix_edit, output_dir, W, H)
                        
                        os.makedirs(full_output_folder, exist_ok=True)
                        
                        counter = 1
                        try:
                            existing_files = [f for f in os.listdir(full_output_folder) if f.startswith(base_filename + "_") and f.endswith(".json")]
                            if existing_files:
                                max_counter = 0
                                for f in existing_files:
                                    try:
                                        num_str = f[len(base_filename)+1:-5]
                                        num = int(num_str)
                                        if num > max_counter:
                                            max_counter = num
                                    except ValueError:
                                        continue
                                counter = max_counter + 1
                        except FileNotFoundError:
                            pass

                        final_filename = f"{base_filename}_{counter:05d}.json"
                        file_path = os.path.join(full_output_folder, final_filename)

                    with open(file_path, 'w') as f:
                        json.dump(data_to_save, f, indent=4)
                    
                    if not is_batch:
                        print(f"IBB_POSE: Saved pose JSON to: {file_path}")

                except Exception as e:
                    print(f"IBB_POSE: ERROR Failed to save editor JSON for frame {frame_idx}: {e}")

        result_tensor = torch.from_numpy(np.stack(result_images, axis=0).astype(np.float32) / 255.0)
        
        # --- 最终清理 ---
        if ibb_pose_model.get("unload_on_finish", False):
            offload_device = torch.device("cpu")
            for key in ["unet", "vae", "decoder"]:
                if key in ibb_pose_model: ibb_pose_model[key].to(offload_device)
        
        del all_jobs
        del pipeline
        gc.collect()
        model_management.soft_empty_cache()

        return (result_tensor, all_frames_json_data)

# --- Node Mappings for ComfyUI ---
NODE_CLASS_MAPPINGS = {
    "IBBPoseModelLoader": IBBPoseModelLoader,
    "IBBPoseProcessor": IBBPoseProcessor,
    "IBBYOLOModelLoader": IBBYOLOModelLoader,
    "IBBGroundingDinoModelLoader": IBBGroundingDinoModelLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IBBPoseModelLoader": "IBB Pose — Load Model",
    "IBBPoseProcessor": "IBB Pose — Run SDPose Estimation",
    "IBBYOLOModelLoader": "IBB Pose — Load YOLO Model",
    "IBBGroundingDinoModelLoader": "IBB Pose — Load GroundingDINO Model",
}
