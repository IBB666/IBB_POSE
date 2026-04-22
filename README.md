# IBB_POSE

IBB_POSE 是对 [judian17/ComfyUI-SDPose-OOD](https://github.com/judian17/ComfyUI-SDPose-OOD) 的 IBB 命名复刻版，保留原仓库的核心节点能力，并统一节点分类/显示名为 `IBB_POSE`。

## 功能

- `IBB Pose — Load Model`
  - 加载 SDPose Body / WholeBody 模型
  - 支持 `fp32 / fp16 / bf16`
  - 支持自动下载 Hugging Face 模型
- `IBB Pose — Run SDPose Estimation`
  - 单图/批量姿态估计
  - 支持全图、YOLO、Florence2、GroundingDINO 四种检测来源
  - 支持 Body(17) / WholeBody(133)
  - 支持 `keep_face / keep_hands / keep_feet`
  - 支持 `pose_scale_factor`、`scale_for_xinsr`
  - 支持输出 OpenPose 风格 `POSE_KEYPOINT`
  - 支持保存可导入 OpenPose Editor 的 JSON
- `IBB Pose — Load YOLO Model`
  - 加载 ComfyUI `models/yolo` 下的 YOLO 模型
- `IBB Pose — Load GroundingDINO Model`
  - 自动下载并加载 GroundingDINO 模型

## 安装

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/IBB666/IBB_POSE.git
cd IBB_POSE
pip install -r requirements.txt
```

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
