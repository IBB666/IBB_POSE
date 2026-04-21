"""
IBB_POSE - ComfyUI Pose Estimation Node
Compatible with Python 3.13, torch 2.10, CUDA 13.x
"""
from .ibb_pose_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print("### Loading: IBB_POSE Nodes ###")
