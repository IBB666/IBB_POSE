# Author: T. S. Liang @ Rama Alpaca
# Emails: tsliang2001@gmail.com | shuangliang@ramaalpaca.com | sliang57@connect.hku.hk
# Date: Jun. 2025
# Description: SDPose heatmap head factory using mmpose `HeatmapHead` + `UDPHeatmap` codec.
#   - mode="body"     → 17 keypoints, in_channels=320
#   - mode="wholebody"→ 133 keypoints, in_channels=640
#   Defaults: image_size=(768,1024), scale=4 (→ heatmap size = (w/4, h/4)), sigma=6.
# License: MIT License (see LICENSE for details)

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
from torch import nn

_mmpose_import_error = None

try:
    from mmpose.registry import MODELS
except ImportError as exc:
    MODELS = None
    _mmpose_import_error = exc


def _build_head_cfg(mode="body"):

    if mode == "body":

        image_size = [768, 1024]  ## width x height
        sigma = 6  ## sigma is 2 for 256
        scale = 4

        embed_dim = 320
        num_keypoints = 17

    elif mode == "wholebody":

        image_size = [768, 1024]  ## width x height
        sigma = 6  ## sigma is 2 for 256
        scale = 4

        embed_dim = 640
        num_keypoints = 133

    else:
        raise ValueError(f"Unsupported heatmap head mode: {mode}")

    codec = dict(
        type="UDPHeatmap",
        input_size=(image_size[0], image_size[1]),
        heatmap_size=(int(image_size[0] / scale), int(image_size[1] / scale)),
        sigma=sigma,
    )

    return dict(
        type="HeatmapHead",
        in_channels=embed_dim,
        out_channels=num_keypoints,
        deconv_out_channels=(embed_dim,),
        deconv_kernel_sizes=(4,),
        conv_out_channels=(embed_dim,),
        conv_kernel_sizes=(1,),
        loss=dict(type="KeypointMSELoss", use_target_weight=True),
        decoder=codec,
    )


class _CompatInstanceData(SimpleNamespace):
    def set_field(self, value, name):
        setattr(self, name, value)

    def all_items(self):
        return self.__dict__.items()


class _FallbackUDPHeatmap:
    def __init__(self, input_size, heatmap_size, sigma=2.0):
        self.input_size = np.asarray(input_size, dtype=np.float32)
        self.heatmap_size = np.asarray(heatmap_size, dtype=np.float32)
        self.sigma = sigma

    def decode(self, encoded):
        heatmaps = encoded.astype(np.float32, copy=False)
        num_keypoints, height, width = heatmaps.shape

        flattened = heatmaps.reshape(num_keypoints, -1)
        flat_indices = flattened.argmax(axis=1)
        scores = flattened[np.arange(num_keypoints), flat_indices]

        xs = (flat_indices % width).astype(np.float32)
        ys = (flat_indices // width).astype(np.float32)

        for idx, (x, y) in enumerate(zip(xs.astype(np.int32), ys.astype(np.int32))):
            if 0 < x < width - 1:
                xs[idx] += np.sign(heatmaps[idx, y, x + 1] - heatmaps[idx, y, x - 1]) * 0.25
            if 0 < y < height - 1:
                ys[idx] += np.sign(heatmaps[idx, y + 1, x] - heatmaps[idx, y - 1, x]) * 0.25

        # Match MMPose UDPHeatmap decoding, which maps heatmap coordinates from
        # the [0, W-1] / [0, H-1] grid back into input image space.
        scale = self.input_size / np.maximum(self.heatmap_size - 1.0, 1.0)
        keypoints = np.stack((xs, ys), axis=-1) * scale

        return keypoints[None], scores[None].astype(np.float32)


class _FallbackHeatmapHead(nn.Module):
    _version = 2

    def __init__(
        self,
        in_channels,
        out_channels,
        deconv_out_channels=(256, 256, 256),
        deconv_kernel_sizes=(4, 4, 4),
        conv_out_channels=None,
        conv_kernel_sizes=None,
        final_layer=None,
        loss=None,
        decoder=None,
        init_cfg=None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss_module = None
        self.decoder = (
            _FallbackUDPHeatmap(
                input_size=decoder["input_size"],
                heatmap_size=decoder["heatmap_size"],
                sigma=decoder.get("sigma", 2.0),
            )
            if decoder is not None
            else None
        )

        if deconv_out_channels:
            self.deconv_layers = self._make_deconv_layers(
                in_channels=in_channels,
                layer_out_channels=deconv_out_channels,
                layer_kernel_sizes=deconv_kernel_sizes,
            )
            in_channels = deconv_out_channels[-1]
        else:
            self.deconv_layers = nn.Identity()

        if conv_out_channels:
            self.conv_layers = self._make_conv_layers(
                in_channels=in_channels,
                layer_out_channels=conv_out_channels,
                layer_kernel_sizes=conv_kernel_sizes,
            )
            in_channels = conv_out_channels[-1]
        else:
            self.conv_layers = nn.Identity()

        final_layer = final_layer or {"kernel_size": 1}
        self.final_layer = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=final_layer.get("kernel_size", 1),
            stride=final_layer.get("stride", 1),
            padding=final_layer.get("padding", 0),
        )

        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    @staticmethod
    def _make_conv_layers(in_channels, layer_out_channels, layer_kernel_sizes):
        layers = []
        for out_channels, kernel_size in zip(layer_out_channels, layer_kernel_sizes):
            padding = (kernel_size - 1) // 2
            layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                )
            )
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layers)

    @staticmethod
    def _make_deconv_layers(in_channels, layer_out_channels, layer_kernel_sizes):
        layers = []
        for out_channels, kernel_size in zip(layer_out_channels, layer_kernel_sizes):
            if kernel_size == 4:
                padding = 1
                output_padding = 0
            elif kernel_size == 3:
                padding = 1
                output_padding = 1
            elif kernel_size == 2:
                padding = 0
                output_padding = 0
            else:
                raise ValueError(f"Unsupported kernel size {kernel_size} for deconv layers")

            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, feats):
        x = feats[-1] if isinstance(feats, (list, tuple)) else feats
        x = self.deconv_layers(x)
        x = self.conv_layers(x)
        x = self.final_layer(x)
        return x

    def decode(self, batch_heatmaps):
        heatmaps_np = batch_heatmaps.detach().float().cpu().numpy()
        preds = []
        for heatmaps in heatmaps_np:
            keypoints, scores = self.decoder.decode(heatmaps)
            preds.append(_CompatInstanceData(keypoints=keypoints, keypoint_scores=scores))
        return preds

    def predict(self, feats, batch_data_samples=None, test_cfg=None):
        test_cfg = test_cfg or {}

        if test_cfg.get("flip_test", False):
            raise RuntimeError(
                "The lightweight fallback heatmap head does not support MMPose flip_test. "
                "Disable flip_test for ComfyUI inference, or install mmpose/mmcv for evaluation workflows."
            )

        batch_heatmaps = self.forward(feats)
        preds = self.decode(batch_heatmaps)

        if test_cfg.get("output_heatmaps", False):
            pred_fields = [SimpleNamespace(heatmaps=hm) for hm in batch_heatmaps.detach()]
            return preds, pred_fields

        return preds

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args, **kwargs):
        version = local_meta.get("version", None)
        if version and version >= self._version:
            return

        keys = list(state_dict.keys())
        for key in keys:
            if not key.startswith(prefix):
                continue

            value = state_dict.pop(key)
            short_key = key[len(prefix):]
            key_parts = short_key.split(".")

            if key_parts[0] == "final_layer":
                if len(key_parts) == 3:
                    try:
                        idx = int(key_parts[1])
                    except ValueError as exc:
                        raise RuntimeError(
                            f"Incompatible state_dict key: {key!r}. "
                            "Expected a numeric final_layer index."
                        ) from exc
                    num_conv_layers = len(self.conv_layers) if isinstance(self.conv_layers, nn.Sequential) else 0
                    if idx < num_conv_layers:
                        new_key = "conv_layers." + ".".join(key_parts[1:])
                    else:
                        new_key = "final_layer." + key_parts[2]
                else:
                    new_key = short_key
            else:
                new_key = short_key

            state_dict[prefix + new_key] = value


def get_heatmap_head(mode="body"):
    head_cfg = _build_head_cfg(mode=mode)

    if MODELS is not None:
        return MODELS.build(head_cfg)

    warnings.warn(
        "IBB_POSE: mmpose/mmcv is not available "
        f"({_mmpose_import_error}). Falling back to the built-in heatmap head "
        "for ComfyUI inference. Install mmpose/mmcv/mmengine only if you need the evaluation scripts.",
        RuntimeWarning,
        stacklevel=2,
    )

    return _FallbackHeatmapHead(
        in_channels=head_cfg["in_channels"],
        out_channels=head_cfg["out_channels"],
        deconv_out_channels=head_cfg["deconv_out_channels"],
        deconv_kernel_sizes=head_cfg["deconv_kernel_sizes"],
        conv_out_channels=head_cfg["conv_out_channels"],
        conv_kernel_sizes=head_cfg["conv_kernel_sizes"],
        loss=head_cfg["loss"],
        decoder=head_cfg["decoder"],
    )
