# rCnnExportOnnx.py
import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from src.pytorch.exporters.exporter import Exporter


def _build_maskrcnn(num_classes: int, device: torch.device):
    """
    Mirrors your training setup:
      torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
      + replace box/mask heads to the required num_classes (e.g., 2 for PennFudan).
    """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model.to(device)


class _DetectionsWrapper(nn.Module):
    """
    ONNX-friendly wrapper around torchvision detection models.

    Input:
      images: float32 tensor [B, 3, H, W] in [0,1]

    Outputs (padded to max_detections):
      boxes : [B, N, 4]   float32 (XYXY)
      labels: [B, N]      int64
      scores: [B, N]      float32
      masks : [B, N, H, W] float32 (probabilities in [0,1]); empty tensor if export_masks=False
    """
    def __init__(self, model: nn.Module, max_detections: int = 100, export_masks: bool = True):
        super().__init__()
        self.model = model
        self.max_detections = max_detections
        self.export_masks = export_masks

    @torch.no_grad()
    def forward(self, images: torch.Tensor):
        # Avoid "iterating over a tensor" tracer warning
        imgs_list = list(images.unbind(0))   # tuple -> list of tensors
        self.model.eval()
        preds = self.model(imgs_list)        # list[dict]

        B, _, H, W = images.shape
        device = images.device

        boxes_out  = torch.zeros((B, self.max_detections, 4), dtype=torch.float32, device=device)
        labels_out = torch.zeros((B, self.max_detections),     dtype=torch.long,    device=device)
        scores_out = torch.zeros((B, self.max_detections),     dtype=torch.float32, device=device)
        if self.export_masks:
            masks_out = torch.zeros((B, self.max_detections, H, W), dtype=torch.float32, device=device)
        else:
            masks_out = torch.zeros((0,), dtype=torch.float32, device=device)  # placeholder

        for b, p in enumerate(preds):
            num = int(p["boxes"].shape[0])              # ensure Python int
            n = min(num, self.max_detections)
            if n <= 0:
                continue
            boxes_out[b, :n]  = p["boxes"][:n]
            labels_out[b, :n] = p["labels"][:n]
            scores_out[b, :n] = p["scores"][:n]
            if self.export_masks and "masks" in p and p["masks"].ndim == 4:
                # p["masks"]: [n,1,H,W]; keep probabilities (no binarization)
                masks_out[b, :n] = p["masks"][:n, 0]

        return boxes_out, labels_out, scores_out, masks_out


class RCNNExportOnnx(Exporter):
    """
    Config keys (with defaults):
      - device:          "cpu" | "cuda"                     (default "cpu")
      - num_classes:     int                                 (default 2)   # PennFudan: background+person
      - model_name:      str  (checkpoint stem)              (required)    # e.g. "artifacts/r-cnn-2025-09-18"
      - input_size:      (H, W)                              (default (480, 640))
      - max_detections:  int                                 (default 100)
      - export_masks:    bool                                (default True)
      - opset:           int                                 (default 13)
      - constant_fold:   bool                                (default True)
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.device = torch.device(self.config.get("device", "cpu"))
        self.num_classes: int = int(self.config.get("num_classes", 2))
        self.input_size: Tuple[int, int] = tuple(self.config.get("input_size", (480, 640)))
        self.max_dets: int = int(self.config.get("max_detections", 100))
        self.export_masks: bool = bool(self.config.get("export_masks", True))
        self.opset: int = int(self.config.get("opset", 13))
        self.constant_fold: bool = bool(self.config.get("constant_fold", True))

        self.model: nn.Module | None = None
        self.wrapper: nn.Module | None = None

    def setup(self):
        # 1) Build model exactly as in training (to match checkpoint heads)
        self.model = _build_maskrcnn(self.num_classes, self.device)

        # 2) Load checkpoint
        ckpt_stem = self.config["model_name"]  # required
        ckpt_path = Path(f"{ckpt_stem}.pth")
        print('path', ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        state = torch.load(str(ckpt_path), map_location=self.device)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print("[RCNNExportOnnx] Warning: state_dict not 1:1 with model.")
            if missing:
                print("  missing keys:", missing)
            if unexpected:
                print("  unexpected keys:", unexpected)

        self.model.eval()

        # 3) Wrap for ONNX-friendly IO
        self.wrapper = _DetectionsWrapper(
            self.model,
            max_detections=self.max_dets,
            export_masks=self.export_masks
        ).to(self.device)
        return self

    def run(self):
        H, W = self.input_size
        dummy = torch.rand(1, 3, H, W, device=self.device, dtype=torch.float32)

        # dynamic axes must use INT keys
        dynamic_axes = {
            "images": {0: "batch", 2: "height", 3: "width"},
            "boxes":  {0: "batch", 1: "num_dets"},
            "labels": {0: "batch", 1: "num_dets"},
            "scores": {0: "batch", 1: "num_dets"},
        }
        if self.export_masks:
            dynamic_axes["masks"] = {0: "batch", 1: "num_dets", 2: "height", 3: "width"}
        else:
            dynamic_axes["masks"] = {0: "batch"}  # keep output stable

        output_names = ["boxes", "labels", "scores", "masks"]

        onnx_path = super().getPath("onnx")

        torch.onnx.export(
            self.wrapper,
            dummy,
            onnx_path,
            export_params=True,
            opset_version=self.opset,                # e.g., 13 or 17
            do_constant_folding=self.constant_fold,  # set False to quiet folding warnings
            input_names=["images"],
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

        print(f"[RCNNExportOnnx] Exported ONNX to: {onnx_path}")
        return onnx_path
