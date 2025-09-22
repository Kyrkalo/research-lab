import torch
from anyio import Path
import torchvision
from src.pytorch.exporters.exporter import Exporter

class RCNNExportOnnx(Exporter):

    def __init__(self, config=None):
        self.config = config
        self.device = config["device"] if "device" in config else "cpu"

    def setup(self):
        
        self.model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
        self.model.load_state_dict(torch.load(self.dcgan_config["model_name"] + ".pth", map_location=self.device))
        return self

    def run(self):
        
        dummy_input = [torch.randn(3, 480, 640, device=self.device)]
        
        dynamic_axes = {
            'input': {0: 'batch', 1: 'channels', 2: 'height', 3: 'width'},
            'boxes': {0: 'num_detections'},
            'labels': {0: 'num_detections'},
            'scores': {0: 'num_detections'}
        }

        path = super().getPath("onnx")
        
        torch.onnx.export(
            self.model, 
            (dummy_input,),
            path,
            opset_version=11, 
            input_names=["input"], 
            output_names=["boxes", "labels", "scores"], 
            dynamic_axes=dynamic_axes
        )

        pass