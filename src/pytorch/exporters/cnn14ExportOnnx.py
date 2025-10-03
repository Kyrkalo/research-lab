import torch

class cnn14ExportOnnx:
    def __init__(self, config):
        self.config = config

    def setup(self):
        self.model = torch.load("artifacts/Cnn14_mAP=0.431.pth", map_location="cpu")
        return self

    def run(self):
        # Code to export CNN14 model to ONNX
        pass