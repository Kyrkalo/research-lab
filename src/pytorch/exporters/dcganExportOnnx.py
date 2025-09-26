import torch
import copy
from anyio import Path
import torch
import torch.nn as nn
import torch.optim as optim
from src.pytorch.models.dcgan import GanDiscriminator, GanGenerator

class DcganExportOnnx():

    def __init__(self, config=None):
        self.device = config["device"] if config and "device" in config else self.device
        self.dcgan_config = config if config else self.dcgan_config

    def _unwrap_to_cpu_eval(self, model: nn.Module) -> nn.Module:
        m = copy.deepcopy(model)
        if hasattr(m, "module"):  # DataParallel / DDP wrapper
            m = m.module
        m.to("cpu").eval()
        return m

    def setup(self):
        self.modelG = GanGenerator(self.dcgan_config).to(self.device)
        self.modelG.load_state_dict(torch.load(self.dcgan_config["model_name"] + "_G.pth", map_location=self.device))

        self.modelD = GanDiscriminator(self.dcgan_config).to(self.device)
        self.modelD.load_state_dict(torch.load(self.dcgan_config["model_name"] + "_D.pth", map_location=self.device))
        return self

    def run(self):

        opset = 13

        g_onnx_path = Path("g_" + self.dcgan_config["model_name"] + ".onnx")
        d_onnx_path = Path("d_" + self.dcgan_config["model_name"] + ".onnx")

        G = self._unwrap_to_cpu_eval(self.modelG)
        D = self._unwrap_to_cpu_eval(self.modelD)

        nz = int(self.dcgan_config["nz"])
        nc = int(self.dcgan_config["nc"])
        
        H = W = 64

        dummy_z   = torch.randn(1, nz, 1, 1, dtype=torch.float32)
        dummy_img = torch.randn(1, nc, H, W, dtype=torch.float32)

        D_wrap = DForExport(D)

        with torch.no_grad():
            # ---- Generator ----
            torch.onnx.export(
                G, dummy_z, str(g_onnx_path),
                input_names=["z"], output_names=["img"],
                opset_version=opset, do_constant_folding=True,
                dynamic_axes={"z": {0: "batch"}, "img": {0: "batch"}},
            )

            # ---- Discriminator ----
            torch.onnx.export(
                D_wrap, dummy_img, str(d_onnx_path),
                input_names=["img"], output_names=["prob"],
                opset_version=opset, do_constant_folding=True,
                dynamic_axes={"img": {0: "batch"}, "prob": {0: "batch"}},
            )

        return self

class DForExport(nn.Module):
    def __init__(self, d): 
        super().__init__(); self.d = d
    def forward(self, x):
        y = self.d(x)
        return y.view(x.size(0), 1) 