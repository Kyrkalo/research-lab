from dataloaders.mnistDataLoader import MnistDataLoader
from models.mdl_mnist_202520 import Mdl_mnist_202520
from test import MNISTTester
from train import MNISTTrainer
import torch.optim as optim
import torch
from pathlib import Path


class MnistPipeline:

    def __init__(self, config=None):
        self.device = config["device"] if config and "device" in config else self.device
        self.mnist_config = config if config else self.mnist_config
        pass

    def setup(self):
        self.model = Mdl_mnist_202520()
        self.model.to(self.device)

        self.optimizer = optim.SGD(self.model.parameters(), lr=self.mnist_config["learning_rate"], momentum=self.mnist_config["momentum"])
        self.train_loader, self.test_loader = MnistDataLoader(self.mnist_config).Get()
        
        torch.backends.cudnn.enabled = False
        torch.manual_seed(self.mnist_config["random_seed"])        

        self.train = MNISTTrainer(self.model, self.optimizer, self.train_loader, self.mnist_config)
        self.test = MNISTTester(self.model, self.test_loader, self.device)

        return self

    def run(self):
        for epoch in range(1, self.mnist_config["n_epochs"] + 1):
            self.train.train(epoch)
            self.test.test(epoch)

        return self
    
    def metric(self):
        return {
            "test_loss": self.test.test_loss,
            "accuracy": self.test.accuracy,
            "n_epochs": self.mnist_config["n_epochs"],
            "model_name": self.mnist_config["model_name"],
            "optimizer_name": self.mnist_config["optimizer_name"],
            "optimizer": "SGD",
            "train_stats": self.train.get_train_stats(),
            "test_stats": self.test.get_test_stats()
        }
    
class MnistExportOnnx():

    def __init__(self, config=None):
        self.device = config["device"] if config and "device" in config else self.device
        self.mnist_config = config if config else self.mnist_config

    def setup(self):
        self.model = Mdl_mnist_202520().to(self.device)
        self.model.load_state_dict(torch.load(self.mnist_config["model_name"] + ".pth", map_location=self.device))
        self.model.eval()
        return self

    def run(self):
        dummy_input = torch.randn(1, 1, 28, 28, device=self.device)
        
        onnx_path = Path(self.mnist_config["model_name"] + ".onnx")
        torch.onnx.export(
            self.model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=12,
            do_constant_folding=True
)