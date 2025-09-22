
from enum import Enum
import torch
from src.pytorch.exporters.dcganExportOnnx import DcganExportOnnx
from src.pytorch.pipelines.mnistPipeline import MnistPipeline, MnistExportOnnx
from src.pytorch.pipelines.dcganPipeline import DcganPipeline
from src.pytorch.pipelines.rCnnPipeline import RCnnPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelTypes(Enum):
    MNIST = "mnist"
    DCGAN = "dcgan"
    RCNN = "rCnn"


configs = {
    "mnist": {
        "learning_rate": 0.01,
        "momentum": 0.5,
        "n_epochs": 20,
        "batch_size_train": 128,
        "batch_size_test": 1000,
        "log_interval": 10,
        "random_seed": 1,
        "device": device,
        "data_dir": "./data",
        "model_name": "Mdl_mnist_2025_2",
        "optimizer_name": "Mdl_mnist_2025_2_optimizer",
    },
    "dcgan": {
        "device": device,
        "dataroot": "C://Users/RomanKyrkalo/source/repos/ai-study/src/notebook/ImageFolder/data/celeba",
        "workers": 2,
        "batch_size": 256,
        "image_size": 64,
        "nc": 3, # Number of channels in the training images. For color images this is 3
        "nz": 100, # Size of z latent vector (i.e. size of generator input)
        "ngf": 64, # Size of feature maps in generator
        "ndf": 64,
        "num_epochs": 1,
        "lr": 0.0002,
        "beta1": 0.5,
        "ngpu": 1, # Number of GPUs available. Use 0 for CPU mode.
        "random_seed": 999,
        "model_name": "dcgan_model_faces",
    },
    "rCnn": {
        "device": device,
        "out_channels": 1280,
        "num_classes": 2,
        "data_root": "data/PennFudanPed",
        "model_name": "rCnn_model_pedestrian",
        "learning_rate": 0.005,
        "num_epochs": 10,
    }
}

def run(modeltype: ModelTypes):
    if modeltype == ModelTypes.MNIST:
        mnist_pipeline = MnistPipeline(configs["mnist"])
        mnist_pipeline.setup().run()

        export_to_onnx = MnistExportOnnx(configs["mnist"])
        export_to_onnx.setup().run()

    elif modeltype == ModelTypes.DCGAN:
        dcgan_pipeline = DcganPipeline(configs["dcgan"])
        dcgan_pipeline.setup().run()

        export_to_onnx = DcganExportOnnx(configs["dcgan"])
        export_to_onnx.setup().run()

    elif modeltype == ModelTypes.RCNN:
        rcnn_pipeline = RCnnPipeline(configs["rCnn"])
        rcnn_pipeline.setup().run()
    pass