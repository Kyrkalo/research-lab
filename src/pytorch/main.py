
import torch
from pipelines.mnistPipeline import MnistPipeline, MnistExportOnnx

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mnist_config = {
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
}

mnist_pipeline = MnistPipeline(mnist_config)
mnist_pipeline.setup().run()

export_to_onnx = MnistExportOnnx(mnist_config)
export_to_onnx.setup().run()


# train_losses, train_counter = train.get_train_stats()
# test_losses, test_counter = test.get_test_stats()

# fig = plt.figure()
# plt.plot(train_counter, train_losses, color='blue')
# plt.scatter(test_counter, test_losses, color='red')
# plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
# plt.xlabel('number of training examples seen')
# plt.ylabel('negative log likelihood loss')
# fig