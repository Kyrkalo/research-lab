using Microsoft.ML.OnnxRuntime;
using OnnxHub.Onnx.Converter;

namespace OnnxHub.Onnx
{
    public static class OnnxSetupService
    {
        public static void AddSingletonOnnx(this IServiceCollection service)
        {
            service.AddSingleton<IModelRegistry>(e =>
            {
                var sessionOptions = new Microsoft.ML.OnnxRuntime.SessionOptions()
                {
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
                    EnableMemoryPattern = true,
                    ExecutionMode = ExecutionMode.ORT_PARALLEL
                };

                //sessionOptions.AppendExecutionProvider_CUDA();

                var mdl_mnist_202520 = new InferenceSession(Path.Combine(AppContext.BaseDirectory, "Onnx/Models", "mdl_mnist_202520.onnx"), sessionOptions);
                var gen = new InferenceSession(Path.Combine(AppContext.BaseDirectory, "Onnx/Models", "generator.onnx"), sessionOptions);
                var rcnnPedestrians = new InferenceSession(Path.Combine(AppContext.BaseDirectory, "Onnx/Models", "FasterRCNN-10.onnx"), sessionOptions);

                return new ModelRegistry()
                .Add("mdl_mnist_202520", mdl_mnist_202520, new MnistImageConverter())
                .Add("gen_faces", gen, new GanConverter())
                //https://github.com/onnx/models/blob/main/validated/vision/object_detection_segmentation/faster-rcnn/model/FasterRCNN-10.onnx
                .Add("FasterRCNN-10", rcnnPedestrians, new RCNNConveror());
            });
        }
    }
}
