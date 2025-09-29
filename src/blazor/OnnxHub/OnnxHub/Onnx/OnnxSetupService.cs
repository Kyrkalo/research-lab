using OnnxHub.Onnx.Converter;
using OnnxHub.Services;

namespace OnnxHub.Onnx
{
    public static class OnnxSetupService
    {
        /// <summary>
        /// Registers a singleton instance of <see cref="IModelRegistry"/> in the service collection.
        /// </summary>
        /// <remarks>This method configures and registers an <see cref="IModelRegistry"/> implementation
        /// that provides pre-configured ONNX models for inference. The registry includes models for tasks such as image
        /// classification, generative adversarial networks, and object detection.</remarks>
        /// <param name="service">The <see cref="IServiceCollection"/> to which the singleton instance will be added.</param>
        public static void AddSingletonOnnx(this IServiceCollection service)
        {
            service.AddSingleton<IModelRegistry>(e =>
            {
                // todo: will be reafacoterd to load model from configuration
                return new ModelRegistry()
                .Add(nameof(MnistService), "mdl_mnist_202520.onnx", new MnistImageConverter())
                .Add(nameof(GanGeneratorService), "generator.onnx", new GanConverter())
                .Add(nameof(RCnnService), "FasterRCNN-10.onnx", new RCNNConveror());
            });
        }
    }
}
