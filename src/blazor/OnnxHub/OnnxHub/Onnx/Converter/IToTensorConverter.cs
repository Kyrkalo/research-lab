using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxHub.Onnx.Converter;

public interface IToTensorConverter
{
    DenseTensor<float> Convert(byte[] bytes);
}
