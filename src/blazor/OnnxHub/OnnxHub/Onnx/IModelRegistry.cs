using Microsoft.ML.OnnxRuntime;
using OnnxHub.Onnx.Converter;

namespace OnnxHub.Onnx;

public interface IModelRegistry
{
    bool TryGet(string name, out InferenceSession session, out IToTensorConverter converter);
}
