using Microsoft.ML.OnnxRuntime;

namespace OnnxHub.Onnx;

public interface IModelRegistry
{
    bool TryGet(string name, out InferenceSession session);
}
