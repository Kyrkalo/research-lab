namespace OnnxHub.Onnx;

public interface IModelRegistry
{
    bool TryGet(string name, out Entry entry);
}
