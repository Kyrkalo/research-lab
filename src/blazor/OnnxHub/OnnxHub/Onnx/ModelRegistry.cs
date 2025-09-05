using Microsoft.ML.OnnxRuntime;

namespace OnnxHub.Onnx;

public sealed class ModelRegistry(IDictionary<string, InferenceSession> sessions) : IModelRegistry, IDisposable
{
    private readonly IDictionary<string, InferenceSession> _sessions = sessions;

    public bool TryGet(string name, out InferenceSession s) => _sessions.TryGetValue(name, out s!);

    public void Dispose()
    {
        foreach (var s in _sessions.Values) s.Dispose();
    }
}
