using Microsoft.ML.OnnxRuntime;
using OnnxHub.Onnx.Converter;

namespace OnnxHub.Onnx;

public sealed class ModelRegistry : IModelRegistry, IDisposable
{
    private sealed record Entry(InferenceSession Session, IToTensorConverter Converter);

    private readonly Dictionary<string, Entry> _map = new(StringComparer.OrdinalIgnoreCase);

    public ModelRegistry Add(string name, InferenceSession session, IToTensorConverter converter)
    {
        _map[name] = new Entry(session, converter);
        return this;
    }

    public bool TryGet(string name, out InferenceSession session, out IToTensorConverter converter)
    {
        if (_map.TryGetValue(name, out var e))
        {
            session = e.Session;
            converter = e.Converter;
            return true;
        }
        session = default!;
        converter = default!;
        return false;
    }

    public void Dispose()
    {
        foreach (var e in _map.Values) e.Session.Dispose();
        _map.Clear();
    }
}
