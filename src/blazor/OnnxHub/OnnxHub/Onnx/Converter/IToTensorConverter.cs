using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxHub.Onnx.Converter;

public enum InputParamType
{
    Height,
    Width,
    Image
}

public interface IToTensorConverter
{
    DenseTensor<float> Convert(byte[] bytes);
    DenseTensor<float> Convert(int[] bytes);
}

public interface IParams
{
    Dictionary<InputParamType, object> GetParams(); 
}
