using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace OnnxHub.Onnx.Converter;

public class RCNNConveror : IToTensorConverter, IParams
{
    private readonly float[] _mean = new[] { 102.9801f, 115.9465f, 122.7717f };
    private readonly Dictionary<InputParamType, object> _params;

    public RCNNConveror()
    {
        _params = new Dictionary<InputParamType, object>();
    }

    public DenseTensor<float> Convert(byte[] bytes)
    {
        var image = Image.Load<Rgb24>(bytes);
        float ratio = 800f / Math.Min(image.Width, image.Height);

        image.Mutate(x => x.Resize((int)(ratio * image.Width), (int)(ratio * image.Height)));

        var paddedHeight = (int)(Math.Ceiling(image.Height / 32f) * 32f);
        var paddedWidth = (int)(Math.Ceiling(image.Width / 32f) * 32f);

        _params[InputParamType.Height] = paddedHeight;
        _params[InputParamType.Width] = paddedWidth;

        Tensor<float> input = new DenseTensor<float>(new[] { 3, paddedHeight, paddedWidth });

        image.ProcessPixelRows(accessor =>
        {
            for (int y = paddedHeight - accessor.Height; y < accessor.Height; y++)
            {
                Span<Rgb24> pixelSpan = accessor.GetRowSpan(y);
                for (int x = paddedWidth - accessor.Width; x < accessor.Width; x++)
                {
                    input[0, y, x] = pixelSpan[x].B - _mean[0];
                    input[1, y, x] = pixelSpan[x].G - _mean[1];
                    input[2, y, x] = pixelSpan[x].R - _mean[2];
                }
            }
        });

        _params[InputParamType.Image] = image;
        return (DenseTensor<float>)input;
    }

    public DenseTensor<float> Convert(int[] bytes)
    {
        throw new NotImplementedException();
    }

    public Dictionary<InputParamType, object> GetParams() => _params;
}
