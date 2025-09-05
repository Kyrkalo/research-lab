using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;

namespace OnnxHub.Onnx.Converter
{
    public class MnistImageConverter : IToTensorConverter
    {
        private const int W = 28, H = 28;
        private const float Mean = 0.1307f, Std = 0.3081f;

        public DenseTensor<float> Convert(byte[] bytes)
        {
            using var img = Image.Load<Rgba32>(bytes);
            img.Mutate(x => x.Resize(W, H).Grayscale());

            var data = new float[1 * 1 * H * W]; // NCHW
            img.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < H; y++)
                {
                    var row = accessor.GetRowSpan(y);
                    for (int x = 0; x < W; x++)
                    {
                        float gray01 = row[x].R / 255f; // after Grayscale(), R=G=B
                        data[y * W + x] = (gray01 - Mean) / Std;
                    }
                }
            });

            return new DenseTensor<float>(data, new[] { 1, 1, H, W });
        }
    }
}
