using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxHub.Onnx.Converter;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

public class MnistImageConverter : IToTensorConverter
{
    private const int W = 28, H = 28;
    private const float Mean = 0.1307f, Std = 0.3081f;

    // Set to false if your model expects non-inverted digits
    public bool Invert { get; init; } = true;

    public DenseTensor<float> Convert(byte[] bytes)
    {
        using var img = Image.Load<Rgba32>(bytes);

        if (HasAlpha(img))
        {
            using var white = new Image<Rgba32>(img.Width, img.Height, new Rgba32(255, 255, 255, 255));
            white.Mutate(x => x.DrawImage(img, 1f));
            img.Mutate(x => x.DrawImage(white, 1f));
        }

        img.Mutate(x => x.Grayscale());

        if (Invert)
            img.Mutate(x => x.Invert());

        img.Mutate(x => x.Resize(W, H));

        var data = new float[1 * 1 * H * W]; // NCHW
        img.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < H; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < W; x++)
                {
                    float gray01 = row[x].R / 255f;
                    data[y * W + x] = (gray01 - Mean) / Std;
                }
            }
        });

        return new DenseTensor<float>(data, new[] { 1, 1, H, W });
    }

    private static bool HasAlpha(Image<Rgba32> img)
    {
        bool hasAlpha = false;
        img.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < img.Height && !hasAlpha; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < img.Width; x++)
                {
                    if (row[x].A < 255) { hasAlpha = true; break; }
                }
            }
        });
        return hasAlpha;
    }

    public DenseTensor<float> Convert(int[] bytes)
    {
        throw new NotImplementedException();
    }

    public DenseTensor<float> Convert(Stream stream)
    {
        throw new NotImplementedException();
    }

    public Task<DenseTensor<float>> ConvertAsync(Stream stream, CancellationToken cancellationToken)
    {
        throw new NotImplementedException();
    }
}
