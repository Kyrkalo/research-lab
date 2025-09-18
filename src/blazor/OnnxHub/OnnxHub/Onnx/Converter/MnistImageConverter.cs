using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxHub.Onnx.Converter;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;

public class MnistImageConverter : IToTensorConverter
{
    private const int W = 28, H = 28;
    private const float Mean = 0.1307f, Std = 0.3081f;

    // Set to false if your model expects non-inverted digits
    public bool Invert { get; init; } = true;

    public DenseTensor<float> Convert(byte[] bytes)
    {
        using var img = Image.Load<Rgba32>(bytes);

        // 1) If the image has transparency, flatten it on a white background
        if (HasAlpha(img))
        {
            using var white = new Image<Rgba32>(img.Width, img.Height, new Rgba32(255, 255, 255, 255));
            white.Mutate(x => x.DrawImage(img, 1f));
            img.Mutate(x => x.DrawImage(white, 1f));
        }

        // 2) Grayscale
        img.Mutate(x => x.Grayscale());

        // 3) Invert (so background is white, digit is black) if desired
        if (Invert)
            img.Mutate(x => x.Invert());

        // 4) Resize to 28x28 (model input)
        img.Mutate(x => x.Resize(W, H));

        // 5) Read pixels → tensor [1,1,28,28], normalize with MNIST stats
        var data = new float[1 * 1 * H * W]; // NCHW
        img.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < H; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < W; x++)
                {
                    // After grayscale, R=G=B. Use R channel.
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
}
