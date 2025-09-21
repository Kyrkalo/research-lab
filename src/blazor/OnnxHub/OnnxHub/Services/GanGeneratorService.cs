using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxHub.Onnx;
using OnnxHub.Onnx.Converter;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace OnnxHub.Services;

public class GanGeneratorService
{
    private readonly InferenceSession _session;
    private readonly IToTensorConverter _converter;

    public GanGeneratorService(IModelRegistry modelRegistry) 
    {
        modelRegistry.TryGet("gen_faces", out InferenceSession session, out IToTensorConverter converter);
        _session = session;
        _converter = converter;
    }

    public async Task<List<string>> Generate()
    {        
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("z", _converter.Convert(new int[] { 4, 100, 1, 1 })) };

        var results = _session.Run(inputs);

        Tensor<float>  output = results.First().AsTensor<float>();
        
        return await Task.FromResult(ToPngBase64Batch([.. output]));
    }

    private static List<string> ToPngBase64Batch(float[] data, int B = 4, int nc = 3, int H = 64, int W = 64)
    {
        var list = new List<string>(B);
        int plane = H * W;
        int sampleStride = nc * plane;

        for (int b = 0; b < B; b++)
        {
            int baseOff = b * sampleStride;
            list.Add(ChwToPngBase64(data, baseOff, nc, H, W));
        }
        return list;
    }

    private static string ChwToPngBase64(float[] chw, int offset, int nc, int H, int W)
    {
        using var img = new Image<Rgba32>(W, H);

        int plane = H * W;
        int rBase = offset + 0 * plane;
        int gBase = (nc > 1) ? offset + 1 * plane : rBase;
        int bBase = (nc > 2) ? offset + 2 * plane : rBase;

        img.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < H; y++)
            {
                Span<Rgba32> row = accessor.GetRowSpan(y);
                for (int x = 0; x < W; x++)
                {
                    int idx = y * W + x;

                    // [-1,1] -> [0,1]
                    float rf = Clamp01((chw[rBase + idx] + 1f) * 0.5f);
                    float gf = Clamp01((chw[gBase + idx] + 1f) * 0.5f);
                    float bf = Clamp01((chw[bBase + idx] + 1f) * 0.5f);

                    row[x] = new Rgba32(
                        (byte)Math.Round(rf * 255f),
                        (byte)Math.Round(gf * 255f),
                        (byte)Math.Round(bf * 255f),
                        255);
                }
            }
        });

        using var ms = new MemoryStream();
        img.SaveAsPng(ms);
        return Convert.ToBase64String(ms.ToArray());
    }

    private static float Clamp01(float v) => v < 0f ? 0f : (v > 1f ? 1f : v);
}
