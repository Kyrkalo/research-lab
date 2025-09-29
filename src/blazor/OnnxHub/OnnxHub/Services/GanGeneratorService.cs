using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxHub.Infrastructure;
using OnnxHub.Onnx;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace OnnxHub.Services;

/// <summary>
/// Represents a request to be processed by a Generative Adversarial Network (GAN) service.
/// </summary>
/// <remarks>This request is used as part of the service contract for operations that involve GAN-based
/// processing. Implementations of the service should define the specific behavior and requirements for handling this
/// request.</remarks>
public record GanRequest() : IServiceRequest;

/// <summary>
/// Represents the response from a Generative Adversarial Network (GAN) service, containing a collection of generated
/// images.
/// </summary>
/// <remarks>This response is typically returned by a GAN-based service after processing a request to generate
/// images.  The <see cref="Images"/> property contains the generated image data as a list of strings, where each string
/// represents an image in a specific format (e.g., base64-encoded or a URL, depending on the service
/// implementation).</remarks>
/// <param name="Images">A list of strings representing the generated images. The format of each string depends on the service
/// implementation.</param>
public record GanResponse(List<string> Images) : IServiceResponse;

public class GanGeneratorService : BaseSarvice
{
    private readonly SysConfigurations _sysConfigurations;
    private readonly Entry _entry;

    public GanGeneratorService(IModelRegistry modelRegistry, IOptions<SysConfigurations> options) 
    {
        modelRegistry.TryGet(nameof(GanGeneratorService), out var entry);
        _sysConfigurations = options.Value;
        _entry = entry;
    }

    public override IServiceResponse Run(IServiceRequest request, CancellationToken cancellationToken)
    {
        using var _session = GetInferenceSession(_sysConfigurations, _entry.Model);
        var inputName = _session.InputMetadata.First().Key;
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(inputName, _entry.Converter.Convert(new int[] { 4, 100, 1, 1 })) };

        var results = _session.Run(inputs);

        Tensor<float> output = results.First().AsTensor<float>();

        return new GanResponse(ToPngBase64Batch([.. output]));
    }

    /// <summary>
    /// Converts a batch of image data represented as a flattened array into a list of Base64-encoded PNG images.
    /// </summary>
    /// <remarks>The input data is expected to be in CHW format, where the channels are stored first, followed
    /// by height and width dimensions. Each image in the batch is extracted from the input array based on its index and
    /// converted to a PNG format before being encoded as a Base64 string.</remarks>
    /// <param name="data">A one-dimensional array containing the image data in CHW (Channel-Height-Width) format. The array must have
    /// sufficient data to represent all images in the batch.</param>
    /// <param name="B">The number of images in the batch. Defaults to 4.</param>
    /// <param name="nc">The number of color channels per image (e.g., 3 for RGB). Defaults to 3.</param>
    /// <param name="H">The height of each image in pixels. Defaults to 64.</param>
    /// <param name="W">The width of each image in pixels. Defaults to 64.</param>
    /// <returns>A list of Base64-encoded strings, where each string represents a PNG image from the batch.</returns>
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

    /// <summary>
    /// Converts a CHW (Channel-Height-Width) formatted array of pixel data into a PNG image encoded as a Base64 string.
    /// </summary>
    /// <remarks>The method assumes the input data is in CHW format, where the first channel corresponds to
    /// red, the second to green, and the third to blue. If fewer than three channels are provided, the missing channels
    /// are replicated from the first channel. The pixel values are normalized from the range [-1, 1] to [0, 255] before
    /// being written to the image.</remarks>
    /// <param name="chw">The input array containing pixel data in CHW format. The values are expected to be in the range [-1, 1].</param>
    /// <param name="offset">The starting index in the <paramref name="chw"/> array where the pixel data begins.</param>
    /// <param name="nc">The number of channels in the input data. Must be 1 (grayscale), 2, or 3 (RGB).</param>
    /// <param name="H">The height of the image, in pixels.</param>
    /// <param name="W">The width of the image, in pixels.</param>
    /// <returns>A Base64-encoded string representing the PNG image created from the input pixel data.</returns>
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

    /// <summary>
    /// Clamps a value to the range [0, 1].
    /// </summary>
    /// <param name="v">The value to clamp.</param>
    /// <returns>The clamped value, which will be 0 if <paramref name="v"/> is less than 0, 1 if <paramref name="v"/> is greater
    /// than 1, or <paramref name="v"/> itself if it is within the range [0, 1].</returns>
    private static float Clamp01(float v) => v < 0f ? 0f : (v > 1f ? 1f : v);
}
