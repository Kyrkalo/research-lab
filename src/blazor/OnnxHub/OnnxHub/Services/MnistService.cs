using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxHub.Infrastructure;
using OnnxHub.Onnx;

namespace OnnxHub.Services;

/// <summary>
/// Represents a request containing an image in Base64 format for processing in the MNIST service.
/// </summary>
/// <remarks>This request is typically used to send an image encoded as a Base64 string to the MNIST service for
/// tasks such as digit recognition. Ensure that the provided image is properly encoded and adheres to the expected
/// format for accurate processing.</remarks>
/// <param name="Image_base64">A Base64-encoded string representing the image to be processed. The image should be in a format supported by the
/// MNIST service.</param>
public record MnistRequest(string Image_base64) : IServiceRequest;

/// <summary>
/// Represents the response containing an MNIST image.
/// </summary>
/// <remarks>This response is typically used to return an image from the MNIST dataset. The image is represented
/// as a string, which may encode the image data in a specific format.</remarks>
/// <param name="Image">The MNIST image data, represented as a string. The format of the string depends on the implementation.</param>
public record MnistResponse(string Image) : IServiceResponse;

public class MnistService: BaseSarvice, INNService
{
    private readonly SysConfigurations _sysConfigurations;
    private readonly Entry _entry;

    public MnistService(IModelRegistry registry, IOptions<SysConfigurations> options)
    {
        registry.TryGet(nameof(MnistService), out var entry);
        _sysConfigurations = options.Value;
        _entry = entry;
    }

    public override IServiceResponse Run(IServiceRequest request, CancellationToken cancellationToken)
    {
        if (request is not MnistRequest req)
            throw new ArgumentException("Invalid request type");

        var comma = req.Image_base64.IndexOf(',');
        var b64 = comma >= 0 ? req.Image_base64[(comma + 1)..] : req.Image_base64;
        byte[] bytes = default;

        bytes = Convert.FromBase64String(b64);

        DenseTensor<float> inputTensor = _entry.Converter.Convert(bytes);

        using var session = GetInferenceSession(_sysConfigurations, _entry.Model);
        var inputName = session.InputMetadata.First().Key;
        
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
        };
        using var results = session.Run(inputs);
        var outTensor = results[0].AsEnumerable<float>().ToArray();

        var probs = Softmax(outTensor);
        var predicted = ArgMax(probs);

        return new MnistResponse(predicted.ToString());
    }

    public Task<IServiceResponse> RunAsync(IServiceRequest request, CancellationToken cancellationToken)
    {
        throw new NotImplementedException();
    }

    /// <summary>
    /// Returns the softmax of the input logits.
    /// </summary>
    /// <param name="logits"></param>
    /// <returns></returns>
    private static float[] Softmax(float[] logits)
    {
        var max = logits.Max();
        var exps = logits.Select(v => MathF.Exp(v - max)).ToArray();
        var sum = exps.Sum();
        for (int i = 0; i < exps.Length; i++) exps[i] /= sum;
        return exps;
    }


    /// <summary>
    /// Returns the index of the maximum value in the array.
    /// </summary>
    /// <param name="arr"></param>
    /// <returns></returns>
    private static int ArgMax(float[] arr)
    {
        int idx = 0; float best = arr[0];
        for (int i = 1; i < arr.Length; i++)
            if (arr[i] > best) { best = arr[i]; idx = i; }
        return idx;
    }
}
