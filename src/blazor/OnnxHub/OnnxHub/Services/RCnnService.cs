using Microsoft.Extensions.Options;
using Microsoft.ML.OnnxRuntime;
using OnnxHub.Infrastructure;
using OnnxHub.Onnx;
using OnnxHub.Onnx.Converter;
using SixLabors.Fonts;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace OnnxHub.Services;

/// <summary>
/// 
/// </summary>
/// <param name="Box"></param>
/// <param name="Label"></param>
/// <param name="Confidence"></param>
public record Prediction(Box Box, string Label, float Confidence);

/// <summary>
/// Represents a rectangular bounding box defined by its minimum and maximum coordinates.
/// </summary>
/// <remarks>The box is defined in a 2D coordinate space, where <paramref name="Xmin"/> and <paramref
/// name="Ymin"/>  specify the bottom-left corner, and <paramref name="Xmax"/> and <paramref name="Ymax"/> specify the 
/// top-right corner. It is assumed that <paramref name="Xmin"/> is less than or equal to <paramref name="Xmax"/>,  and
/// <paramref name="Ymin"/> is less than or equal to <paramref name="Ymax"/>.</remarks>
/// <param name="Xmin">The minimum X-coordinate of the box.</param>
/// <param name="Ymin">The minimum Y-coordinate of the box.</param>
/// <param name="Xmax">The maximum X-coordinate of the box.</param>
/// <param name="Ymax">The maximum Y-coordinate of the box.</param>
public record Box(float Xmin, float Ymin, float Xmax, float Ymax);

/// <summary>
/// Represents a request for an RCNN (Region-based Convolutional Neural Network) operation,  containing the input data
/// and a threshold for filtering results based on confidence scores.
/// </summary>
/// <param name="Bytes">The input data as a byte array, typically representing an image or other binary content. This parameter cannot be
/// null.</param>
/// <param name="scoreThreshold">The minimum confidence score required for a result to be included.  Must be a value between 0 and 1, where higher
/// values result in stricter filtering.  The default value is 0.5.</param>
public record RCnnRequest(byte[] Bytes, float scoreThreshold = 0.5f) : IServiceRequest;

/// <summary>
/// Represents the response returned by an R-CNN image processing operation.
/// </summary>
/// <remarks>This response contains the processed image data as a string. The format and content of the string 
/// depend on the specific implementation of the R-CNN operation.</remarks>
/// <param name="Image">The processed image data, typically encoded as a string. The exact format of the string  (e.g., base64, JSON, or
/// another format) depends on the operation producing the response.</param>
public record RCnnResponse(string Image) : IServiceResponse;

public class RCnnService : BaseSarvice, INNService
{
    private readonly SysConfigurations _sysConfigurations;
    private readonly Entry _entry;
    private readonly float _minConfidence = 0.7f;

    public RCnnService(IModelRegistry modelRegistry, IOptions<SysConfigurations> options)
    {
        modelRegistry.TryGet(nameof(RCnnService), out var entry);
        _sysConfigurations = options.Value;
        _entry = entry;
    }

    public override IServiceResponse Run(IServiceRequest request, CancellationToken cancellationToken)
    {
        if (request is not RCnnRequest rCnnRequest)
            throw new ArgumentException("Invalid request type");

        var input = _entry.Converter.Convert(rCnnRequest.Bytes);
        var inputParams = GetParams(_entry.Converter);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("image", input)
        };

        using var _session = GetInferenceSession(_sysConfigurations, _entry.Model);
        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = _session.Run(inputs);

        var resultsArray = results.ToArray();
        float[] boxes = resultsArray[0].AsEnumerable<float>().ToArray();
        long[] labels = resultsArray[1].AsEnumerable<long>().ToArray();
        float[] confidences = resultsArray[2].AsEnumerable<float>().ToArray();
        var predictions = new List<Prediction>();
        

        for (int i = 0; i < boxes.Length - 4; i += 4)
        {
            var index = i / 4;
            if (confidences[index] >= _minConfidence)
            {
                predictions.Add(new Prediction(new Box(boxes[i], boxes[i + 1], boxes[i + 2], boxes[i + 3]), Labels[labels[index]], confidences[index]));
            }
        }

        Font font = SystemFonts.CreateFont("Arial", 16);
        var image = inputParams[InputParamType.Image] as Image<Rgb24>;
        foreach (var p in predictions)
        {
            image.Mutate(x =>
            {
                x.DrawLine(Color.Red, 2f, new PointF[] {

                        new PointF(p.Box.Xmin, p.Box.Ymin),
                        new PointF(p.Box.Xmax, p.Box.Ymin),

                        new PointF(p.Box.Xmax, p.Box.Ymin),
                        new PointF(p.Box.Xmax, p.Box.Ymax),

                        new PointF(p.Box.Xmax, p.Box.Ymax),
                        new PointF(p.Box.Xmin, p.Box.Ymax),

                        new PointF(p.Box.Xmin, p.Box.Ymax),
                        new PointF(p.Box.Xmin, p.Box.Ymin)
                    });
                x.DrawText($"{p.Label}, {p.Confidence:0.00}", font, Color.White, new PointF(p.Box.Xmin, p.Box.Ymin));
            });
        }
        using var ms = new MemoryStream();
        image.SaveAsPng(ms);
        var base64 =  Convert.ToBase64String(ms.ToArray());

        return new RCnnResponse(base64);
    }

    public Task<IServiceResponse> RunAsync(IServiceRequest request, CancellationToken cancellationToken)
    {
        throw new NotImplementedException();
    }

    /// <summary>
    /// Retrieves a dictionary of input parameters from the specified tensor converter.
    /// </summary>
    /// <param name="toTensorConverter">An object that converts data to tensors. Must implement the <see cref="IParams"/> interface.</param>
    /// <returns>A dictionary where the keys represent parameter types and the values represent the corresponding parameter
    /// values.</returns>
    /// <exception cref="ArgumentException">Thrown if <paramref name="toTensorConverter"/> does not implement the <see cref="IParams"/> interface.</exception>
    private Dictionary<InputParamType, object> GetParams(IToTensorConverter toTensorConverter)
    {
        if (toTensorConverter is IParams paramConverter)
        {
            return paramConverter.GetParams();
        }
        throw new ArgumentException("Converter does not implement IParams");
    }

    /// <summary>
    /// This is temporary solution, must be moved to configuration or database
    /// </summary>
    private static readonly string[] Labels = new[] {"__background",
                                                        "person",
                                                        "bicycle",
                                                        "car",
                                                        "motorcycle",
                                                        "airplane",
                                                        "bus",
                                                        "train",
                                                        "truck",
                                                        "boat",
                                                        "traffic light",
                                                        "fire hydrant",
                                                        "stop sign",
                                                        "parking meter",
                                                        "bench",
                                                        "bird",
                                                        "cat",
                                                        "dog",
                                                        "horse",
                                                        "sheep",
                                                        "cow",
                                                        "elephant",
                                                        "bear",
                                                        "zebra",
                                                        "giraffe",
                                                        "backpack",
                                                        "umbrella",
                                                        "handbag",
                                                        "tie",
                                                        "suitcase",
                                                        "frisbee",
                                                        "skis",
                                                        "snowboard",
                                                        "sports ball",
                                                        "kite",
                                                        "baseball bat",
                                                        "baseball glove",
                                                        "skateboard",
                                                        "surfboard",
                                                        "tennis racket",
                                                        "bottle",
                                                        "wine glass",
                                                        "cup",
                                                        "fork",
                                                        "knife",
                                                        "spoon",
                                                        "bowl",
                                                        "banana",
                                                        "apple",
                                                        "sandwich",
                                                        "orange",
                                                        "broccoli",
                                                        "carrot",
                                                        "hot dog",
                                                        "pizza",
                                                        "donut",
                                                        "cake",
                                                        "chair",
                                                        "couch",
                                                        "potted plant",
                                                        "bed",
                                                        "dining table",
                                                        "toilet",
                                                        "tv",
                                                        "laptop",
                                                        "mouse",
                                                        "remote",
                                                        "keyboard",
                                                        "cell phone",
                                                        "microwave",
                                                        "oven",
                                                        "toaster",
                                                        "sink",
                                                        "refrigerator",
                                                        "book",
                                                        "clock",
                                                        "vase",
                                                        "scissors",
                                                        "teddy bear",
                                                        "hair drier",
                                                        "toothbrush"};
}
