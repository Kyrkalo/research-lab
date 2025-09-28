using Microsoft.ML.OnnxRuntime;
using OnnxHub.Onnx;
using OnnxHub.Onnx.Converter;
using SixLabors.Fonts;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Drawing.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace OnnxHub.Services;

public record Prediction(Box Box, string Label, float Confidence);

public record Box(float Xmin, float Ymin, float Xmax, float Ymax);

public record RCnnRequest(byte[] Bytes, float scoreThreshold = 0.5f) : IServiceRequest;

public record RCnnResponse(string Image) : IServiceResponse;

public class RCnnService : INNService
{
    private readonly InferenceSession _session;
    private readonly IToTensorConverter _converter;
    private readonly float _minConfidence = 0.7f;

    public RCnnService(IModelRegistry modelRegistry)
    {
        modelRegistry.TryGet("FasterRCNN-10", out InferenceSession session, out IToTensorConverter converter);
        _session = session;
        _converter = converter;
    }

    public IServiceResponse Run(IServiceRequest request, CancellationToken cancellationToken)
    {
        if (request is not RCnnRequest rCnnRequest)
            throw new ArgumentException("Invalid request type");

        var input = _converter.Convert(rCnnRequest.Bytes);
        var inputParams = GetParams(_converter);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("image", input)
        };

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
