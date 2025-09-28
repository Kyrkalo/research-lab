using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxHub.Onnx.Converter;

public class GanConverter : IToTensorConverter
{
    private static void Gaussian(Span<float> span)
    {
        var rng = new Random();
        int i = 0;
        while (i < span.Length)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            double r = Math.Sqrt(-2.0 * Math.Log(u1));
            double th = 2.0 * Math.PI * u2;
            float n0 = (float)(r * Math.Cos(th));
            float n1 = (float)(r * Math.Sin(th));
            span[i++] = n0;
            if (i < span.Length)
            {
                span[i++] = n1;
            }
        }
    }

    public DenseTensor<float> Convert(params int[] bytes)
    {
        var denseTensors = new DenseTensor<float>(bytes);
        Gaussian(denseTensors.Buffer.Span);
        return denseTensors;
    }

    public DenseTensor<float> Convert(Stream stream)
    {
        throw new NotImplementedException();
    }

    public DenseTensor<float> Convert(byte[] bytes)
    {
        throw new NotImplementedException();
    }

    public Task<DenseTensor<float>> ConvertAsync(Stream stream, CancellationToken cancellationToken)
    {
        throw new NotImplementedException();
    }
}
