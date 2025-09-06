using Microsoft.AspNetCore.Mvc;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxHub.Http;
using OnnxHub.Onnx;

namespace OnnxHub.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class MNISTController : ControllerBase
    {
        private readonly IModelRegistry _registry;

        public MNISTController(IModelRegistry registry)
        {
            _registry = registry;
        }

        [HttpPost("")]
        public string Post([FromBody] MnistRequest req)
        {
            var comma = req.Image_base64.IndexOf(',');
            var b64 = comma >= 0 ? req.Image_base64[(comma + 1)..] : req.Image_base64;
            byte[] bytes = default;
            try { 
                bytes = Convert.FromBase64String(b64); 
            }
            catch (Exception e) {
            }

            // Get the MNIST model + converter from the registry
            if (!_registry.TryGet("mdl_mnist_202520", out var session, out var info))
            {
                return null;
            }

            // Preprocess → tensor
            DenseTensor<float> inputTensor;
            try { 
                inputTensor = info.Convert(bytes); 
            }
            catch (Exception e) {
                return null;
            }

            var inputName = session.InputMetadata.First().Key;
            // Run inference
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(inputName, inputTensor)
            };
            using var results = session.Run(inputs);
            var outTensor = results[0].AsEnumerable<float>().ToArray();

            // Softmax + argmax
            var probs = Softmax(outTensor);
            var predicted = ArgMax(probs);

            return predicted.ToString();
        }

        private static float[] Softmax(float[] logits)
        {
            var max = logits.Max();
            var exps = logits.Select(v => MathF.Exp(v - max)).ToArray();
            var sum = exps.Sum();
            for (int i = 0; i < exps.Length; i++) exps[i] /= sum;
            return exps;
        }

        private static int ArgMax(float[] arr)
        {
            int idx = 0; float best = arr[0];
            for (int i = 1; i < arr.Length; i++)
                if (arr[i] > best) { best = arr[i]; idx = i; }
            return idx;
        }
    }
}
