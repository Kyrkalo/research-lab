using OnnxHub.Dto;
using OnnxHub.Dto.LLama32;
using System;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text.Json;
using System.Threading.Tasks;

namespace OnnxHub.Integration
{
    public class LLamaApi : BaseApi
    {

        public LLamaApi(HttpClient httpClient) : base(httpClient)
        { }

        public async Task<string> Generate(string text, CancellationToken ct = default)
        {
            BodyRequest body = new BodyRequest()
            {
                Model = "llama3.2",
                Prompt = text,
            };
            var result = await PostAsync<BodyResponse>("/api/generate", body, ct);
            return result.Response;
        }

    }
}
