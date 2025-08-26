using System;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Threading.Tasks;
using System.Text.Json;

namespace OnnxHub.Integration
{
    public class LLamaApi : HttpClient
    {
        private readonly HttpClient _httpClient;
        private string _model;

        public LLamaApi(HttpClient httpClient, string model)
        {
            _httpClient = httpClient;
            _model = model;
        }

        public async Task<string> SendMessageAsync(string prompt, CancellationToken cancellationToken = default)
        {
            var payload = new
            {
                model = _model,
                message = new[] 
                {
                    new { role = "user", content = prompt }
                },
                stream = false
            };

            HttpResponseMessage response = await _httpClient.PostAsJsonAsync("message", payload, cancellationToken);

            using var stream = await response.Content.ReadAsStreamAsync(cancellationToken);
            using var doc = await JsonDocument.ParseAsync(stream, cancellationToken: cancellationToken);

            return doc.RootElement.GetProperty("message")
                                  .GetProperty("content")
                                  .GetString();
        }
    }
}
