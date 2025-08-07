using OnnxHub.Infrastructure;
using System.Text.Json.Serialization;

namespace OnnxHub.Services
{
    public class TokenizerApiService
    {
        private readonly IHttpClientFactory _httpClientFactory;

        public TokenizerApiService(IHttpClientFactory factory) => _httpClientFactory = factory;

        public async Task<long[]> EncodeAsync(string text)
        {
            var client = _httpClientFactory.CreateClient("Encode");
            var response = await client.PostAsJsonAsync("", new { text });
            response.EnsureSuccessStatusCode();
            var result = await response.Content.ReadFromJsonAsync<EncodeResponse>();
            return result?.Tokens ?? Array.Empty<long>();
        }

        public async Task<string> DecodeAsync(long[] tokens)
        {
            var client = _httpClientFactory.CreateClient("Decode");
            var response = await client.PostAsJsonAsync("", new { tokens });
            response.EnsureSuccessStatusCode();
            var result = await response.Content.ReadFromJsonAsync<DecodeResponse>();
            return result?.Text ?? "";
        }
    }

    public class EncodeResponse
    {
        [JsonPropertyName("tokens")]
        public long[] Tokens { get; set; }
    }

    public class DecodeResponse
    {
        [JsonPropertyName("text")]
        public string Text { get; set; }
    }
}
