using Newtonsoft.Json;
using System.Text;

namespace OnnxHub.Integration
{
    public abstract class BaseApi: HttpClient
    {
        protected readonly HttpClient _httpClient;

        public BaseApi(HttpClient httpClient) => _httpClient = httpClient;

        protected async Task<T> GetAsync<T>(string path, CancellationToken ct)
        {
            using var res = await _httpClient.GetAsync(path, ct);
            //await EnsureOk(res);
            var json = await res.Content.ReadAsStringAsync();
            return JsonConvert.DeserializeObject<T>(json);
        }

        protected async Task<T> PostAsync<T>(string path, object payload, CancellationToken ct)
        {
            var content = new StringContent(JsonConvert.SerializeObject(payload,
                new JsonSerializerSettings { NullValueHandling = NullValueHandling.Ignore }),
                Encoding.UTF8, "application/json");

            using var res = await _httpClient.PostAsync(path, content, ct);
            //await EnsureOk(res);
            var json = await res.Content.ReadAsStringAsync();
            return JsonConvert.DeserializeObject<T>(json);
        }

        protected static async Task EnsureOk(HttpResponseMessage res)
        {
            if (res.IsSuccessStatusCode) return;
            var body = res.Content != null ? await res.Content.ReadAsStringAsync() : "";
            throw new HttpRequestException($"HTTP {(int)res.StatusCode} {res.ReasonPhrase}: {body}");
        }
    }
}
