using OnnxHub.Dto.LLama32;

namespace OnnxHub.Integration;

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
            Temperature = 0.1
        };

        var result = await PostAsync<BodyResponse>("/api/generate", body, ct);

        return result.Response;
    }
}
