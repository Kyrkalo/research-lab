using Newtonsoft.Json;

namespace OnnxHub.Dto.LLama32
{
    public class BodyResponse
    {
        public string Model { get; set; }

        [JsonProperty("created_at")]
        public string CreatedAt { get; set; }

        public string Response { get; set; }
    }
}
