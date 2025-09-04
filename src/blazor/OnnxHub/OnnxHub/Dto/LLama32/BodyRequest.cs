namespace OnnxHub.Dto.LLama32
{
    public class BodyRequest
    {
        public string Model { get; set; }
        public string Prompt { get; set; }
        public bool Stream { get; set; }
        public double Temperature { get; set; }
    }
}
