using Microsoft.AspNetCore.Mvc;
using OnnxHub.Services;

namespace OnnxHub.Controllers;

[Route("api/[controller]")]
[ApiController]
public class MNISTController : ControllerBase
{
    private readonly MnistService _mnistService;

    public MNISTController(MnistService mnistService)
    {
        _mnistService = mnistService;
    }

    [HttpPost("")]
    public string Post([FromBody] Services.MnistRequest req)
    {
        return _mnistService.Run(req, CancellationToken.None) is MnistResponse res ? res.Image : string.Empty;
    }
}
