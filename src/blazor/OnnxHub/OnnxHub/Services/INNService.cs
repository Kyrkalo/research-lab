namespace OnnxHub.Services
{
    public interface INNService
    {
        IServiceResponse Run(IServiceRequest request, CancellationToken cancellationToken);

        Task<IServiceResponse> RunAsync(IServiceRequest request, CancellationToken cancellationToken);
    }

    public interface IServiceRequest
    {
    }

    public interface IServiceResponse
    {
    }
}
