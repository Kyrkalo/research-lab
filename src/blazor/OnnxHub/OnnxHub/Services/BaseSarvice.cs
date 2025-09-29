using Microsoft.ML.OnnxRuntime;
using OnnxHub.Infrastructure;

namespace OnnxHub.Services;

/// <summary>
/// Provides a base class for implementing services that execute operations based on service requests.
/// </summary>
/// <remarks>This abstract class defines the contract for executing operations through the <see cref="Run"/>
/// method,  which must be implemented by derived classes. It also includes utility methods for managing inference 
/// sessions and configuring session options. Derived classes should provide specific implementations  tailored to their
/// respective service logic.</remarks>
public abstract class BaseSarvice
{
    /// <summary>
    /// Executes the operation defined by the specified service request.
    /// </summary>
    /// <remarks>The behavior of the operation is determined by the implementation of the derived class. 
    /// Callers should ensure that the <paramref name="cancellationToken"/> is monitored if cancellation is
    /// required.</remarks>
    /// <param name="request">The service request containing the parameters and context for the operation. Cannot be <see langword="null"/>.</param>
    /// <param name="cancellationToken">A token that can be used to cancel the operation.</param>
    /// <returns>An <see cref="IServiceResponse"/> representing the result of the operation.</returns>
    public abstract IServiceResponse Run(IServiceRequest request, CancellationToken cancellationToken);

    /// <summary>
    /// Returns an InferenceSession for the specified model using the provided system configurations.
    /// </summary>
    /// <param name="sysConfigurations"></param>
    /// <param name="model"></param>
    /// <returns></returns>
    protected InferenceSession GetInferenceSession(SysConfigurations sysConfigurations, string model)
    {
        var sessionOptions = GetSessionOptions(sysConfigurations);
        return new InferenceSession(Path.Combine(AppContext.BaseDirectory, sysConfigurations.ModelPath, model), sessionOptions);
    }

    /// <summary>
    /// Returns session options configured for optimal performance.
    /// </summary>
    /// <param name="sysConfigurations"></param>
    /// <returns></returns>
    private Microsoft.ML.OnnxRuntime.SessionOptions GetSessionOptions(SysConfigurations sysConfigurations)
    {
        return new Microsoft.ML.OnnxRuntime.SessionOptions()
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            EnableMemoryPattern = true,
            ExecutionMode = ExecutionMode.ORT_PARALLEL
        };
    }
}
