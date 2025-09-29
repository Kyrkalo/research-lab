using OnnxHub.Onnx.Converter;

namespace OnnxHub.Onnx;

/// <summary>
/// Represents an entry that associates a model identifier with a tensor conversion strategy.
/// </summary>
/// <param name="Model">The identifier of the model associated with this entry. This value cannot be null.</param>
/// <param name="Converter">The tensor converter used to transform data for the associated model. This value cannot be null.</param>
public sealed record Entry(string Model, IToTensorConverter Converter);

/// <summary>
/// Represents a registry for managing models and their associated converters.
/// </summary>
/// <remarks>The <see cref="ModelRegistry"/> class provides functionality to add models with unique names and
/// retrieve them along with their associated converters. It supports case-insensitive name lookups and allows method
/// chaining when adding models.</remarks>
public sealed class ModelRegistry : IModelRegistry
{
    private readonly Dictionary<string, Entry> _map = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// Adds a model to the registry with the specified name and converter.
    /// </summary>
    /// <param name="name">The unique name used to identify the model in the registry. Cannot be <see langword="null"/> or empty.</param>
    /// <param name="model">The model to be added to the registry. Cannot be <see langword="null"/>.</param>
    /// <param name=""></param>
    /// <param name="converter">The converter used to transform input data to tensors for the model. Cannot be <see langword="null"/>.</param>
    /// <returns>The current <see cref="ModelRegistry"/> instance, allowing for method chaining.</returns>
    public ModelRegistry Add(string name, string model, IToTensorConverter converter)
    {
        _map[name] = new Entry(model, converter);
        return this;
    }

    /// <summary>
    /// Attempts to retrieve an entry associated with the specified name.
    /// </summary>
    /// <param name="name">The name of the entry to retrieve.</param>
    /// <param name="entry">When this method returns, contains the entry associated with the specified name,  if the name is found;
    /// otherwise, the default value for the <see cref="Entry"/> type. This parameter is passed uninitialized.</param>
    /// <returns><see langword="true"/> if an entry with the specified name is found; otherwise, <see langword="false"/>.</returns>
    public bool TryGet(string name, out Entry entry)
    {
        if (_map.TryGetValue(name, out var e))
        {
            entry = e;
            return true;
        }
        entry = default!;
        return false;
    }
}
