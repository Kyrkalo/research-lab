using Microsoft.ML.OnnxRuntime;
using OnnxHub.Components;
using OnnxHub.Infrastructure;
using OnnxHub.Integration;
using OnnxHub.Onnx;
using OnnxHub.Onnx.Converter;
using OnnxHub.Services;

var builder = WebApplication.CreateBuilder(args);

var tokenizerApiSettings = builder.Configuration.GetSection("TokenizerApi").Get<TokenizerApiSettings>();

builder.Services.AddHttpClient<TokenizerApiService>("Encode", client => { client.BaseAddress = new Uri($"{tokenizerApiSettings.BaseUrl}/encode"); });
builder.Services.AddHttpClient<TokenizerApiService>("Decode", client => { client.BaseAddress = new Uri($"{tokenizerApiSettings.BaseUrl}/decode"); });
builder.Services.AddSingleton<IToTensorConverter, MnistImageConverter>();
builder.Services.AddTransient<GanGeneratorService>();

builder.Services.AddHttpClient<LLamaApi>("llama3.2", client => {
    client.BaseAddress = new Uri("http://localhost:11434");
    client.Timeout = TimeSpan.FromSeconds(30);
});

builder.Services.AddSingleton<IModelRegistry>(e =>
{
    var sessionOptions = new Microsoft.ML.OnnxRuntime.SessionOptions() 
    {
        GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
        EnableMemoryPattern = true,
        ExecutionMode = ExecutionMode.ORT_PARALLEL
    };

    var mdl_mnist_202520 = new InferenceSession(Path.Combine(AppContext.BaseDirectory, "Onnx/Models", "mdl_mnist_202520.onnx"), sessionOptions);
    var gen = new InferenceSession(Path.Combine(AppContext.BaseDirectory, "Onnx/Models", "generator.onnx"), sessionOptions);

    return new ModelRegistry()
    .Add("mdl_mnist_202520", mdl_mnist_202520, new MnistImageConverter())
    .Add("gen_faces", gen, new GanConverter());
});

builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

builder.Services.AddSingleton<TokenizerApiService>();
builder.Services.AddControllers();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error", createScopeForErrors: true);
    // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
    app.UseHsts();
}

app.UseHttpsRedirection();
app.MapControllers();

app.UseStaticFiles();
app.UseAntiforgery();

app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.Run();
