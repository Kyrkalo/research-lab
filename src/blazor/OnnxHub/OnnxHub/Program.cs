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

builder.Services.AddSingletonOnnx();

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
