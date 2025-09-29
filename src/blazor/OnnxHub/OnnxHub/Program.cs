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
builder.Services.AddTransient<RCnnService>();
builder.Services.AddTransient<MnistService>();

builder.Services.AddHttpClient<LLamaApi>("llama3.2", client => {
    client.BaseAddress = new Uri("http://localhost:11434");
    client.Timeout = TimeSpan.FromSeconds(30);
});

builder.Services.Configure<SysConfigurations>(builder.Configuration.GetSection("SysConfigurations"));

builder.Services.AddSingletonOnnx();

builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

builder.Services.AddSingleton<TokenizerApiService>();
builder.Services.AddControllers();
builder.Services.AddServerSideBlazor()
    .AddHubOptions(o => o.MaximumReceiveMessageSize = 1024 * 1024 * 100);
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
