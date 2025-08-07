using OnnxHub.Components;
using OnnxHub.Infrastructure;
using OnnxHub.Services;
using System.Runtime;

var builder = WebApplication.CreateBuilder(args);

var tokenizerApiSettings = builder.Configuration.GetSection("TokenizerApi").Get<TokenizerApiSettings>();

builder.Services.AddHttpClient("Encode", client => { client.BaseAddress = new Uri($"{tokenizerApiSettings.BaseUrl}/encode"); });
builder.Services.AddHttpClient("Decode", client => { client.BaseAddress = new Uri($"{tokenizerApiSettings.BaseUrl}/decode"); });

builder.Services.AddRazorComponents()
    .AddInteractiveServerComponents();

builder.Services.AddSingleton<TokenizerApiService>();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Error", createScopeForErrors: true);
    // The default HSTS value is 30 days. You may want to change this for production scenarios, see https://aka.ms/aspnetcore-hsts.
    app.UseHsts();
}

app.UseHttpsRedirection();

app.UseStaticFiles();
app.UseAntiforgery();

app.MapRazorComponents<App>()
    .AddInteractiveServerRenderMode();

app.Run();
