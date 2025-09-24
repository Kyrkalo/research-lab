# GPT2-small (the 124M configuration we already implemented):

# "emb_dim" = 768
# "n_layers" = 12
# "n_heads" = 12
# GPT2-medium:

# "emb_dim" = 1024
# "n_layers" = 24
# "n_heads" = 16
# GPT2-large:

# "emb_dim" = 1280
# "n_layers" = 36
# "n_heads" = 20
# GPT2-XL:

# "emb_dim" = 1600
# "n_layers" = 48
# "n_heads" = 25


GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 1024, # Context length
    "emb_dim": 1600,         # Embedding dimension
    "n_heads": 25,          # Number of attention heads
    "n_layers": 48,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

import torch
import torch.nn as nn
import tiktoken

from analysis.benchmark import calculate_with_fixed_batch_size
from models.dummyGPTModel import DummyGPTModel
from models.feedForward import ExampleDeepNeuralNetwork, FeedForward, print_gradients
from models.gpt_model import GPTModel
from models.layerNorm import LayerNorm
#from models.tools import calculate_size
from models.transformerBlock import TransformerBlock
from poviders import GPTConfig, get_config

# tokenizer = tiktoken.get_encoding("gpt2")

# batch = []

# txt1 = "Every effort moves you"
# txt2 = "Every day holds a"

# batch.append(torch.tensor(tokenizer.encode(txt1)))
# batch.append(torch.tensor(tokenizer.encode(txt2)))
# batch = torch.stack(batch, dim=0)
# print(batch)

# torch.manual_seed(123)
# model = DummyGPTModel(GPT_CONFIG_124M)

# logits = model(batch)
# print("Output shape:", logits.shape)
# print(logits)

# # 4.2 Normalizing activations with layer normalization

# torch.manual_seed(123)

# # create 2 training examples with 5 dimensions (features) each
# batch_example = torch.randn(2, 5) 
# print(batch_example)
# layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
# out = layer(batch_example)
# print(out)

# mean = out.mean(dim=-1, keepdim=True)
# var = out.var(dim=-1, keepdim=True)

# print("Mean:\n", mean)
# print("Variance:\n", var)

# ln = LayerNorm(emb_dim=5)
# out_ln = ln(batch_example)

# mean = out_ln.mean(dim=-1, keepdim=True)
# var = out_ln.var(dim=-1, unbiased=False, keepdim=True)

# print("Mean:\n", mean)
# print("Variance:\n", var)

# #4.3 Implementing a feed forward network with GELU activations


# print(GPT_CONFIG_124M["emb_dim"])

# ffn = FeedForward(GPT_CONFIG_124M)

# # input shape: [batch_size, num_token, emb_size]
# x = torch.rand(2, 3, 1600) 
# out = ffn(x)
# print(out.shape)

# # 4.4 Adding shortcut connections

# layer_sizes = [3, 3, 3, 3, 3, 1]  

# sample_input = torch.tensor([[1., 0., -1.]])

# torch.manual_seed(123)
# model_without_shortcut = ExampleDeepNeuralNetwork(
#     layer_sizes, use_shortcut=False
# )
# print_gradients(model_without_shortcut, sample_input)

# # 4.5 Connecting attention and linear layers in a transformer block

# torch.manual_seed(123)

# x = torch.rand(2, 4, 1600)  # Shape: [batch_size, num_tokens, emb_dim]
# block = TransformerBlock(GPT_CONFIG_124M)
# output = block(x)

# print("Input shape:", x.shape)
# print("Output shape:", output.shape)

# torch.manual_seed(123)
# model = GPTModel(GPT_CONFIG_124M)


# for model_abbrev in (GPTConfig.GPT2_SMALL, GPTConfig.GPT2_MEDIUM, GPTConfig.GPT2_LARGE, GPTConfig.GPT2_XLARGE):
#     CONFIG = get_config(GPT_CONFIG_124M, model_name=model_abbrev)
#     model = GPTModel(CONFIG)
#     print(f"\n\n{model_abbrev}:")
#     calculate_size(model)

BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

for size in model_configs:
    BASE_CONFIG.update(model_configs[size])
    model = GPTModel(BASE_CONFIG).bfloat16()
    flops, params = calculate_with_fixed_batch_size(model, batch_size=2)
    del model
    torch.cuda.empty_cache()
    print(f"{size:18}: {flops:.1e} FLOPS; {params/1e6:.1f} M parameters")

