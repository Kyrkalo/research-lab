from enum import Enum

class GPTConfig(Enum):
    GPT2_SMALL = "gpt2_small"
    GPT2_MEDIUM = "gpt2_medium"
    GPT2_LARGE = "gpt2_large"
    GPT2_XLARGE = "gpt2_xlarge"

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

def get_config(base_config, model_name: GPTConfig):
    config = base_config.copy()
    if model_name == GPTConfig.GPT2_SMALL:
        config.update({
            "emb_dim": 768,
            "n_layers": 12,
            "n_heads": 12
        })
    elif model_name == GPTConfig.GPT2_MEDIUM:
        config.update({
            "emb_dim": 1024,
            "n_layers": 24,
            "n_heads": 16
        })
    elif model_name == GPTConfig.GPT2_LARGE:
        config.update({
            "emb_dim": 1280,
            "n_layers": 36,
            "n_heads": 20
        })
    elif model_name == GPTConfig.GPT2_XLARGE:
        config.update({
            "emb_dim": 1600,
            "n_layers": 48,
            "n_heads": 25
        })
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return config