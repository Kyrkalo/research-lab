def calculate_size(model): # based on chapter code
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    total_params_gpt2 =  total_params - sum(p.numel() for p in model.out_head.parameters())
    print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")
    
    # Calculate the total size in bytes (assuming float32, 4 bytes per parameter)
    total_size_bytes = total_params * 4
    
    # Convert to megabytes
    total_size_mb = total_size_bytes / (1024 * 1024)
    
    print(f"Total size of the model: {total_size_mb:.2f} MB")