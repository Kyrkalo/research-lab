'''FLOPS to calculate the number of floating point operations per second (FLOPS) for a given model.
This is useful for understanding the computational complexity of the model.
This code is based on the chapter code.
It is not a complete implementation, but it provides a starting point for calculating FLOPS.
'''

import torch
import torch.nn as nn
from thop import profile

def calculate_with_fixed_batch_size(model: nn.Module, batch_size: int = 2, input_tensor: torch.Tensor = None, device: str = '') -> tuple:
    """
    Calculate the FLOPS and parameters of a model with a fixed input size.
    
    Args:
        model (nn.Module): The model to benchmark.
        batch_size (int): The batch size for the input tensor.
        input_tensor (torch.Tensor, optional): A sample input tensor. If None, a random tensor will be created.
        
    Returns:
        tuple: A tuple containing the number of FLOPS and parameters.
    """

    if device == '':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")

    if input_tensor is None:
        input_tensor =torch.randint(0, 50257, (batch_size, 1024)).to(device)

    model.to(device)

    macs, params = profile(model, inputs=(input_tensor,), verbose=False)
    flops = 2*macs
    return flops, params
