"""
Optimization techniques for text-to-3D models to reduce training time and resource usage.
Implements 2025 state-of-the-art techniques for parameter-efficient fine-tuning.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List
import math

# Direct implementation of LoRA that doesn't rely on external packages


class LoRALayer(nn.Module):
    """
    Implementation of LoRA (Low-Rank Adaptation) layer.
    This adds low-rank adaptation matrices to a frozen linear layer.
    """
    def __init__(self, in_features, out_features, rank=4, alpha=16, dropout=0.05):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices (A is in_features x rank, B is rank x out_features)
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.dropout = nn.Dropout(dropout)
        
        # Initialize A with random values and B with zeros for stable training
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        # Low-rank adaptation: x -> A -> dropout -> B -> scaled output
        return self.dropout(x @ self.lora_A) @ self.lora_B * self.scaling


class LinearWithLoRA(nn.Module):
    """
    Linear layer with LoRA adaptation.
    Freezes the original layer and adds LoRA adaptation matrices.
    """
    def __init__(self, linear_layer, rank=4, alpha=16, dropout=0.05):
        super().__init__()
        self.linear = linear_layer
        self.lora = LoRALayer(
            linear_layer.in_features, 
            linear_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
        # Freeze the original linear layer
        for param in self.linear.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # Original frozen path + LoRA path
        return self.linear(x) + self.lora(x)


def apply_lora_to_model(model: nn.Module, 
                         rank: int = 4, 
                         alpha: int = 16,
                         dropout: float = 0.05,
                         target_modules: Optional[Union[List[str], str]] = None) -> nn.Module:
    """
    Apply LoRA (Low-Rank Adaptation) to model to drastically reduce trainable parameters.
    
    LoRA is a parameter-efficient fine-tuning technique that reduces training time and memory
    requirements by freezing the pre-trained weights and injecting trainable rank decomposition
    matrices into each layer.
    
    Args:
        model: The model to apply LoRA to
        rank: LoRA attention dimension (r)
        alpha: LoRA alpha value
        dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA to, if None, will auto-detect
        
    Returns:
        Model with LoRA adapters attached
    """
    # Freeze all parameters by default
    for param in model.parameters():
        param.requires_grad = False
    
    # Auto-detect target modules if not specified
    if target_modules is None:
        # Target any linear layer by default
        target_modules = []
        for name, module in model.named_modules():
            # Focus on key parameter-heavy parts of the model
            if isinstance(module, nn.Linear) and any(
                keyword in name for keyword in 
                ["proj", "query", "key", "value", "output", "dense", "attention", "mlp", "fc"]
            ):
                target_modules.append(name)
    
    # Handle the case of a single string target
    if isinstance(target_modules, str):
        target_modules = [target_modules]
    
    replaced_modules = {}
    
    # Helper function to recursively replace modules
    def replace_module(model, name, module, replaced_modules):
        # Traverse the module hierarchy to find and replace the target module
        name_parts = name.split('.')
        if len(name_parts) == 1:
            # Direct child module
            if hasattr(model, name_parts[0]):
                original_module = getattr(model, name_parts[0])
                if isinstance(original_module, nn.Linear):
                    # Replace with LoRA-enhanced version
                    lora_module = LinearWithLoRA(
                        original_module, rank=rank, alpha=alpha, dropout=dropout
                    )
                    setattr(model, name_parts[0], lora_module)
                    replaced_modules[name] = lora_module
                    return True
            return False
        else:
            # Nested module, recursively descend
            child_name = name_parts[0]
            if hasattr(model, child_name):
                child_module = getattr(model, child_name)
                remaining_path = '.'.join(name_parts[1:])
                return replace_module(child_module, remaining_path, module, replaced_modules)
            return False
    
    # Apply LoRA to all target modules
    for module_name in target_modules:
        try:
            # Get the module
            for name, module in model.named_modules():
                if name == module_name:
                    # Skip if not a linear layer
                    if not isinstance(module, nn.Linear):
                        continue
                        
                    # Find the parent module and replace the target
                    parent_path = '.'.join(module_name.split('.')[:-1])
                    child_name = module_name.split('.')[-1]
                    
                    parent = model
                    if parent_path:
                        for part in parent_path.split('.'):
                            parent = getattr(parent, part)
                    
                    # Replace with LoRA version
                    if hasattr(parent, child_name):
                        original = getattr(parent, child_name)
                        lora_module = LinearWithLoRA(
                            original, rank=rank, alpha=alpha, dropout=dropout
                        )
                        setattr(parent, child_name, lora_module)
                        replaced_modules[module_name] = lora_module
                    
                    break
        except (AttributeError, IndexError) as e:
            print(f"Error applying LoRA to {module_name}: {e}")
    
    # Count and print parameter statistics
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"LoRA applied to {len(replaced_modules)} modules with rank {rank} and alpha {alpha}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
    print(f"Total parameters: {total_params:,}")
    
    return model


def apply_quantization_for_inference(model, quantize_type="dynamic"):
    """
    Apply quantization to reduce model size and speed up inference.
    
    Note: Dynamic quantization in PyTorch currently only supports CPU models.
    If the model is on CUDA, we will copy it to CPU for quantization.
    
    Args:
        model: Model to quantize
        quantize_type: Type of quantization ('dynamic' or 'static')
        
    Returns:
        Quantized model
    """
    import torch.quantization
    import torch.nn as nn
    
    # Store original device
    original_device = next(model.parameters()).device
    was_on_cuda = original_device.type == 'cuda'
    
    # Move to CPU if on CUDA (dynamic quantization only works on CPU)
    if was_on_cuda:
        print("Moving model to CPU for quantization (dynamic quantization only supported on CPU)...")
        model = model.cpu()
    
    if quantize_type == "dynamic":
        # Use direct quantization of specific module types instead of full model conversion
        # This avoids issues with embedding layers that aren't compatible with default quantization
        print("Using selective dynamic quantization for Linear and LSTM layers only...")
        
        # Quantize only linear layers which are the most parameter-heavy
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            {nn.Linear}, # Only quantize Linear layers, skip embeddings
            dtype=torch.qint8
        )
        
        print("Model quantized successfully using dynamic quantization")
        
        # If original model was on CUDA but quantization requires CPU, warn the user
        if was_on_cuda:
            print("WARNING: Quantized model will run on CPU, not GPU. "
                  "Dynamic quantization is not supported on CUDA devices.")
            
        return quantized_model
    
    elif quantize_type == "static":
        print("Static quantization requires calibration data and is not implemented")
        # Move back to original device
        if was_on_cuda:
            model = model.to(original_device)
        return model
    
    else:
        print(f"Unknown quantization type: {quantize_type}")
        # Move back to original device
        if was_on_cuda:
            model = model.to(original_device)
        return model


def progressive_generation(model, text, num_points=4096, device="cuda"):
    """
    Generate point cloud progressively from coarse to fine for better quality and efficiency.
    
    This approach starts with a low-resolution point cloud and iteratively refines it,
    resulting in better structural coherence and faster generation.
    
    Args:
        model: Text-to-3D model
        text: Input text prompt
        num_points: Target number of points in final point cloud
        device: Device to run generation on
        
    Returns:
        Generated point cloud tensor
    """
    # Coarse resolution for initial structure (1/8 of final resolution)
    coarse_points = max(512, num_points // 8)
    
    # Medium resolution for refinement (1/2 of final resolution)
    medium_points = max(1024, num_points // 2)
    
    # Generate initial coarse point cloud
    with torch.no_grad():
        if hasattr(model, 'generate_with_resolution'):
            coarse_pc = model.generate_with_resolution(text, num_points=coarse_points)
            
            # Refine to medium resolution
            medium_pc = model.refine_point_cloud(coarse_pc, text, num_points=medium_points)
            
            # Final high-resolution point cloud
            fine_pc = model.refine_point_cloud(medium_pc, text, num_points=num_points)
        else:
            # Fallback if model doesn't have resolution-specific generation
            fine_pc = model(text, num_points=num_points)
    
    return fine_pc
