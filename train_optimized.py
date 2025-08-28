#!/usr/bin/env python
"""
Optimized training script for Text-to-3D model generation.
This script applies all optimizations (LoRA, mixed precision, progressive generation)
in a single runnable file for the best performance/quality trade-off.
"""

import os
import argparse
import yaml
import torch
import datetime
import time
from types import SimpleNamespace
from tqdm import tqdm
from accelerate import Accelerator

# Import model and utilities
from model.text_to_3d_model import TextTo3DModel
from utils.dataset import ModelNet10Dataset
from torch.utils.data import DataLoader


# Function to save model checkpoints
def save_checkpoint(model, optimizer, checkpoint_path, epoch):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)


def train_optimized(args):
    """Run optimized training for Text-to-3D model."""
    print("Starting optimized training with all performance enhancements...")
    
    # Load configuration
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # YAML config has nested dictionaries that need to be converted to namespace
    # for attribute-style access
    def dict_to_namespace(d):
        if isinstance(d, dict):
            for key, value in d.items():
                d[key] = dict_to_namespace(value)
            return SimpleNamespace(**d)
        return d
    
    config = dict_to_namespace(config_dict)
    
    # Set up Accelerator with mixed precision
    accelerator = Accelerator(
        mixed_precision='fp16',
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )
    
    # Set up TensorBoard logging separately if needed
    if args.log_dir:
        from torch.utils.tensorboard import SummaryWriter
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.log_dir)
    else:
        writer = None
    
    # Print setup info
    print(f"Device: {accelerator.device}")
    print(f"Mixed precision: {accelerator.mixed_precision}")
    print(f"Num devices: {accelerator.num_processes}")
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        
    # Setup logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"text2shape_optimized_{timestamp}"
    
    # Create datasets and data loaders
    print("Loading datasets...")
    train_dataset = ModelNet10Dataset(
        data_dir=config.dataset.data_dir,
        split='train'
    )
    
    val_dataset = ModelNet10Dataset(
        data_dir=config.dataset.data_dir,
        split='test'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    print("Creating model...")
    model = TextTo3DModel(config.model)
    
    # Apply LoRA for parameter-efficient training
    print(f"Applying LoRA with rank {args.lora_rank}...")
    model.apply_lora(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout
    )
    
    # Print parameter stats
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
    
    # Optimizer (AdamW works better with LoRA)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99)
    )
    
    # Define loss function (Chamfer Distance for point clouds)
    def chamfer_distance(x, y):
        """Compute chamfer distance between two point clouds."""
        x = x.unsqueeze(2)  # [B, N, 1, 3]
        y = y.unsqueeze(1)  # [B, 1, M, 3]
        
        # Compute pairwise distances
        dist = torch.sum((x - y) ** 2, dim=-1)  # [B, N, M]
        
        # Compute min distances for each point in each point cloud
        min_y_to_x = torch.min(dist, dim=1)[0]  # [B, M]
        min_x_to_y = torch.min(dist, dim=2)[0]  # [B, N]
        
        # Average over points
        chamfer_dist = torch.mean(min_y_to_x, dim=1) + torch.mean(min_x_to_y, dim=1)
        
        # Average over batch to get a scalar loss
        chamfer_dist = torch.mean(chamfer_dist)
        
        return chamfer_dist
    
    criterion = chamfer_distance
    
    # Prepare for training with accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    # Use a more robust CosineAnnealingWarmRestarts scheduler instead
    # This avoids step counting issues that OneCycleLR can have with gradient accumulation
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=args.epochs,  # First restart cycle length
        T_mult=1,        # Keep the same cycle length
        eta_min=1e-8     # Minimum learning rate
    )
    
    # Training loop
    best_val_loss = float('inf')
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train phase
        model.train()
        train_loss = 0
        train_batches = 0
        
        # Progress bar for training
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        # Use accelerator.accumulate for gradient accumulation
        for batch_idx, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                text_prompts, point_clouds, _ = batch  # Unpack text, point cloud, and ignore label
                
                # Forward pass with progressive generation during training
                outputs = model(text_list=text_prompts, progressive=True)
                
                # Compute loss
                loss = criterion(outputs, point_clouds.to(accelerator.device))
                
                # Backward pass with accelerator (which handles gradient unscaling internally)
                accelerator.backward(loss)
                
                # Note: We're not manually clipping gradients here as that can cause conflicts
                # with Accelerator's automatic gradient unscaling in mixed precision training
                
                # Update model (only after gradient accumulation is complete)
                optimizer.step()
                
                # For batch-level schedulers like OneCycleLR, we'd step here
                # But for epoch-level schedulers like CosineAnnealingWarmRestarts,
                # we'll step once per epoch at the end of the epoch instead
                
                optimizer.zero_grad()
                
                # Update metrics
                train_loss += accelerator.gather(loss.detach()).mean().item()
                train_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": loss.item(), 
                    "lr": optimizer.param_groups[0]['lr']
                })
        
        # Calculate average training loss
        train_loss /= train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                text_prompts, point_clouds, _ = batch  # Unpack text, point cloud, and ignore label
                
                # Forward pass - use standard generation for validation
                outputs = model(text_list=text_prompts)
                
                # Compute loss
                loss = criterion(outputs, point_clouds.to(accelerator.device))
                
                # Update metrics
                val_loss += accelerator.gather(loss.detach()).mean().item()
                val_batches += 1
        
        # Calculate average validation loss
        val_loss /= val_batches
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Time: {epoch_time:.2f}s")
              
        # Log with TensorBoard
        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/validation", val_loss, epoch)
            writer.add_scalar("Learning_rate", optimizer.param_groups[0]['lr'], epoch)
            
            # Additional metrics can be logged here
            writer.add_scalar("Time/epoch", epoch_time, epoch)
        
        # Step the scheduler once per epoch (for CosineAnnealingWarmRestarts)
        scheduler.step()
        
        # Save best model
        if val_loss < best_val_loss and accelerator.is_local_main_process:
            best_val_loss = val_loss
            best_model_path = os.path.join(
                args.output_dir,
                f"{run_name}_best.pt"
            )
            
            # Unwrap model before saving
            unwrapped_model = accelerator.unwrap_model(model)
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': unwrapped_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config_dict,
            }, best_model_path)
            
            print(f"✓ Best model saved with loss: {val_loss:.4f}")
        
        # Synchronize across processes
        accelerator.wait_for_everyone()
    
    # End training
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved at: {best_model_path}")
    
    # Helper function to convert LoRA state dict to regular state dict
    def convert_lora_state_dict_to_regular(state_dict):
        # Create a new state dict
        regular_state_dict = {}
        
        # Process each key in the state dict
        for key, value in state_dict.items():
            # Check if this is a LoRA parameter
            if '.linear.weight' in key:
                # Replace with the regular parameter name
                new_key = key.replace('.linear.weight', '.weight')
                regular_state_dict[new_key] = value
            elif '.linear.bias' in key:
                # Replace with the regular parameter name
                new_key = key.replace('.linear.bias', '.bias')
                regular_state_dict[new_key] = value
            elif '.lora.' not in key:  # Skip LoRA-specific parameters
                # Keep other parameters as is
                regular_state_dict[key] = value
        
        return regular_state_dict
    
    # Save the final quantized model for efficient inference
    if accelerator.is_local_main_process:
        print("Creating quantized model for efficient inference...")
        final_model = TextTo3DModel(config.model)
        
        # Convert the LoRA state dict to a regular state dict and load it
        lora_state_dict = unwrapped_model.state_dict()
        regular_state_dict = convert_lora_state_dict_to_regular(lora_state_dict)
        
        # Load the converted state dict with strict=False to allow missing keys
        final_model.load_state_dict(regular_state_dict, strict=False)
        final_model.eval()
        
        # Quantize the model
        quantized_model = final_model.quantize_for_inference()
        
        # Save quantized model
        quantized_model_path = os.path.join(
            args.output_dir,
            f"{run_name}_quantized.pt"
        )
        torch.save({
            'model_state_dict': quantized_model.state_dict(),
            'config': config_dict,
        }, quantized_model_path)
        
        print(f"✓ Quantized model saved at: {quantized_model_path}")
        
        # Save final model
        final_model_path = os.path.join(
            args.output_dir,
            f"{run_name}_final.pt"
        )
        save_checkpoint(unwrapped_model, optimizer, final_model_path, epoch)
        print(f"Final model saved at {final_model_path}")
    
    # Close TensorBoard writer
    if writer is not None:
        writer.close()
        print("TensorBoard logs saved to {}".format(args.log_dir))
    
    return best_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized training for Text-to-3D model")
    
    # Basic configuration
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Directory to save models")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory for logs (leave empty to disable)")
    
    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Peak learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for regularization")
    parser.add_argument("--gradient_clip", type=float, default=1.0,
                        help="Gradient clipping value")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    
    # Low-resource optimization parameters
    parser.add_argument("--lora_rank", type=int, default=4,
                        help="Rank for LoRA adapters")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="Alpha scaling for LoRA")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="Dropout probability for LoRA layers")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of steps to accumulate gradients")
                        
    args = parser.parse_args()
    
    # Run optimized training
    best_model_path = train_optimized(args)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Best model saved at: {best_model_path}")
    print("\nTo generate 3D models, run:")
    print(f"python generate.py --checkpoint {best_model_path} --prompt \"your text prompt\" --progressive --quantize")
    print("="*50)
