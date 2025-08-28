import json
import os
import sys
import time
import torch
import random
import argparse
import numpy as np
import yaml
import glob
import scipy
import trimesh
from types import SimpleNamespace
from tqdm import tqdm
from datetime import datetime
from utils.report_generator import generate_research_report

# Import model and utilities
from model.text_to_3d_model import TextTo3DModel
from utils.visualization import visualize_point_cloud, visualize_mesh
from utils.visualization import save_point_cloud_as_ply, save_mesh_as_obj
from utils.mesh_refinement import point_cloud_to_mesh

# Import metrics for evaluation
from utils.metrics import chamfer_distance, hausdorff_distance, f_score


def load_off_file(file_path):
    """
    Load an OFF file and convert it to a trimesh mesh.
    
    Args:
        file_path: Path to the OFF file
        
    Returns:
        trimesh.Trimesh object
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    # Check header
    if not lines[0].strip() == 'OFF':
        raise ValueError(f"Invalid OFF file format: {file_path}")
        
    # Parse vertex and face count
    counts = lines[1].strip().split()
    n_vertices = int(counts[0])
    n_faces = int(counts[1])
    
    # Read vertices
    vertices = []
    for i in range(n_vertices):
        line = lines[i + 2].strip().split()
        vertices.append([float(line[0]), float(line[1]), float(line[2])])
    vertices = np.array(vertices)
    
    # Read faces
    faces = []
    for i in range(n_faces):
        line = lines[i + 2 + n_vertices].strip().split()
        if int(line[0]) == 3:  # Only triangular faces
            faces.append([int(line[1]), int(line[2]), int(line[3])])
    faces = np.array(faces)
    
    # Create trimesh mesh
    return trimesh.Trimesh(vertices=vertices, faces=faces)


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert to namespace for easier access
    config = SimpleNamespace(**{
        k: SimpleNamespace(**v) if isinstance(v, dict) else v 
        for k, v in config_dict.items()
    })
    
    return config


def convert_lora_state_dict(state_dict, model_state_dict=None):
    """Convert state dict with LoRA keys to regular model keys with improved handling
    
    Args:
        state_dict: State dict from LoRA-enhanced model
        model_state_dict: Optional existing model state dict to merge with
        
    Returns:
        Converted state dict compatible with regular model
    """
    new_state_dict = {} if model_state_dict is None else model_state_dict.copy()
    lora_prefixes = {}
    
    # First pass: identify all LoRA prefixes
    for key in state_dict.keys():
        if '.lora_A' in key or '.lora_B' in key:
            # Extract the module name without the lora suffix
            prefix = key.split('.lora_')[0]
            lora_prefixes[prefix] = True
    
    # Second pass: handle all key types properly
    for key, value in state_dict.items():
        # Skip LoRA-specific parameters - we'll handle these separately
        if '.lora_A' in key or '.lora_B' in key:
            continue
            
        # Handle direct linear layer weights
        if '.linear.weight' in key:
            new_key = key.replace('.linear.weight', '.weight')
            new_state_dict[new_key] = value
        elif '.linear.bias' in key:
            new_key = key.replace('.linear.bias', '.bias')
            new_state_dict[new_key] = value
        # Handle other weights
        elif any(prefix in key for prefix in lora_prefixes) and '.linear.' not in key:
            # These are the base weights in a LoRA-adapted layer
            # Keep as is, since they're already in the right format
            new_state_dict[key] = value
        # Copy any non-LoRA weights directly
        elif '.lora.' not in key:
            new_state_dict[key] = value
    
    # Print statistics
    print(f"Converted {len(state_dict)} LoRA parameters to {len(new_state_dict)} standard parameters")
    
    return new_state_dict

def load_model(checkpoint_path, config, quantize=False):
    """Load trained model from checkpoint with optional quantization
    
    Args:
        checkpoint_path: Path to model checkpoint
        config: Model configuration
        quantize: Whether to quantize the model for faster inference
        
    Returns:
        Loaded model ready for inference
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint first to inspect contents
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if the state dict has LoRA adapters
    state_dict = checkpoint['model_state_dict']
    has_lora = any('.lora_' in k or '.linear.' in k for k in state_dict.keys())
    
    # Create model based on checkpoint type
    model = TextTo3DModel(config.model).to(device)
    
    # Get initial state for debugging
    initial_state = {k: v.clone() for k, v in model.state_dict().items() if 'shape_encoder' in k}
    
    if has_lora:
        print("Found LoRA-enhanced model checkpoint, processing weights...")
        print(f"LoRA checkpoint has {len(state_dict)} parameters")
        
        # Get a few LoRA keys to understand the structure
        lora_keys = [k for k in state_dict.keys() if '.lora_' in k][:5]
        linear_keys = [k for k in state_dict.keys() if '.linear.' in k][:5]
        if lora_keys:
            print(f"Sample LoRA keys: {lora_keys}")
        if linear_keys:
            print(f"Sample Linear keys: {linear_keys}")
            
        # Get base model state dict for merging
        base_state_dict = model.state_dict()
        
        # Convert and merge state dicts with improved conversion
        converted_state_dict = convert_lora_state_dict(state_dict, base_state_dict)
        
        # Load with strict=False to handle any remaining mismatches
        missing, unexpected = model.load_state_dict(converted_state_dict, strict=False)
        
        if missing:
            print(f"Warning: {len(missing)} missing keys in model: {missing[:3]}...")
        if unexpected:
            print(f"Warning: {len(unexpected)} unexpected keys in checkpoint: {unexpected[:3]}...")
    else:
        # Regular model checkpoint
        print("Loading standard model checkpoint...")
        model.load_state_dict(state_dict)
    
    # Set to evaluation mode
    model.eval()
    
    # Debug: check if key parameters were loaded correctly by comparing before/after values
    # Focus on shape encoder weights as they're critical for geometric understanding
    changed_keys = 0
    for k, v in model.state_dict().items():
        if 'shape_encoder' in k and k in initial_state:
            if not torch.allclose(initial_state[k], v):
                changed_keys += 1
    print(f"Shape encoder had {changed_keys} parameters changed during loading")
    
    # Apply quantization if requested (only after successful loading)
    if quantize:
        print("Applying quantization for faster inference...")
        model = model.quantize_for_inference(quantize_type="dynamic")
    
    print(f"Model successfully loaded and prepared for inference")
    return model


def generate_3d_from_text(model, text_prompt, config, num_samples=1, temperature=0.8, 
progressive=False, resolution=None, guidance_scale=3.0, num_inference_steps=50, unconditional=False):
    """Generate 3D model from text prompt with optional progressive generation
    
    Args:
        model: Text-to-3D model
        text_prompt: Text prompt for generation
        config: Model configuration
        num_samples: Number of samples to generate
        temperature: Sampling temperature
        progressive: Whether to use progressive generation
        resolution: Target resolution for point cloud generation
        guidance_scale: Text conditioning strength (higher = more faithful to text)
        num_inference_steps: Number of refinement steps during generation
        unconditional: Whether to ignore text prompt and generate unconditionally
        
    Returns:
        Generated 3D object (point cloud or mesh)
    """
    device = next(model.parameters()).device
    
    # Create batch for multiple samples
    text_prompts = [text_prompt] * num_samples
    
    # Generate shape
    with torch.no_grad():
        if config.model.output_type == 'point_cloud':
            if progressive and hasattr(model, 'coarse_decoder'):
                print("Using progressive generation for better quality...")
                # Initial coarse generation
                coarse_pc = model.generate_with_resolution(text_prompts, num_points=512, 
                                                       temperature=temperature, 
                                                       guidance_scale=guidance_scale,
                                                       num_inference_steps=num_inference_steps,
                                                       unconditional_generation=unconditional)
                
                # Refine to medium resolution if needed
                medium_pc = None
                if config.model.num_points > 512:
                    medium_pc = model.refine_point_cloud(
                        coarse_pc, text_prompts, 
                        num_points=min(2048, config.model.num_points // 2)
                    )
                
                # Final high-resolution output
                if resolution is not None:
                    target_points = resolution
                else:
                    target_points = config.model.num_points
                
                if target_points > 2048 and medium_pc is not None:
                    outputs = model.refine_point_cloud(medium_pc, text_prompts, target_points)
                elif target_points > 512:
                    outputs = model.refine_point_cloud(coarse_pc, text_prompts, target_points)
                else:
                    outputs = coarse_pc
            else:
                # Standard generation
                if resolution is not None:
                    outputs = model.generate_with_resolution(text_prompts, num_points=resolution,
                                                         temperature=temperature,
                                                         guidance_scale=guidance_scale,
                                                         num_inference_steps=num_inference_steps,
                                                         unconditional_generation=unconditional)
                else:
                    outputs = model.generate_with_resolution(text_prompts,
                                                         temperature=temperature,
                                                         guidance_scale=guidance_scale,
                                                         num_inference_steps=num_inference_steps,
                                                         unconditional_generation=unconditional)
        else:  # mesh generation
            outputs = model(text_list=text_prompts)
    
    # Process output based on model type
    if config.model.output_type == 'point_cloud':
        return outputs.detach().cpu().numpy()
    elif config.model.output_type == 'mesh':
        vertices, faces = outputs
        return vertices.detach().cpu().numpy(), faces.detach().cpu().numpy()


def main(args):
    # Load configuration
    config = load_config(args.config)
    
    # Set output name based on prompt if not provided
    if args.output_name is None:
        # Clean prompt for use as filename
        args.output_name = args.prompt.lower().replace(' ', '_').replace('"', '').replace('\'', '').replace('/', '_')
        # Truncate if too long
        if len(args.output_name) > 30:
            args.output_name = args.output_name[:30]
    
    # Load model
    model = load_model(args.checkpoint, config, quantize=args.quantize)
    
    # Generate 3D model from text with optional progressive generation
    outputs = generate_3d_from_text(
        model, 
        args.prompt, 
        config,
        num_samples=args.num_samples,
        temperature=args.temperature,
        progressive=args.progressive,
        resolution=args.resolution,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        unconditional=args.unconditional
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process and save outputs
    if config.model.output_type == 'point_cloud':
        # Handle point cloud output
        point_clouds = outputs
        
        for i in range(len(point_clouds)):
            # Get sample name
            if args.num_samples > 1:
                sample_name = f"{args.output_name}_{i}"
            else:
                sample_name = args.output_name
            
            # Save raw point cloud
            save_point_cloud_as_ply(
                point_clouds[i],
                os.path.join(args.output_dir, f"{sample_name}_raw.ply")
            )
            
            # Visualize raw point cloud
            img = visualize_point_cloud(
                point_clouds[i], 
                title=f"Generated Point Cloud: {args.prompt}",
                save_path=os.path.join(args.output_dir, f"{sample_name}_pointcloud.png")
            )
            
            print(f"Saved point cloud as {os.path.join(args.output_dir, f'{sample_name}_raw.ply')}")
            
            # Apply mesh refinement if requested
            if args.generate_mesh:
                print(f"Generating mesh from point cloud...")
                
                try:
                    # Convert point cloud to mesh using refinement techniques
                    refined_mesh = point_cloud_to_mesh(
                        point_clouds[i],
                        method=args.refinement_method,
                        poisson_depth=args.poisson_depth,
                        alpha_value=args.alpha_value,
                        smooth_iterations=args.smooth_iterations,
                        smooth_lambda=args.smooth_lambda,
                        simplify_fraction=args.simplify_fraction,
                        fill_hole_size=args.fill_hole_size
                    )
                    
                    # Visualize refined mesh
                    img = visualize_mesh(
                        refined_mesh.vertices, 
                        refined_mesh.faces,
                        title=f"Refined Mesh: {args.prompt}",
                        save_path=os.path.join(args.output_dir, f"{sample_name}_refined.png")
                    )
                    
                    # Save refined mesh
                    save_mesh_as_obj(
                        refined_mesh.vertices,
                        refined_mesh.faces,
                        os.path.join(args.output_dir, f"{sample_name}_refined.obj")
                    )
                    
                    print(f"Saved refined mesh as {os.path.join(args.output_dir, f'{sample_name}_refined.obj')}")
                except Exception as e:
                    print(f"Mesh refinement failed: {e}")
    
    elif config.model.output_type == 'mesh':
        # Handle mesh output
        vertices, faces = outputs
        vertices = vertices.detach().cpu().numpy()
        faces = faces.detach().cpu().numpy()
        
        for i in range(len(vertices)):
            # Get sample name
            if args.num_samples > 1:
                sample_name = f"{args.output_name}_{i}"
            else:
                sample_name = args.output_name
            
            # Visualize original mesh
            img = visualize_mesh(
                vertices[i], 
                faces[i],
                title=f"Generated Mesh: {args.prompt}",
                save_path=os.path.join(args.output_dir, f"{sample_name}_original.png")
            )
            
            # Save original mesh
            save_mesh_as_obj(
                vertices[i],
                faces[i],
                os.path.join(args.output_dir, f"{sample_name}_original.obj")
            )
            
            print(f"Saved original mesh as {os.path.join(args.output_dir, f'{sample_name}_original.obj')}")
            
            # Apply mesh refinement if requested
            if args.generate_mesh:
                print(f"Applying mesh refinement to smooth and improve quality...")
                
                try:
                    # Create a trimesh object from the generated mesh
                    original_mesh = trimesh.Trimesh(vertices=vertices[i], faces=faces[i])
                    
                    # Apply smoothing to the mesh
                    from utils.mesh_refinement import smooth_mesh, fill_holes, simplify_mesh
                    
                    refined_mesh = smooth_mesh(
                        original_mesh, 
                        iterations=args.smooth_iterations, 
                        lambda_value=args.smooth_lambda
                    )
                    
                    # Fill holes if present
                    refined_mesh = fill_holes(refined_mesh, hole_size=args.fill_hole_size)
                    
                    # Visualize refined mesh
                    img = visualize_mesh(
                        refined_mesh.vertices, 
                        refined_mesh.faces,
                        title=f"Refined Mesh: {args.prompt}",
                        save_path=os.path.join(args.output_dir, f"{sample_name}_refined.png")
                    )
                    
                    # Save refined mesh
                    save_mesh_as_obj(
                        refined_mesh.vertices,
                        refined_mesh.faces,
                        os.path.join(args.output_dir, f"{sample_name}_refined.obj")
                    )
                    
                    print(f"Saved refined mesh as {os.path.join(args.output_dir, f'{sample_name}_refined.obj')}")
                except Exception as e:
                    print(f"Mesh refinement failed: {e}")
            
    # Calculate Chamfer distance if requested
    if args.calculate_chamfer:
        print("\nCalculating Chamfer distance against reference models...")
        try:
                # Find reference models
            if args.category is not None:
                category = args.category.lower()
                reference_patterns = [
                    os.path.join(args.reference_dir, category, "*.off"),
                    os.path.join(args.reference_dir, category, "*.ply"),
                    os.path.join(args.reference_dir, category, "*.obj")
                ]
            else:
                # Try to infer category from prompt
                category = None
                if not args.unconditional:
                    # List of ModelNet10 categories
                    modelnet_categories = ["bathtub", "bed", "chair", "desk", "dresser", "monitor", "night_stand", "sofa", "table", "toilet"]
                    prompt_lower = args.prompt.lower()
                    
                    # Check if any category appears in the prompt
                    for cat in modelnet_categories:
                        if cat in prompt_lower:
                            category = cat
                            break
                    
                    if category:
                        print(f"Inferred category '{category}' from prompt for evaluation")
                        reference_patterns = [
                            os.path.join(args.reference_dir, category, "*.off"),
                            os.path.join(args.reference_dir, category, "*.ply"),
                            os.path.join(args.reference_dir, category, "*.obj")
                        ]
                    else:
                        print("Could not infer category from prompt, comparing against all reference models")
                        reference_patterns = [
                            os.path.join(args.reference_dir, "*", "*.off"),
                            os.path.join(args.reference_dir, "*", "*.ply"),
                            os.path.join(args.reference_dir, "*", "*.obj")
                        ]
                else:
                    reference_patterns = [
                        os.path.join(args.reference_dir, "*", "*.off"),
                        os.path.join(args.reference_dir, "*", "*.ply"),
                        os.path.join(args.reference_dir, "*", "*.obj")
                    ]
                
            # Collect all matching files from all patterns
            reference_files = []
            for pattern in reference_patterns:
                reference_files.extend(glob.glob(pattern))
            
            if len(reference_files) == 0:
                print(f"Warning: No reference models found matching the patterns for category '{category}'")
            else:
                print(f"Found {len(reference_files)} reference models for comparison")
                
                # Load and process reference models
                reference_clouds = []
                target_point_count = min(2048, args.resolution if args.resolution else 2048)  # Match point count to generated models
                print(f"Loading reference models with {target_point_count} points each...")
                
                # Process with batching for memory efficiency
                batch_size = 5  # Process references in batches
                for batch_idx in range(0, min(len(reference_files), 20), batch_size):  # Limit to 20 references max
                    batch_files = reference_files[batch_idx:batch_idx + batch_size]
                    for ref_file in batch_files:
                        try:
                            # Get file format
                            ref_format = os.path.splitext(ref_file)[1].lower()
                            
                            # Load mesh based on file format
                            if ref_format in ['.ply', '.obj']:
                                ref_data = trimesh.load(ref_file)
                            elif ref_format == '.off':
                                # ModelNet10 uses .off files - handle them specifically
                                try:
                                    # First try with trimesh loader
                                    ref_data = trimesh.load(ref_file)
                                except Exception:
                                    # If trimesh fails, use a custom OFF file reader
                                    ref_data = load_off_file(ref_file)
                            else:
                                print(f"Unsupported file format: {ref_format} for file {ref_file}")
                                continue
                                
                            # Extract points based on the loaded data type
                            if isinstance(ref_data, trimesh.Trimesh):
                                # Sample points uniformly from mesh surface using Poisson disk sampling
                                ref_points = ref_data.sample(target_point_count)
                            elif hasattr(ref_data, 'vertices'):
                                ref_points = np.array(ref_data.vertices)
                                if len(ref_points) > target_point_count:
                                    # Use farthest point sampling for better coverage (fall back to random)
                                    indices = np.random.choice(len(ref_points), target_point_count, replace=False)
                                    ref_points = ref_points[indices]
                            else:
                                print(f"Could not extract points from {ref_file}")
                                continue
                                
                            # Normalize point cloud to unit cube for consistent distance measures
                            if ref_points is not None and len(ref_points) > 0:
                                # Center the point cloud
                                centroid = np.mean(ref_points, axis=0)
                                ref_points = ref_points - centroid
                                
                                # Scale to unit cube
                                scale = np.max(np.abs(ref_points)) * 1.05  # Add 5% margin
                                ref_points = ref_points / scale
                                
                                reference_clouds.append(ref_points)
                        except Exception as e:
                            print(f"Error loading reference model {ref_file}: {e}")
                            continue
                
                print(f"Successfully loaded {len(reference_clouds)} reference models")
                
                if len(reference_clouds) == 0:
                    print("No valid reference models could be loaded")
                else:
                    # Calculate Chamfer distances for each generated point cloud
                    all_metrics = {}
                    
                    for i in range(len(point_clouds)):
                        # Get sample name
                        if args.num_samples > 1:
                            sample_name = f"{args.output_name}_{i}"
                        else:
                            sample_name = args.output_name
                            
                        # Normalize point cloud similar to references
                        points = point_clouds[i]
                        centroid = np.mean(points, axis=0)
                        points_centered = points - centroid
                        scale = np.max(np.abs(points_centered)) * 1.05  # Add 5% margin
                        points_normalized = points_centered / scale
                        
                        # Convert to tensor - use CUDA only if available
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        pc_tensor = torch.tensor(points_normalized).float().to(device)
                        
                        # Process in batches to avoid memory issues
                        batch_size = 5  # Process references in batches
                        chamfer_distances = []
                        
                        for batch_idx in range(0, len(reference_clouds), batch_size):
                            batch_refs = reference_clouds[batch_idx:batch_idx + batch_size]
                            
                            # Create batch tensor of reference clouds
                            ref_batch = np.stack(batch_refs, axis=0)
                            ref_tensor = torch.tensor(ref_batch).float().to(device)
                            pc_tensor_expanded = pc_tensor.unsqueeze(0).expand(len(batch_refs), -1, -1)
                            
                            try:
                                # Calculate bidirectional Chamfer distance
                                dist_batch = chamfer_distance(pc_tensor_expanded, ref_tensor)
                                # Add batch results to overall list
                                if isinstance(dist_batch, torch.Tensor) and dist_batch.ndim > 0:
                                    chamfer_distances.extend(dist_batch.detach().cpu().numpy().tolist())
                                else:
                                    chamfer_distances.append(dist_batch.item() if isinstance(dist_batch, torch.Tensor) else dist_batch)
                            except Exception as e:
                                print(f"Error calculating Chamfer distance: {e}")
                                
                        # Free up memory
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        if len(chamfer_distances) > 0:
                            # Basic statistics
                            min_dist = min(chamfer_distances)
                            avg_dist = sum(chamfer_distances) / len(chamfer_distances)
                            max_dist = max(chamfer_distances)
                            median_dist = sorted(chamfer_distances)[len(chamfer_distances) // 2]
                            std_dev = np.std(chamfer_distances)
                            
                            # Calculate percentiles for distribution analysis
                            p25 = np.percentile(chamfer_distances, 25)
                            p75 = np.percentile(chamfer_distances, 75)
                            iqr = p75 - p25  # Interquartile range
                            
                            # Calculate additional metrics if we have enough samples
                            if len(chamfer_distances) > 5:
                                # Confidence interval (95%)
                                ci_low, ci_high = scipy.stats.t.interval(
                                    0.95, len(chamfer_distances)-1, loc=avg_dist,
                                    scale=scipy.stats.sem(chamfer_distances)
                                )
                            else:
                                ci_low, ci_high = None, None
                            
                            # Calculate other metrics
                            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                            
                            # Calculate F-scores at different thresholds
                            try:
                                # Prepare tensors for F-score calculation
                                pc_tensor_for_f = torch.tensor(points_normalized).float().unsqueeze(0).to(device)
                                
                                # Calculate F-scores for different reference models and take the average
                                f_scores_05 = []
                                f_scores_10 = []
                                
                                # Process in smaller batches to avoid memory issues
                                f_score_batch_size = 3
                                for f_batch_idx in range(0, min(len(reference_clouds), 10), f_score_batch_size):
                                    f_batch_refs = reference_clouds[f_batch_idx:f_batch_idx + f_score_batch_size]
                                    f_ref_batch = np.stack(f_batch_refs, axis=0)
                                    f_ref_tensor = torch.tensor(f_ref_batch).float().to(device)
                                    f_pc_expanded = pc_tensor_for_f.expand(len(f_batch_refs), -1, -1)
                                    
                                    # Calculate F-scores using the imported function
                                    try:
                                        # Threshold 0.05 (5% of unit cube)
                                        fs_05 = f_score(f_pc_expanded, f_ref_tensor, threshold=0.05)
                                        if isinstance(fs_05, torch.Tensor):
                                            fs_05 = fs_05.detach().cpu().item()
                                        f_scores_05.append(fs_05)
                                        
                                        # Threshold 0.10 (10% of unit cube)
                                        fs_10 = f_score(f_pc_expanded, f_ref_tensor, threshold=0.10)
                                        if isinstance(fs_10, torch.Tensor):
                                            fs_10 = fs_10.detach().cpu().item()
                                        f_scores_10.append(fs_10)
                                    except Exception as e:
                                        print(f"Error calculating F-score: {e}")
                                
                                # Average F-scores
                                f_score_05 = sum(f_scores_05) / len(f_scores_05) if f_scores_05 else 0.0
                                f_score_10 = sum(f_scores_10) / len(f_scores_10) if f_scores_10 else 0.0
                                
                                # Add to metrics dictionary
                                all_metrics[sample_name] = {
                                    "chamfer_min": min_dist,
                                    "chamfer_avg": avg_dist,
                                    "chamfer_max": max_dist,
                                    "chamfer_median": median_dist,
                                    "chamfer_std": std_dev,
                                    "f_score_05": f_score_05,
                                    "f_score_10": f_score_10
                                }
                                
                                if ci_low is not None and ci_high is not None:
                                    all_metrics[sample_name]["chamfer_ci_low"] = ci_low
                                    all_metrics[sample_name]["chamfer_ci_high"] = ci_high
                            except Exception as e:
                                print(f"Error during F-score calculation: {e}")
                                # Initialize metrics dictionary without F-scores
                                all_metrics[sample_name] = {
                                    "chamfer_min": min_dist,
                                    "chamfer_avg": avg_dist,
                                    "chamfer_max": max_dist,
                                    "chamfer_median": median_dist,
                                    "chamfer_std": std_dev
                                }
                            
                            # Add generation parameters
                            all_metrics[sample_name]["generation_params"] = {
                                "temperature": args.temperature,
                                "guidance_scale": args.guidance_scale,
                                "inference_steps": args.num_inference_steps,
                                "resolution": args.resolution or 2048,
                                "unconditional": args.unconditional
                            }
                            
                    # Export metrics to JSON if requested
                    if args.export_metrics and all_metrics:
                        metrics_file = os.path.join(args.output_dir, "chamfer_metrics.json")
                        with open(metrics_file, 'w') as f:
                            json.dump(all_metrics, f, indent=2)
                        print(f"\nMetrics saved to {metrics_file}")
                    
                    # Generate research evaluation report if requested
                    if args.evaluation_report and all_metrics:
                        report_file = os.path.join(args.output_dir, "research_evaluation.txt")
                        generate_research_report(all_metrics, args, report_file)
                        print(f"\nResearch evaluation report saved to {report_file}")
                            
        except Exception as e:
            print(f"Error during Chamfer distance calculation: {e}")
            
    print("\nGeneration completed successfully!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate 3D models from text')
    # Basic generation parameters
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--prompt', type=str, required=True,
                        help='Text prompt for 3D generation')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Directory to save outputs')
    parser.add_argument('--output_name', type=str, default=None,
                        help='Base name for output files (defaults to prompt text if not provided)')
    parser.add_argument('--num_samples', type=int, default=1,
                        help='Number of samples to generate')
    
    # Mesh refinement parameters
    parser.add_argument('--generate_mesh', action='store_true',
                        help='Generate mesh from point cloud')
    parser.add_argument('--refinement_method', type=str, choices=['alpha', 'poisson', 'none'], 
                        default='poisson', help='Mesh refinement method')
    parser.add_argument('--poisson_depth', type=int, default=8,
                        help='Depth parameter for Poisson reconstruction')
    parser.add_argument('--alpha_value', type=float, default=0.5,
                        help='Alpha value for alpha shape reconstruction')
    parser.add_argument('--simplify', action='store_true',
                        help='Apply mesh simplification')
    parser.add_argument('--simplify_fraction', type=float, default=0.5,
                        help='Target fraction of faces to keep after simplification')
    parser.add_argument('--target_faces', type=int, default=5000,
                        help='Target number of faces for simplification')
    parser.add_argument('--smooth', action='store_true',
                        help='Apply Laplacian smoothing')
    parser.add_argument('--smooth_iterations', type=int, default=5,
                        help='Number of smoothing iterations')
    parser.add_argument('--smooth_lambda', type=float, default=0.5,
                        help='Lambda parameter for Laplacian smoothing')
    parser.add_argument('--fill_holes', action='store_true',
                        help='Fill holes in mesh')
    parser.add_argument('--fill_hole_size', type=float, default=0.1,
                        help='Maximum size of holes to fill')
    
    # Evaluation parameters
    parser.add_argument('--calculate_chamfer', action='store_true', help='Calculate Chamfer distance to reference models')
    parser.add_argument('--reference_dir', type=str, default='reference_models', help='Directory containing reference models')
    parser.add_argument('--category', type=str, default='', help='Category for reference models (e.g., chair, table)')
    parser.add_argument('--export_metrics', action='store_true', help='Export metrics to JSON file')
    parser.add_argument('--evaluation_report', action='store_true', help='Generate research-quality evaluation report')
    
    # Visualization parameters
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize generated 3D model')
    
    # Generation parameters
    parser.add_argument('--quantize', action='store_true',
                        help='Apply quantization for faster inference')
    parser.add_argument('--progressive', action='store_true',
                        help='Use progressive generation for better quality')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature (lower = more deterministic)')
    parser.add_argument('--guidance_scale', type=float, default=3.0,
                        help='Text guidance scale (higher = more faithful to text)')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                        help='Number of refinement steps during generation')
    parser.add_argument('--unconditional', action='store_true',
                        help='Generate shapes without text conditioning')
    parser.add_argument('--resolution', type=int, default=None,
                        help='Target resolution for point cloud generation')
    parser.add_argument('--max_hole_size', type=float, default=0.05,
                        help='Maximum size of holes to fill')
    
    args = parser.parse_args()
    main(args)
