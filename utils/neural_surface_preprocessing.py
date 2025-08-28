import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
import glob

class NeuralSurfaceExtractor(nn.Module):
    """
    Neural implicit function to learn continuous surface representation from point clouds
    This is based on the Neural-Pull method for improved surface reconstruction
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        # Multi-layer perceptron for implicit function
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x):
        # x: batch of 3D points [B, N, 3]
        batch_size, num_points, _ = x.shape
        x_flat = x.reshape(-1, 3)
        signed_distance = self.mlp(x_flat).reshape(batch_size, num_points, 1)
        return signed_distance

def extract_surface_features(point_clouds, category_ids, device='cuda'):
    """
    Extract surface features from point clouds using neural surface reconstruction
    Args:
        point_clouds: Tensor of shape [B, N, 3]
        category_ids: Tensor of shape [B]
        device: Device to run computation on
    Returns:
        Surface-aware features [B, N, C]
    """
    batch_size, num_points, _ = point_clouds.shape
    
    # Create neural surface model
    surface_model = NeuralSurfaceExtractor().to(device)
    
    # Sample points near the surface
    sigma = 0.05  # Standard deviation for sampling
    num_samples = 1024  # Number of samples per point cloud
    
    # Generate random offsets
    offsets = torch.randn(batch_size, num_samples, 3, device=device) * sigma
    
    # Sample query points (randomly select surface points and add offset)
    indices = torch.randint(0, num_points, (batch_size, num_samples), device=device)
    surface_points = torch.gather(
        point_clouds, 1, 
        indices.unsqueeze(-1).expand(batch_size, num_samples, 3)
    )
    query_points = surface_points + offsets
    
    # Compute implicit function values (signed distance field)
    with torch.enable_grad():
        query_points.requires_grad_(True)
        sdf = surface_model(query_points)
        
        # Compute gradients (surface normals)
        normals = torch.zeros_like(query_points, device=device)
        for i in range(batch_size):
            grad = torch.autograd.grad(
                outputs=sdf[i].sum(),
                inputs=query_points,
                create_graph=True
            )[0][i]
            normals[i] = F.normalize(grad, dim=1)
    
    # Compute distance to surface
    distances = torch.min(
        torch.cdist(query_points, point_clouds), dim=2
    )[0].unsqueeze(-1)
    
    # Extract surface features [position, normal, distance]
    surface_features = torch.cat([query_points, normals, distances], dim=2)
    
    # Project back to original points using attention
    attention_weights = torch.softmax(
        -torch.cdist(point_clouds, query_points) / 0.1, dim=2
    )
    aggregated_features = torch.bmm(attention_weights, surface_features)
    
    return aggregated_features

def preprocess_modelnet10_with_surface_features(data_dir, output_dir):
    """
    Enhance ModelNet10 dataset with neural surface features
    """
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Find all point cloud files
    all_files = []
    categories = []
    for category in os.listdir(data_dir):
        category_dir = os.path.join(data_dir, category)
        if os.path.isdir(category_dir):
            categories.append(category)
            point_cloud_files = glob.glob(os.path.join(category_dir, '*.npy'))
            all_files.extend([(f, category) for f in point_cloud_files])
    
    # Create category to ID mapping
    category_to_id = {cat: i for i, cat in enumerate(categories)}
    print(f"Found {len(all_files)} files in {len(categories)} categories")
    
    # Process in batches
    batch_size = 8
    for i in tqdm(range(0, len(all_files), batch_size)):
        batch_files = all_files[i:i+batch_size]
        
        # Load point clouds and category IDs
        point_clouds = []
        category_ids = []
        file_paths = []
        
        for file_path, category in batch_files:
            try:
                # Load point cloud
                point_cloud = np.load(file_path).astype(np.float32)
                if point_cloud.shape[1] > 3:  # If it has normals, just use xyz
                    point_cloud = point_cloud[:, :3]
                
                # Normalize to unit cube
                center = np.mean(point_cloud, axis=0)
                point_cloud = point_cloud - center
                scale = np.max(np.abs(point_cloud))
                point_cloud = point_cloud / scale
                
                # Add to batch
                point_clouds.append(point_cloud)
                category_ids.append(category_to_id[category])
                file_paths.append(file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        if not point_clouds:
            continue
        
        # Pad or sample to fixed size
        target_points = 2048
        processed_clouds = []
        for pc in point_clouds:
            if pc.shape[0] > target_points:
                # Randomly sample points
                indices = np.random.choice(pc.shape[0], target_points, replace=False)
                pc = pc[indices]
            elif pc.shape[0] < target_points:
                # Pad by repeating points
                repeat_indices = np.random.choice(pc.shape[0], target_points - pc.shape[0], replace=True)
                pc = np.vstack([pc, pc[repeat_indices]])
            processed_clouds.append(pc)
        
        # Convert to tensors
        point_clouds_tensor = torch.tensor(np.stack(processed_clouds), device=device)
        category_ids_tensor = torch.tensor(category_ids, device=device)
        
        # Extract surface features
        surface_features = extract_surface_features(point_clouds_tensor, category_ids_tensor, device)
        
        # Save enhanced point clouds
        for j, (file_path, category) in enumerate(batch_files):
            if j >= len(surface_features):
                continue
                
            # Create output directory
            category_output_dir = os.path.join(output_dir, category)
            os.makedirs(category_output_dir, exist_ok=True)
            
            # Get filename
            filename = os.path.basename(file_path)
            output_path = os.path.join(category_output_dir, filename)
            
            # Save features
            enhanced_point_cloud = torch.cat([
                point_clouds_tensor[j],
                surface_features[j]
            ], dim=1).cpu().numpy()
            
            np.save(output_path, enhanced_point_cloud)
    
    print(f"Enhanced dataset saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhance ModelNet10 with neural surface features")
    parser.add_argument("--data_dir", type=str, required=True, help="Input directory with ModelNet10 point clouds")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for enhanced data")
    
    args = parser.parse_args()
    preprocess_modelnet10_with_surface_features(args.data_dir, args.output_dir)
