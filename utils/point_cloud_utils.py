import torch
import numpy as np

def remove_dispersed_points(point_cloud, std_multiplier=2.0):
    """
    Remove or fix dispersed points in a point cloud
    
    Args:
        point_cloud: Tensor of shape [N, 3] or [B, N, 3]
        std_multiplier: Points farther than mean + std_multiplier*std will be adjusted
        
    Returns:
        Cleaned point cloud with same shape as input
    """
    is_batched = len(point_cloud.shape) == 3
    device = point_cloud.device
    
    if not is_batched:
        # Add batch dimension for processing
        point_cloud = point_cloud.unsqueeze(0)
    
    batch_size = point_cloud.shape[0]
    result = point_cloud.clone()
    
    for i in range(batch_size):
        # Find the centroid of the point cloud
        centroid = torch.mean(point_cloud[i], dim=0)
        
        # Calculate distance from each point to centroid
        dists = torch.norm(point_cloud[i] - centroid, dim=1)
        
        # Find outliers (points far from center)
        threshold = torch.mean(dists) + std_multiplier * torch.std(dists)
        outlier_mask = dists > threshold
        
        # Replace outliers with points closer to the centroid
        if outlier_mask.sum() > 0:
            # Move outliers closer to the center
            direction = point_cloud[i][outlier_mask] - centroid
            direction = direction / torch.norm(direction, dim=1, keepdim=True)
            result[i][outlier_mask] = centroid + direction * threshold * 0.8
    
    if not is_batched:
        # Remove batch dimension
        result = result.squeeze(0)
    
    return result

def enforce_category_geometry(point_cloud, category, modelnet_categories):
    """
    Apply category-specific geometric constraints to ensure ModelNet10 resemblance
    
    Args:
        point_cloud: Tensor of shape [N, 3] or [B, N, 3]
        category: Category name or index
        modelnet_categories: List of category names
        
    Returns:
        Point cloud with category-specific geometric constraints applied
    """
    is_batched = len(point_cloud.shape) == 3
    device = point_cloud.device
    
    if not is_batched:
        # Add batch dimension for processing
        point_cloud = point_cloud.unsqueeze(0)
        category = [category]
    
    batch_size = point_cloud.shape[0]
    result = point_cloud.clone()
    
    for i in range(batch_size):
        # Get category for this point cloud
        if isinstance(category[i], int):
            cat_name = modelnet_categories[category[i]]
        else:
            cat_name = category[i]
        
        # Apply category-specific adjustments
        if cat_name == 'chair':
            # Ensure chair has clear seat and back
            seat_mask = (result[i, :, 1] > -0.1) & (result[i, :, 1] < 0.1)
            back_mask = (result[i, :, 2] > 0.3) & (result[i, :, 1] > 0.0)
            # Make seat more flat
            result[i, seat_mask, 1] = 0.0
            # Make back more vertical
            back_points = result[i, back_mask, :]
            if back_points.shape[0] > 0:
                result[i, back_mask, 2] = torch.clamp(result[i, back_mask, 2], min=0.4)
        
        elif cat_name == 'table':
            # Ensure table has flat top
            top_mask = result[i, :, 1] > 0.2
            if top_mask.sum() > 0:
                result[i, top_mask, 1] = 0.3
            
            # Ensure table has legs
            leg_points = result[i, result[i, :, 1] < -0.2, :]
            if leg_points.shape[0] < 20:
                # Add some leg points if missing
                corners = torch.tensor([
                    [0.3, -0.5, 0.3], [0.3, -0.5, -0.3], 
                    [-0.3, -0.5, 0.3], [-0.3, -0.5, -0.3]
                ], device=device)
                
                for corner in corners:
                    # Create leg points
                    for h in torch.linspace(-0.5, 0.2, 10):
                        idx = torch.randint(0, result.shape[1], (1,))
                        result[i, idx, :] = corner.clone()
                        result[i, idx, 1] = h
        
        elif cat_name == 'toilet':
            # Ensure toilet has proper bowl shape
            # Find points in the middle section
            bowl_mask = (result[i, :, 1] > -0.2) & (result[i, :, 1] < 0.1)
            bowl_points = result[i, bowl_mask, :]
            
            # Make more bowl-like by adjusting to circular pattern
            if bowl_points.shape[0] > 0:
                center = torch.mean(bowl_points[:, [0, 2]], dim=0)
                for j in range(bowl_points.shape[0]):
                    # Get vector from center to point (xz-plane)
                    vec = bowl_points[j, [0, 2]] - center
                    # Normalize to consistent radius
                    vec = vec / (torch.norm(vec) + 1e-6) * 0.3
                    # Update point
                    new_xz = center + vec
                    result[i, bowl_mask, 0][j] = new_xz[0]
                    result[i, bowl_mask, 2][j] = new_xz[1]
    
    if not is_batched:
        # Remove batch dimension
        result = result.squeeze(0)
    
    return result

def uniform_density(point_cloud, num_regions=5, density_factor=0.8):
    """
    Adjust point cloud to have more uniform density, preventing clumping
    
    Args:
        point_cloud: Tensor of shape [N, 3] or [B, N, 3]
        num_regions: Number of regions to divide the point cloud into
        density_factor: Controls how aggressively to redistribute points
        
    Returns:
        Point cloud with more uniform density
    """
    is_batched = len(point_cloud.shape) == 3
    device = point_cloud.device
    
    if not is_batched:
        # Add batch dimension for processing
        point_cloud = point_cloud.unsqueeze(0)
    
    batch_size = point_cloud.shape[0]
    result = point_cloud.clone()
    
    for i in range(batch_size):
        # Calculate point density
        x_min, x_max = torch.min(result[i, :, 0]), torch.max(result[i, :, 0])
        y_min, y_max = torch.min(result[i, :, 1]), torch.max(result[i, :, 1])
        z_min, z_max = torch.min(result[i, :, 2]), torch.max(result[i, :, 2])
        
        # Create grid for density estimation
        x_bins = torch.linspace(x_min, x_max, num_regions+1)
        y_bins = torch.linspace(y_min, y_max, num_regions+1)
        z_bins = torch.linspace(z_min, z_max, num_regions+1)
        
        # Count points in each region
        region_counts = torch.zeros((num_regions, num_regions, num_regions), device=device)
        
        for x_idx in range(num_regions):
            x_mask = (result[i, :, 0] >= x_bins[x_idx]) & (result[i, :, 0] < x_bins[x_idx+1])
            for y_idx in range(num_regions):
                y_mask = (result[i, :, 1] >= y_bins[y_idx]) & (result[i, :, 1] < y_bins[y_idx+1])
                for z_idx in range(num_regions):
                    z_mask = (result[i, :, 2] >= z_bins[z_idx]) & (result[i, :, 2] < z_bins[z_idx+1])
                    region_mask = x_mask & y_mask & z_mask
                    region_counts[x_idx, y_idx, z_idx] = region_mask.sum()
        
        # Find regions with too many points
        mean_count = torch.mean(region_counts[region_counts > 0])
        high_density_regions = (region_counts > 2 * mean_count) & (region_counts > 0)
        
        # Redistribute points from high density regions
        if high_density_regions.sum() > 0:
            high_x, high_y, high_z = torch.where(high_density_regions)
            
            for idx in range(high_x.shape[0]):
                x_idx, y_idx, z_idx = high_x[idx], high_y[idx], high_z[idx]
                
                # Find points in this high density region
                x_mask = (result[i, :, 0] >= x_bins[x_idx]) & (result[i, :, 0] < x_bins[x_idx+1])
                y_mask = (result[i, :, 1] >= y_bins[y_idx]) & (result[i, :, 1] < y_bins[y_idx+1])
                z_mask = (result[i, :, 2] >= z_bins[z_idx]) & (result[i, :, 2] < z_bins[z_idx+1])
                region_mask = x_mask & y_mask & z_mask
                
                # Randomly select points to redistribute
                points_to_move = int((region_counts[x_idx, y_idx, z_idx] - mean_count) * density_factor)
                if points_to_move > 0 and region_mask.sum() > points_to_move:
                    # Select random points from this region
                    region_indices = torch.where(region_mask)[0]
                    move_indices = region_indices[torch.randperm(region_indices.shape[0])[:points_to_move]]
                    
                    # Find low density regions
                    low_density_regions = (region_counts < 0.5 * mean_count) & (region_counts >= 0)
                    if low_density_regions.sum() > 0:
                        low_x, low_y, low_z = torch.where(low_density_regions)
                        
                        # Move points to random low density regions
                        for j, point_idx in enumerate(move_indices):
                            # Select random low density region
                            low_idx = j % low_x.shape[0]
                            lx, ly, lz = low_x[low_idx], low_y[low_idx], low_z[low_idx]
                            
                            # Generate random position in that region
                            new_x = torch.rand(1, device=device) * (x_bins[lx+1] - x_bins[lx]) + x_bins[lx]
                            new_y = torch.rand(1, device=device) * (y_bins[ly+1] - y_bins[ly]) + y_bins[ly]
                            new_z = torch.rand(1, device=device) * (z_bins[lz+1] - z_bins[lz]) + z_bins[lz]
                            
                            # Move point
                            result[i, point_idx, 0] = new_x
                            result[i, point_idx, 1] = new_y
                            result[i, point_idx, 2] = new_z
                            
                            # Update counts
                            region_counts[x_idx, y_idx, z_idx] -= 1
                            region_counts[lx, ly, lz] += 1
    
    if not is_batched:
        # Remove batch dimension
        result = result.squeeze(0)
    
    return result
