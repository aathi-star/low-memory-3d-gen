"""
Metrics for evaluating 3D model quality

This module implements common metrics for evaluating the quality of generated 3D models,
including Chamfer distance for point cloud comparison.
"""

import math
import torch
import numpy as np
import scipy.spatial as spatial
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
from functools import partial

def chamfer_distance(x, y):
    """
    Calculate bidirectional Chamfer distance between two point clouds.
    
    Args:
        x: First point cloud tensor of shape (batch_size, num_points_x, 3)
        y: Second point cloud tensor of shape (batch_size, num_points_y, 3)
    
    Returns:
        Bidirectional Chamfer distance (scalar)
    """
    # Input checking
    if not torch.is_tensor(x):
        x = torch.tensor(x).float()
    if not torch.is_tensor(y):
        y = torch.tensor(y).float()
        
    # Make sure inputs are 3D with batch dimension
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if y.dim() == 2:
        y = y.unsqueeze(0)
        
    # Move to same device if needed
    if x.device != y.device:
        y = y.to(x.device)
    
    # Validate shapes
    batch_size = x.shape[0]
    if y.shape[0] != batch_size:
        raise ValueError(f"Batch sizes don't match: {x.shape[0]} vs {y.shape[0]}")
    
    # Extract dimensions    
    num_points_x = x.shape[1]
    num_points_y = y.shape[1]
    
    # Reshape tensors for pairwise distance calculations
    x_expanded = x.unsqueeze(2)  # (batch_size, num_points_x, 1, 3)
    y_expanded = y.unsqueeze(1)  # (batch_size, 1, num_points_y, 3)
    
    # Compute pairwise distances - ||x_i - y_j||^2
    dist = torch.sum((x_expanded - y_expanded) ** 2, dim=3)  # (batch_size, num_points_x, num_points_y)
    
    # Compute minimum distance from x to y
    x_to_y = torch.min(dist, dim=2)[0]  # (batch_size, num_points_x)
    
    # Compute minimum distance from y to x
    y_to_x = torch.min(dist, dim=1)[0]  # (batch_size, num_points_y)
    
    # Compute mean over all points
    chamfer_x_to_y = torch.mean(x_to_y, dim=1)  # (batch_size,)
    chamfer_y_to_x = torch.mean(y_to_x, dim=1)  # (batch_size,)
    
    # Bidirectional chamfer distance
    chamfer_dist = chamfer_x_to_y + chamfer_y_to_x  # (batch_size,)
    
    # Return mean over batch
    return torch.mean(chamfer_dist)


def hausdorff_distance(x, y, bidirectional=True):
    """
    Calculate Hausdorff distance between two point clouds.
    
    Args:
        x: First point cloud tensor of shape (batch_size, num_points_x, 3)
        y: Second point cloud tensor of shape (batch_size, num_points_y, 3)
        bidirectional: If True, compute bidirectional Hausdorff distance
        
    Returns:
        Hausdorff distance (scalar)
    """
    # Input checking
    if not torch.is_tensor(x):
        x = torch.tensor(x).float()
    if not torch.is_tensor(y):
        y = torch.tensor(y).float()
        
    # Make sure inputs are 3D with batch dimension
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if y.dim() == 2:
        y = y.unsqueeze(0)
        
    # Move to same device if needed
    if x.device != y.device:
        y = y.to(x.device)
    
    # Reshape tensors for pairwise distance calculations
    x_expanded = x.unsqueeze(2)  # (batch_size, num_points_x, 1, 3)
    y_expanded = y.unsqueeze(1)  # (batch_size, 1, num_points_y, 3)
    
    # Compute pairwise distances - ||x_i - y_j||^2
    dist = torch.sum((x_expanded - y_expanded) ** 2, dim=3)  # (batch_size, num_points_x, num_points_y)
    
    # Forward Hausdorff: max of min distances from x to y
    x_to_y = torch.min(dist, dim=2)[0]  # (batch_size, num_points_x)
    forward_hausdorff = torch.max(x_to_y, dim=1)[0]  # (batch_size,)
    
    if not bidirectional:
        return torch.mean(forward_hausdorff)
    
    # Backward Hausdorff: max of min distances from y to x
    y_to_x = torch.min(dist, dim=1)[0]  # (batch_size, num_points_y)
    backward_hausdorff = torch.max(y_to_x, dim=1)[0]  # (batch_size,)
    
    # Bidirectional Hausdorff
    hausdorff_dist = torch.max(forward_hausdorff, backward_hausdorff)  # (batch_size,)
    
    # Return mean over batch
    return torch.mean(hausdorff_dist)


def f_score(x, y, threshold=0.01):
    """
    Calculate F-score between two point clouds at the specified threshold.
    
    Args:
        x: First point cloud tensor of shape (batch_size, num_points_x, 3)
        y: Second point cloud tensor of shape (batch_size, num_points_y, 3)
        threshold: Distance threshold for determining point matches
        
    Returns:
        F-score value (scalar)
    """
    # Input checking
    if not torch.is_tensor(x):
        x = torch.tensor(x).float()
    if not torch.is_tensor(y):
        y = torch.tensor(y).float()
        
    # Make sure inputs are 3D with batch dimension
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if y.dim() == 2:
        y = y.unsqueeze(0)
        
    # Move to same device if needed
    if x.device != y.device:
        y = y.to(x.device)
    
    # Reshape tensors for pairwise distance calculations
    x_expanded = x.unsqueeze(2)  # (batch_size, num_points_x, 1, 3)
    y_expanded = y.unsqueeze(1)  # (batch_size, 1, num_points_y, 3)
    
    # Compute pairwise distances - ||x_i - y_j||^2
    dist = torch.sum((x_expanded - y_expanded) ** 2, dim=3)  # (batch_size, num_points_x, num_points_y)
    
    # Compute minimum distance from x to y
    x_to_y = torch.min(dist, dim=2)[0]  # (batch_size, num_points_x)
    # Compute minimum distance from y to x
    y_to_x = torch.min(dist, dim=1)[0]  # (batch_size, num_points_y)
    
    # Calculate precision and recall
    precision = torch.mean((x_to_y < threshold**2).float(), dim=1)  # (batch_size,)
    recall = torch.mean((y_to_x < threshold**2).float(), dim=1)  # (batch_size,)
    
    # Calculate F-score
    f_score = 2 * precision * recall / (precision + recall + 1e-8)  # (batch_size,)
    
    # Return mean over batch
    return torch.mean(f_score)


def compute_f_score(predicted_points, reference_points, threshold=0.01):
    """
    Compute F-score between two point clouds
    Args:
        predicted_points: (N, 3) tensor of predicted points
        reference_points: (M, 3) tensor of reference points
        threshold: Distance threshold for precision/recall calculation
    Returns:
        f_score: F-score value
    """
    # Calculate pairwise distances
    dist1, _ = chamfer_distance_one_sided(predicted_points, reference_points)
    dist2, _ = chamfer_distance_one_sided(reference_points, predicted_points)
    
    # Calculate precision and recall
    precision = torch.mean((dist2 < threshold).float())
    recall = torch.mean((dist1 < threshold).float())
    
    # Calculate F-score
    if precision + recall > 0:
        f_score = 2 * precision * recall / (precision + recall)
    else:
        f_score = torch.tensor(0.0, device=predicted_points.device)
    
    return f_score.item()


def evaluate_mesh_integrity(mesh):
    """
    Evaluate the structural integrity of a mesh to determine if it's a valid, closed 3D object.
    
    Args:
        mesh: A trimesh.Trimesh object
        
    Returns:
        dict: Dictionary containing multiple mesh integrity metrics
    """
    results = {}
    
    # Check if mesh exists and has faces
    if mesh is None or len(mesh.faces) == 0:
        results['is_watertight'] = False
        results['is_manifold'] = False
        results['euler_number'] = 0
        results['genus'] = 0
        results['volume'] = 0
        results['surface_area'] = 0
        results['compactness'] = 0
        results['connected_components'] = 0
        results['structural_integrity_score'] = 0
        results['is_convex'] = False
        results['convexity_ratio'] = 0
        return results
    
    # Calculate basic mesh properties
    results['is_watertight'] = mesh.is_watertight
    results['is_manifold'] = mesh.is_manifold
    results['euler_number'] = mesh.euler_number
    
    # Calculate topological properties
    # Genus = (2 - Euler characteristic) / 2
    results['genus'] = int((2 - mesh.euler_number) / 2) if mesh.is_watertight else -1
    
    # Volume and surface area (only meaningful for watertight meshes)
    try:
        results['volume'] = mesh.volume if mesh.is_watertight else 0
        results['surface_area'] = mesh.area
        
        # Compactness ratio: ratio of volume to surface area (normalized)
        # Higher values indicate more compact (sphere-like) shapes
        if mesh.is_watertight and mesh.volume > 0 and mesh.area > 0:
            # Normalized compactness: ratio of the mesh's compactness to a sphere's compactness
            # For a sphere: V²/(S³/36π) = 1
            # For any other shape: V²/(S³/36π) < 1
            compactness = (36 * np.pi * mesh.volume**2) / (mesh.area**3)
            results['compactness'] = min(compactness, 1.0)  # Clamp to 1.0
        else:
            results['compactness'] = 0
    except:
        results['volume'] = 0
        results['surface_area'] = 0
        results['compactness'] = 0
    
    # Count connected components (single component is better)
    try:
        components = mesh.split(only_watertight=False)
        results['connected_components'] = len(components)
    except:
        results['connected_components'] = 0
    
    # Check convexity (fully convex shapes are simpler)
    try:
        convex_hull = mesh.convex_hull
        results['is_convex'] = np.isclose(mesh.volume, convex_hull.volume, rtol=0.05) if mesh.is_watertight else False
        results['convexity_ratio'] = mesh.volume / convex_hull.volume if mesh.is_watertight and convex_hull.volume > 0 else 0
    except:
        results['is_convex'] = False
        results['convexity_ratio'] = 0
    
    # Calculate a structural integrity score (0.0 to 1.0)
    # This considers: watertightness, manifoldness, connectedness, and compactness
    integrity_score = 0.0
    if mesh.is_watertight:
        integrity_score += 0.4  # 40% for being watertight
    if mesh.is_manifold:
        integrity_score += 0.2  # 20% for being manifold
    if results['connected_components'] == 1:
        integrity_score += 0.2  # 20% for being a single connected component
    integrity_score += 0.2 * results['compactness']  # Up to 20% for compactness
    
    results['structural_integrity_score'] = integrity_score
    
    return results


def compute_point_cloud_coherence(points):
    """
    Compute metrics that measure how coherent (not dispersed) a point cloud is.
    
    Args:
        points: (N, 3) tensor or array of point coordinates
        
    Returns:
        dict: Dictionary containing point cloud coherence metrics
    """
    import numpy as np
    
    # Convert to numpy if tensor
    if hasattr(points, 'cpu') and hasattr(points, 'numpy'):
        points = points.cpu().numpy()
    
    results = {}
    
    # Empty point cloud check
    if len(points) < 4:
        results['spatial_uniformity'] = 0.0
        results['local_density_variation'] = 1.0  # Maximum variation (bad)
        results['nearest_neighbor_consistency'] = 0.0
        results['coherence_score'] = 0.0
        return results
        
    # Center and normalize the point cloud
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    scale = np.max(np.linalg.norm(points_centered, axis=1))
    points_normalized = points_centered / (scale if scale > 0 else 1.0)
    
    # Compute nearest neighbor distances (k=6 neighbors)
    try:
        kdtree = spatial.KDTree(points_normalized)
        k = min(7, len(points))  # k+1 because the point itself is included
        distances, _ = kdtree.query(points_normalized, k=k)
        nn_distances = distances[:, 1:]  # Skip the first column (distance to self = 0)
    except:
        # Fallback if KDTree fails
        nn_distances = np.zeros((len(points), min(6, len(points)-1)))
    
    # Spatial uniformity: variance of distances to centroid
    # Lower values indicate more uniform distribution
    distances_to_center = np.linalg.norm(points_normalized, axis=1)
    uniformity = 1.0 - min(np.std(distances_to_center), 0.5) / 0.5
    results['spatial_uniformity'] = max(0.0, min(1.0, uniformity))
    
    # Local density variation: variance of nearest neighbor distances
    # Lower values indicate more consistent local density
    if nn_distances.size > 0:
        mean_nn_distances = np.mean(nn_distances, axis=1)
        density_variation = np.std(mean_nn_distances) / np.mean(mean_nn_distances) if np.mean(mean_nn_distances) > 0 else 1.0
        results['local_density_variation'] = max(0.0, min(1.0, 1.0 - density_variation))
    else:
        results['local_density_variation'] = 0.0
    
    # Nearest neighbor consistency: how consistent the distance is to k nearest neighbors
    # Higher values indicate more coherent structure
    if nn_distances.size > 0:
        std_per_point = np.std(nn_distances, axis=1)
        consistency = 1.0 - np.mean(std_per_point / np.mean(nn_distances)) if np.mean(nn_distances) > 0 else 0.0
        results['nearest_neighbor_consistency'] = max(0.0, min(1.0, consistency))
    else:
        results['nearest_neighbor_consistency'] = 0.0
    
    # Overall coherence score (combination of above metrics)
    results['coherence_score'] = (results['spatial_uniformity'] * 0.3 + 
                                 results['local_density_variation'] * 0.3 + 
                                 results['nearest_neighbor_consistency'] * 0.4)
    
    return results


def compute_convergence_metrics(points, mesh=None):
    """
    Compute metrics that indicate how well-converged a 3D model is,
    useful for justifying that a model has converged even with fewer epochs.
    
    Args:
        points: (N, 3) tensor or array of point cloud points
        mesh: A trimesh.Trimesh object (optional)
        
    Returns:
        dict: Dictionary containing convergence quality metrics
    """
    import numpy as np
    
    # Convert to numpy if tensor
    if hasattr(points, 'cpu') and hasattr(points, 'numpy'):
        points = points.cpu().numpy()
    
    results = {}
    
    # Get point cloud coherence metrics
    coherence_metrics = compute_point_cloud_coherence(points)
    
    # Get mesh integrity metrics if mesh is provided
    if mesh is not None:
        mesh_metrics = evaluate_mesh_integrity(mesh)
    else:
        mesh_metrics = {'structural_integrity_score': 0.0, 'is_watertight': False}
    
    # Compute point cloud statistical properties that indicate convergence
    
    # 1. Normal distribution test on each coordinate
    # Well-converged models often have more normally distributed coordinates
    # in at least one or two dimensions (not all three, as shapes aren't spheres)
    normality_scores = []
    for dim in range(3):
        if len(points) > 8:  # Minimum sample size for normality test
            _, p_value = stats.normaltest(points[:, dim])
            # Convert p-value to a score (higher p-value = more normal = better)
            # We use a threshold because perfect normality isn't expected
            norm_score = min(p_value / 0.1, 1.0)
            normality_scores.append(norm_score)
        else:
            normality_scores.append(0.0)
    
    # Use the best dimension's normality (we expect at least one dimension to be somewhat normal)
    results['coordinate_normality'] = max(normality_scores) if normality_scores else 0.0
    
    # 2. Spatial entropy: measure of information content in the point distribution
    # Lower entropy generally indicates more structure and better convergence
    try:
        # Use a simple binning approach for entropy calculation
        bins = 10
        H, _ = np.histogramdd(points, bins=[bins, bins, bins])
        H = H / len(points)  # Normalize
        H = H.flatten()
        # Remove zeros (log(0) is undefined)
        H = H[H > 0]
        entropy = -np.sum(H * np.log2(H))
        # Normalize by maximum possible entropy
        max_entropy = np.log2(bins**3)
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 1.0
        # Convert to a score (lower entropy = higher score)
        results['spatial_entropy_score'] = 1.0 - min(norm_entropy, 1.0)
    except:
        results['spatial_entropy_score'] = 0.0
    
    # 3. Shape consistency: consistency of the object's bounding box aspect ratio
    # Well-converged models maintain consistent proportions
    try:
        # Calculate axis-aligned bounding box extents
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        extents = max_coords - min_coords
        
        # Check for zero extents
        if np.any(extents < 1e-6):
            results['shape_consistency'] = 0.0
        else:
            # Sort extents to get consistent dimension ordering
            extents = np.sort(extents)
            
            # Calculate aspect ratios between dimensions
            aspect_1 = extents[0] / extents[2] if extents[2] > 0 else 0
            aspect_2 = extents[1] / extents[2] if extents[2] > 0 else 0
            
            # A well-formed object usually has reasonable aspect ratios
            # (not too thin in any dimension)
            aspect_score = (min(aspect_1 / 0.25, 1.0) + min(aspect_2 / 0.25, 1.0)) / 2
            results['shape_consistency'] = aspect_score
    except:
        results['shape_consistency'] = 0.0
    
    # 4. Overall convergence score (combination of metrics)
    # This weights mesh properties heavily because they're critical for convergence quality
    coherence_contribution = coherence_metrics['coherence_score'] * 0.4
    structure_contribution = 0.0
    
    if mesh is not None and mesh_metrics['is_watertight']:
        # If we have a watertight mesh, give it more weight
        structure_contribution = mesh_metrics['structural_integrity_score'] * 0.4
    else:
        # Otherwise rely more on point cloud metrics
        structure_contribution = results['spatial_entropy_score'] * 0.2 + results['shape_consistency'] * 0.2
    
    normality_contribution = results['coordinate_normality'] * 0.2
    
    results['convergence_quality_score'] = coherence_contribution + structure_contribution + normality_contribution
    
    # Add the coherence metrics for completeness
    for key, value in coherence_metrics.items():
        results[f'point_cloud_{key}'] = value
    
    # Add the mesh metrics if available
    if mesh is not None:
        for key, value in mesh_metrics.items():
            results[f'mesh_{key}'] = value
    
    return results
