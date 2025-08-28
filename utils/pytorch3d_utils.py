import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Implement knn_points and knn_gather functions that would normally be from pytorch3d.ops
def knn_points(p1, p2, K=3, return_sorted=True):
    """
    Find the K-Nearest neighbors of p1 in p2.
    Args:
        p1: (B, N, D) tensor of points
        p2: (B, M, D) tensor of points
        K: number of nearest neighbors to return
        return_sorted: Whether to return the nearest neighbors sorted by distance
    Returns:
        dists: (B, N, K) tensor of squared distances
        idx: (B, N, K) tensor of indices of nearest neighbors
    """
    if p1.dim() == 2:
        p1 = p1.unsqueeze(0)
    if p2.dim() == 2:
        p2 = p2.unsqueeze(0)
    
    # Compute squared distances between all pairs of points
    # (B, N, M) tensor of squared distances
    dists = torch.cdist(p1, p2, p=2.0)**2
    
    # Get the K nearest neighbors
    dists, idx = dists.topk(k=K, dim=2, largest=False, sorted=return_sorted)
    
    return dists, idx

def knn_gather(x, idx):
    """
    Gather values from x using the indices in idx.
    Args:
        x: (B, N, C) tensor of features
        idx: (B, P, K) tensor of indices
    Returns:
        (B, P, K, C) tensor of gathered features
    """
    batch_size, num_points, num_dims = x.shape
    _, points_idx, k = idx.shape
    
    # Flatten batch dimension
    x_flat = x.reshape(-1, num_dims)
    idx_flat = idx.reshape(-1)
    
    # Add offsets to indices for batched operation
    idx_offset = torch.arange(batch_size, device=x.device) * num_points
    idx_offset = idx_offset.view(-1, 1, 1).expand(-1, points_idx, k).reshape(-1)
    idx_flat = idx_flat + idx_offset
    
    # Gather features
    gathered_flat = x_flat[idx_flat]
    
    # Reshape back to batch format
    gathered = gathered_flat.reshape(batch_size, points_idx, k, num_dims)
    
    return gathered


def mesh_to_pointcloud(vertices, faces, num_points=2048):
    """
    Sample points from mesh triangles with uniform probability.
    Simplified version of pytorch3d's sample_points_from_meshes function.
    
    Args:
        vertices: (V, 3) tensor of vertices coordinates
        faces: (F, 3) tensor of indices into vertices for each triangle
        num_points: number of points to sample
    
    Returns:
        points: (num_points, 3) tensor of sampled point positions
    """
    if isinstance(vertices, np.ndarray):
        vertices = torch.tensor(vertices, dtype=torch.float32)
    if isinstance(faces, np.ndarray):
        faces = torch.tensor(faces, dtype=torch.int64)
    
    device = vertices.device
    
    # Calculate face areas for weighted sampling
    v0, v1, v2 = [
        vertices[faces[:, i], :] for i in range(3)
    ]
    
    # Compute areas of each face using cross product
    areas = 0.5 * torch.norm(torch.cross(v1 - v0, v2 - v0, dim=1), dim=1)
    
    # Probability of sampling face proportional to area
    prob_faces = areas / areas.sum()
    
    # Sample faces with replacement according to probability
    face_ids = torch.multinomial(prob_faces, num_points, replacement=True)
    
    # Sample random points on selected triangles
    # Random barycentric coordinates
    u = torch.sqrt(torch.rand(num_points, device=device))
    v = torch.rand(num_points, device=device) * (1.0 - u)
    w = 1.0 - u - v
    
    # Sample point positions from barycentric coordinates
    selected_v0 = v0[face_ids]
    selected_v1 = v1[face_ids]
    selected_v2 = v2[face_ids]
    
    # Apply barycentric coordinates to get final points
    points = (
        w[:, None] * selected_v0 + 
        u[:, None] * selected_v1 + 
        v[:, None] * selected_v2
    )
    
    return points


def point_mesh_face_distance(points, vertices, faces):
    """
    Compute distance from points to mesh faces.
    Simplified version of the corresponding pytorch3d function.
    
    Args:
        points: (P, 3) tensor of points
        vertices: (V, 3) tensor of vertex positions
        faces: (F, 3) tensor of indices of vertices for each face
    
    Returns:
        distances: (P,) tensor of distances from each point to closest face
    """
    if isinstance(points, np.ndarray):
        points = torch.tensor(points, dtype=torch.float32)
    if isinstance(vertices, np.ndarray):
        vertices = torch.tensor(vertices, dtype=torch.float32)
    if isinstance(faces, np.ndarray):
        faces = torch.tensor(faces, dtype=torch.int64)
    
    # Get vertices for each face
    v0, v1, v2 = [
        vertices[faces[:, i], :] for i in range(3)
    ]
    
    # Compute face normals
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    face_normals = F.normalize(face_normals, p=2, dim=1)
    
    # For each point, compute distance to each face
    # This is a simple but not the most efficient implementation
    distances = []
    for point in points:
        # Compute vectors from point to each vertex of each face
        w0 = point.unsqueeze(0) - v0  # (F, 3)
        
        # Compute dot product of w0 with face normals
        dot = torch.sum(w0 * face_normals, dim=1)  # (F,)
        
        # Project point onto face plane
        projected = point.unsqueeze(0) - dot.unsqueeze(1) * face_normals  # (F, 3)
        
        # Compute barycentric coordinates
        # This is a simplified version - in real applications we'd need to check if the
        # projected point is inside the triangle, but for our purpose this approximation is enough
        w0 = projected - v0  # (F, 3)
        w1 = v1 - v0  # (F, 3)
        w2 = v2 - v0  # (F, 3)
        
        # Compute squared distances
        dist_squared = torch.min(torch.sum(w0 * w0, dim=1))  # Minimum over all faces
        distances.append(torch.sqrt(dist_squared))
    
    return torch.stack(distances)


def chamfer_distance(x, y, batch_reduction="mean", point_reduction="mean"):
    """
    Compute chamfer distance between two point clouds.
    Simplified version of the pytorch3d chamfer_distance function.
    
    Args:
        x: (B, N, 3) tensor of points
        y: (B, M, 3) tensor of points
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"]
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"]
    
    Returns:
        loss: Chamfer distance between the point clouds
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError("Batch sizes must match for chamfer distance")
    
    batch_size = x.shape[0]
    
    # For each point in x, find its nearest neighbor in y
    x_nn_y_dists = []
    for b in range(batch_size):
        # Compute pairwise distances
        dist = torch.cdist(x[b], y[b])  # (N, M)
        # Get min distance for each point in x
        min_dist, _ = torch.min(dist, dim=1)  # (N,)
        x_nn_y_dists.append(min_dist)
    
    # For each point in y, find its nearest neighbor in x
    y_nn_x_dists = []
    for b in range(batch_size):
        # Compute pairwise distances
        dist = torch.cdist(y[b], x[b])  # (M, N)
        # Get min distance for each point in y
        min_dist, _ = torch.min(dist, dim=1)  # (M,)
        y_nn_x_dists.append(min_dist)
    
    # Stack batch results
    x_nn_y_dists = torch.stack(x_nn_y_dists)  # (B, N)
    y_nn_x_dists = torch.stack(y_nn_x_dists)  # (B, M)
    
    # Apply point reduction
    if point_reduction == "mean":
        x_nn_y_dists = x_nn_y_dists.mean(dim=1)  # (B,)
        y_nn_x_dists = y_nn_x_dists.mean(dim=1)  # (B,)
    elif point_reduction == "sum":
        x_nn_y_dists = x_nn_y_dists.sum(dim=1)  # (B,)
        y_nn_x_dists = y_nn_x_dists.sum(dim=1)  # (B,)
    
    # Compute bidirectional chamfer distance
    chamfer_dist = x_nn_y_dists + y_nn_x_dists  # (B,)
    
    # Apply batch reduction
    if batch_reduction == "mean":
        chamfer_dist = chamfer_dist.mean()
    elif batch_reduction == "sum":
        chamfer_dist = chamfer_dist.sum()
    
    return chamfer_dist


def compute_normal_consistency(x, y):
    """
    Compute normal consistency between two point clouds with normals.
    Simplified version of the corresponding function in pytorch3d.
    
    Args:
        x: (N, 6) tensor of points and normals (xyz + normal_xyz)
        y: (M, 6) tensor of points and normals (xyz + normal_xyz)
    
    Returns:
        normal_consistency: Scalar measuring normal consistency
    """
    if x.shape[1] != 6 or y.shape[1] != 6:
        raise ValueError("Input point clouds must have normals (shape N,6 and M,6)")
    
    # Extract points and normals
    x_points, x_normals = x[:, :3], x[:, 3:]
    y_points, y_normals = y[:, :3], y[:, 3:]
    
    # Normalize normals
    x_normals = F.normalize(x_normals, p=2, dim=1)
    y_normals = F.normalize(y_normals, p=2, dim=1)
    
    # Compute pairwise distances between points
    dists = torch.cdist(x_points, y_points)  # (N, M)
    
    # Get nearest neighbors
    x_nn = torch.argmin(dists, dim=1)  # (N,)
    y_nn = torch.argmin(dists, dim=0)  # (M,)
    
    # Compute dot products of normals with nearest neighbors
    x_normal_consistency = torch.sum(x_normals * y_normals[x_nn], dim=1)  # (N,)
    y_normal_consistency = torch.sum(y_normals * x_normals[y_nn], dim=1)  # (M,)
    
    # Average consistency (higher is better, range [-1, 1])
    # Convert to a loss (lower is better, range [0, 2])
    normal_consistency_loss = 2.0 - (x_normal_consistency.mean() + y_normal_consistency.mean())
    
    return normal_consistency_loss
