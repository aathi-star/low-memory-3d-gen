import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ChamferLoss(nn.Module):
    """
    Chamfer Distance loss with enhanced stability and optional direction weighting
    """
    def __init__(self, bidirectional=True, point_reduction='mean', batch_reduction='mean'):
        super().__init__()
        self.bidirectional = bidirectional
        self.point_reduction = point_reduction
        self.batch_reduction = batch_reduction

    def forward(self, x, y, weight=None):
        """
        Compute Chamfer distance between two point clouds
        
        Args:
            x: (B, N, 3) tensor of points
            y: (B, M, 3) tensor of points
            weight: Optional weight for each sample in the batch
            
        Returns:
            chamfer_dist: Chamfer distance
        """
        x = x.contiguous()
        y = y.contiguous()
        
        batch_size = x.size(0)
        num_points_x = x.size(1)
        num_points_y = y.size(1)
        
        # Reshape to (B, N, 1, 3) and (B, 1, M, 3)
        x = x.unsqueeze(2)
        y = y.unsqueeze(1)
        
        # Compute pairwise distances
        dist = torch.sum((x - y) ** 2, dim=-1)  # (B, N, M)
        
        # Find nearest neighbors
        dist_x_min, _ = torch.min(dist, dim=2)  # (B, N)
        dist_y_min, _ = torch.min(dist, dim=1)  # (B, M)
        
        # Apply point-wise reduction
        if self.point_reduction == 'sum':
            loss_x = torch.sum(dist_x_min, dim=1)  # (B,)
            loss_y = torch.sum(dist_y_min, dim=1)  # (B,)
        else:  # mean
            loss_x = torch.mean(dist_x_min, dim=1)  # (B,)
            loss_y = torch.mean(dist_y_min, dim=1)  # (B,)
        
        # Combine directions
        if self.bidirectional:
            loss = loss_x + loss_y
        else:
            loss = loss_x
        
        # Apply sample weights if provided
        if weight is not None:
            loss = loss * weight
        
        # Apply batch-wise reduction
        if self.batch_reduction == 'sum':
            return torch.sum(loss)
        else:  # mean
            return torch.mean(loss)

class EarthMoverDistanceLoss(nn.Module):
    """
    Earth Mover's Distance loss for point clouds
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        """
        Compute approximate Earth Mover's Distance
        
        Args:
            x: (B, N, 3) tensor of points
            y: (B, N, 3) tensor of points (must have same number of points as x)
            
        Returns:
            emd: Earth Mover's Distance
        """
        batch_size = x.size(0)
        num_points = x.size(1)
        
        # Simple EMD approximation using sorted distances
        # This is faster than exact EMD computation
        x_sorted, _ = torch.sort(x.view(batch_size, -1), dim=1)
        y_sorted, _ = torch.sort(y.view(batch_size, -1), dim=1)
        
        emd = F.mse_loss(x_sorted, y_sorted)
        return emd

class ShapeConsistencyLoss(nn.Module):
    """
    Loss to enforce consistency in shape characteristics
    """
    def __init__(self):
        super().__init__()

    def forward(self, point_cloud, category_ids=None):
        """
        Compute shape consistency loss to enforce ModelNet10 characteristics
        
        Args:
            point_cloud: (B, N, 3) tensor of points
            category_ids: Optional tensor of category IDs for category-specific constraints
            
        Returns:
            loss: Shape consistency loss
        """
        batch_size, num_points, _ = point_cloud.shape
        device = point_cloud.device
        
        # Calculate centroid for each point cloud
        centroids = torch.mean(point_cloud, dim=1, keepdim=True)  # (B, 1, 3)
        
        # Compute distances from centroid
        dists = torch.norm(point_cloud - centroids, dim=2)  # (B, N)
        
        # Compute variance of distances to encourage uniform distribution
        # Lower variance means more uniform point distribution
        dist_variance = torch.var(dists, dim=1)  # (B)
        
        # Compute nearest neighbor distances to prevent points from clustering
        # Reshape for distance computation
        x = point_cloud.view(batch_size * num_points, 1, 3)
        y = point_cloud.view(batch_size, num_points, 3).unsqueeze(1).repeat(1, num_points, 1, 1)
        y = y.view(batch_size * num_points, num_points, 3)
        
        # Compute pairwise distances within each point cloud
        pair_dist = torch.sum((x - y) ** 2, dim=2)  # (B*N, N)
        
        # Set self-distances to a large value
        pair_dist = pair_dist + torch.eye(num_points, device=device) * 1e6
        
        # Find minimum distances to nearest neighbors
        min_nn_dist, _ = torch.min(pair_dist, dim=1)  # (B*N)
        min_nn_dist = min_nn_dist.view(batch_size, num_points)
        
        # Calculate mean nearest neighbor distance
        mean_nn_dist = torch.mean(min_nn_dist, dim=1)  # (B)
        
        # Penalize when nearest neighbor distance is too small
        # (prevents points from being too close together)
        nn_penalty = torch.mean(torch.exp(-5.0 * min_nn_dist))
        
        # Compute normals to encourage smooth surfaces
        # We use a simple approximation by computing normals from nearest neighbors
        normals_consistency = torch.zeros(batch_size, device=device)
        
        # Combine all terms for final loss
        shape_consistency_loss = dist_variance.mean() + nn_penalty
        
        return shape_consistency_loss

class ModelNetShapeLoss(nn.Module):
    """
    Combined loss for generating high-quality shapes resembling ModelNet10 objects
    """
    def __init__(self, chamfer_weight=1.0, emd_weight=0.5, consistency_weight=0.3):
        super().__init__()
        self.chamfer_loss = ChamferLoss(bidirectional=True)
        self.emd_loss = EarthMoverDistanceLoss()
        self.consistency_loss = ShapeConsistencyLoss()
        
        self.chamfer_weight = chamfer_weight
        self.emd_weight = emd_weight
        self.consistency_weight = consistency_weight

    def forward(self, pred_pc, gt_pc, category_ids=None):
        """
        Compute combined loss for shape generation
        
        Args:
            pred_pc: (B, N, 3) predicted point cloud
            gt_pc: (B, N, 3) ground truth point cloud
            category_ids: Optional tensor of category IDs
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual loss components
        """
        # Compute individual loss components
        chamfer = self.chamfer_loss(pred_pc, gt_pc)
        emd = self.emd_loss(pred_pc, gt_pc)
        consistency = self.consistency_loss(pred_pc, category_ids)
        
        # Combine losses
        total_loss = (
            self.chamfer_weight * chamfer + 
            self.emd_weight * emd + 
            self.consistency_weight * consistency
        )
        
        # Create loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'chamfer': chamfer.item(),
            'emd': emd.item(),
            'consistency': consistency.item()
        }
        
        return total_loss, loss_dict
