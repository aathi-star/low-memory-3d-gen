import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class ModelNet10Prototypes(nn.Module):
    """
    Learns and maintains prototype representations for each ModelNet10 category
    Uses these prototypes for contrastive learning to ensure exact resemblance
    """
    def __init__(self, num_categories=10, feature_dim=256, temperature=0.07):
        super().__init__()
        self.num_categories = num_categories
        self.feature_dim = feature_dim
        self.temperature = temperature
        
        # Initialize prototypes (learnable parameters)
        self.prototypes = nn.Parameter(
            torch.randn(num_categories, feature_dim),
            requires_grad=True
        )
        
        # Normalize prototypes
        with torch.no_grad():
            self.prototypes.data = F.normalize(self.prototypes.data, dim=1)
        
        # Memory bank for each category
        self.register_buffer('memory_bank', torch.zeros(num_categories, 512, feature_dim))
        self.register_buffer('memory_ptr', torch.zeros(num_categories, dtype=torch.long))
        
    def forward(self, features, category_ids):
        """
        Apply contrastive learning with category prototypes
        
        Args:
            features: Tensor of shape [B, feature_dim]
            category_ids: Tensor of shape [B]
            
        Returns:
            Contrastive loss, updated features
        """
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Update memory bank
        with torch.no_grad():
            for i in range(batch_size):
                cat_id = category_ids[i].item()
                ptr = self.memory_ptr[cat_id]
                self.memory_bank[cat_id, ptr] = features[i].detach()
                self.memory_ptr[cat_id] = (ptr + 1) % 512
        
        # Update prototypes (moving average)
        with torch.no_grad():
            for cat_id in range(self.num_categories):
                # Get all examples of this category in the current batch
                mask = (category_ids == cat_id)
                if mask.sum() > 0:
                    cat_features = features[mask]
                    # Update prototype with moving average
                    new_prototype = F.normalize(
                        0.9 * self.prototypes[cat_id] + 0.1 * cat_features.mean(dim=0),
                        dim=0
                    )
                    self.prototypes.data[cat_id] = new_prototype
        
        # Compute similarity between features and prototypes
        sim_matrix = features @ self.prototypes.t() / self.temperature
        
        # Compute contrastive loss
        # Target: each feature should be close to its category prototype
        labels = torch.zeros(batch_size, self.num_categories, device=features.device)
        for i in range(batch_size):
            labels[i, category_ids[i]] = 1
        
        # Cross-entropy loss
        loss = -torch.sum(labels * F.log_softmax(sim_matrix, dim=1)) / batch_size
        
        # Pull features towards their prototypes
        aligned_features = features.clone()
        for i in range(batch_size):
            cat_id = category_ids[i].item()
            # Pull feature towards its prototype (weighted average)
            aligned_features[i] = F.normalize(
                0.7 * features[i] + 0.3 * self.prototypes[cat_id],
                dim=0
            )
        
        return loss, aligned_features
    
    def get_prototype(self, category_id):
        """
        Get the prototype for a specific category
        """
        return self.prototypes[category_id]
    
    def align_with_prototype(self, features, category_ids, strength=0.5):
        """
        Align features with their category prototypes
        
        Args:
            features: Tensor of shape [B, feature_dim]
            category_ids: Tensor of shape [B]
            strength: How strongly to pull towards prototype
            
        Returns:
            Aligned features
        """
        features = F.normalize(features, dim=1)
        batch_size = features.shape[0]
        aligned_features = features.clone()
        
        for i in range(batch_size):
            cat_id = category_ids[i].item()
            # Pull feature towards its prototype
            aligned_features[i] = F.normalize(
                (1-strength) * features[i] + strength * self.prototypes[cat_id],
                dim=0
            )
        
        return aligned_features
    
    def save_prototypes(self, save_path):
        """
        Save learned prototypes to file
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.prototypes.detach().cpu(), save_path)
        print(f"Saved ModelNet10 prototypes to {save_path}")
    
    def load_prototypes(self, load_path):
        """
        Load prototypes from file
        """
        if os.path.exists(load_path):
            self.prototypes.data = torch.load(load_path, map_location=self.prototypes.device)
            print(f"Loaded ModelNet10 prototypes from {load_path}")
        else:
            print(f"No prototypes found at {load_path}, using random initialization")


class ShapeConsistencyRegularizer(nn.Module):
    """
    Enforces shape consistency with multiple regularization techniques
    to ensure exact ModelNet10 resemblance
    """
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
    def forward(self, pred_points, gt_points, category_ids):
        """
        Apply multiple consistency regularization terms
        
        Args:
            pred_points: Generated point clouds [B, N, 3]
            gt_points: Ground truth point clouds [B, N, 3]
            category_ids: Category IDs [B]
            
        Returns:
            Total regularization loss, dict of individual losses
        """
        batch_size = pred_points.shape[0]
        
        # 1. Surface normal consistency
        normal_loss = self.normal_consistency_loss(pred_points, gt_points)
        
        # 2. Local structure preservation
        structure_loss = self.local_structure_loss(pred_points, gt_points)
        
        # 3. Category-specific shape prior enforcement
        prior_loss = self.shape_prior_loss(pred_points, category_ids)
        
        # 4. Surface curvature consistency
        curvature_loss = self.curvature_consistency(pred_points, gt_points)
        
        # Total loss
        total_loss = normal_loss + structure_loss + prior_loss + curvature_loss
        
        loss_dict = {
            'normal_consistency': normal_loss.item(),
            'structure': structure_loss.item(),
            'shape_prior': prior_loss.item(),
            'curvature': curvature_loss.item(),
            'total_regularization': total_loss.item()
        }
        
        return total_loss, loss_dict
    
    def estimate_normals(self, points, k=20):
        """
        Estimate point normals using PCA on local neighborhoods
        """
        batch_size, num_points, _ = points.shape
        normals = torch.zeros_like(points)
        
        for b in range(batch_size):
            # Compute pairwise distances
            dist = torch.cdist(points[b], points[b])
            
            # Get k nearest neighbors
            _, nn_idx = torch.topk(dist, k=k+1, dim=1, largest=False)
            nn_idx = nn_idx[:, 1:]  # exclude self
            
            # Get neighbors
            for i in range(num_points):
                neighbors = points[b, nn_idx[i]]
                
                # Center points
                centered = neighbors - neighbors.mean(dim=0, keepdim=True)
                
                # Compute covariance matrix
                cov = centered.t() @ centered
                
                # Compute eigenvectors and eigenvalues
                try:
                    eigvals, eigvecs = torch.linalg.eigh(cov)
                    # Smallest eigenvector = normal direction
                    normal = eigvecs[:, 0]
                    normals[b, i] = normal
                except:
                    # Fallback if eigendecomposition fails
                    normals[b, i] = torch.tensor([0., 1., 0.], device=points.device)
        
        # Normalize normals
        normals = F.normalize(normals, dim=2)
        return normals
    
    def normal_consistency_loss(self, pred_points, gt_points):
        """
        Ensure consistent normal directions
        """
        # Estimate normals
        pred_normals = self.estimate_normals(pred_points)
        gt_normals = self.estimate_normals(gt_points)
        
        # Find closest points in ground truth for each predicted point
        batch_size = pred_points.shape[0]
        loss = 0
        
        for b in range(batch_size):
            # Find nearest neighbor in gt for each pred point
            dist = torch.cdist(pred_points[b], gt_points[b])
            min_idx = torch.argmin(dist, dim=1)
            
            # Get corresponding normals
            corresponding_normals = gt_normals[b, min_idx]
            
            # Compute normal consistency (1 - |dot product|)
            # We use absolute value because normals can point in opposite directions
            dot_product = torch.sum(pred_normals[b] * corresponding_normals, dim=1)
            normal_loss = 1.0 - torch.abs(dot_product).mean()
            
            loss += normal_loss
        
        return loss / batch_size
    
    def local_structure_loss(self, pred_points, gt_points, k=20):
        """
        Preserve local geometric structures
        """
        batch_size, num_points, _ = pred_points.shape
        loss = 0
        
        for b in range(batch_size):
            # Compute pairwise distances for both point clouds
            pred_dist = torch.cdist(pred_points[b], pred_points[b])
            gt_dist = torch.cdist(gt_points[b], gt_points[b])
            
            # Get k nearest neighbors
            _, pred_nn = torch.topk(pred_dist, k=k+1, dim=1, largest=False)
            _, gt_nn = torch.topk(gt_dist, k=k+1, dim=1, largest=False)
            
            # Skip self (first neighbor is the point itself)
            pred_nn = pred_nn[:, 1:]
            gt_nn = gt_nn[:, 1:]
            
            # Compute average distance to neighbors
            pred_neighbor_dists = torch.gather(pred_dist, 1, pred_nn)
            gt_neighbor_dists = torch.gather(gt_dist, 1, gt_nn)
            
            # Distance distribution should be similar
            pred_dist_mean = pred_neighbor_dists.mean(dim=1)
            gt_dist_mean = gt_neighbor_dists.mean(dim=1)
            
            # Find closest points in ground truth for each predicted point
            dist = torch.cdist(pred_points[b], gt_points[b])
            min_idx = torch.argmin(dist, dim=1)
            
            # Get corresponding distance means
            corresponding_dist_means = gt_dist_mean[min_idx]
            
            # Compute loss as relative difference in average neighbor distance
            structure_loss = torch.abs(pred_dist_mean - corresponding_dist_means) / (corresponding_dist_means + 1e-8)
            loss += structure_loss.mean()
        
        return loss / batch_size
    
    def shape_prior_loss(self, points, category_ids):
        """
        Apply category-specific shape priors
        """
        batch_size = points.shape[0]
        loss = 0
        
        # ModelNet10 categories
        categories = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']
        
        for b in range(batch_size):
            cat_id = category_ids[b].item()
            category = categories[cat_id] if cat_id < len(categories) else 'unknown'
            
            # Category-specific priors
            if category == 'chair':
                # Chairs should have a flat horizontal surface (seat) and vertical surface (back)
                y_coords = points[b, :, 1]
                z_coords = points[b, :, 2]
                
                # Find potential seat points (mid-height, y around 0)
                seat_mask = (y_coords > -0.1) & (y_coords < 0.1)
                if seat_mask.sum() > 0:
                    # Seat should be flat (low y-variance)
                    seat_variance = torch.var(y_coords[seat_mask])
                    loss += seat_variance * 10.0
                
                # Find potential back points (higher z values)
                back_mask = z_coords > 0.3
                if back_mask.sum() > 0:
                    # Back should be more vertical (high z-variance)
                    back_z_variance = torch.var(z_coords[back_mask])
                    loss += torch.max(torch.tensor(0.05, device=points.device) - back_z_variance, torch.tensor(0.0, device=points.device)) * 5.0
            
            elif category == 'table':
                # Tables should have a flat top and legs
                y_coords = points[b, :, 1]
                
                # Find potential table top points (higher y values)
                top_mask = y_coords > 0.2
                if top_mask.sum() > 0:
                    # Top should be flat (low y-variance)
                    top_variance = torch.var(y_coords[top_mask])
                    loss += top_variance * 10.0
                
                # Find potential leg points (lower y values)
                leg_mask = y_coords < -0.2
                if leg_mask.sum() < 10:
                    # Encourage more leg points
                    loss += 0.1
            
            elif category == 'sofa':
                # Sofas should have a specific structure with seat and back
                y_coords = points[b, :, 1]
                z_coords = points[b, :, 2]
                
                # Seats should be horizontal
                seat_mask = (y_coords > -0.1) & (y_coords < 0.1)
                if seat_mask.sum() > 0:
                    seat_variance = torch.var(y_coords[seat_mask])
                    loss += seat_variance * 5.0
                
                # Back should be more vertical and present
                back_mask = (z_coords > 0.3) & (y_coords > 0.0)
                if back_mask.sum() < 10 or back_mask.sum() / points.shape[1] < 0.1:
                    loss += 0.1
        
        return loss / batch_size
    
    def curvature_consistency(self, pred_points, gt_points, k=20):
        """
        Enforce similar surface curvature between prediction and ground truth
        """
        batch_size = pred_points.shape[0]
        loss = 0
        
        for b in range(batch_size):
            # Estimate curvature for both point clouds
            pred_curvature = self.estimate_curvature(pred_points[b], k)
            gt_curvature = self.estimate_curvature(gt_points[b], k)
            
            # Find closest points in ground truth for each predicted point
            dist = torch.cdist(pred_points[b], gt_points[b])
            min_idx = torch.argmin(dist, dim=1)
            
            # Get corresponding curvatures
            corresponding_curvature = gt_curvature[min_idx]
            
            # Compute loss as mean absolute difference
            curvature_loss = torch.abs(pred_curvature - corresponding_curvature).mean()
            loss += curvature_loss
        
        return loss / batch_size
    
    def estimate_curvature(self, points, k=20):
        """
        Estimate local curvature at each point
        """
        num_points = points.shape[0]
        curvature = torch.zeros(num_points, device=points.device)
        
        # Compute pairwise distances
        dist = torch.cdist(points, points)
        
        # Get k nearest neighbors
        _, nn_idx = torch.topk(dist, k=k+1, dim=1, largest=False)
        nn_idx = nn_idx[:, 1:]  # exclude self
        
        for i in range(num_points):
            neighbors = points[nn_idx[i]]
            centered = neighbors - neighbors.mean(dim=0, keepdim=True)
            
            # Compute covariance matrix
            cov = centered.t() @ centered
            
            # Compute eigenvalues
            try:
                eigvals, _ = torch.linalg.eigh(cov)
                # Curvature is approximated by ratio of smallest eigenvalue to sum
                curvature[i] = eigvals[0] / (torch.sum(eigvals) + 1e-8)
            except:
                curvature[i] = 0.0
        
        return curvature
