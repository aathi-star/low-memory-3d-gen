import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ModelNet10ShapeRefinement(nn.Module):
    """
    Advanced shape refinement module specifically designed for ModelNet10 shapes.
    Takes point clouds and category information to apply category-specific refinements
    that ensure generated shapes exactly resemble ModelNet10 objects.
    """
    def __init__(self, input_dim=3, latent_dim=256, hidden_dim=128, use_category=True, num_categories=10):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.use_category = use_category
        self.num_categories = num_categories
        
        # Point-wise refinement network (operates on each point)
        self.point_refinement = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2)
        )
        
        # Category-specific refinement
        if use_category:
            # Category embedding
            self.category_embedding = nn.Embedding(num_categories, hidden_dim)
            
            # Combined refinement network (point features + category features)
            self.combined_refinement = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim // 2, 3)  # XYZ offset
            )
        else:
            # Point-only refinement network
            self.point_only_refinement = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim // 2, 3)  # XYZ offset
            )
        
        # Shape symmetry enforcement
        self.symmetry_net = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 3)
        )
        
        # ModelNet10-specific shape priors
        self.shape_prior = ModelNet10ShapePrior(hidden_dim, num_categories)
    
    def forward(self, points, category_ids=None):
        """
        Apply refinement to ensure exact ModelNet10 resemblance
        
        Args:
            points: point cloud tensor of shape [B, N, input_dim]
            category_ids: category IDs tensor of shape [B]
            
        Returns:
            Refinement offsets of shape [B, N, 3]
        """
        batch_size, num_points = points.shape[:2]
        
        # Extract xyz coordinates for shape operations
        xyz = points[:, :, :3]
        
        # Get point-wise features
        point_features = self.point_refinement(points)
        
        # Apply category-specific refinement if available
        if self.use_category and category_ids is not None:
            # Get category embeddings
            cat_emb = self.category_embedding(category_ids)  # [B, hidden_dim]
            
            # Expand category embedding to all points
            cat_emb = cat_emb.unsqueeze(1).expand(-1, num_points, -1)  # [B, N, hidden_dim]
            
            # Concatenate point features and category features
            combined_features = torch.cat([point_features, cat_emb], dim=2)  # [B, N, hidden_dim*2]
            
            # Generate refinement offsets
            offsets = self.combined_refinement(combined_features)  # [B, N, 3]
        else:
            # Generate refinement offsets without category information
            offsets = self.point_only_refinement(point_features)  # [B, N, 3]
        
        # Apply shape priors based on category
        if category_ids is not None:
            prior_offsets = self.shape_prior(xyz, category_ids)  # [B, N, 3]
            offsets = offsets + prior_offsets * 0.5  # Blend with shape priors
        
        # Enforce symmetry for specific categories
        if category_ids is not None:
            # Categories that typically have symmetry (chairs, tables, etc.)
            symmetric_categories = [2, 8]  # chair and table indices
            
            for i, cat_id in enumerate(category_ids):
                if cat_id in symmetric_categories:
                    # Create mirrored points along x-axis
                    mirrored_xyz = xyz[i].clone()
                    mirrored_xyz[:, 0] = -mirrored_xyz[:, 0]  # Flip x-coordinate
                    
                    # Get refinement for mirrored points
                    mirrored_refine = self.symmetry_net(mirrored_xyz)  # [N, 3]
                    
                    # Apply symmetric refinement to original points
                    mirrored_refine[:, 0] = -mirrored_refine[:, 0]  # Flip x-coordinate back
                    offsets[i] = (offsets[i] + mirrored_refine) * 0.5
        
        return offsets


class ModelNet10ShapePrior(nn.Module):
    """
    Encodes shape priors for ModelNet10 categories to guide refinement
    toward category-specific characteristics.
    """
    def __init__(self, hidden_dim=128, num_categories=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_categories = num_categories
        
        # Category-specific shape priors
        self.category_priors = nn.ModuleList([
            self._create_category_prior(hidden_dim) for _ in range(num_categories)
        ])
    
    def _create_category_prior(self, hidden_dim):
        return nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, 3)
        )
    
    def forward(self, points, category_ids):
        """
        Apply category-specific shape priors
        
        Args:
            points: point cloud xyz coordinates [B, N, 3]
            category_ids: category IDs tensor of shape [B]
            
        Returns:
            Prior-based offsets of shape [B, N, 3]
        """
        batch_size, num_points = points.shape[:2]
        offsets = torch.zeros_like(points)  # [B, N, 3]
        
        # Apply appropriate prior for each sample in the batch
        for i, cat_id in enumerate(category_ids):
            cat_prior = self.category_priors[cat_id]
            offsets[i] = cat_prior(points[i])
        
        return offsets
