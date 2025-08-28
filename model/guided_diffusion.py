import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class GuidedDiffusionGenerator(nn.Module):
    """
    Neural diffusion process with category-guided sampling for exact ModelNet10 resemblance
    Uses a denoising diffusion process with category conditioning and template guidance
    """
    def __init__(self, model, num_steps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model  # ShapeGenerator model for denoising
        self.num_steps = num_steps
        
        # Define noise schedule (linear beta schedule)
        self.betas = torch.linspace(beta_start, beta_end, num_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Precompute values for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod[:-1]) / (1.0 - self.alphas_cumprod[1:])
        self.posterior_variance = torch.cat([self.posterior_variance, torch.tensor([0.0])])
        
        # Register templates and prototypes buffers
        self.register_buffer('category_templates', torch.zeros(10, 2048, 3))
        self.has_templates = False
    
    def load_templates(self, templates_path):
        """
        Load category-specific templates for guidance
        """
        if os.path.exists(templates_path):
            self.category_templates = torch.load(templates_path)
            self.has_templates = True
            print(f"Loaded {self.category_templates.shape[0]} category templates")
        else:
            print(f"No templates found at {templates_path}")
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process (add noise at timestep t)
        x_start: [B, N, 3] clean point cloud
        t: [B] timesteps
        noise: [B, N, 3] noise to add (if None, random noise is used)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_sample(self, x, t, category_ids=None, template_strength=0.3, classifier_guidance_scale=1.0):
        """
        Reverse diffusion sampling step (denoise)
        x: [B, N, 3] noisy point cloud at timestep t
        t: [B] timesteps
        category_ids: [B] category IDs for conditional sampling
        template_strength: weight for template guidance
        classifier_guidance_scale: weight for category guidance
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Predict noise using the model
        with torch.no_grad():
            predicted_noise = self.model.denoise(x, t, category_ids)
        
        # Apply classifier guidance (category conditioning)
        if category_ids is not None and classifier_guidance_scale > 1.0:
            # Get unconditional prediction
            uncond_noise = self.model.denoise(x, t, None)
            
            # Apply classifier guidance
            predicted_noise = uncond_noise + classifier_guidance_scale * (predicted_noise - uncond_noise)
        
        # Compute denoised x_0 prediction
        alpha_t = self.alphas[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        # Predict x_0
        pred_x0 = (x - sqrt_one_minus_alpha_t * predicted_noise) / torch.sqrt(alpha_t)
        
        # Apply template guidance if available
        if self.has_templates and category_ids is not None and template_strength > 0:
            for i in range(batch_size):
                cat_id = category_ids[i].item()
                if cat_id < self.category_templates.shape[0]:
                    template = self.category_templates[cat_id].to(device)
                    
                    # Gradual template strength that decreases over time
                    current_strength = template_strength * (1.0 - t[i].float() / self.num_steps)
                    
                    # Blend with template
                    pred_x0[i] = (1 - current_strength) * pred_x0[i] + current_strength * template
        
        # Get the mean for the posterior distribution
        beta_t = self.betas[t].view(-1, 1, 1)
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].view(-1, 1, 1)
        posterior_mean = sqrt_recip_alphas_t * (x - beta_t * predicted_noise / sqrt_one_minus_alpha_t)
        
        # Add noise for the next step (if not final step)
        posterior_variance_t = self.posterior_variance[t].view(-1, 1, 1)
        noise = torch.randn_like(x) if t[0] > 0 else torch.zeros_like(x)
        
        return posterior_mean + torch.sqrt(posterior_variance_t) * noise
    
    def generate(self, batch_size=1, categories=None, classifier_guidance_scale=2.5, template_strength=0.3, device='cuda'):
        """
        Generate point clouds using the guided diffusion process
        
        Args:
            batch_size: Number of point clouds to generate
            categories: Category IDs (if None, unconditional generation)
            classifier_guidance_scale: Weight for category guidance
            template_strength: Weight for template guidance
            device: Device to run generation on
            
        Returns:
            Generated point clouds [B, N, 3]
        """
        # Set device
        device = torch.device(device)
        
        # Set up shape and device
        num_points = self.model.num_points
        
        # Initialize with pure noise
        x = torch.randn(batch_size, num_points, 3, device=device)
        
        # Convert categories to tensor if provided
        if categories is not None:
            if isinstance(categories, list):
                categories = torch.tensor(categories, device=device)
            elif not isinstance(categories, torch.Tensor):
                categories = torch.tensor([categories], device=device).expand(batch_size)
        
        # Sampling loop
        for t_idx in reversed(range(self.num_steps)):
            # Set same timestep for entire batch
            t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
            
            # Sample from p(x_{t-1} | x_t)
            x = self.p_sample(
                x, t, 
                category_ids=categories,
                template_strength=template_strength,
                classifier_guidance_scale=classifier_guidance_scale
            )
        
        # Ensure the point cloud is normalized to unit cube
        for i in range(batch_size):
            max_abs_val = torch.max(torch.abs(x[i]))
            x[i] = x[i] / max_abs_val
        
        return x
    
    def apply_category_refinement(self, point_clouds, category_ids=None):
        """
        Apply final category-specific refinements to ensure exact ModelNet10 resemblance
        
        Args:
            point_clouds: [B, N, 3] generated point clouds
            category_ids: [B] category IDs
            
        Returns:
            Refined point clouds [B, N, 3]
        """
        batch_size = point_clouds.shape[0]
        refined = point_clouds.clone()
        
        if category_ids is None:
            return refined
        
        # ModelNet10 categories
        categories = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']
        
        for i in range(batch_size):
            cat_id = category_ids[i].item()
            category = categories[cat_id] if cat_id < len(categories) else 'unknown'
            
            # Apply category-specific refinements
            if category == 'chair':
                # Ensure chair has clear seat and back
                
                # 1. Flatten seat area
                seat_mask = (refined[i, :, 1] > -0.1) & (refined[i, :, 1] < 0.1)
                if seat_mask.sum() > 0:
                    refined[i, seat_mask, 1] = 0.0
                
                # 2. Ensure chair back is vertical
                back_mask = (refined[i, :, 2] > 0.3) & (refined[i, :, 1] > 0.1)
                if back_mask.sum() > 0:
                    # Make back more vertical
                    back_center = refined[i, back_mask, :].mean(dim=0)
                    for j in torch.where(back_mask)[0]:
                        # Adjust points to be more vertical
                        p = refined[i, j]
                        # Keep x position, adjust y-z to be more vertical
                        p[1] = max(p[1], 0.2)  # Ensure back is high enough
                        # Move z value to make back more vertical
                        p[2] = max(p[2], 0.3)
                        refined[i, j] = p
            
            elif category == 'table':
                # 1. Ensure flat table top
                top_mask = refined[i, :, 1] > 0.2
                if top_mask.sum() > 0:
                    # Make top perfectly flat
                    top_y = refined[i, top_mask, 1].mean()
                    refined[i, top_mask, 1] = top_y
                
                # 2. Ensure table has proper legs
                leg_mask = refined[i, :, 1] < -0.2
                if leg_mask.sum() < 20:
                    # Add or reinforce legs
                    # Find corners of table top bounding box
                    if top_mask.sum() > 0:
                        top_points = refined[i, top_mask]
                        min_x, min_z = top_points[:, 0].min(), top_points[:, 2].min()
                        max_x, max_z = top_points[:, 0].max(), top_points[:, 2].max()
                        
                        # Create leg at each corner
                        corners = [
                            [min_x, max_x, min_x, max_x],
                            [-0.5, -0.5, -0.5, -0.5],  # y (bottom)
                            [min_z, min_z, max_z, max_z]
                        ]
                        
                        leg_count = min(20, refined.shape[1]//10)  # Use up to 20 points for legs
                        leg_points_per_corner = leg_count // 4
                        
                        # Find points to replace with leg points
                        candidates = torch.where(~top_mask)[0]
                        if len(candidates) >= leg_count:
                            leg_indices = candidates[:leg_count]
                            
                            # Distribute leg points
                            for leg in range(4):
                                for j in range(leg_points_per_corner):
                                    if leg*leg_points_per_corner + j < len(leg_indices):
                                        idx = leg_indices[leg*leg_points_per_corner + j]
                                        # Set leg point
                                        leg_x = corners[0][leg]
                                        leg_z = corners[2][leg]
                                        # Vary y from bottom to near top
                                        leg_y = -0.5 + j * 0.5 / leg_points_per_corner
                                        refined[i, idx, 0] = leg_x
                                        refined[i, idx, 1] = leg_y
                                        refined[i, idx, 2] = leg_z
            
            elif category == 'toilet':
                # 1. Ensure toilet has bowl shape
                bowl_mask = (refined[i, :, 1] > -0.2) & (refined[i, :, 1] < 0.1)
                if bowl_mask.sum() > 0:
                    # Get bowl center in xz plane
                    bowl_points = refined[i, bowl_mask]
                    bowl_center_xz = torch.mean(bowl_points[:, [0, 2]], dim=0)
                    
                    # Make bowl more circular
                    for j in torch.where(bowl_mask)[0]:
                        # Current xz position
                        xz = refined[i, j, [0, 2]]
                        # Vector from center to point
                        vec = xz - bowl_center_xz
                        # Normalize and set to consistent radius
                        dist = torch.norm(vec)
                        if dist > 1e-6:  # Avoid division by zero
                            normalized = vec / dist
                            # Set to consistent radius but vary slightly
                            radius = 0.25 + torch.rand(1, device=refined.device) * 0.05
                            refined[i, j, 0] = bowl_center_xz[0] + normalized[0] * radius
                            refined[i, j, 2] = bowl_center_xz[1] + normalized[1] * radius
                
                # 2. Ensure toilet has tank at back
                tank_mask = (refined[i, :, 2] > 0.2) & (refined[i, :, 1] > 0.0)
                if tank_mask.sum() < 20:
                    # Add tank points
                    non_bowl = ~bowl_mask
                    candidates = torch.where(non_bowl)[0]
                    if len(candidates) >= 20:
                        tank_indices = candidates[:20]
                        # Set tank points in a rectangular shape
                        for j, idx in enumerate(tank_indices):
                            refined[i, idx, 0] = (j % 5 - 2) * 0.1  # x: distribute horizontally
                            refined[i, idx, 1] = 0.2 + (j // 10) * 0.1  # y: height
                            refined[i, idx, 2] = 0.4  # z: back of toilet
        
        return refined
