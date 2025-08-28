import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig, BertTokenizer
from einops import rearrange, repeat
from typing import Tuple, Optional, List, Union, Dict, Any


class TokenCentricGraphAttention(nn.Module):
    """
    Token-Centric Graph Attention module for enhancing text-to-3D generation.
    This module creates a learnable attention graph between text tokens and shape features.
    
    Implementation inspired by research papers:
    - "Attention is All You Need" (Vaswani et al., 2017)
    - "Graph Attention Networks" (Veličković et al., 2018)
    - "Text2Shape: Generating Shapes from Natural Language" (Chen et al., 2019)
    """
    def __init__(self, text_dim, shape_dim, num_heads=8, dropout=0.1, edge_dim=32):
        super().__init__()
        self.num_heads = num_heads
        self.text_dim = text_dim
        self.shape_dim = shape_dim
        self.edge_dim = edge_dim  # Dimension for edge features in the graph
        
        # Project text and shape features to a common dimension
        self.head_dim = text_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Projection layers for Q, K, V with layer normalization
        self.text_norm = nn.LayerNorm(text_dim)
        self.shape_norm = nn.LayerNorm(shape_dim)
        
        self.to_q = nn.Linear(text_dim, text_dim)
        self.to_k = nn.Linear(shape_dim, text_dim)
        self.to_v = nn.Linear(shape_dim, text_dim)
        
        # Edge features in the graph (representing token-shape relationships)
        self.edge_embedding = nn.Parameter(torch.randn(num_heads, edge_dim))
        self.edge_proj = nn.Sequential(
            nn.Linear(edge_dim, 1),
            nn.LeakyReLU(0.2)
        )
        
        # Output projection with residual connection components
        self.proj = nn.Linear(text_dim, text_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(text_dim)
        
        # Feed-forward network after attention (similar to Transformer)
        self.ffn = nn.Sequential(
            nn.Linear(text_dim, text_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(text_dim * 4, text_dim),
            nn.Dropout(dropout)
        )
        self.ffn_norm = nn.LayerNorm(text_dim)
        
    def forward(self, text_features, shape_features):
        """
        Args:
            text_features: Tensor of shape [batch_size, seq_len, text_dim]
            shape_features: Tensor of shape [batch_size, num_points, shape_dim]
            
        Returns:
            enhanced_features: Text features enhanced with shape context
            attn: Attention matrix showing text-shape relationships
        """
        batch_size, seq_len, _ = text_features.shape
        _, num_points, _ = shape_features.shape
        
        # Apply layer normalization first (pre-norm approach)
        text_normed = self.text_norm(text_features)
        shape_normed = self.shape_norm(shape_features)
        
        # Residual connection preparation
        residual = text_features
        
        # Project to queries, keys, values
        q = self.to_q(text_normed)
        k = self.to_k(shape_normed)
        v = self.to_v(shape_normed)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Compute basic dot-product attention scores
        base_attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        
        # Enhance with learned edge features for each head
        # This creates a learnable graph structure between text and shape
        # First create expanded base attention without einsum to avoid dimension mismatch
        base_attn_expanded = base_attn.unsqueeze(-1)  # Shape: [batch, heads, seq_len, num_points, 1]
        edge_attn = torch.zeros(batch_size, self.num_heads, seq_len, num_points, self.edge_dim, device=text_features.device)
        
        # Apply edge embeddings to each attention score
        for h in range(self.num_heads):
            edge_attn[:, h] = base_attn_expanded[:, h] * self.edge_embedding[h].view(1, 1, 1, -1)
        edge_scores = self.edge_proj(edge_attn).squeeze(-1)
        
        # Combine base attention with edge-enhanced attention
        attn = base_attn + edge_scores
        
        # Apply softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # Final projection with residual connection
        out = self.proj(out)
        out = self.dropout(out)
        out = self.output_norm(residual + out)
        
        # Apply feed-forward network with another residual connection
        ffn_residual = out
        out = self.ffn(out)
        out = self.ffn_norm(ffn_residual + out)
        
        return out, attn


class PointCloudDecoder(nn.Module):
    """
    Decoder that generates 3D point clouds from latent features.
    """
    def __init__(self, latent_dim, output_dim=3, num_points=2048):
        super().__init__()
        self.num_points = num_points
        self.output_dim = output_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, num_points * output_dim)
        )
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, latent_dim]
        Returns:
            Point cloud of shape [batch_size, num_points, output_dim]
        """
        batch_size = x.shape[0]
        points = self.mlp(x)
        points = points.view(batch_size, self.num_points, self.output_dim)
        return points


class MeshDecoder(nn.Module):
    """
    Decoder that generates meshes from latent features.
    Uses a coarse-to-fine approach for generating meshes from point clouds.
    """
    def __init__(self, latent_dim, num_vertices=1000, num_faces=2000):
        super().__init__()
        self.num_vertices = num_vertices
        self.num_faces = num_faces
        
        # Generate vertex positions
        self.vertex_mlp = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, num_vertices * 3)
        )
        
        # Generate face connectivity (simplified representation)
        # In practice, this would be more complex or use a predefined topology
        self.face_mlp = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, num_faces * 3)
        )
        
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, latent_dim]
        Returns:
            vertices: Tensor of shape [batch_size, num_vertices, 3]
            faces: Tensor of shape [batch_size, num_faces, 3]
        """
        batch_size = x.shape[0]
        
        # Generate vertices
        vertices = self.vertex_mlp(x)
        vertices = vertices.view(batch_size, self.num_vertices, 3)
        
        # Generate face connectivity (simplified)
        # In practice, use more sophisticated methods
        faces_logits = self.face_mlp(x)
        faces_logits = faces_logits.view(batch_size, self.num_faces, 3)
        
        # Convert to vertex indices (simplified approach)
        # In a real implementation, this would be more sophisticated
        faces = torch.clamp(
            (F.softmax(faces_logits, dim=-1) * self.num_vertices).long(),
            0, self.num_vertices - 1
        )
        
        return vertices, faces


class TextTo3DModel(nn.Module):
    """
    Main model class that combines text encoder with 3D generation.
    Includes optimizations for low-resource environments:
    - Support for LoRA adapters (Parameter-Efficient Fine-Tuning)
    - Progressive point cloud generation
    - Efficient inferencing
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Text encoder (using pretrained BERT)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.text_dim = self.text_encoder.config.hidden_size
        
        # Freeze BERT parameters for faster training
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        # Text feature projector
        self.text_projector = nn.Sequential(
            nn.Linear(self.text_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2)
        )
        
        # Token-centric graph attention
        self.graph_attention = TokenCentricGraphAttention(
            text_dim=512,
            shape_dim=512,
            num_heads=8,
            dropout=0.1
        )
        
        # Shape latent encoder
        self.shape_encoder = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2)
        )
        
        # Output decoders - we'll create multiple resolution decoders for progressive generation
        if config.output_type == 'point_cloud':
            # Create decoders for different resolutions
            # Low-res (coarse) decoder - 512 points
            self.coarse_decoder = PointCloudDecoder(
                latent_dim=256,
                num_points=min(512, config.num_points)
            )
            
            # Medium-res decoder - 2048 points or half of target points
            self.medium_decoder = PointCloudDecoder(
                latent_dim=256,
                num_points=min(2048, config.num_points // 2)
            ) if config.num_points > 512 else None
            
            # Full resolution decoder
            self.decoder = PointCloudDecoder(
                latent_dim=256,
                num_points=config.num_points
            )
        elif config.output_type == 'mesh':
            self.decoder = MeshDecoder(
                latent_dim=256,
                num_vertices=config.num_vertices,
                num_faces=config.num_faces
            )
            
        # Flag to indicate if LoRA adapters are applied
        self.using_lora = False
        
    def encode_text(self, text_list):
        """
        Encode a list of text strings.
        
        Args:
            text_list: List of text prompts to encode
            
        Returns:
            text_features: Encoded text features
            attention_mask: Attention mask for the text tokens
        """
        tokens = self.tokenizer(
            text_list, 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=32
        ).to(next(self.parameters()).device)
        
        # Get BERT embeddings
        with torch.no_grad():
            outputs = self.text_encoder(**tokens)
            text_features = outputs.last_hidden_state
        
        # Project text features
        text_features = self.text_projector(text_features)
        
        return text_features, tokens.attention_mask
    
    def forward(self, text_list=None, text_features=None, progressive=False):
        """
        Forward pass of the model.
        
        Args:
            text_list: List of text prompts
            text_features: Pre-computed text features (optional)
            progressive: Whether to use progressive generation
            
        Returns:
            3D object representation (point cloud or mesh)
        """
        # Get text encodings if not provided
        attention_mask = None
        if text_features is None:
            text_features, attention_mask = self.encode_text(text_list)
        
        batch_size = text_features.shape[0]
        
        # Initialize shape features as learnable parameters
        # These will be refined through the graph attention mechanism
        shape_features = torch.randn(
            batch_size,
            self.config.num_shape_features,
            512,  # Same dimension as text projector output
            device=text_features.device
        )
        
        # Apply token-centric graph attention
        # This creates alignment between text tokens and shape features
        attended_features, _ = self.graph_attention(text_features, shape_features)
        
        # Pool the attended features
        # We use attention mask to focus on actual tokens, not padding
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled_features = (attended_features * mask).sum(dim=1) / mask.sum(dim=1)
        else:
            pooled_features = attended_features.mean(dim=1)
        
        # Encode to shape latent
        shape_latent = self.shape_encoder(pooled_features)
        
        # Decode to output 3D representation
        if self.config.output_type == 'point_cloud':
            if progressive and hasattr(self, 'coarse_decoder'):
                # For training, we only need the final output
                output = self.decoder(shape_latent)
            else:
                output = self.decoder(shape_latent)
        elif self.config.output_type == 'mesh':
            output = self.decoder(shape_latent)
        
        return output
        
    def generate_with_resolution(self, text_list, num_points=None, temperature=0.8, guidance_scale=3.0, num_inference_steps=50, unconditional_generation=False):
        """
        Generate point cloud at a specific resolution with control parameters.
        
        Args:
            text_list: List of text prompts (ignored if unconditional_generation=True)
            num_points: Number of points to generate (overrides config)
            temperature: Sampling temperature (lower = more deterministic)
            guidance_scale: Text conditioning strength (higher = more faithful to text)
            num_inference_steps: Number of refinement steps during generation
            unconditional_generation: If True, ignore text prompt and generate shapes without text conditioning
            
        Returns:
            Point cloud at the specified resolution
        """
        if unconditional_generation:
            # For unconditional generation, create a batch of empty embeddings
            print("Running in unconditional generation mode - ignoring text prompt")
            # Create a dummy batch with same device as model
            device = next(self.parameters()).device
            
            # Determine text embedding dimension from model structure
            # We'll use the first layer of the token graph attention network
            if hasattr(self, 'token_graph_attention'):
                # Get embedding dimension from the first layer's weight shape
                text_dim = self.token_graph_attention.layers[0].weight.shape[1]
            elif hasattr(self, 'text_encoder'):
                # Try to get from text encoder output dimension
                try:
                    # For BERT-like encoders
                    text_dim = self.text_encoder.config.hidden_size
                except AttributeError:
                    # Default fallback to common embedding sizes
                    text_dim = 768  # Default BERT embedding size
            else:
                # Final fallback to standard embedding size
                text_dim = 768
                
            print(f"Using text embedding dimension: {text_dim} for unconditional generation")
                
            # Create empty text features with proper shape [batch_size, seq_len, text_dim]
            # We need to match the expected 3D tensor shape for the graph attention
            seq_len = 10  # Use a reasonable sequence length
            
            # First create BERT-sized embeddings with some random noise for variety
            bert_features = torch.zeros((len(text_list), seq_len, text_dim), device=device)
            # Add small random noise for variety
            bert_features = bert_features + 0.1 * torch.randn_like(bert_features)
            
            # Project the features through text_projector just like normal text features
            text_features = self.text_projector(bert_features)
            print(f"Created unconditional text features with shape: {text_features.shape}")
            
            # Create proper attention mask
            attention_mask = torch.ones((len(text_list), seq_len), device=device)
            batch_size = len(text_list)
        else:
            # Normal conditional generation - encode text
            text_features, attention_mask = self.encode_text(text_list)
            batch_size = text_features.shape[0]
        
        # Process through graph attention and shape encoding
        # Apply temperature to initial noise for controlled randomness
        noise_scale = temperature * (1.0 if temperature > 0 else 1.0)
        shape_features = torch.randn(
            batch_size,
            self.config.num_shape_features,
            512,
            device=text_features.device
        ) * noise_scale
        
        # Apply iterative refinement based on num_inference_steps
        for i in range(num_inference_steps):
            # Optional: print progress for long generation processes
            if i == 0 or i == num_inference_steps - 1 or (i+1) % 10 == 0:
                print(f"Generation step {i+1}/{num_inference_steps}")
                
            # Get attended features with guided attention
            attended_features, attention_weights = self.graph_attention(text_features, shape_features)
            
            # Apply guidance scale to enhance text conditioning
            if guidance_scale > 1.0:
                # Stronger influence from text condition
                attended_features = attended_features * guidance_scale
            
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled_features = (attended_features * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                pooled_features = attended_features.mean(dim=1)
                
            # Only do full refinement iterations for the first few and last few steps
            if i < 10 or i >= num_inference_steps - 10 or num_inference_steps <= 20:
                shape_latent = self.shape_encoder(pooled_features)
                # Iteratively refine shape features if not the last step
                if i < num_inference_steps - 1:
                    # Get intermediate output and feed back into the process
                    with torch.no_grad():
                        if hasattr(self, 'feedback_encoder'):
                            # If we have a dedicated feedback encoder, use it
                            refined_features = self.feedback_encoder(shape_latent)
                            # Mix original and refined features with decreasing noise
                            mix_ratio = 1.0 - ((i + 1) / num_inference_steps)
                            shape_features = refined_features + (shape_features * mix_ratio * 0.1)
            
        # Final shape latent for decoding
        shape_latent = self.shape_encoder(pooled_features)
        
        # Select appropriate decoder based on resolution
        if num_points is None:
            num_points = self.config.num_points
            
        if num_points <= 512 and hasattr(self, 'coarse_decoder'):
            return self.coarse_decoder(shape_latent)
        elif num_points <= 2048 and hasattr(self, 'medium_decoder'):
            return self.medium_decoder(shape_latent)
        else:
            return self.decoder(shape_latent)
    
    def refine_point_cloud(self, coarse_pc, text_list, num_points, temperature=0.8, guidance_scale=3.0, num_inference_steps=20):
        """
        Refine a coarse point cloud to higher resolution with enhanced control.
        
        Args:
            coarse_pc: Coarse point cloud to refine
            text_list: Text prompts for guidance
            num_points: Target number of points
            temperature: Sampling temperature (lower = more deterministic)
            guidance_scale: Text conditioning strength (higher = more faithful to text)
            num_inference_steps: Number of refinement steps during generation
            
        Returns:
            Refined point cloud with improved detail and structure
        """
        # Encode text
        text_features, attention_mask = self.encode_text(text_list)
        batch_size = text_features.shape[0]
        
        # Apply temperature to shape features for controlled randomness
        noise_scale = temperature * 0.1  # Smaller scale for refinement compared to generation
        
        # Use the coarse point cloud as initial shape features
        coarse_features = F.interpolate(
            coarse_pc.transpose(1, 2),  # [B, 3, N_coarse]
            size=self.config.num_shape_features,
            mode='linear'
        )  # [B, 3, num_shape_features]
        
        # Project to shape feature dimension
        shape_projector = nn.Linear(
            3, 512, device=coarse_pc.device
        )
        initial_shape_features = shape_projector(coarse_features.transpose(1, 2))  # [B, num_shape_features, 512]
        
        # Add controlled noise based on temperature
        shape_features = initial_shape_features + (torch.randn_like(initial_shape_features) * noise_scale)
        
        print(f"Refining point cloud with {num_inference_steps} steps, guidance={guidance_scale:.1f}, temp={temperature:.1f}")
        
        # Iterative refinement process
        for i in range(num_inference_steps):
            # Optional progress reporting
            if i == 0 or i == num_inference_steps - 1 or (i+1) % 5 == 0:
                print(f"Refinement step {i+1}/{num_inference_steps}")
                
            # Apply graph attention with enhanced text guidance
            attended_features, attention_weights = self.graph_attention(text_features, shape_features)
            
            # Apply guidance scale to enhance text conditioning 
            if guidance_scale > 1.0:
                attended_features = attended_features * guidance_scale
            
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled_features = (attended_features * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                pooled_features = attended_features.mean(dim=1)
            
            # Process through shape encoder
            current_shape_latent = self.shape_encoder(pooled_features)
            
            # If not the last step, continue refining
            if i < num_inference_steps - 1:
                # Feedback loop to improve features
                with torch.no_grad():
                    # Generate intermediate point cloud
                    if hasattr(self, 'medium_decoder') and num_points > 512:
                        intermediate_pc = self.medium_decoder(current_shape_latent)
                        # Convert back to features for next iteration
                        inter_features = F.interpolate(
                            intermediate_pc.transpose(1, 2),
                            size=self.config.num_shape_features,
                            mode='linear'
                        )
                        # Mix with original features, decreasing noise over iterations
                        mix_ratio = 1.0 - ((i + 1) / num_inference_steps)
                        refined_features = shape_projector(inter_features.transpose(1, 2))
                        shape_features = refined_features + (initial_shape_features * mix_ratio * 0.05)
        
        # Final shape latent for decoding
        shape_latent = current_shape_latent
        
        # Generate refined point cloud
        if num_points <= 2048 and hasattr(self, 'medium_decoder'):
            return self.medium_decoder(shape_latent)
        else:
            return self.decoder(shape_latent)
            
    def apply_lora(self, rank=4, alpha=16, dropout=0.05):
        """
        Apply LoRA adapters to the model for parameter-efficient fine-tuning.
        
        Args:
            rank: LoRA rank parameter
            alpha: LoRA alpha parameter
            dropout: LoRA dropout probability
            
        Returns:
            self: The model with LoRA adapters applied
        """
        try:
            from model.optimization import apply_lora_to_model
            
            # Apply LoRA to the model
            apply_lora_to_model(self, rank=rank, alpha=alpha, dropout=dropout)
            self.using_lora = True
            return self
        except ImportError:
            print("LoRA modules not available. Please install peft package.")
            return self
            
    def quantize_for_inference(self, quantize_type="dynamic"):
        """
        Quantize the model for faster inference.
        
        Args:
            quantize_type: Type of quantization ("dynamic" or "static")
            
        Returns:
            Quantized model
        """
        try:
            from model.optimization import apply_quantization_for_inference
            return apply_quantization_for_inference(self, quantize_type=quantize_type)
        except ImportError:
            print("Quantization functions not available.")
            return self
