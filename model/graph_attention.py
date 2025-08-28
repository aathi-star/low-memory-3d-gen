import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """
    Graph attention layer for point cloud processing
    Enables better structural understanding of ModelNet10 shapes
    """
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # Transform matrices
        self.W = nn.Parameter(torch.zeros(in_features, out_features))
        nn.init.xavier_uniform_(self.W.data)
        
        # Attention parameters
        self.a = nn.Parameter(torch.zeros(2*out_features, 1))
        nn.init.xavier_uniform_(self.a.data)
        
        # Learnable parameters for positional encoding
        self.pos_mlp = nn.Sequential(
            nn.Linear(3, out_features),
            nn.ReLU()
        )
        
        # Leaky ReLU for attention
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, x, pos, adj=None, k=20):
        """
        Forward pass with dynamic graph construction
        Args:
            x: Node features [B, N, F]
            pos: Node positions [B, N, 3]
            adj: Optional adjacency matrix [B, N, N]
            k: Number of nearest neighbors for dynamic graph
        Returns:
            Updated node features [B, N, F']
        """
        batch_size, num_nodes, _ = x.shape
        device = x.device
        
        # Transform node features
        Wh = torch.matmul(x, self.W)  # [B, N, F']
        
        # Generate adjacency matrix from positions if not provided
        if adj is None:
            # Compute pairwise distances
            dist = torch.cdist(pos, pos)  # [B, N, N]
            
            # Get k nearest neighbors
            _, indices = torch.topk(dist, k=k+1, dim=2, largest=False)  # [B, N, k+1]
            indices = indices[:, :, 1:]  # Remove self-loop (first neighbor is the node itself)
            
            # Create adjacency matrix (sparse format for efficiency)
            adj = torch.zeros(batch_size, num_nodes, num_nodes, device=device)
            for b in range(batch_size):
                for i in range(num_nodes):
                    adj[b, i, indices[b, i]] = 1.0
        
        # Prepare for attention
        Wh1 = Wh.unsqueeze(2).repeat(1, 1, num_nodes, 1)  # [B, N, N, F']
        Wh2 = Wh.unsqueeze(1).repeat(1, num_nodes, 1, 1)  # [B, N, N, F']
        
        # Concatenate features for attention computation
        concat_features = torch.cat([Wh1, Wh2], dim=3)  # [B, N, N, 2*F']
        
        # Compute attention coefficients
        e = self.leakyrelu(torch.matmul(concat_features, self.a)).squeeze(3)  # [B, N, N]
        
        # Apply mask and softmax
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)  # [B, N, N]
        attention = F.softmax(attention, dim=2)  # [B, N, N]
        attention = F.dropout(attention, self.dropout, training=self.training)  # [B, N, N]
        
        # Apply attention to features
        h_prime = torch.matmul(attention, Wh)  # [B, N, F']
        
        # Add positional encoding
        pos_encoding = self.pos_mlp(pos)
        h_prime = h_prime + pos_encoding
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class MultiHeadGraphAttention(nn.Module):
    """
    Multi-head graph attention for capturing different aspects of shape structure
    Critical for exact ModelNet10 resemblance
    """
    def __init__(self, in_features, hidden_features, out_features, num_heads=4, dropout=0.1):
        super(MultiHeadGraphAttention, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Multiple attention heads
        self.attention_heads = nn.ModuleList([
            GraphAttentionLayer(
                in_features, 
                hidden_features // num_heads, 
                dropout=dropout, 
                concat=True
            ) for _ in range(num_heads)
        ])
        
        # Output projection
        self.out_proj = nn.Linear(hidden_features, out_features)
        
        # Skip connection
        self.skip = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_features)
        self.layer_norm2 = nn.LayerNorm(out_features)
        
        # MLP for post-processing
        self.mlp = nn.Sequential(
            nn.Linear(out_features, out_features * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_features * 2, out_features)
        )
    
    def forward(self, x, pos, adj=None, k=20):
        """
        Forward pass with multiple attention heads
        Args:
            x: Node features [B, N, F]
            pos: Node positions [B, N, 3]
            adj: Optional adjacency matrix [B, N, N]
            k: Number of nearest neighbors for dynamic graph
        Returns:
            Updated node features [B, N, F']
        """
        # Apply each attention head
        head_outputs = []
        for head in self.attention_heads:
            head_outputs.append(head(x, pos, adj, k))
        
        # Concatenate head outputs
        multi_head = torch.cat(head_outputs, dim=2)
        multi_head = self.layer_norm1(multi_head)
        
        # Apply output projection
        output = self.out_proj(multi_head)
        
        # Add skip connection
        output = output + self.skip(x)
        output = self.layer_norm2(output)
        
        # Apply MLP
        output = output + self.mlp(output)
        
        return output

class PointCloudGraphNetwork(nn.Module):
    """
    Full graph network for point cloud processing with Halton sampling integration
    Specialized for exact ModelNet10 resemblance
    """
    def __init__(self, input_dim=3, output_dim=3, hidden_dim=64, num_layers=3, num_heads=4, dropout=0.1):
        super(PointCloudGraphNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Initial feature embedding
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Graph attention layers
        self.graph_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.graph_layers.append(
                MultiHeadGraphAttention(
                    hidden_dim, 
                    hidden_dim * 2,
                    hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout
                )
            )
        
        # Final output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
        # Global feature aggregation
        self.global_features = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, points, features=None, k=20):
        """
        Process point cloud through graph network
        Args:
            points: Point positions [B, N, 3]
            features: Optional input features [B, N, F]
            k: Number of nearest neighbors
        Returns:
            Updated points [B, N, 3], global features [B, hidden_dim]
        """
        batch_size, num_points, _ = points.shape
        
        # Initial features (use provided features or positions)
        if features is None:
            x = self.embedding(points)
        else:
            x = self.embedding(torch.cat([points, features], dim=2))
        
        # Process through graph layers
        for layer in self.graph_layers:
            x = layer(x, points, adj=None, k=k)
        
        # Generate output points
        output_points = self.output_layer(x)
        
        # Global feature pooling (max and mean)
        global_max = torch.max(x, dim=1)[0]
        global_avg = torch.mean(x, dim=1)
        global_features = self.global_features(global_max + global_avg)
        
        return output_points, global_features

class ModelNet10GraphEncoder(nn.Module):
    """
    Graph-based encoder for ModelNet10 shape understanding
    Captures category-specific geometric structure
    """
    def __init__(self, hidden_dim=128, latent_dim=256, num_layers=3, num_heads=4, dropout=0.1):
        super(ModelNet10GraphEncoder, self).__init__()
        
        # Point cloud graph network
        self.graph_network = PointCloudGraphNetwork(
            input_dim=3,
            output_dim=3,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Latent space mapping
        self.latent_mapping = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, latent_dim)
        )
        
        # Category-specific adapter
        self.category_adapter = nn.Parameter(torch.randn(10, latent_dim))
    
    def forward(self, points, category_ids=None):
        """
        Encode point cloud into latent representation
        Args:
            points: Point cloud [B, N, 3]
            category_ids: Optional category IDs [B]
        Returns:
            Latent representation [B, latent_dim]
        """
        # Process through graph network
        _, global_features = self.graph_network(points)
        
        # Map to latent space
        latent = self.latent_mapping(global_features)
        
        # Apply category-specific adaptation if provided
        if category_ids is not None:
            category_embeddings = self.category_adapter[category_ids]
            latent = latent + category_embeddings
        
        return latent

class ModelNet10GraphDecoder(nn.Module):
    """
    Graph-based decoder for ModelNet10 shape generation
    Uses Halton sampling for better point distribution
    """
    def __init__(self, latent_dim=256, hidden_dim=128, num_points=2048, num_layers=3, num_heads=4, dropout=0.1):
        super(ModelNet10GraphDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_points = num_points
        
        # Latent to initial points
        self.initial_points = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_points * 3)
        )
        
        # Point cloud refinement network
        self.refinement_network = PointCloudGraphNetwork(
            input_dim=3 + latent_dim,  # Points + latent
            output_dim=3,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Import Halton sampler here to avoid circular imports
        from utils.halton_sampling import HaltonSampler
        self.halton_sampler = HaltonSampler(dim=3, scale=0.8, center=True)
    
    def forward(self, latent, template=None, num_refinement_steps=2):
        """
        Decode latent representation into point cloud
        Args:
            latent: Latent representation [B, latent_dim]
            template: Optional template point cloud [B, N, 3]
            num_refinement_steps: Number of refinement steps
        Returns:
            Generated point cloud [B, num_points, 3]
        """
        batch_size = latent.shape[0]
        device = latent.device
        
        # Generate initial points
        if template is None:
            # Use Halton sampling for better distribution of initial points
            halton_points = self.halton_sampler.sample(self.num_points).to(device)
            halton_points = halton_points.unsqueeze(0).repeat(batch_size, 1, 1)
            
            # Transform halton points based on latent
            initial_offsets = self.initial_points(latent).view(batch_size, self.num_points, 3)
            points = halton_points + 0.2 * initial_offsets
        else:
            # Use template but perturb with Halton sequence for better diversity
            points = template.clone()
            points = self.halton_sampler.perturb_points(points, noise_scale=0.1)
        
        # Broadcast latent to all points
        latent_expanded = latent.unsqueeze(1).expand(-1, self.num_points, -1)
        
        # Refinement steps
        for _ in range(num_refinement_steps):
            # Combine points with latent
            point_features = torch.cat([points, latent_expanded], dim=2)
            
            # Refine through graph network
            refined_points, _ = self.refinement_network(points, point_features)
            
            # Residual update
            points = points + refined_points
            
            # Re-normalize to unit cube after each step
            points_max, _ = torch.max(torch.abs(points), dim=1, keepdim=True)
            points = points / torch.max(points_max, torch.tensor(1e-6, device=device))
        
        return points
