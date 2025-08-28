import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetEncoder(nn.Module):
    """PointNet-based encoder for point cloud feature extraction.
    Compatible with Python 3.12.
    """
    def __init__(self, input_channels=3, global_feat=True, feature_transform=True, channel_multiplier=1):
        super(PointNetEncoder, self).__init__()
        self.input_channels = input_channels
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        c = channel_multiplier
        
        # Basic convolutional layers
        self.conv1 = nn.Conv1d(input_channels, 64 * c, 1)
        self.conv2 = nn.Conv1d(64 * c, 128 * c, 1)
        self.conv3 = nn.Conv1d(128 * c, 256 * c, 1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(64 * c)
        self.bn2 = nn.BatchNorm1d(128 * c)
        self.bn3 = nn.BatchNorm1d(256 * c)
        
        # Feature transform layers
        if self.feature_transform:
            self.fstn = STNkd(k=64 * c)
        
        # For input transform
        self.stn = STNkd(k=input_channels)
    
    def forward(self, x):
        # Input shape: batch_size x num_points x input_channels
        # Convert to: batch_size x input_channels x num_points
        if x.dim() == 3 and x.size(2) == self.input_channels:
            x = x.transpose(1, 2)
        
        batch_size = x.size(0)
        num_points = x.size(2)
        
        # Input transform
        trans = self.stn(x)
        x = torch.bmm(trans, x)
        
        # MLP with pointwise convolutions
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Feature transform if enabled
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = torch.bmm(trans_feat, x)
        else:
            trans_feat = None
        
        # Additional layers
        point_feat = x  # Save point features for local features
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        # Global max pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 256 * channel_multiplier)
        
        # For global features, return the global feature
        # For segmentation, also return point features
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 256 * channel_multiplier, 1).repeat(1, 1, num_points)
            return torch.cat([x, point_feat], 1), trans, trans_feat


class STNkd(nn.Module):
    """Spatial Transformer Network for k dimensions."""
    def __init__(self, k=3):
        super(STNkd, self).__init__()
        self.k = k
        
        # Shared MLP for feature extraction
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        # Regressor network
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global pooling
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        
        # Regressor
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        
        # Initialize as identity transformation
        identity = torch.eye(self.k, requires_grad=True).repeat(batch_size, 1, 1)
        if x.is_cuda:
            identity = identity.cuda()
        
        x = x.view(-1, self.k, self.k) + identity
        return x


class PointNet2Encoder(nn.Module):
    """PointNet++ encoder for point cloud feature extraction.
    Simplified version compatible with Python 3.12.
    """
    def __init__(self, input_channels=3, use_xyz=True):
        super(PointNet2Encoder, self).__init__()
        self.input_channels = input_channels
        self.use_xyz = use_xyz
        
        # Simplified architecture without requiring complex operations
        # Simulating set abstraction with standard PyTorch operations
        self.mlp1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        self.mlp2 = nn.Sequential(
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        
        # Final MLP for feature aggregation
        self.final_mlp = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
    
    def forward(self, x):
        # Input shape: batch_size x num_points x input_channels
        # Convert to: batch_size x input_channels x num_points
        if x.dim() == 3 and x.size(2) == self.input_channels:
            x = x.transpose(1, 2)
        
        # First level processing
        features = self.mlp1(x)
        
        # Second level processing with subsampling
        # Simple max pooling to simulate grouping and sampling
        features = self.mlp2(features)
        
        # Global features
        global_features = torch.max(features, dim=2)[0]  # Max pooling along points
        
        # Feature transformation
        global_features = self.final_mlp(global_features)
        
        return global_features
