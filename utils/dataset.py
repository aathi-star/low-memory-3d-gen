import os
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset
import random
from transformers import BertTokenizer


class ModelNet10Dataset(Dataset):
    """
    Dataset class for ModelNet10 with text labels.
    """
    def __init__(self, data_dir, split='train', augment=False):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing processed dataset
            split: 'train' or 'test'
            augment: Whether to use data augmentation
        """
        self.data_dir = data_dir
        self.split = split
        self.augment = augment
        
        # Load H5 file with processed data
        h5_path = os.path.join(data_dir, 'modelnet10.h5')
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Dataset file {h5_path} not found. Run download_modelnet10.py first.")
        
        self.h5_file = h5py.File(h5_path, 'r')
        
        # Load category mapping
        mapping_path = os.path.join(data_dir, 'category_mapping.txt')
        self.categories = []
        self.label_to_category = {}
        
        with open(mapping_path, 'r') as f:
            for line in f:
                category, label = line.strip().split(',')
                self.categories.append(category)
                self.label_to_category[int(label)] = category
        
        # Get dataset keys
        self.data_keys = []
        self.labels = []
        
        for category in self.categories:
            key = f"{category}_{split}_pointclouds"
            if key in self.h5_file:
                num_samples = len(self.h5_file[key])
                self.data_keys.extend([(key, i) for i in range(num_samples)])
                label = self.categories.index(category)
                self.labels.extend([label] * num_samples)
        
        # Initialize tokenizer for text inputs
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        print(f"Loaded {len(self.data_keys)} {split} samples across {len(self.categories)} categories")

    def __len__(self):
        return len(self.data_keys)

    def __getitem__(self, idx):
        # Get point cloud
        key, sample_idx = self.data_keys[idx]
        point_cloud = self.h5_file[key][sample_idx]
        
        # Get label and category name
        label = self.labels[idx]
        category = self.label_to_category[label]
        
        # Create text prompt
        # We'll use a variety of text prompts for the same category to improve generalization
        prompts = [
            f"a {category}",
            f"3d model of {category}",
            f"{category} object",
            f"a {category} model",
            f"this is a {category}"
        ]
        text = random.choice(prompts)
        
        point_cloud = torch.from_numpy(point_cloud.astype(np.float32))
        
        # Apply data augmentation if enabled
        if self.augment:
            point_cloud = self._augment_point_cloud(point_cloud)
        
        return text, point_cloud, label

    def _augment_point_cloud(self, point_cloud):
        """Apply data augmentation to point cloud."""
        # Random rotation around z-axis
        theta = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = torch.tensor([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ], dtype=torch.float32)
        
        point_cloud = torch.matmul(point_cloud, rotation_matrix)
        
        # Random scaling
        scale = torch.tensor(np.random.uniform(0.8, 1.2), dtype=torch.float32)
        point_cloud = point_cloud * scale
        
        # Random jittering
        jitter = torch.randn_like(point_cloud) * 0.01
        point_cloud = point_cloud + jitter
        
        return point_cloud

    def category_name(self, label):
        """Convert integer label to category name."""
        return self.label_to_category[label]

    def get_all_categories(self):
        """Return list of all categories."""
        return self.categories

    def close(self):
        """Close H5 file."""
        self.h5_file.close()
