import os
import yaml
from easydict import EasyDict as edict

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert to EasyDict for attribute access
    config = edict(config)
    
    # Set default values for required fields if not present
    if 'data' not in config:
        config.data = edict()
    if 'dataset_path' not in config.data:
        config.data.dataset_path = 'data'
    
    if 'model' not in config:
        config.model = edict()
    if 'num_points' not in config.model:
        config.model.num_points = 2048
    if 'use_halton' not in config.model:
        config.model.use_halton = True
    if 'use_graph_attention' not in config.model:
        config.model.use_graph_attention = True
    
    if 'training' not in config:
        config.training = edict()
    if 'batch_size' not in config.training:
        config.training.batch_size = 32
    if 'num_epochs' not in config.training:
        config.training.num_epochs = 100
    if 'learning_rate' not in config.training:
        config.training.learning_rate = 0.001
    if 'checkpoint_dir' not in config.training:
        config.training.checkpoint_dir = 'checkpoints'
    if 'resume_checkpoint' not in config.training:
        config.training.resume_checkpoint = None
    if 'log_interval' not in config.training:
        config.training.log_interval = 10
    if 'sample_interval' not in config.training:
        config.training.sample_interval = 5
    if 'num_workers' not in config.training:
        config.training.num_workers = 4
    if 'weight_decay' not in config.training:
        config.training.weight_decay = 1e-5
    if 'lr_step_size' not in config.training:
        config.training.lr_step_size = 30
    if 'lr_gamma' not in config.training:
        config.training.lr_gamma = 0.5
    if 'use_shape_prior' not in config.training:
        config.training.use_shape_prior = True
    if 'shape_prior_weight' not in config.training:
        config.training.shape_prior_weight = 0.1
    
    if 'generation' not in config:
        config.generation = edict()
    if 'output_dir' not in config.generation:
        config.generation.output_dir = 'output'
    if 'inference_steps' not in config.generation:
        config.generation.inference_steps = 50
    if 'temperature' not in config.generation:
        config.generation.temperature = 0.8
    
    if 'visualization' not in config:
        config.visualization = edict()
    if 'show_examples' not in config.visualization:
        config.visualization.show_examples = True
    
    # Add configurations for halton sampling if not present
    if 'halton' not in config:
        config.halton = edict()
    if 'scale' not in config.halton:
        config.halton.scale = 0.8
    if 'center' not in config.halton:
        config.halton.center = True
    if 'noise_scale' not in config.halton:
        config.halton.noise_scale = 0.05
    
    # Add configurations for graph attention if not present
    if 'graph_attention' not in config:
        config.graph_attention = edict()
    if 'num_heads' not in config.graph_attention:
        config.graph_attention.num_heads = 4
    if 'num_layers' not in config.graph_attention:
        config.graph_attention.num_layers = 3
    if 'hidden_dim' not in config.graph_attention:
        config.graph_attention.hidden_dim = 128
    if 'dropout' not in config.graph_attention:
        config.graph_attention.dropout = 0.1
    if 'k_neighbors' not in config.graph_attention:
        config.graph_attention.k_neighbors = 20
    
    return config


def save_config(config, config_path):
    """Save configuration to YAML file"""
    # Convert EasyDict to dict
    if isinstance(config, edict):
        config_dict = dict(config)
        for k, v in config_dict.items():
            if isinstance(v, edict):
                config_dict[k] = dict(v)
    else:
        config_dict = config
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Save to file
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)
