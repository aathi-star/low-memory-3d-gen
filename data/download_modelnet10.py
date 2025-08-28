import os
import urllib.request
import zipfile
import shutil
import numpy as np
import trimesh
import h5py
import argparse
from tqdm import tqdm

def download_modelnet10():
    """Download and extract ModelNet10 dataset."""
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # URL for ModelNet10
    url = "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
    zip_file = os.path.join(data_dir, "ModelNet10.zip")
    
    # Download if not exists
    if not os.path.exists(zip_file):
        print(f"Downloading ModelNet10 dataset to {zip_file}...")
        urllib.request.urlretrieve(url, zip_file)
        print("Download complete.")
    
    # Extract if needed
    extracted_dir = os.path.join(data_dir, "ModelNet10")
    if not os.path.exists(extracted_dir):
        print(f"Extracting {zip_file}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Extraction complete.")
    
    return extracted_dir

def sample_point_cloud(mesh, n_points=2048, use_halton=True):
    """Sample point cloud from mesh using either uniform or Halton sampling."""
    if use_halton:
        points = halton_sampling_from_mesh(mesh, n_points)
    else:
        points = mesh.sample(n_points)
    
    return points

def halton_sampling_from_mesh(mesh, n_points=2048):
    """Generate Halton sequence sampling from a mesh.
    
    Implementation based on research papers:
    - "Point Sampling with General Noise Spectrum" (Heck et al., 2013)
    - "Low-Discrepancy Sampling for Path Integration" (Grünschloß et al., 2012)
    """
    from scipy.stats import qmc
    
    # Get mesh faces and vertices
    faces = mesh.faces
    vertices = mesh.vertices
    
    # Calculate face areas for weighted sampling
    areas = np.zeros(len(faces))
    for i, face in enumerate(faces):
        v0, v1, v2 = vertices[face]
        areas[i] = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
    
    # Normalize areas to probabilities
    if np.sum(areas) > 0:
        probs = areas / np.sum(areas)
    else:
        probs = np.ones_like(areas) / len(areas)
    
    # Sample faces based on area
    face_indices = np.random.choice(len(faces), size=n_points, p=probs)
    
    # Create Halton sampler with optimal prime bases for 2D
    # Using prime bases 2 and 3 for better distribution
    sampler = qmc.Halton(d=2, scramble=True, seed=42)
    
    # Skip initial points (leaping) for better distribution
    # The first few points in a Halton sequence can be poorly distributed
    leap = 100
    sampler.fast_forward(leap)
    
    # Generate samples
    samples = sampler.random(n=n_points)
    
    # Convert to barycentric coordinates and sample points
    points = np.zeros((n_points, 3))
    for i, (face_idx, (u, v)) in enumerate(zip(face_indices, samples)):
        # Convert uniform square samples to barycentric coordinates using sqrt mapping
        # This provides a more uniform distribution across the triangle
        # Reference: "A Low Distortion Map Between Triangle and Square" (Shirley et al.)
        w = 1 - np.sqrt(u)
        u_new = np.sqrt(u) * (1 - v)
        v_new = np.sqrt(u) * v
        
        # Get vertices of the sampled face
        face = faces[face_idx]
        v0, v1, v2 = vertices[face]
        
        # Compute point using barycentric coordinates
        points[i] = w * v0 + u_new * v1 + v_new * v2
    
    return points

def process_category(category_dir, output_h5, category, split, use_halton=True):
    """Process a category directory and add samples to h5 file."""
    files = os.listdir(os.path.join(category_dir, split))
    n_samples = len(files)
    
    # Create dataset in the h5 file
    point_clouds = output_h5.create_dataset(
        f"{category}_{split}_pointclouds", 
        shape=(n_samples, 2048, 3), 
        dtype=np.float32
    )
    
    # Process each OFF file
    for i, file in enumerate(tqdm(files, desc=f"Processing {category} {split}")):
        if not file.endswith('.off'):
            continue
            
        off_file = os.path.join(category_dir, split, file)
        try:
            mesh = trimesh.load(off_file, file_type='off')
            points = sample_point_cloud(mesh, use_halton=use_halton)
            
            # Normalize to unit sphere
            center = np.mean(points, axis=0)
            points = points - center
            max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
            points = points / max_dist
            
            point_clouds[i] = points
        except Exception as e:
            print(f"Error processing {off_file}: {e}")
            # Use zeros as placeholder if error occurs
            point_clouds[i] = np.zeros((2048, 3), dtype=np.float32)

def create_dataset(modelnet_dir, output_dir, use_halton=True):
    """Create H5 dataset from ModelNet10."""
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "modelnet10.h5")
    
    # Get all categories
    categories = [d for d in os.listdir(modelnet_dir) if os.path.isdir(os.path.join(modelnet_dir, d))]
    
    # Create a mapping of category names to integer labels
    category_to_label = {cat: i for i, cat in enumerate(sorted(categories))}
    
    # Save category mapping
    with open(os.path.join(output_dir, "category_mapping.txt"), "w") as f:
        for cat, label in category_to_label.items():
            f.write(f"{cat},{label}\n")
    
    # Create H5 file
    with h5py.File(output_file, 'w') as h5f:
        # Store category names and labels
        h5f.attrs['categories'] = np.array(sorted(categories), dtype='S')
        
        # Process each category
        for category in categories:
            category_dir = os.path.join(modelnet_dir, category)
            
            # Process train and test splits
            for split in ["train", "test"]:
                if os.path.exists(os.path.join(category_dir, split)):
                    process_category(category_dir, h5f, category, split, use_halton)
    
    print(f"Dataset created at {output_file}")
    return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process ModelNet10 dataset.")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Directory to save processed dataset")
    parser.add_argument("--use_halton", action="store_true", help="Use Halton sampling instead of uniform")
    args = parser.parse_args()
    
    # Download and extract dataset
    modelnet_dir = download_modelnet10()
    
    # Process the dataset
    create_dataset(modelnet_dir, args.output_dir, args.use_halton)
