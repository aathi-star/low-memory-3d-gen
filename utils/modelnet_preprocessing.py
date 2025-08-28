import os
import numpy as np
import torch
import glob
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

def normalize_point_cloud(points):
    """
    Normalize point cloud to unit cube centered at origin
    """
    # Center at centroid
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    # Scale to unit cube
    max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
    points = points / max_dist
    
    return points

def remove_outliers(points, k=5, std_ratio=2.0):
    """
    Remove outlier points that are far from their neighbors
    """
    # Find distances to k nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    distances, _ = nbrs.kneighbors(points)
    
    # Calculate average distance for each point
    avg_distances = np.mean(distances, axis=1)
    
    # Find points within standard deviation threshold
    mean_dist = np.mean(avg_distances)
    std_dist = np.std(avg_distances)
    threshold = mean_dist + std_ratio * std_dist
    mask = avg_distances < threshold
    
    return points[mask]

def uniform_sample(points, n_samples):
    """
    Sample points with uniform density using farthest point sampling
    """
    if len(points) <= n_samples:
        return points
    
    # Start with a random point
    sampled_indices = [np.random.randint(0, len(points))]
    sampled_points = [points[sampled_indices[0]]]
    
    # Compute distances to this point
    dists = np.sum((points - sampled_points[0])**2, axis=1)
    
    # Iteratively add the farthest point
    for i in range(1, n_samples):
        # Select the point with the largest minimum distance to sampled points
        new_idx = np.argmax(dists)
        sampled_indices.append(new_idx)
        sampled_points.append(points[new_idx])
        
        # Update distances
        new_dists = np.sum((points - sampled_points[-1])**2, axis=1)
        dists = np.minimum(dists, new_dists)
    
    return np.array(sampled_points)

def add_surface_normals(points, k=30):
    """
    Estimate surface normals for each point
    Returns points with normals [x, y, z, nx, ny, nz]
    """
    normals = np.zeros_like(points)
    
    # Compute normals using PCA on local neighborhoods
    nbrs = NearestNeighbors(n_neighbors=k).fit(points)
    _, indices = nbrs.kneighbors(points)
    
    for i in range(len(points)):
        # Get neighborhood
        neighbors = points[indices[i]]
        
        # Center points
        centered = neighbors - np.mean(neighbors, axis=0)
        
        # Compute covariance matrix
        cov = np.cov(centered, rowvar=False)
        
        # Compute eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Smallest eigenvector approximates the normal
        normals[i] = eigenvectors[:, 0]
    
    # Ensure consistent normal orientation (pointing outward)
    centroid = np.mean(points, axis=0)
    for i in range(len(points)):
        direction = points[i] - centroid
        if np.dot(direction, normals[i]) < 0:
            normals[i] = -normals[i]
    
    # Combine points and normals
    return np.hstack([points, normals])

def enhance_modelnet10_dataset(data_dir, output_dir, num_points=2048, add_normals=True):
    """
    Preprocess ModelNet10 dataset for better training results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all OFF files
    all_files = []
    for category in os.listdir(data_dir):
        category_dir = os.path.join(data_dir, category)
        if os.path.isdir(category_dir):
            off_files = glob.glob(os.path.join(category_dir, '*/*.off'))
            all_files.extend([(f, category) for f in off_files])
    
    print(f"Found {len(all_files)} files to process")
    
    # Process each file
    for file_path, category in tqdm(all_files):
        # Extract file name
        file_name = os.path.basename(file_path).replace('.off', '')
        
        # Create category directory in output
        category_dir = os.path.join(output_dir, category)
        os.makedirs(category_dir, exist_ok=True)
        
        # Read OFF file
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Skip header
            if lines[0].strip() == 'OFF':
                start_idx = 1
            else:
                start_idx = 0
            
            # Get number of vertices and faces
            parts = lines[start_idx].strip().split()
            n_vertices = int(parts[0])
            
            # Read vertices
            vertices = []
            for i in range(start_idx + 1, start_idx + 1 + n_vertices):
                vertex = list(map(float, lines[i].strip().split()))
                vertices.append(vertex)
            
            # Convert to numpy array
            points = np.array(vertices)
            
            # Clean and normalize point cloud
            points = remove_outliers(points)
            points = normalize_point_cloud(points)
            
            # Sample uniformly
            points = uniform_sample(points, num_points)
            
            # Add surface normals if requested
            if add_normals:
                points_with_normals = add_surface_normals(points)
                output_data = points_with_normals
            else:
                output_data = points
            
            # Save as NPY file
            output_path = os.path.join(category_dir, f"{file_name}.npy")
            np.save(output_path, output_data)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"Enhanced dataset saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhance ModelNet10 dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to ModelNet10 dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_points", type=int, default=2048, help="Number of points per cloud")
    parser.add_argument("--add_normals", action="store_true", help="Add surface normals to points")
    
    args = parser.parse_args()
    enhance_modelnet10_dataset(args.data_dir, args.output_dir, args.num_points, args.add_normals)
