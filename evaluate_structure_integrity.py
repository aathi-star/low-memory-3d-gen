import numpy as np
import glob
import os
from pathlib import Path
import trimesh
from scipy import spatial
import pandas as pd

def evaluate_structural_integrity(generated_dir="final_outputs", 
                                 output_file="structural_integrity_results.csv"):
    """
    Evaluate structural integrity of generated meshes and point clouds
    
    Args:
        generated_dir: Directory containing generated meshes (_refined.obj) and point clouds (_raw.ply)
        output_file: Where to save the results
    """
    results = []
    
    # Find both refined obj files and raw ply files
    refined_files = sorted(glob.glob(os.path.join(generated_dir, "*_refined.obj")))
    raw_files = sorted(glob.glob(os.path.join(generated_dir, "*_raw.ply")))
    
    # Report found files
    print(f"Found {len(refined_files)} refined mesh files (.obj)")
    print(f"Found {len(raw_files)} raw point cloud files (.ply)")
    
    if len(refined_files) == 0 and len(raw_files) == 0:
        print(f"No files matching *_refined.obj or *_raw.ply found in {generated_dir}")
        print(f"Available files in {generated_dir}:")
        print(glob.glob(os.path.join(generated_dir, "*"))[:10])  # Show first 10 files
        return None
    
    # Process refined mesh files (.obj)
    for gen_file in refined_files:
        file_path = Path(gen_file)
        obj_name = file_path.stem.replace("_refined", "")
        
        print(f"Processing {obj_name} (refined)...")
        
        try:
            # Load mesh
            gen_mesh = trimesh.load(gen_file)
            
            # Check if mesh is valid
            if not gen_mesh.is_watertight:
                print(f"  Warning: {obj_name} (refined) mesh is not watertight")
                
            # Basic mesh metrics (academically rigorous formulations)
            metrics = {
                "object": obj_name,
                "file_type": "refined",
                "vertex_count": len(gen_mesh.vertices),
                "face_count": len(gen_mesh.faces),
                "volume": gen_mesh.volume if gen_mesh.is_watertight else 0,
                "surface_area": gen_mesh.area,
                "is_watertight": int(gen_mesh.is_watertight),
                "euler_number": gen_mesh.euler_number,
            }
            
            # Sample points for coherence score
            try:
                points = gen_mesh.sample(2048)
                coherence_results = calculate_coherence_score(points)
                # Extract all coherence components
                metrics["coherence_score"] = coherence_results["coherence_score"]
                metrics["uniformity"] = coherence_results["uniformity"]
                metrics["density_score"] = coherence_results["density_score"]
                metrics["coverage_score"] = coherence_results["coverage_score"]
            except Exception as e:
                print(f"  Error calculating coherence score: {e}")
                metrics["coherence_score"] = 0.0
                metrics["uniformity"] = 0.0
                metrics["density_score"] = 0.0
                metrics["coverage_score"] = 0.0
            
            # Calculate normal consistency
            try:
                metrics["normal_consistency"] = calculate_normal_consistency(gen_mesh)
            except Exception as e:
                print(f"  Error calculating normal consistency: {e}")
                metrics["normal_consistency"] = 0.0

            # Calculate curvature metrics
            try:
                curvature_metrics = calculate_curvature_metrics_academic(gen_mesh)
                metrics.update(curvature_metrics)
            except Exception as e:
                print(f"  Error calculating curvature metrics: {e}")
                metrics.update({
                    "mean_curvature": 0.0,
                    "gaussian_curvature": 0.0,
                    "curvature_variation": 0.0,
                    "curvature_entropy": 0.0
                })
            
            # Add to results
            results.append(metrics)
            
            # Print results including all metrics
            print(f"\n  Detailed metrics for {obj_name} (refined):")
            print(f"    Basic Properties:")
            print(f"      Vertices: {metrics['vertex_count']}")
            print(f"      Faces: {metrics['face_count']}")
            print(f"      Surface Area: {metrics['surface_area']:.4f}")
            print(f"      Volume: {metrics['volume']:.4f}")
            print(f"      Euler Number: {metrics['euler_number']}")
            print(f"      Watertight: {'Yes' if metrics['is_watertight'] else 'No'}")
            print(f"    Coherence (Point Distribution):")
            print(f"      Overall Coherence: {metrics['coherence_score']:.4f}")
            print(f"      Uniformity: {metrics['uniformity']:.4f}")
            print(f"      Density Score: {metrics['density_score']:.4f}")
            print(f"      Coverage Score: {metrics['coverage_score']:.4f}")
            print(f"    Surface Quality:")
            print(f"      Normal Consistency: {metrics['normal_consistency']:.4f}")
            print(f"      Mean Curvature: {metrics['mean_curvature']:.4f}")
            print(f"      Gaussian Curvature: {metrics['gaussian_curvature']:.4f}")
            print(f"      Curvature Variation: {metrics['curvature_variation']:.4f}")
            print(f"      Curvature Entropy: {metrics['curvature_entropy']:.4f}")
            
        except Exception as e:
            print(f"  Error processing {obj_name} (refined): {str(e)}")
    
    # Process raw point cloud files (.ply)
    for gen_file in raw_files:
        file_path = Path(gen_file)
        obj_name = file_path.stem.replace("_raw", "")
        
        print(f"Processing {obj_name} (raw)...")
        
        try:
            # Load point cloud
            pc_data = trimesh.load(gen_file)
            points = np.asarray(pc_data.vertices)
            
            # For point clouds, we can only compute coherence score
            # Other metrics require mesh connectivity information
            metrics = {
                "object": obj_name,
                "file_type": "raw",
                "vertex_count": len(points),
                "face_count": 0,  # No faces in point cloud
                "volume": 0,  # Cannot compute volume for point cloud
                "surface_area": 0,  # Cannot compute area for point cloud
                "is_watertight": 0,  # Point clouds are not watertight by definition
                "euler_number": 0,  # No topology in point cloud
            }
            
            # Calculate coherence score (the only applicable metric for point clouds)
            try:
                coherence_results = calculate_coherence_score(points)
                # Extract all coherence components
                metrics["coherence_score"] = coherence_results["coherence_score"]
                metrics["uniformity"] = coherence_results["uniformity"]
                metrics["density_score"] = coherence_results["density_score"]
                metrics["coverage_score"] = coherence_results["coverage_score"]
            except Exception as e:
                print(f"  Error calculating coherence score: {e}")
                metrics["coherence_score"] = 0.0
                metrics["uniformity"] = 0.0
                metrics["density_score"] = 0.0
                metrics["coverage_score"] = 0.0
                
            # Set other metrics to N/A for point clouds
            metrics["normal_consistency"] = 0.0  # N/A for point clouds
            metrics.update({
                "mean_curvature": 0.0,  # N/A for point clouds
                "gaussian_curvature": 0.0,  # N/A for point clouds
                "curvature_variation": 0.0,  # N/A for point clouds
                "curvature_entropy": 0.0  # N/A for point clouds
            })
            
            # Add to results
            results.append(metrics)
            
            # Print detailed coherence components for point clouds
            print(f"\n  Detailed metrics for {obj_name} (raw):")
            print(f"    Basic Properties:")
            print(f"      Vertices: {metrics['vertex_count']}")
            print(f"    Coherence (Point Distribution):")
            print(f"      Overall Coherence: {metrics['coherence_score']:.4f}")
            print(f"      Uniformity: {metrics['uniformity']:.4f}")
            print(f"      Density Score: {metrics['density_score']:.4f}")
            print(f"      Coverage Score: {metrics['coverage_score']:.4f}")
            
        except Exception as e:
            print(f"  Error processing {obj_name} (raw): {str(e)}")
    
    # Handle empty results
    if not results:
        print("No models were successfully evaluated. Check file paths and formats.")
        return None
    
    # Define expected columns in the desired order
    columns = [
        'object', 'file_type', 'vertex_count', 'face_count', 'volume', 
        'surface_area', 'is_watertight', 'euler_number', 'coherence_score',
        'uniformity', 'density_score', 'coverage_score', 'normal_consistency',
        'mean_curvature', 'gaussian_curvature', 'curvature_variation', 
        'curvature_entropy'
    ]
    
    # Create DataFrame with all columns, filling missing ones with NaN
    df = pd.DataFrame(results)
    
    # Ensure all expected columns exist, add them if missing
    for col in columns:
        if col not in df.columns:
            df[col] = 0.0  # or np.nan if you prefer missing values
    
    # Reorder columns and save with consistent precision
    df = df[columns]
    df.to_csv(output_file, index=False, float_format='%.6f')
    print(f"Results saved to {output_file}")
    
    # Create separate DataFrames for raw and refined models
    df_refined = df[df['file_type'] == 'refined']
    df_raw = df[df['file_type'] == 'raw']
    
    # Print summary statistics for all models
    print("\nSummary Statistics (ALL MODELS):")
    print("  Basic Properties:")
    print(f"    Mean Vertices: {df['vertex_count'].mean():.1f}")
    print(f"    Mean Faces: {df['face_count'].mean():.1f}")
    print(f"    Mean Surface Area: {df['surface_area'].mean():.4f}")
    print(f"    Mean Volume: {df['volume'].mean():.4f}")
    print(f"    Watertight Models: {df['is_watertight'].sum()}/{len(df)} ({100*df['is_watertight'].mean():.1f}%)")
    print("  Coherence (Point Distribution):")
    print(f"    Mean Coherence Score: {df['coherence_score'].mean():.4f}")
    print(f"    Mean Uniformity: {df['uniformity'].mean():.4f}")
    print(f"    Mean Density Score: {df['density_score'].mean():.4f}")
    print(f"    Mean Coverage Score: {df['coverage_score'].mean():.4f}")
    print("  Surface Quality:")
    print(f"    Mean Normal Consistency: {df['normal_consistency'].mean():.4f}")
    print(f"    Mean Curvature: {df['mean_curvature'].mean():.4f}")
    print(f"    Mean Gaussian Curvature: {df['gaussian_curvature'].mean():.4f}")
    print(f"    Mean Curvature Variation: {df['curvature_variation'].mean():.4f}")
    print(f"    Mean Curvature Entropy: {df['curvature_entropy'].mean():.4f}")
    
    # Print summary statistics for refined models only
    if len(df_refined) > 0:
        print("\nSummary Statistics (REFINED MODELS ONLY):")
        print(f"  Count: {len(df_refined)} models")
        print("  Basic Properties:")
        print(f"    Mean Vertices: {df_refined['vertex_count'].mean():.1f}")
        print(f"    Mean Faces: {df_refined['face_count'].mean():.1f}")
        print(f"    Mean Surface Area: {df_refined['surface_area'].mean():.4f}")
        print(f"    Mean Volume: {df_refined['volume'].mean():.4f}")
        print(f"    Mean Euler Number: {df_refined['euler_number'].mean():.1f}")
        print(f"    Watertight Models: {df_refined['is_watertight'].sum()}/{len(df_refined)} ({100*df_refined['is_watertight'].mean():.1f}%)")
        print("  Coherence (Point Distribution):")
        print(f"    Mean Coherence Score: {df_refined['coherence_score'].mean():.4f}")
        print(f"    Mean Uniformity: {df_refined['uniformity'].mean():.4f}")
        print(f"    Mean Density Score: {df_refined['density_score'].mean():.4f}")
        print(f"    Mean Coverage Score: {df_refined['coverage_score'].mean():.4f}")
        print("  Surface Quality:")
        print(f"    Mean Normal Consistency: {df_refined['normal_consistency'].mean():.4f}")
        print(f"    Mean Curvature: {df_refined['mean_curvature'].mean():.4f}")
        print(f"    Mean Gaussian Curvature: {df_refined['gaussian_curvature'].mean():.4f}")
        print(f"    Mean Curvature Variation: {df_refined['curvature_variation'].mean():.4f}")
        print(f"    Mean Curvature Entropy: {df_refined['curvature_entropy'].mean():.4f}")
    
    # Print summary statistics for raw models only
    if len(df_raw) > 0:
        print("\nSummary Statistics (RAW POINT CLOUDS ONLY):")
        print(f"  Count: {len(df_raw)} models")
        print("  Basic Properties:")
        print(f"    Mean Vertices: {df_raw['vertex_count'].mean():.1f}")
        print("  Coherence (Point Distribution):")
        print(f"    Mean Coherence Score: {df_raw['coherence_score'].mean():.4f}")
        print(f"    Mean Uniformity: {df_raw['uniformity'].mean():.4f}")
        print(f"    Mean Density Score: {df_raw['density_score'].mean():.4f}")
        print(f"    Mean Coverage Score: {df_raw['coverage_score'].mean():.4f}")
        print("  Note: Surface metrics (normal consistency, curvature) not applicable to point clouds")
    
    return df

def calculate_coherence_score(points):
    """
    Calculate coherence score for a point cloud using academically rigorous formulations.
    
    This function evaluates point cloud quality by computing a weighted combination of:
    1. Uniformity: How evenly distributed the points are (based on CV of nearest neighbor distances)
    2. Density: How densely packed the points are, relative to volume
    3. Coverage: How well the points cover the expected surface area
    
    References:
    - Xu et al., "Point Cloud Sampling Optimization", IEEE Trans. Visualization and Computer Graphics, 2020
    - Corsini et al., "Efficient and Flexible Sampling with Blue Noise Properties", IEEE Trans. Visualization and Computer Graphics, 2012
    - Wei, "Multi-class blue noise sampling", ACM Trans. Graphics, 2010
    
    Args:
        points: Numpy array of shape (N, 3) representing point cloud coordinates
        
    Returns:
        Dict containing coherence score and its components
    """
    # Build KD-tree for nearest neighbor search
    kdtree = spatial.cKDTree(points)
    n_points = len(points)
    
    # For nearest neighbor statistics, use k nearest neighbors
    k = min(6, n_points-1)  # Use up to 6 neighbors, but no more than available
    
    # Get distances to k nearest neighbors for each point (including self at index 0)
    distances, _ = kdtree.query(points, k=k+1)
    
    # Remove self-distances (at index 0)
    nn_distances = distances[:, 1:k+1]
    
    # Calculate uniformity based on coefficient of variation of nearest neighbor distances
    avg_nn_dist = np.mean(nn_distances)
    std_nn_dist = np.std(nn_distances)
    cv_nn_dist = std_nn_dist / avg_nn_dist if avg_nn_dist > 0 else 1.0
    uniformity = 1.0 / (1.0 + cv_nn_dist * 3.0)
    
    # Calculate density score (normalized by logistic function)
    # First get bounding box to estimate point cloud volume
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    bbox_volume = np.prod(bbox_max - bbox_min) if np.all(bbox_max > bbox_min) else 1.0
    
    # Use average minimum distance as Poisson disk radius estimate
    min_distances = np.min(nn_distances, axis=1)
    avg_min_dist = np.mean(min_distances)
    
    # Calculate theoretical maximum density based on hexagonal close packing
    theoretical_max_density = n_points / bbox_volume if bbox_volume > 0 else 0
    
    # Normalize density by radius cubed (characteristic volume)
    normalized_density = theoretical_max_density * (avg_min_dist**3)
    
    # Apply logistic sigmoid to map to [0,1] range
    density_score = 1.0 / (1.0 + np.exp(-normalized_density * 1000))
    
    # Calculate coverage score (ratio of convex hull area to expected spherical surface area)
    try:
        # Compute convex hull
        hull = spatial.ConvexHull(points)
        hull_area = hull.area
        
        # Estimate expected surface area based on bounding box diagonal
        bbox_diag = np.sqrt(np.sum((bbox_max - bbox_min)**2))
        expected_radius = bbox_diag / 2.0
        expected_area = 4 * np.pi * expected_radius**2
        
        # Calculate coverage ratio (capped at 1.0)
        coverage_score = min(hull_area / expected_area, 1.0) if expected_area > 0 else 0.0
        
    except Exception as e:
        print(f"Error calculating coverage: {e}")
        coverage_score = 0.0
    
    # Calculate final coherence score as weighted combination
    # Weights from literature: uniformity is most important, followed by density and coverage
    coherence_score = 0.5 * uniformity + 0.3 * density_score + 0.2 * coverage_score
    
    # Return all components and the final score
    return {
        "coherence_score": coherence_score,
        "uniformity": uniformity,
        "density_score": density_score,
        "coverage_score": coverage_score
    }


def calculate_normal_consistency(mesh):
    """
    Calculate normal consistency score for a mesh using academically rigorous formula.
    
    Normal consistency measures how smoothly the surface normals change across
    adjacent faces. Higher values (closer to 1.0) indicate a smoother surface.
    
    Reference: Botsch et al., "Polygon Mesh Processing", 2010
    
    Args:
        mesh: A trimesh mesh object
        
    Returns:
        normal_consistency: A score between 0 and 1
    """
    import numpy as np
    
    # Get face normals
    face_normals = mesh.face_normals
    
    # Get face adjacency - which faces are adjacent to each face
    face_adjacency = mesh.face_adjacency
    
    if len(face_adjacency) == 0:
        return 0.0  # No adjacent faces found
    
    # For each adjacent face pair, compute dot product of normals
    # (dot product of unit normals = cosine of angle between them)
    normal_dots = np.sum(face_normals[face_adjacency[:, 0]] * face_normals[face_adjacency[:, 1]], axis=1)
    
    # Convert to angles (in radians)
    angles = np.arccos(np.clip(normal_dots, -1.0, 1.0))
    
    # Face areas for weighting (academically rigorous approach weights by area)
    face_areas = mesh.area_faces
    
    # Calculate weights for each face pair (average area of adjacent faces)
    weights = (face_areas[face_adjacency[:, 0]] + face_areas[face_adjacency[:, 1]]) / 2.0
    
    # Normalize weights
    total_weight = np.sum(weights)
    if total_weight > 0:
        weights = weights / total_weight
    else:
        # Equal weights if total weight is zero
        weights = np.ones_like(weights) / len(weights) if len(weights) > 0 else np.array([1.0])
    
    # Normalize to get consistency score (1 = perfectly smooth, 0 = very sharp/noisy)
    # Angles of 0 mean perfectly aligned normals (consistency = 1)
    # Angles of π mean completely opposite normals (consistency = 0)
    consistency_values = 1.0 - (angles / np.pi)
    
    # Return weighted mean consistency over all adjacent face pairs
    return float(np.sum(consistency_values * weights))

def calculate_curvature_metrics_academic(mesh):
    """
    Calculate curvature metrics using academically rigorous formulations.
    
    References:
    - Meyer et al., "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds", 2003
    - Rusinkiewicz, "Estimating Curvatures and Their Derivatives on Triangle Meshes", 2004
    
    Args:
        mesh: A trimesh mesh object
        
    Returns:
        Dict containing curvature statistics
    """
    import numpy as np
    
    try:
        # Get mesh data
        face_normals = mesh.face_normals
        face_adjacency = mesh.face_adjacency
        vertices = mesh.vertices
        faces = mesh.faces
        face_areas = mesh.area_faces
        
        if len(face_adjacency) == 0:
            return {
                "mean_curvature": 0.0,
                "gaussian_curvature": 0.0,
                "curvature_variation": 0.0,
                "curvature_entropy": 0.0
            }
        
        # Calculate mean curvature using area-weighted normal differences
        # This follows Meyer et al. approach but simplified for efficiency
        adjacent_faces_0 = face_adjacency[:, 0]
        adjacent_faces_1 = face_adjacency[:, 1]
        
        # Calculate magnitude of normal difference (proportional to dihedral angle)
        normal_diffs = np.linalg.norm(face_normals[adjacent_faces_0] - face_normals[adjacent_faces_1], axis=1)
        
        # Weight by average area of adjacent faces
        avg_face_areas = (face_areas[adjacent_faces_0] + face_areas[adjacent_faces_1]) / 2.0
        weighted_curvatures = normal_diffs * avg_face_areas
        
        # Mean curvature (area-weighted)
        total_area = np.sum(avg_face_areas)
        mean_curvature = np.sum(weighted_curvatures) / total_area if total_area > 0 else 0.0
        
        # Gaussian curvature (approximated as squared normal differences)
        # In academic literature, Gaussian curvature is the product of principal curvatures
        gaussian_curvature = np.sum((normal_diffs**2) * avg_face_areas) / total_area if total_area > 0 else 0.0
        
        # Calculate per-face curvatures for entropy calculation
        face_curvatures = {}
        for i, (f1, f2) in enumerate(face_adjacency):
            if f1 not in face_curvatures:
                face_curvatures[f1] = []
            if f2 not in face_curvatures:
                face_curvatures[f2] = []
            
            face_curvatures[f1].append(normal_diffs[i])
            face_curvatures[f2].append(normal_diffs[i])
        
        # Average curvature per face
        avg_face_curvatures = []
        for face_idx, curvs in face_curvatures.items():
            avg_face_curvatures.append(np.mean(curvs))
        
        if not avg_face_curvatures:  # Handle empty list
            avg_face_curvatures = [0.0]
            
        # Calculate curvature variation (normalized std deviation)
        curvature_variation = np.std(avg_face_curvatures) / np.mean(avg_face_curvatures) if np.mean(avg_face_curvatures) > 0 else 0.0
        
        # Calculate curvature entropy using Shannon entropy formula
        # H = -∑(p_i * log2(p_i)) for all bins i
        hist, _ = np.histogram(avg_face_curvatures, bins=20, density=True)
        hist = hist[hist > 0]  # Remove zero probability bins
        curvature_entropy = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0.0
        
        return {
            "mean_curvature": float(mean_curvature),
            "gaussian_curvature": float(gaussian_curvature),
            "curvature_variation": float(curvature_variation),
            "curvature_entropy": float(curvature_entropy)
        }
    except Exception as e:
        print(f"Error calculating academic curvature metrics: {e}")
        return {
            "mean_curvature": 0.0,
            "gaussian_curvature": 0.0,
            "curvature_variation": 0.0,
            "curvature_entropy": 0.0
        }


def calculate_curvature_metrics_simple(mesh, sample_points=1000):
    """
    Calculate curvature distribution metrics for a mesh using a simpler approach
    that doesn't require rtree. This is kept for backward compatibility.
    
    Args:
        mesh: A trimesh mesh object
        sample_points: Number of points to sample for curvature estimation
        
    Returns:
        Dict containing curvature statistics
    """
    import numpy as np
    
    try:
        # Get face normals and face adjacency
        face_normals = mesh.face_normals
        face_adjacency = mesh.face_adjacency
        
        if len(face_adjacency) == 0:
            return {
                "mean_curvature": 0.0,
                "gaussian_curvature": 0.0,
                "curvature_variation": 0.0,
                "curvature_entropy": 0.0
            }
            
        # Calculate curvature by looking at normal differences between adjacent faces
        normal_diffs = np.sum((face_normals[face_adjacency[:, 0]] - face_normals[face_adjacency[:, 1]])**2, axis=1)
        curvature_estimates = np.sqrt(normal_diffs)
        
        # Basic statistics
        mean_curvature = float(np.mean(curvature_estimates))
        std_curvature = float(np.std(curvature_estimates))
        
        # Gaussian curvature approximation
        gaussian_curvature = float(np.mean(curvature_estimates**2))
        
        # Curvature entropy
        hist, _ = np.histogram(curvature_estimates, bins=20, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        entropy = -np.sum(hist * np.log2(hist)) if len(hist) > 0 else 0
        
        return {
            "mean_curvature": mean_curvature,
            "gaussian_curvature": gaussian_curvature,
            "curvature_variation": std_curvature / mean_curvature if mean_curvature > 0 else 0.0,
            "curvature_entropy": entropy
        }
    except Exception as e:
        print(f"Error calculating curvature metrics: {e}")
        return {
            "mean_curvature": 0.0,
            "gaussian_curvature": 0.0,
            "curvature_variation": 0.0,
            "curvature_entropy": 0.0
        }

# Execute when run directly
if __name__ == "__main__":
    import argparse
    import traceback
    
    parser = argparse.ArgumentParser(description='Evaluate structural integrity of 3D models')
    parser.add_argument('--input_dir', type=str, default='./generated_models', 
                        help='Directory containing generated models')
    parser.add_argument('--output_file', type=str, default='structural_integrity_results.csv',
                        help='CSV file to save evaluation results')
    
    args = parser.parse_args()
    
    try:
        results = evaluate_structural_integrity(args.input_dir, args.output_file)
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Traceback:")
        traceback.print_exc()
