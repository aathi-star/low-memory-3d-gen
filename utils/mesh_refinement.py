import numpy as np
import trimesh
import torch
from scipy.spatial import Delaunay
import scipy.ndimage
from collections import Counter

def poisson_surface_reconstruction(points, depth=8, scale=1.0, samples_per_node=1.0):
    """
    Perform Poisson surface reconstruction using scipy Delaunay triangulation and trimesh.
    This is a simplified version as the true Poisson reconstruction requires more complex libraries.
    
    Args:
        points: [N, 3] array of point positions
        depth: Octree depth for reconstruction detail (higher = more detail)
        scale: Scale factor for the point cloud
        samples_per_node: Samples per octree node (higher = smoother)
        
    Returns:
        trimesh.Trimesh: Reconstructed mesh
    """
    print(f"Running Poisson surface reconstruction with depth {depth}...")
    
    # Ensure points are numpy array
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
        
    # Scale points if necessary
    if scale != 1.0:
        points = points * scale
    
    # Estimate normals for the point cloud - critical for Poisson reconstruction
    # Since we don't have true normals, we'll use a simple approach
    # to estimate them from local neighborhoods
    pc = trimesh.PointCloud(points)
    
    # Create a KDTree for nearest neighbor search
    from scipy.spatial import cKDTree
    kdtree = cKDTree(points)
    
    # Get nearest neighbors for each point
    k = min(30, len(points) - 1)  # Number of neighbors to consider
    distances, indices = kdtree.query(points, k=k)
    
    # Compute normals using PCA on local neighborhoods
    normals = np.zeros_like(points)
    for i in range(len(points)):
        # Get neighbors
        neighbors = points[indices[i]]
        # Center the neighbors
        centered = neighbors - neighbors.mean(axis=0)
        # Compute covariance matrix
        cov = np.dot(centered.T, centered)
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # The eigenvector with the smallest eigenvalue is the normal
        normal = eigenvectors[:, 0]
        normals[i] = normal / np.linalg.norm(normal)
    
    # Ensure consistent normal orientation (all pointing outward)
    center = points.mean(axis=0)
    for i in range(len(normals)):
        if np.dot(points[i] - center, normals[i]) < 0:
            normals[i] = -normals[i]
    
    # Based on depth parameter, determine sampling density
    # Higher depth means we need more points for accurate reconstruction
    target_points = min(len(points), 2**(depth+2))
    if len(points) > target_points:
        # Subsample points for efficiency
        indices = np.random.choice(len(points), target_points, replace=False)
        points = points[indices]
        normals = normals[indices]
    
    print(f"Using {len(points)} points for Poisson reconstruction")
    
    # Create a Delaunay triangulation as a simplified Poisson reconstruction
    try:
        # Add points slightly offset along their normals to create volume
        offset = 0.01 * scale  # Small offset
        points_with_offset = np.vstack([
            points,
            points + normals * offset
        ])
        
        # Create a Delaunay triangulation
        tri = Delaunay(points_with_offset)
        
        # Extract the mesh faces
        faces = tri.simplices
        
        # Create the mesh
        mesh = trimesh.Trimesh(vertices=points_with_offset, faces=faces)
        
        # Remove duplicate faces and vertices
        mesh = mesh.process()
        
        print(f"Initial mesh created with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces")
        
        # Check if mesh is valid
        if not mesh.is_watertight:
            print("Warning: Mesh is not watertight, attempting to repair")
            # Try to fix non-watertight mesh
            # mesh.fill_holes() returns a boolean indicating success, not a new mesh
            success = mesh.fill_holes()
            if success:
                print("Successfully filled holes in mesh")
            else:
                print("Could not fill all holes automatically")
        
        return mesh
    
    except Exception as e:
        print(f"Poisson reconstruction failed: {e}")
        # Fallback to alpha shape if Poisson fails
        return alpha_shape(points, alpha=1.0)

def point_cloud_to_mesh(points, method='ball_pivot', max_hole_size=0.05, poisson_depth=8, alpha_value=0.1, smooth_iterations=0, smooth_lambda=0.5, simplify_fraction=None, fill_hole_size=None):
    """
    Convert a point cloud to a mesh using one of several methods.
    
    Args:
        points: [N, 3] array of point positions
        method: Method to use ('ball_pivot', 'alpha_shape', 'poisson', or 'convex_hull')
        max_hole_size: Maximum size of holes to fill (for post-processing)
        poisson_depth: Octree depth for Poisson reconstruction (higher = more detail)
        alpha_value: Alpha value for alpha shape reconstruction (smaller = more detailed)
        smooth_iterations: Number of smoothing iterations
        smooth_lambda: Lambda value for Laplacian smoothing
        simplify_fraction: Fraction of faces to keep when simplifying
        fill_hole_size: Maximum size of holes to fill
        
    Returns:
        trimesh.Trimesh: Reconstructed mesh or point cloud if conversion fails
    """
    # Ensure points are numpy array
    if isinstance(points, torch.Tensor):
        points = points.detach().cpu().numpy()
    
    # Choose reconstruction method
    try:
        if method == 'alpha_shape':
            mesh = alpha_shape(points, alpha=alpha_value)
        elif method == 'poisson':
            mesh = poisson_surface_reconstruction(points, depth=poisson_depth)
        elif method == 'ball_pivot':
            # Use convex hull as fallback since ball pivoting needs Open3D
            mesh = simple_convex_hull(points)
            print("Note: Ball pivoting requires Open3D; using convex hull instead")
        else:  # convex_hull
            mesh = simple_convex_hull(points)
            
        # Apply basic smoothing if requested
        if smooth_iterations > 0:
            mesh = simple_smooth_mesh(mesh, iterations=smooth_iterations)
            
        return mesh
    except Exception as e:
        print(f"Mesh reconstruction failed: {e}")
        # Return a point cloud as fallback
        return trimesh.PointCloud(points)


def simple_convex_hull(points):
    """
    Create a simple convex hull mesh from points using SciPy's Delaunay triangulation.
    
    Args:
        points: [N, 3] array of point positions
        
    Returns:
        trimesh.Trimesh: Reconstructed mesh
    """
    # Use Delaunay triangulation to create a convex hull
    try:
        hull = Delaunay(points)
        # Convert to trimesh and remove duplicate faces
        mesh = trimesh.Trimesh(vertices=points, faces=hull.simplices)
        mesh.remove_duplicate_faces()
        return mesh
    except Exception as e:
        print(f"Warning: Could not create convex hull: {e}")
        # Fallback to creating a point cloud trimesh
        return trimesh.PointCloud(points)

def alpha_shape(points, alpha=0.5):
    """
    Create mesh using alpha shapes algorithm, good for concave objects.
    
    Args:
        points: [N, 3] array of point positions
        alpha: Alpha value, controls level of detail (smaller = more detail)
        
    Returns:
        trimesh.Trimesh: Reconstructed mesh
    """
    # Compute Delaunay triangulation
    try:
        tri = Delaunay(points)
        
        # Get all tetrahedrons in the triangulation
        tetrahedrons = points[tri.simplices]
        
        # Compute circumradius for each tetrahedron
        circumcenters = np.zeros((len(tetrahedrons), 3))
        radii = np.zeros(len(tetrahedrons))
        
        for i, tetra in enumerate(tetrahedrons):
            # Use circumsphere computation
            a, b, c, d = tetra
            A = np.array([
                [2*(b[0]-a[0]), 2*(b[1]-a[1]), 2*(b[2]-a[2])],
                [2*(c[0]-a[0]), 2*(c[1]-a[1]), 2*(c[2]-a[2])],
                [2*(d[0]-a[0]), 2*(d[1]-a[1]), 2*(d[2]-a[2])]
            ])
            
            b_sq = np.sum(b**2) - np.sum(a**2)
            c_sq = np.sum(c**2) - np.sum(a**2)
            d_sq = np.sum(d**2) - np.sum(a**2)
            
            b_vec = np.array([b_sq, c_sq, d_sq])
            
            try:
                center_relative = np.linalg.solve(A, b_vec)
                circumcenters[i] = center_relative + a
                radii[i] = np.linalg.norm(circumcenters[i] - a)
            except np.linalg.LinAlgError:
                # Skip degenerate cases
                radii[i] = np.inf
        
        # Filter tetrahedrons by alpha value
        valid_tetra = radii < (1.0/alpha)
        valid_simplices = tri.simplices[valid_tetra]
        
        # Extract boundary faces
        faces_list = []
        for tetra in valid_simplices:
            for i in range(4):
                face = np.delete(tetra, i)
                # Sort faces for easier duplicate detection
                face = np.sort(face)
                faces_list.append(tuple(face))
        
        # Count occurrences of each face
        face_counts = Counter(faces_list)
        
        # Keep only faces that appear exactly once (boundary faces)
        boundary_faces = [face for face, count in face_counts.items() if count == 1]
        boundary_faces = np.array(boundary_faces)
        
        # Create mesh with boundary faces
        if len(boundary_faces) > 0:
            return trimesh.Trimesh(vertices=points, faces=boundary_faces)
        else:
            # Fallback to convex hull
            return simple_convex_hull(points)
    except Exception as e:
        print(f"Alpha shape failed: {e}")
        # Fallback to convex hull
        return simple_convex_hull(points)


def simple_smooth_mesh(mesh, iterations=2):
    """
    Apply simple Laplacian smoothing to a mesh without using Open3D.
    
    Args:
        mesh: trimesh.Trimesh object
        iterations: Number of smoothing iterations
        
    Returns:
        trimesh.Trimesh: Smoothed mesh
    """
    # Create adjacency matrix to find neighboring vertices
    adjacency = {i: set() for i in range(len(mesh.vertices))}
    
    # For each face, add connections between all vertices
    for face in mesh.faces:
        for i in range(3):
            for j in range(3):
                if i != j:
                    adjacency[face[i]].add(face[j])
    
    # Copy original vertices
    smoothed_vertices = np.copy(mesh.vertices)
    
    # Perform smoothing iterations
    for _ in range(iterations):
        new_vertices = np.copy(smoothed_vertices)
        
        # For each vertex, average with neighbors
        for i in range(len(smoothed_vertices)):
            neighbors = list(adjacency[i])
            if neighbors:
                neighbor_positions = smoothed_vertices[neighbors]
                new_vertices[i] = np.mean(neighbor_positions, axis=0) * 0.5 + smoothed_vertices[i] * 0.5
        
        smoothed_vertices = new_vertices
    
    # Create new mesh with smoothed vertices
    return trimesh.Trimesh(vertices=smoothed_vertices, faces=mesh.faces)

# End of module - all Open3D dependencies removed

def fill_holes(mesh, hole_size=0.05):
    """
    Fill small holes in a mesh using trimesh's hole filling capabilities.
    
    Args:
        mesh: trimesh.Trimesh object
        hole_size: Maximum area of holes to fill as a proportion of total mesh area
        
    Returns:
        trimesh.Trimesh: Mesh with holes filled
    """
    if not isinstance(mesh, trimesh.Trimesh):
        print("Warning: Input is not a valid mesh, returning as-is")
        return mesh
        
    try:
        # Get total mesh area for relative hole size calculation
        total_area = mesh.area
        if total_area <= 0:
            print("Warning: Mesh has zero area, cannot determine relative hole size")
            return mesh
            
        max_hole_area = total_area * hole_size
        
        # Make a copy of the mesh to avoid modifying the original
        fixed_mesh = mesh.copy()
        
        # Fill holes in the mesh - note this modifies the mesh in-place
        # and returns a boolean indicating success
        success = fixed_mesh.fill_holes(max_hole_area)
        
        if success:
            print(f"Successfully filled holes in mesh (max size: {max_hole_area:.6f})")
        else:
            print("No holes were filled or filling was unsuccessful")
            
        return fixed_mesh
    
    except Exception as e:
        print(f"Error filling holes: {e}")
        return mesh
    
def simplify_mesh(mesh, target_faces=None, fraction=None):
    """
    Simplify mesh to reduce complexity while preserving shape.
    
    Args:
        mesh: trimesh.Trimesh object
        target_faces: Target number of faces
        fraction: Alternatively, fraction of original faces to keep
        
    Returns:
        trimesh.Trimesh: Simplified mesh
    """
    if target_faces is None and fraction is not None:
        target_faces = int(len(mesh.faces) * fraction)
    elif target_faces is None and fraction is None:
        # Default to 50% reduction
        target_faces = len(mesh.faces) // 2
    
    # Use trimesh's built-in simplification
    simplified_mesh = mesh.simplify_quadratic_decimation(target_faces)
    
    return simplified_mesh

def apply_refinement_pipeline(points, refinement_type="full", **kwargs):
    """
    Apply a complete refinement pipeline to convert point cloud to high-quality mesh.
    
    Args:
        points: [N, 3] array of point positions
        refinement_type: Type of refinement to apply ("full", "poisson", "alpha")
        **kwargs: Additional parameters for refinement methods
        
    Returns:
        trimesh.Trimesh: Refined mesh
    """
    # Extract parameters with defaults
    poisson_depth = kwargs.get("poisson_depth", 9)
    alpha_value = kwargs.get("alpha_value", 0.5)
    smooth_iterations = kwargs.get("smooth_iterations", 3)
    smooth_lambda = kwargs.get("smooth_lambda", 0.5)
    simplify_fraction = kwargs.get("simplify_fraction", 0.8)
    fill_hole_size = kwargs.get("fill_hole_size", 100)
    
    if refinement_type == "poisson":
        # Use Poisson surface reconstruction
        mesh = poisson_surface_reconstruction(points, depth=poisson_depth)
    elif refinement_type == "alpha":
        # Use alpha shapes
        mesh = alpha_shape(points, alpha=alpha_value)
    elif refinement_type == "full":
        # Try Poisson first, fall back to alpha shapes if it fails
        try:
            mesh = poisson_surface_reconstruction(points, depth=poisson_depth)
            # Apply smoothing
            mesh = smooth_mesh(mesh, iterations=smooth_iterations, lambda_value=smooth_lambda)
            # Fill holes
            mesh = fill_holes(mesh, hole_size=fill_hole_size)
            # Simplify if needed
            if simplify_fraction < 1.0:
                mesh = simplify_mesh(mesh, fraction=simplify_fraction)
        except Exception as e:
            print(f"Poisson reconstruction failed: {e}. Falling back to alpha shapes.")
            mesh = alpha_shape(points, alpha=alpha_value)
            mesh = smooth_mesh(mesh, iterations=smooth_iterations, lambda_value=smooth_lambda)
    else:
        raise ValueError(f"Unknown refinement type: {refinement_type}")
    
    return mesh

def point_cloud_to_mesh(point_cloud, method="poisson", **kwargs):
    """
    Convert point cloud to mesh using specified method.
    
    Args:
        point_cloud: [N, 3] numpy array or torch.Tensor of points
        method: Reconstruction method ("poisson", "alpha", or "full")
        **kwargs: Additional parameters for the method
        
    Returns:
        trimesh.Trimesh: Reconstructed mesh
    """
    # Convert to numpy if tensor
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.detach().cpu().numpy()
    
    return apply_refinement_pipeline(point_cloud, refinement_type=method, **kwargs)
