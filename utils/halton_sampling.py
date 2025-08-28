import numpy as np
import torch

def next_prime():
    """Generator for prime numbers"""
    def is_prime(num):
        "Check if num is prime"
        if num <= 1:
            return False
        if num <= 3:
            return True
        if num % 2 == 0 or num % 3 == 0:
            return False
        i = 5
        while i * i <= num:
            if num % i == 0 or num % (i + 2) == 0:
                return False
            i += 6
        return True
    
    prime = 2
    yield prime
    
    prime = 3
    yield prime
    
    while True:
        prime += 2
        if is_prime(prime):
            yield prime

def van_der_corput(n, base):
    """Van der Corput sequence for a given base"""
    vdc, denom = 0, 1
    while n:
        denom *= base
        n, remainder = divmod(n, base)
        vdc += remainder / denom
    return vdc

def halton_sequence(dim, n_samples):
    """Generate Halton sequence in dim dimensions with n_samples"""
    seq = np.zeros((n_samples, dim))
    primes = []
    
    # Get first dim prime numbers for bases
    prime_gen = next_prime()
    for _ in range(dim):
        primes.append(next(prime_gen))
    
    # Generate sequence
    for i in range(n_samples):
        for j in range(dim):
            seq[i, j] = van_der_corput(i, primes[j])
    
    return seq

def halton_points_3d(n_points, scale=1.0, center=True):
    """Generate 3D points using Halton sequence with scaling"""
    # Generate raw sequence
    points = halton_sequence(3, n_points)
    
    # Scale to [-scale, scale] or [0, scale]
    if center:
        points = 2 * points - 1  # Map from [0,1] to [-1,1]
        points = points * scale
    else:
        points = points * scale
    
    return points

def sample_surface_points_halton(vertices, faces, n_points):
    """Sample points on mesh surface using Halton sequence for better uniformity"""
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.detach().cpu().numpy()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy()
    
    # Calculate face areas for weighted sampling
    face_vertices = vertices[faces]
    v0, v1, v2 = face_vertices[:, 0], face_vertices[:, 1], face_vertices[:, 2]
    face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    face_areas = face_areas / np.sum(face_areas)
    
    # Sample faces based on area
    face_indices = np.random.choice(len(faces), size=n_points, p=face_areas)
    selected_faces = face_vertices[face_indices]
    
    # Use Halton sequence for barycentric coordinates (better distribution)
    # Generate 2D Halton sequence (barycentric coords need only 2 dimensions)
    halton_2d = halton_sequence(2, n_points)
    
    # Convert to barycentric coordinates
    u = np.sqrt(halton_2d[:, 0])
    v = halton_2d[:, 1] * u
    barycentric = np.stack([1 - u, u * (1 - v), v * u], axis=1)
    
    # Compute points on triangles
    sampled_points = np.sum(selected_faces * barycentric[:, :, np.newaxis], axis=1)
    
    return sampled_points

def halton_downsample(points, target_count):
    """Downsample a point cloud using Halton sequence indices"""
    if len(points) <= target_count:
        return points
    
    # Generate 1D Halton sequence
    halton_1d = np.array([van_der_corput(i, 2) for i in range(len(points))])
    
    # Get indices of the largest Halton values
    indices = np.argsort(halton_1d)[-target_count:]
    
    # Return points at these indices
    if isinstance(points, torch.Tensor):
        return points[indices]
    else:
        return points[indices]

class HaltonSampler:
    """Class for sampling point clouds using Halton sequence"""
    def __init__(self, dim=3, scale=1.0, center=True):
        self.dim = dim
        self.scale = scale
        self.center = center
        self.primes = []
        
        # Initialize primes
        prime_gen = next_prime()
        for _ in range(dim):
            self.primes.append(next(prime_gen))
    
    def sample(self, n_points):
        """Sample n_points in dim dimensions"""
        return torch.tensor(halton_points_3d(n_points, self.scale, self.center))
    
    def sample_from_mesh(self, mesh, n_points, return_indices=False):
        """
        Sample points from a mesh surface using Halton-based sampling for better uniformity
        
        Args:
            mesh: trimesh.Trimesh object
            n_points: number of points to sample
            return_indices: if True, return face indices for each sampled point
            
        Returns:
            sampled points, and optionally face indices
        """
        # Get mesh faces and vertices
        faces = np.array(mesh.faces)
        vertices = np.array(mesh.vertices)
        
        # Calculate face areas for weighted sampling
        face_areas = np.zeros(len(faces))
        for i, face in enumerate(faces):
            # Get vertices of this face
            v0, v1, v2 = vertices[face]
            # Calculate area using cross product
            face_areas[i] = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        
        # Normalize areas for probability distribution
        face_areas = face_areas / np.sum(face_areas)
        
        # Sample faces based on area (larger faces get more samples)
        face_indices = np.random.choice(len(faces), size=n_points, p=face_areas)
        
        # Generate barycentric coordinates using Halton sequence for better uniformity
        halton_points = np.array(halton_sequence(2, n_points))  # 2D Halton sequence for barycentric coords
        
        # Convert to barycentric coordinates (u,v,w) where u+v+w=1
        u = halton_points[:,0]
        v = halton_points[:,1] * (1-u)
        w = 1 - u - v
        
        # Generate sample points using barycentric coordinates
        sampled_points = np.zeros((n_points, 3))
        for i in range(n_points):
            face_idx = face_indices[i]
            face = faces[face_idx]
            # Get vertices of this face
            v0, v1, v2 = vertices[face]
            # Apply barycentric coordinates
            sampled_points[i] = u[i]*v0 + v[i]*v1 + w[i]*v2
        
        if return_indices:
            return sampled_points, face_indices
        else:
            return sampled_points
            
    def sample_on_shape(self, template_points, n_points, noise_scale=0.05):
        """Sample near a template shape using Halton perturbation"""
        if isinstance(template_points, np.ndarray):
            template_points = torch.tensor(template_points)
        
        # If template has fewer points, duplicate some
        if len(template_points) < n_points:
            # Repeat points with small variations
            indices = torch.randint(len(template_points), (n_points,))
            base_points = template_points[indices]
        else:
            # Downsample if needed
            base_points = template_points[:n_points]
        
        # Generate Halton noise
        halton_noise = torch.tensor(
            halton_sequence(self.dim, n_points), 
            dtype=base_points.dtype, 
            device=base_points.device
        )
        
        # Convert from [0,1] to [-noise_scale, noise_scale]
        halton_noise = (2 * halton_noise - 1) * noise_scale
        
        # Add noise to base points
        sampled_points = base_points + halton_noise
        
        return sampled_points
    
    def perturb_points(self, points, noise_scale=0.05):
        """Perturb existing points using Halton sequence"""
        n_points = len(points)
        
        # Generate Halton noise
        halton_noise = torch.tensor(
            halton_sequence(self.dim, n_points), 
            dtype=points.dtype, 
            device=points.device if isinstance(points, torch.Tensor) else 'cpu'
        )
        
        # Convert from [0,1] to [-noise_scale, noise_scale]
        halton_noise = (2 * halton_noise - 1) * noise_scale
        
        # Add noise
        if isinstance(points, torch.Tensor):
            return points + halton_noise
        else:
            return points + halton_noise.numpy()
