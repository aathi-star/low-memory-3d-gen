import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
import trimesh
from PIL import Image


def visualize_point_cloud(points, title=None, save_path=None):
    """
    Visualize a point cloud.
    
    Args:
        points: NumPy array of shape [N, 3]
        title: Optional title for the plot
        save_path: Optional path to save the visualization
        
    Returns:
        PIL Image of the visualization
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=5, c=points[:, 2], cmap='viridis')
    
    # Set equal aspect ratio
    max_range = np.max(np.abs(points)) * 1.1
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_zlim(-max_range, max_range)
    
    if title:
        ax.set_title(title)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path)
    
    # Convert figure to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    
    plt.close(fig)
    
    return img


def visualize_mesh(vertices, faces, title=None, save_path=None):
    """
    Visualize a mesh using matplotlib for headless environments where OpenGL isn't available.
    
    Args:
        vertices: NumPy array of shape [N, 3]
        faces: NumPy array of shape [M, 3]
        title: Optional title for the plot
        save_path: Optional path to save the visualization
        
    Returns:
        PIL Image of the visualization
    """
    try:
        # First try the trimesh renderer which gives better quality
        # Create mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Create scene
        scene = trimesh.Scene(mesh)
        
        # Render
        try:
            resolution = (1024, 1024)
            img = scene.save_image(
                resolution=resolution,
                visible=True
            )
            # Convert to PIL Image
            img = Image.fromarray(img)
        except Exception as e:
            print(f"Trimesh rendering failed (this is normal on headless systems): {e}")
            # Fall back to matplotlib rendering
            raise RuntimeError("Falling back to matplotlib")
    
    except Exception as e:
        print(f"Using matplotlib fallback renderer for mesh visualization")
        
        # Use matplotlib for headless rendering
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot mesh as a collection of triangles
        # First get triangle coordinates
        triangles = vertices[faces]
        
        # Plot triangles
        for tri in triangles:
            # Plot each face as a separate polygon
            x = tri[:, 0]
            y = tri[:, 1]
            z = tri[:, 2]
            ax.plot_trisurf(x, y, z, color='lightblue', alpha=0.8, shade=True)
            
        # Plot a subset of edges for better visualization
        # Sample edges to avoid cluttering
        if len(faces) > 500:
            sample_size = 500
            indices = np.random.choice(len(faces), sample_size, replace=False)
            sample_faces = faces[indices]
        else:
            sample_faces = faces
            
        # Plot wireframe for sampled faces
        for face in sample_faces:
            for i in range(3):
                # Get indices of edge vertices
                idx1 = face[i]
                idx2 = face[(i+1)%3]
                # Plot edge
                ax.plot([vertices[idx1, 0], vertices[idx2, 0]],
                        [vertices[idx1, 1], vertices[idx2, 1]],
                        [vertices[idx1, 2], vertices[idx2, 2]], 'k-', alpha=0.2)
                
        # Set equal aspect ratio
        max_range = np.max(np.abs(vertices)) * 1.1
        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-max_range, max_range)
        ax.set_zlim(-max_range, max_range)
        
        if title:
            ax.set_title(title)
            
        # Save if requested
        if save_path:
            plt.savefig(save_path)
            
        # Convert figure to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
    
    # This line can be removed as we already convert to PIL Image in both code paths
    # No need to convert again
    pass
    
    # Add title if provided
    if title:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
        
        # Convert figure to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
    
    # Save if requested
    if save_path:
        img.save(save_path)
    
    return img


def save_point_cloud_as_ply(point_cloud, output_path):
    """
    Save a point cloud as a PLY file.
    
    Args:
        point_cloud: NumPy array of shape [N, 3]
        output_path: Path to save the PLY file
    """
    # Create point cloud object
    pc_trimesh = trimesh.PointCloud(point_cloud)
    
    # Save as PLY
    pc_trimesh.export(output_path)


def save_mesh_as_obj(vertices, faces, output_path):
    """
    Save a mesh as an OBJ file.
    
    Args:
        vertices: NumPy array of shape [N, 3]
        faces: NumPy array of shape [M, 3]
        output_path: Path to save the OBJ file
    """
    # Create mesh
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Export as OBJ
    mesh.export(output_path)


def save_point_cloud_obj(points, output_path):
    """
    Save a point cloud as an OBJ file by creating a mesh using ball pivoting algorithm.
    Falls back to vertices-only OBJ if mesh creation fails.
    
    Args:
        points: NumPy array of shape [N, 3]
        output_path: Path to save the OBJ file
    """
    try:
        # Try to create a mesh from the point cloud using ball pivoting
        # This works better for well-distributed points (like those from Halton sampling)
        cloud = trimesh.PointCloud(points)
        mesh = trimesh.voxel.ops.points_to_marching_cubes(points, pitch=0.05)
        
        if mesh.is_empty:
            raise ValueError("Could not create mesh with marching cubes")
            
        # Export as OBJ
        mesh.export(output_path)
        
    except Exception as e:
        print(f"Warning: Mesh creation failed, saving as vertices-only OBJ: {e}")
        
        # Fall back to saving as vertices only
        with open(output_path, 'w') as f:
            for point in points:
                f.write(f"v {point[0]} {point[1]} {point[2]}\n")
            
        print(f"Saved point cloud as vertices-only OBJ to {output_path}")
