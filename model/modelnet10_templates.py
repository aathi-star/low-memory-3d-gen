import torch
import numpy as np

# ModelNet10 template shapes for better resemblance
# These templates provide a starting point for shape generation
# Each template is a simplified point cloud that captures the essence of the shape

class ModelNet10Templates:
    def __init__(self, device='cuda'):
        self.device = device
        self.categories = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self):
        """Initialize simplified template shapes for each ModelNet10 category"""
        templates = {}
        
        # Generate templates for all categories
        templates['bathtub'] = self._create_bathtub_template()
        templates['bed'] = self._create_bed_template()
        templates['chair'] = self._create_chair_template()
        templates['desk'] = self._create_desk_template()
        templates['dresser'] = self._create_dresser_template()
        templates['monitor'] = self._create_monitor_template()
        templates['night_stand'] = self._create_night_stand_template()
        templates['sofa'] = self._create_sofa_template()
        templates['table'] = self._create_table_template()
        templates['toilet'] = self._create_toilet_template()
        
        return templates
    
    def get_template(self, category_name=None, category_id=None, num_points=2048):
        """
        Get template points for a specific category
        Args:
            category_name: string name of category
            category_id: integer ID of category
            num_points: number of points to sample from template
        Returns:
            tensor of shape [num_points, 3]
        """
        if category_name is None and category_id is None:
            raise ValueError("Must provide either category_name or category_id")
            
        if category_name is None:
            category_name = self.categories[category_id]
            
        if category_name not in self.templates:
            raise ValueError(f"Unknown category: {category_name}")
            
        template = self.templates[category_name]
        
        # If template has more points than requested, subsample
        if len(template) > num_points:
            indices = torch.randperm(len(template))[:num_points]
            return template[indices]
        # If template has fewer points than requested, duplicate with small variations
        elif len(template) < num_points:
            # Repeat points with small noise
            repeats = int(np.ceil(num_points / len(template)))
            repeated = template.repeat(repeats, 1)[:num_points]
            noise = torch.randn_like(repeated) * 0.02  # Small noise
            return repeated + noise
        else:
            return template
        return templates
    
    def _create_chair_template(self):
        """Create a basic chair template with seat, back, and legs"""
        # Start with a grid of points
        points = []
        
        # Seat
        for x in np.linspace(-0.4, 0.4, 5):
            for z in np.linspace(-0.4, 0.4, 5):
                points.append([x, 0.0, z])
        
        # Back
        for x in np.linspace(-0.4, 0.4, 5):
            for y in np.linspace(0.0, 0.8, 5):
                points.append([x, y, 0.4])
        
        # Legs
        for x in [-0.4, 0.4]:
            for z in [-0.4, 0.4]:
                for y in np.linspace(-0.5, 0.0, 3):
                    points.append([x, y, z])
        
        # Convert to tensor
        return torch.tensor(points, dtype=torch.float32).to(self.device)
    
    def _create_table_template(self):
        """Create a basic table template with top and legs"""
        points = []
        
        # Table top
        for x in np.linspace(-0.5, 0.5, 8):
            for z in np.linspace(-0.3, 0.3, 5):
                points.append([x, 0.0, z])
        
        # Legs
        for x in [-0.5, 0.5]:
            for z in [-0.3, 0.3]:
                for y in np.linspace(-0.5, 0.0, 4):
                    points.append([x, y, z])
        
        return torch.tensor(points, dtype=torch.float32).to(self.device)
    
    def _create_bed_template(self):
        """Create a basic bed template"""
        points = []
        
        # Mattress
        for x in np.linspace(-0.5, 0.5, 8):
            for z in np.linspace(-0.8, 0.8, 12):
                points.append([x, 0.0, z])
        
        # Headboard
        for x in np.linspace(-0.5, 0.5, 8):
            for y in np.linspace(0.0, 0.4, 4):
                points.append([x, y, 0.8])
        
        # Frame
        for x in [-0.5, 0.5]:
            for z in np.linspace(-0.8, 0.8, 12):
                for y in np.linspace(-0.2, 0.0, 2):
                    points.append([x, y, z])
        
        for x in np.linspace(-0.5, 0.5, 8):
            for z in [-0.8, 0.8]:
                for y in np.linspace(-0.2, 0.0, 2):
                    points.append([x, y, z])
        
        return torch.tensor(points, dtype=torch.float32).to(self.device)
    
    def _create_sofa_template(self):
        """Create a basic sofa template"""
        points = []
        
        # Seat
        for x in np.linspace(-0.8, 0.8, 10):
            for z in np.linspace(-0.4, 0.4, 5):
                points.append([x, 0.0, z])
        
        # Back
        for x in np.linspace(-0.8, 0.8, 10):
            for y in np.linspace(0.0, 0.6, 5):
                points.append([x, y, 0.4])
        
        # Arms
        for x in [-0.8, 0.8]:
            for z in np.linspace(-0.4, 0.4, 5):
                for y in np.linspace(0.0, 0.3, 3):
                    points.append([x, y, z])
        
        # Base
        for x in np.linspace(-0.8, 0.8, 6):
            for z in np.linspace(-0.4, 0.4, 3):
                points.append([x, -0.2, z])
        
        return torch.tensor(points, dtype=torch.float32).to(self.device)
    
    def _create_bathtub_template(self):
        """Create a basic bathtub template"""
        points = []
        
        # Tub bottom
        for x in np.linspace(-0.4, 0.4, 6):
            for z in np.linspace(-0.8, 0.8, 12):
                points.append([x, -0.3, z])
        
        # Tub sides
        for y in np.linspace(-0.3, 0.1, 5):
            for z in np.linspace(-0.8, 0.8, 12):
                points.append([0.4, y, z])  # Right side
                points.append([-0.4, y, z])  # Left side
        
        for y in np.linspace(-0.3, 0.1, 5):
            for x in np.linspace(-0.4, 0.4, 6):
                points.append([x, y, 0.8])  # Back side
                points.append([x, y, -0.8])  # Front side
        
        return torch.tensor(points, dtype=torch.float32).to(self.device)
    
    def _create_desk_template(self):
        """Create a basic desk template"""
        points = []
        
        # Desk top
        for x in np.linspace(-0.6, 0.6, 10):
            for z in np.linspace(-0.3, 0.3, 5):
                points.append([x, 0.0, z])
        
        # Legs
        for x in [-0.6, 0.6]:
            for z in [-0.3, 0.3]:
                for y in np.linspace(-0.5, 0.0, 4):
                    points.append([x, y, z])
        
        # Optional drawer
        for x in np.linspace(-0.2, 0.2, 3):
            for y in np.linspace(-0.1, 0.0, 2):
                points.append([x, y, 0.3])
        
        return torch.tensor(points, dtype=torch.float32).to(self.device)
    
    def _create_dresser_template(self):
        """Create a basic dresser template"""
        points = []
        
        # Box shape
        for x in np.linspace(-0.5, 0.5, 8):
            for y in np.linspace(-0.5, 0.5, 8):
                points.append([x, y, 0.3])  # Front
            
        for x in np.linspace(-0.5, 0.5, 8):
            for y in np.linspace(-0.5, 0.5, 8):
                points.append([x, y, -0.3])  # Back
        
        for x in np.linspace(-0.5, 0.5, 8):
            for z in np.linspace(-0.3, 0.3, 5):
                points.append([x, 0.5, z])  # Top
                points.append([x, -0.5, z])  # Bottom
        
        for y in np.linspace(-0.5, 0.5, 8):
            for z in np.linspace(-0.3, 0.3, 5):
                points.append([0.5, y, z])  # Right
                points.append([-0.5, y, z])  # Left
        
        # Drawer lines
        for x in np.linspace(-0.5, 0.5, 8):
            for y_val in [-0.17, 0.17]:
                points.append([x, y_val, 0.3])
        
        return torch.tensor(points, dtype=torch.float32).to(self.device)
    
    def _create_monitor_template(self):
        """Create a basic monitor template"""
        points = []
        
        # Screen
        for x in np.linspace(-0.5, 0.5, 8):
            for y in np.linspace(-0.3, 0.3, 5):
                points.append([x, y, 0.0])
        
        # Frame
        for x in [-0.5, 0.5]:
            for y in np.linspace(-0.3, 0.3, 5):
                points.append([x, y, 0.0])
        
        for y in [-0.3, 0.3]:
            for x in np.linspace(-0.5, 0.5, 8):
                points.append([x, y, 0.0])
        
        # Stand
        for y in np.linspace(-0.3, -0.5, 3):
            points.append([0.0, y, 0.0])
        
        # Base
        for x in np.linspace(-0.2, 0.2, 3):
            for z in np.linspace(-0.1, 0.1, 3):
                points.append([x, -0.5, z])
        
        return torch.tensor(points, dtype=torch.float32).to(self.device)
    
    def _create_night_stand_template(self):
        """Create a basic night stand template"""
        points = []
        
        # Box shape
        for x in np.linspace(-0.3, 0.3, 5):
            for y in np.linspace(-0.4, 0.4, 6):
                points.append([x, y, 0.3])  # Front
        
        for x in np.linspace(-0.3, 0.3, 5):
            for y in np.linspace(-0.4, 0.4, 6):
                points.append([x, y, -0.3])  # Back
        
        for x in np.linspace(-0.3, 0.3, 5):
            for z in np.linspace(-0.3, 0.3, 5):
                points.append([x, 0.4, z])  # Top
                points.append([x, -0.4, z])  # Bottom
        
        for y in np.linspace(-0.4, 0.4, 6):
            for z in np.linspace(-0.3, 0.3, 5):
                points.append([0.3, y, z])  # Right
                points.append([-0.3, y, z])  # Left
        
        # Drawer line
        for x in np.linspace(-0.3, 0.3, 5):
            points.append([x, 0.0, 0.3])
        
        return torch.tensor(points, dtype=torch.float32).to(self.device)
    
    def _create_toilet_template(self):
        """Create a basic toilet template"""
        points = []
        
        # Bowl
        for theta in np.linspace(0, 2*np.pi, 12):
            for y in np.linspace(-0.2, 0.0, 3):
                x = 0.3 * np.cos(theta)
                z = 0.3 * np.sin(theta)
                points.append([x, y, z])
        
        # Base
        for x in np.linspace(-0.3, 0.3, 5):
            for z in np.linspace(-0.3, 0.3, 5):
                points.append([x, -0.5, z])
        
        # Tank
        for x in np.linspace(-0.25, 0.25, 4):
            for y in np.linspace(0.0, 0.5, 4):
                points.append([x, y, -0.2])
        
        for x in [-0.25, 0.25]:
            for y in np.linspace(0.0, 0.5, 4):
                for z in np.linspace(-0.2, 0.0, 2):
                    points.append([x, y, z])
        
        for y in [0.0, 0.5]:
            for x in np.linspace(-0.25, 0.25, 4):
                for z in np.linspace(-0.2, 0.0, 2):
                    points.append([x, y, z])
        
        # Seat
        for theta in np.linspace(0, 2*np.pi, 12):
            x = 0.35 * np.cos(theta)
            z = 0.35 * np.sin(theta)
            points.append([x, 0.0, z])
        
        for theta in np.linspace(0, 2*np.pi, 12):
            x = 0.25 * np.cos(theta)
            z = 0.25 * np.sin(theta)
            points.append([x, 0.0, z])
        
        return torch.tensor(points, dtype=torch.float32).to(self.device)
    
    # Main get_template method is defined above
