import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
from typing import Dict, List

class SceneReconstructor(nn.Module):
    def __init__(self):
        super().__init__()
        self.poisson_reconstructor = PoissonReconstructor()
        self.alpha_shapes_reconstructor = AlphaShapesReconstructor()
        self.completion_net = PointCloudCompletionNet()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.completion_net(x)
    
    def to(self, device):
        super().to(device)
        self.completion_net.to(device)
        return self
    
    def pointcloud_to_mesh(self, pointcloud: np.ndarray, method: str = "poisson") -> o3d.geometry.TriangleMesh:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud)
        
        if method == "poisson":
            return self.poisson_reconstructor.reconstruct(pcd)
        elif method == "alpha_shapes":
            return self.alpha_shapes_reconstructor.reconstruct(pcd)
        else:
            raise ValueError(f"Unknown reconstruction method: {method}")
    
    def complete_pointcloud(self, partial_pointcloud: np.ndarray) -> np.ndarray:
        if isinstance(partial_pointcloud, np.ndarray):
            partial_pointcloud = torch.from_numpy(partial_pointcloud).float()
        
        with torch.no_grad():
            completed = self.completion_net(partial_pointcloud.unsqueeze(0))
        
        return completed.squeeze(0).cpu().numpy()

class PoissonReconstructor:
    def reconstruct(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        pcd.estimate_normals()
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        return mesh

class AlphaShapesReconstructor:
    def reconstruct(self, pcd: o3d.geometry.PointCloud, alpha: float = 0.03) -> o3d.geometry.TriangleMesh:
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        return mesh

class PointCloudCompletionNet(nn.Module):
    def __init__(self, latent_dim: int = 512, output_points: int = 2048):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_points = output_points
        
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.ReLU(),
            nn.Conv1d(256, latent_dim, 1)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_points * 3)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        
        x = x.transpose(1, 2)
        encoded = self.encoder(x)
        encoded = torch.max(encoded, 2)[0]
        
        decoded = self.decoder(encoded)
        completed = decoded.view(batch_size, self.output_points, 3)
        
        return completed