import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
from typing import Dict, List, Tuple

class PointCloudProcessor(nn.Module):
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        
        self.pointnet = PointNetFeatureExtractor(feature_dim)
        self.dgcnn = DGCNNFeatureExtractor(feature_dim)
        self.denoising_net = PointCloudDenoisingNet()
        self.normal_estimation_net = NormalEstimationNet()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointnet(x)
    
    def to(self, device):
        super().to(device)
        self.pointnet.to(device)
        self.dgcnn.to(device)
        self.denoising_net.to(device)
        self.normal_estimation_net.to(device)
        return self
    
    def denoise(self, pointcloud: np.ndarray) -> np.ndarray:
        if isinstance(pointcloud, np.ndarray):
            pointcloud = torch.from_numpy(pointcloud).float()
        
        with torch.no_grad():
            denoised = self.denoising_net(pointcloud.unsqueeze(0))
        
        return denoised.squeeze(0).cpu().numpy()
    
    def estimate_normals(self, pointcloud: np.ndarray) -> np.ndarray:
        if isinstance(pointcloud, np.ndarray):
            pointcloud = torch.from_numpy(pointcloud).float()
        
        with torch.no_grad():
            normals = self.normal_estimation_net(pointcloud.unsqueeze(0))
        
        return normals.squeeze(0).cpu().numpy()
    
    def extract_features(self, pointcloud: np.ndarray) -> Dict[str, np.ndarray]:
        if isinstance(pointcloud, np.ndarray):
            pointcloud = torch.from_numpy(pointcloud).float()
        
        with torch.no_grad():
            pointnet_features = self.pointnet(pointcloud.unsqueeze(0))
            dgcnn_features = self.dgcnn(pointcloud.unsqueeze(0))
            geometric_features = self.extract_geometric_features(pointcloud)
        
        return {
            "pointnet_features": pointnet_features.squeeze(0).cpu().numpy(),
            "dgcnn_features": dgcnn_features.squeeze(0).cpu().numpy(),
            "geometric_features": geometric_features
        }
    
    def extract_geometric_features(self, pointcloud: np.ndarray) -> Dict[str, np.ndarray]:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud)
        
        features = {}
        
        if len(pointcloud) > 0:
            features["density"] = self._compute_point_density(pointcloud)
            features["curvature"] = self._compute_curvature(pcd)
            features["linearity"] = self._compute_geometric_features(pcd)
        
        return features
    
    def segment(self, pointcloud: np.ndarray) -> Dict:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud)
        
        labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=10))
        
        segments = {}
        unique_labels = np.unique(labels)
        for label in unique_labels:
            if label >= 0:
                segment_points = pointcloud[labels == label]
                segments[f"segment_{label}"] = {
                    "points": segment_points,
                    "centroid": np.mean(segment_points, axis=0),
                    "size": len(segment_points)
                }
        
        return segments
    
    def _compute_point_density(self, pointcloud: np.ndarray) -> float:
        if len(pointcloud) == 0:
            return 0.0
        
        from scipy.spatial import KDTree
        tree = KDTree(pointcloud)
        distances, _ = tree.query(pointcloud, k=2)
        mean_distance = np.mean(distances[:, 1])
        
        return 1.0 / (mean_distance + 1e-8)
    
    def _compute_curvature(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(10)
        
        curvatures = []
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        
        for i in range(len(points)):
            if i < len(normals):
                curvature = self._compute_point_curvature(points, normals, i)
                curvatures.append(curvature)
        
        return np.array(curvatures)
    
    def _compute_point_curvature(self, points: np.ndarray, normals: np.ndarray, idx: int) -> float:
        return 0.1
    
    def _compute_geometric_features(self, pcd: o3d.geometry.PointCloud) -> Dict[str, float]:
        if len(pcd.points) == 0:
            return {}
        
        points = np.asarray(pcd.points)
        
        bbox = pcd.get_axis_aligned_bounding_box()
        volume = bbox.volume()
        
        eigenvalues = self._compute_covariance_eigenvalues(points)
        
        linearity = (eigenvalues[0] - eigenvalues[1]) / eigenvalues[0] if eigenvalues[0] > 0 else 0
        planarity = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0] if eigenvalues[0] > 0 else 0
        sphericity = eigenvalues[2] / eigenvalues[0] if eigenvalues[0] > 0 else 0
        
        return {
            "linearity": linearity,
            "planarity": planarity,
            "sphericity": sphericity,
            "volume": volume,
            "point_count": len(points)
        }
    
    def _compute_covariance_eigenvalues(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0:
            return np.zeros(3)
        
        centered = points - np.mean(points, axis=0)
        cov = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        return np.sort(eigenvalues)[::-1]

class PointNetFeatureExtractor(nn.Module):
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, feature_dim, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(feature_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        x = torch.max(x, 2)[0]
        return x

class DGCNNFeatureExtractor(nn.Module):
    def __init__(self, feature_dim: int = 128, k: int = 20):
        super().__init__()
        self.k = k
        self.feature_dim = feature_dim
        
        self.conv1 = nn.Conv2d(6, 64, 1)
        self.conv2 = nn.Conv2d(64, 64, 1)
        self.conv3 = nn.Conv2d(64, 128, 1)
        self.conv4 = nn.Conv2d(128, feature_dim, 1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(feature_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, _ = x.shape
        
        knn_indices = self._knn(x, self.k)
        edge_features = self._get_edge_features(x, knn_indices)
        
        x = F.relu(self.bn1(self.conv1(edge_features)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.bn4(self.conv4(x))
        
        x = torch.max(x, -1)[0]
        return x
    
    def _knn(self, x: torch.Tensor, k: int) -> torch.Tensor:
        inner = -2 * torch.matmul(x, x.transpose(2, 1))
        xx = torch.sum(x**2, dim=2, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        
        idx = pairwise_distance.topk(k=k, dim=-1)[1]
        return idx
    
    def _get_edge_features(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, _ = x.shape
        k = idx.shape[-1]
        
        idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        
        x = x.view(batch_size * num_points, -1)
        neighbors = x[idx].view(batch_size, num_points, k, 3)
        x = x.view(batch_size, num_points, 1, 3).repeat(1, 1, k, 1)
        
        edge_features = torch.cat([x, neighbors - x], dim=-1)
        edge_features = edge_features.permute(0, 3, 1, 2)
        
        return edge_features

class PointCloudDenoisingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1)
        )
        
        self.decoder = nn.Sequential(
            nn.Conv1d(256, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.transpose(1, 2)

class NormalEstimationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        normals = self.network(x)
        normals = F.normalize(normals, p=2, dim=1)
        return normals.transpose(1, 2)