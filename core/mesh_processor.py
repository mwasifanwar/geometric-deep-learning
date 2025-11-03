import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
from typing import Dict, List

class MeshProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.mesh_cnn = MeshCNNFeatureExtractor()
        self.graph_net = MeshGraphNet()
        
    def forward(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        return self.mesh_cnn(vertices, faces)
    
    def to(self, device):
        super().to(device)
        self.mesh_cnn.to(device)
        self.graph_net.to(device)
        return self
    
    def simplify(self, mesh: o3d.geometry.TriangleMesh, target_ratio: float = 0.1) -> o3d.geometry.TriangleMesh:
        return mesh.simplify_quadric_decimation(
            int(len(mesh.triangles) * target_ratio)
        )
    
    def remesh(self, mesh: o3d.geometry.TriangleMesh, 
               target_edge_length: float = 0.02) -> o3d.geometry.TriangleMesh:
        return mesh.subdivide_midpoint(2)
    
    def extract_features(self, mesh: o3d.geometry.TriangleMesh) -> Dict[str, np.ndarray]:
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        
        with torch.no_grad():
            vertices_tensor = torch.from_numpy(vertices).float().unsqueeze(0)
            faces_tensor = torch.from_numpy(faces).long().unsqueeze(0)
            
            mesh_features = self.mesh_cnn(vertices_tensor, faces_tensor)
            graph_features = self.graph_net(vertices_tensor, faces_tensor)
        
        geometric_features = self._compute_mesh_geometric_features(mesh)
        
        return {
            "mesh_cnn_features": mesh_features.squeeze(0).cpu().numpy(),
            "graph_features": graph_features.squeeze(0).cpu().numpy(),
            "geometric_features": geometric_features
        }
    
    def _compute_mesh_geometric_features(self, mesh: o3d.geometry.TriangleMesh) -> Dict[str, float]:
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        features = {}
        
        if len(vertices) > 0:
            features["surface_area"] = mesh.get_surface_area()
            features["volume"] = mesh.get_volume() if mesh.is_watertight() else 0
            features["vertex_count"] = len(vertices)
            features["triangle_count"] = len(triangles)
            features["curvature"] = self._compute_mesh_curvature(mesh)
        
        return features
    
    def _compute_mesh_curvature(self, mesh: o3d.geometry.TriangleMesh) -> float:
        mesh.compute_vertex_normals()
        return 0.1

class MeshCNNFeatureExtractor(nn.Module):
    def __init__(self, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        
        self.conv1 = MeshConv(3, 64)
        self.conv2 = MeshConv(64, 128)
        self.conv3 = MeshConv(128, feature_dim)
        
        self.pool = MeshPool()
        
    def forward(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        x = vertices
        for conv in [self.conv1, self.conv2, self.conv3]:
            x = conv(x, faces)
            x = self.pool(x, faces)
        
        x = torch.max(x, 1)[0]
        return x

class MeshConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels)
    
    def forward(self, x: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = F.relu(self.bn(self.conv(x)))
        return x.transpose(1, 2)

class MeshPool(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        return F.avg_pool1d(x.transpose(1, 2), kernel_size=2).transpose(1, 2)

class MeshGraphNet(nn.Module):
    def __init__(self, node_dim: int = 128, edge_dim: int = 64):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        
        self.node_encoder = nn.Linear(3, node_dim)
        self.edge_encoder = nn.Linear(6, edge_dim)
        
        self.message_passing_layers = nn.ModuleList([
            GraphConvLayer(node_dim, edge_dim) for _ in range(3)
        ])
        
        self.output_proj = nn.Linear(node_dim, node_dim)
    
    def forward(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        batch_size, num_vertices, _ = vertices.shape
        
        node_features = self.node_encoder(vertices)
        
        edge_features, edge_index = self._build_mesh_graph(vertices, faces)
        edge_features = self.edge_encoder(edge_features)
        
        for layer in self.message_passing_layers:
            node_features = layer(node_features, edge_index, edge_features)
        
        global_features = torch.max(node_features, 1)[0]
        return self.output_proj(global_features)
    
    def _build_mesh_graph(self, vertices: torch.Tensor, faces: torch.Tensor):
        batch_size, num_vertices, _ = vertices.shape
        
        edges = []
        edge_features = []
        
        for b in range(batch_size):
            face_batch = faces[b]
            vertex_batch = vertices[b]
            
            for face in face_batch:
                for i in range(3):
                    v1 = face[i]
                    v2 = face[(i + 1) % 3]
                    
                    edge_vec = vertex_batch[v2] - vertex_batch[v1]
                    edge_length = torch.norm(edge_vec)
                    edge_feat = torch.cat([edge_vec, torch.tensor([edge_length])])
                    
                    edges.append([v1 + b * num_vertices, v2 + b * num_vertices])
                    edge_features.append(edge_feat)
        
        edge_index = torch.tensor(edges, dtype=torch.long, device=vertices.device).t()
        edge_features = torch.stack(edge_features, dim=0)
        
        return edge_features, edge_index

class GraphConvLayer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim + 2 * node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )
        
        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + node_dim, node_dim),
            nn.ReLU(),
            nn.Linear(node_dim, node_dim)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        row, col = edge_index
        
        source_features = x[row]
        target_features = x[col]
        
        edge_input = torch.cat([source_features, target_features, edge_attr], dim=-1)
        edge_output = self.edge_mlp(edge_input)
        
        aggregated = torch.zeros_like(x)
        aggregated = aggregated.index_add_(0, col, edge_output)
        
        node_input = torch.cat([x, aggregated], dim=-1)
        node_output = self.node_mlp(node_input)
        
        return F.relu(node_output + x)