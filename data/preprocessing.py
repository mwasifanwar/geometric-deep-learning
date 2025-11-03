import numpy as np
import open3d as o3d
from typing import Dict

class PointCloudPreprocessor:
    def __init__(self):
        pass
    
    def process(self, pointcloud: np.ndarray, max_points: int = 1024) -> Dict:
        if len(pointcloud) == 0:
            pointcloud = np.random.randn(max_points, 3).astype(np.float32)
        
        if len(pointcloud) > max_points:
            indices = np.random.choice(len(pointcloud), max_points, replace=False)
            pointcloud = pointcloud[indices]
        elif len(pointcloud) < max_points:
            padding = max_points - len(pointcloud)
            padding_points = np.random.randn(padding, 3) * 0.01 + pointcloud.mean(axis=0)
            pointcloud = np.concatenate([pointcloud, padding_points], axis=0)
        
        normalized_pc = self._normalize(pointcloud)
        
        return {
            'points': normalized_pc.astype(np.float32),
            'original_points': pointcloud.astype(np.float32),
            'num_points': len(normalized_pc)
        }
    
    def _normalize(self, pointcloud: np.ndarray) -> np.ndarray:
        centroid = np.mean(pointcloud, axis=0)
        pointcloud = pointcloud - centroid
        
        max_dist = np.max(np.sqrt(np.sum(pointcloud**2, axis=1)))
        if max_dist > 0:
            pointcloud = pointcloud / max_dist
        
        return pointcloud

class MeshPreprocessor:
    def __init__(self):
        pass
    
    def process(self, mesh: o3d.geometry.TriangleMesh) -> Dict:
        vertices = np.asarray(mesh.vertices).astype(np.float32)
        faces = np.asarray(mesh.triangles).astype(np.int64)
        
        if len(vertices) == 0:
            vertices = np.random.randn(100, 3).astype(np.float32)
            faces = np.random.randint(0, 100, (100, 3)).astype(np.int64)
        
        normalized_vertices = self._normalize(vertices)
        
        return {
            'vertices': normalized_vertices,
            'faces': faces,
            'original_vertices': vertices,
            'num_vertices': len(vertices),
            'num_faces': len(faces)
        }
    
    def _normalize(self, vertices: np.ndarray) -> np.ndarray:
        centroid = np.mean(vertices, axis=0)
        vertices = vertices - centroid
        
        max_dist = np.max(np.sqrt(np.sum(vertices**2, axis=1)))
        if max_dist > 0:
            vertices = vertices / max_dist
        
        return vertices