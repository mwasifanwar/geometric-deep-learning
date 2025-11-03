import torch
from torch.utils.data import Dataset
import numpy as np
import open3d as o3d
import os
import json
from typing import Dict, List, Optional

class PointCloudDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train", max_points: int = 1024):
        self.data_dir = data_dir
        self.split = split
        self.max_points = max_points
        
        self.samples = self._load_samples()
        self.preprocessor = PointCloudPreprocessor()
    
    def _load_samples(self) -> List[Dict]:
        annotation_file = os.path.join(self.data_dir, f"{self.split}_annotations.json")
        
        if os.path.exists(annotation_file):
            with open(annotation_file, 'r') as f:
                samples = json.load(f)
        else:
            samples = self._generate_dummy_samples()
        
        return samples
    
    def _generate_dummy_samples(self) -> List[Dict]:
        dummy_samples = []
        for i in range(100):
            sample = {
                'pointcloud_path': f"pointclouds/sample_{i}.ply",
                'mesh_path': f"meshes/sample_{i}.obj",
                'labels': np.random.randint(0, 10, 100),
                'bounding_boxes': []
            }
            dummy_samples.append(sample)
        
        return dummy_samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        pointcloud = self._load_pointcloud(sample['pointcloud_path'])
        processed_pc = self.preprocessor.process(pointcloud, self.max_points)
        
        item = {
            'pointcloud': processed_pc,
            'labels': sample.get('labels', []),
            'bounding_boxes': sample.get('bounding_boxes', []),
            'metadata': sample.get('metadata', {})
        }
        
        return item
    
    def _load_pointcloud(self, path: str) -> np.ndarray:
        if os.path.exists(path):
            pcd = o3d.io.read_point_cloud(path)
            return np.asarray(pcd.points)
        else:
            return np.random.randn(1000, 3).astype(np.float32)

class MeshDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train"):
        self.data_dir = data_dir
        self.split = split
        
        self.samples = self._load_samples()
        self.preprocessor = MeshPreprocessor()
    
    def _load_samples(self) -> List[Dict]:
        annotation_file = os.path.join(self.data_dir, f"{self.split}_annotations.json")
        
        if os.path.exists(annotation_file):
            with open(annotation_file, 'r') as f:
                samples = json.load(f)
        else:
            samples = self._generate_dummy_samples()
        
        return samples
    
    def _generate_dummy_samples(self) -> List[Dict]:
        dummy_samples = []
        for i in range(50):
            sample = {
                'mesh_path': f"meshes/sample_{i}.obj",
                'vertices': np.random.randn(1000, 3).astype(np.float32),
                'faces': np.random.randint(0, 1000, (2000, 3)),
                'labels': np.random.randint(0, 5, 1000)
            }
            dummy_samples.append(sample)
        
        return dummy_samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        mesh = self._load_mesh(sample['mesh_path'])
        processed_mesh = self.preprocessor.process(mesh)
        
        item = {
            'vertices': processed_mesh['vertices'],
            'faces': processed_mesh['faces'],
            'labels': sample.get('labels', []),
            'metadata': sample.get('metadata', {})
        }
        
        return item
    
    def _load_mesh(self, path: str) -> o3d.geometry.TriangleMesh:
        if os.path.exists(path):
            return o3d.io.read_triangle_mesh(path)
        else:
            return o3d.geometry.TriangleMesh()

class SceneDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "train"):
        self.data_dir = data_dir
        self.split = split
        
        self.samples = self._load_samples()
        self.pc_preprocessor = PointCloudPreprocessor()
    
    def _load_samples(self) -> List[Dict]:
        annotation_file = os.path.join(self.data_dir, f"{self.split}_annotations.json")
        
        if os.path.exists(annotation_file):
            with open(annotation_file, 'r') as f:
                samples = json.load(f)
        else:
            samples = self._generate_dummy_samples()
        
        return samples
    
    def _generate_dummy_samples(self) -> List[Dict]:
        dummy_samples = []
        for i in range(50):
            sample = {
                'scene_path': f"scenes/sample_{i}.ply",
                'pointcloud': np.random.randn(5000, 3).astype(np.float32),
                'object_instances': [
                    {'class': 'chair', 'bbox': [0, 0, 0, 1, 1, 1]},
                    {'class': 'table', 'bbox': [1, 0, 0, 2, 1, 1]}
                ],
                'scene_graph': {}
            }
            dummy_samples.append(sample)
        
        return dummy_samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        pointcloud = sample['pointcloud']
        processed_pc = self.pc_preprocessor.process(pointcloud, 2048)
        
        item = {
            'pointcloud': processed_pc,
            'object_instances': sample.get('object_instances', []),
            'scene_graph': sample.get('scene_graph', {}),
            'metadata': sample.get('metadata', {})
        }
        
        return item