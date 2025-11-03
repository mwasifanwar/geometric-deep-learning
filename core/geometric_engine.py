import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import open3d as o3d

class GeometricEngine:
    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.pointcloud_processor = PointCloudProcessor()
        self.mesh_processor = MeshProcessor()
        self.scene_reconstructor = SceneReconstructor()
        self.scene_understanding = SceneUnderstanding()
        
        self._move_to_device()
        
    def _move_to_device(self):
        self.pointcloud_processor.to(self.device)
        self.mesh_processor.to(self.device)
        self.scene_reconstructor.to(self.device)
        self.scene_understanding.to(self.device)
    
    def process_pointcloud(self, pointcloud: np.ndarray, 
                         processing_steps: List[str] = None) -> Dict:
        if processing_steps is None:
            processing_steps = ["denoising", "normal_estimation", "feature_extraction"]
        
        results = {}
        
        if "denoising" in processing_steps:
            results["denoised"] = self.pointcloud_processor.denoise(pointcloud)
        
        if "normal_estimation" in processing_steps:
            results["normals"] = self.pointcloud_processor.estimate_normals(pointcloud)
        
        if "feature_extraction" in processing_steps:
            results["features"] = self.pointcloud_processor.extract_features(pointcloud)
        
        if "segmentation" in processing_steps:
            results["segments"] = self.pointcloud_processor.segment(pointcloud)
        
        return results
    
    def reconstruct_mesh_from_pointcloud(self, pointcloud: np.ndarray, 
                                       method: str = "poisson") -> o3d.geometry.TriangleMesh:
        return self.scene_reconstructor.pointcloud_to_mesh(pointcloud, method)
    
    def understand_scene(self, pointcloud: np.ndarray, 
                       understanding_tasks: List[str] = None) -> Dict:
        if understanding_tasks is None:
            understanding_tasks = ["object_detection", "semantic_segmentation", "scene_graph"]
        
        results = {}
        
        if "object_detection" in understanding_tasks:
            results["objects"] = self.scene_understanding.detect_objects(pointcloud)
        
        if "semantic_segmentation" in understanding_tasks:
            results["semantics"] = self.scene_understanding.semantic_segmentation(pointcloud)
        
        if "scene_graph" in understanding_tasks:
            results["scene_graph"] = self.scene_understanding.build_scene_graph(pointcloud)
        
        if "spatial_relations" in understanding_tasks:
            results["spatial_relations"] = self.scene_understanding.analyze_spatial_relations(pointcloud)
        
        return results
    
    def process_mesh(self, mesh: o3d.geometry.TriangleMesh,
                   processing_steps: List[str] = None) -> Dict:
        if processing_steps is None:
            processing_steps = ["simplification", "remeshing", "feature_extraction"]
        
        results = {}
        
        if "simplification" in processing_steps:
            results["simplified"] = self.mesh_processor.simplify(mesh)
        
        if "remeshing" in processing_steps:
            results["remeshed"] = self.mesh_processor.remesh(mesh)
        
        if "feature_extraction" in processing_steps:
            results["features"] = self.mesh_processor.extract_features(mesh)
        
        return results
    
    def complete_3d_pipeline(self, input_data, pipeline_type: str = "reconstruction") -> Dict:
        if pipeline_type == "reconstruction":
            return self._reconstruction_pipeline(input_data)
        elif pipeline_type == "understanding":
            return self._understanding_pipeline(input_data)
        elif pipeline_type == "completion":
            return self._completion_pipeline(input_data)
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
    
    def _reconstruction_pipeline(self, pointcloud: np.ndarray) -> Dict:
        processed_pc = self.process_pointcloud(pointcloud)
        mesh = self.reconstruct_mesh_from_pointcloud(processed_pc["denoised"])
        processed_mesh = self.process_mesh(mesh)
        scene_understanding = self.understand_scene(pointcloud)
        
        return {
            "pointcloud_processing": processed_pc,
            "mesh_reconstruction": mesh,
            "mesh_processing": processed_mesh,
            "scene_understanding": scene_understanding
        }
    
    def _understanding_pipeline(self, pointcloud: np.ndarray) -> Dict:
        scene_analysis = self.understand_scene(pointcloud)
        geometric_features = self.pointcloud_processor.extract_geometric_features(pointcloud)
        structural_analysis = self.scene_understanding.analyze_structure(pointcloud)
        
        return {
            "scene_analysis": scene_analysis,
            "geometric_features": geometric_features,
            "structural_analysis": structural_analysis
        }
    
    def _completion_pipeline(self, partial_pointcloud: np.ndarray) -> Dict:
        completed_pc = self.scene_reconstructor.complete_pointcloud(partial_pointcloud)
        reconstructed_mesh = self.reconstruct_mesh_from_pointcloud(completed_pc)
        scene_understanding = self.understand_scene(completed_pc)
        
        return {
            "completed_pointcloud": completed_pc,
            "reconstructed_mesh": reconstructed_mesh,
            "scene_understanding": scene_understanding
        }
