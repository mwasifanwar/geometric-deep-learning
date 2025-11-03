import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import GeometricEngine

def basic_3d_reconstruction_demo():
    print("=== Geometric Deep Learning for 3D Scene Understanding Demo ===")
    print("Advanced 3D reconstruction and scene understanding using geometric deep learning")
    print("Created by mwasifanwar")
    
    engine = GeometricEngine()
    
    print("1. Generating sample point cloud")
    sample_pointcloud = np.random.randn(1000, 3).astype(np.float32)
    print(f"Sample point cloud shape: {sample_pointcloud.shape}")
    
    print("2. Processing point cloud")
    processing_results = engine.process_pointcloud(
        sample_pointcloud,
        processing_steps=["denoising", "normal_estimation", "feature_extraction"]
    )
    
    print("Processing Results:")
    print(f"  Denoised points: {processing_results['denoised'].shape}")
    print(f"  Estimated normals: {processing_results['normals'].shape}")
    print(f"  Extracted features: {len(processing_results['features'])} feature types")
    
    print("3. Mesh reconstruction from point cloud")
    mesh = engine.reconstruct_mesh_from_pointcloud(sample_pointcloud, method="poisson")
    print(f"Reconstructed mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    
    print("4. Scene understanding")
    understanding_results = engine.understand_scene(
        sample_pointcloud,
        understanding_tasks=["object_detection", "semantic_segmentation"]
    )
    
    print("Scene Understanding Results:")
    print(f"  Detected objects: {len(understanding_results['objects'])}")
    print(f"  Semantic segmentation: {understanding_results['semantics']['labels'].shape} labels")
    
    print("5. Complete 3D pipeline")
    pipeline_results = engine.complete_3d_pipeline(sample_pointcloud, pipeline_type="reconstruction")
    print("Pipeline completed successfully")
    
    return engine

if __name__ == "__main__":
    engine = basic_3d_reconstruction_demo()