import torch
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import GeometricEngine

def test_engine_initialization():
    engine = GeometricEngine()
    
    assert engine.pointcloud_processor is not None
    assert engine.mesh_processor is not None
    assert engine.scene_reconstructor is not None
    assert engine.scene_understanding is not None

def test_pointcloud_processing():
    engine = GeometricEngine()
    
    sample_pc = torch.randn(100, 3)
    results = engine.process_pointcloud(sample_pc.numpy())
    
    assert "denoised" in results
    assert "normals" in results
    assert "features" in results

def test_mesh_reconstruction():
    engine = GeometricEngine()
    
    sample_pc = torch.randn(100, 3).numpy()
    mesh = engine.reconstruct_mesh_from_pointcloud(sample_pc)
    
    assert mesh is not None
    assert hasattr(mesh, 'vertices')
    assert hasattr(mesh, 'triangles')