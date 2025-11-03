import torch
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import PointCloudProcessor

def test_pointcloud_processor_initialization():
    processor = PointCloudProcessor()
    
    assert processor.pointnet is not None
    assert processor.dgcnn is not None
    assert processor.denoising_net is not None
    assert processor.normal_estimation_net is not None

def test_feature_extraction():
    processor = PointCloudProcessor()
    
    sample_pc = torch.randn(1, 100, 3)
    features = processor.extract_features(sample_pc.numpy())
    
    assert "pointnet_features" in features
    assert "dgcnn_features" in features
    assert "geometric_features" in features

def test_pointcloud_denoising():
    processor = PointCloudProcessor()
    
    sample_pc = torch.randn(100, 3)
    denoised = processor.denoise(sample_pc.numpy())
    
    assert denoised.shape == sample_pc.shape