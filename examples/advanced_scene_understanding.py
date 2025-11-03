import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import GeometricEngine
from training import GeometricTrainer, ChamferLoss
from data import PointCloudDataset
from torch.utils.data import DataLoader

def advanced_scene_understanding_demo():
    print("=== Advanced 3D Scene Understanding Demo ===")
    print("Complex scene analysis, object detection, and spatial relationship understanding")
    print("Created by mwasifanwar")
    
    engine = GeometricEngine()
    
    train_dataset = PointCloudDataset("data/train", "train")
    val_dataset = PointCloudDataset("data/val", "val")
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    
    optimizer = torch.optim.Adam(engine.parameters(), lr=0.001)
    criterion = ChamferLoss()
    
    trainer = GeometricTrainer(
        model=engine,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    print("Starting geometric deep learning training...")
    trainer.train(num_epochs=5, save_path="best_geometric_model.pth")
    
    print("Training completed. Performing advanced scene analysis...")
    
    test_pointcloud = np.random.randn(2000, 3).astype(np.float32)
    
    print("1. Complete scene understanding pipeline")
    scene_analysis = engine.complete_3d_pipeline(test_pointcloud, pipeline_type="understanding")
    
    print("Scene Analysis Results:")
    print(f"  Object detection: {len(scene_analysis['scene_analysis']['objects'])} objects")
    print(f"  Geometric features: {len(scene_analysis['geometric_features'])} feature types")
    print(f"  Structural analysis: {len(scene_analysis['structural_analysis'])} structural properties")
    
    print("2. Scene graph construction")
    scene_graph = engine.understand_scene(test_pointcloud, understanding_tasks=["scene_graph"])
    print(f"Scene graph: {len(scene_graph['scene_graph']['objects'])} objects, "
          f"{len(scene_graph['scene_graph']['relations'])} relations")
    
    print("3. Spatial relationship analysis")
    spatial_analysis = engine.understand_scene(test_pointcloud, understanding_tasks=["spatial_relations"])
    print(f"Spatial relations: {len(spatial_analysis['spatial_relations'])} relationships analyzed")
    
    return engine, trainer

if __name__ == "__main__":
    engine, trainer = advanced_scene_understanding_demo()