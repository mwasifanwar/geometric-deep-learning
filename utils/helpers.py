import logging
import json
import torch
import numpy as np
from datetime import datetime

def setup_logging(name: str = "geometric_deep_learning"):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{name}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)

def save_results(results: dict, filename: str = "geometric_results.json"):
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)

def load_checkpoint(checkpoint_path: str, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('best_val_loss', float('inf'))

def calculate_metrics(predictions: dict, targets: dict) -> dict:
    metrics = {}
    
    if 'pointcloud' in predictions and 'pointcloud' in targets:
        chamfer_dist = calculate_chamfer_distance(
            predictions['pointcloud'],
            targets['pointcloud']
        )
        metrics['chamfer_distance'] = chamfer_dist
    
    if 'normals' in predictions and 'normals' in targets:
        normal_accuracy = calculate_normal_accuracy(
            predictions['normals'],
            targets['normals']
        )
        metrics['normal_accuracy'] = normal_accuracy
    
    return metrics

def calculate_chamfer_distance(pred_points: np.ndarray, target_points: np.ndarray) -> float:
    from scipy.spatial import KDTree
    tree_pred = KDTree(pred_points)
    tree_target = KDTree(target_points)
    
    dist1, _ = tree_pred.query(target_points)
    dist2, _ = tree_target.query(pred_points)
    
    chamfer_dist = np.mean(dist1) + np.mean(dist2)
    return float(chamfer_dist)

def calculate_normal_accuracy(pred_normals: np.ndarray, target_normals: np.ndarray) -> float:
    if len(pred_normals) == 0 or len(target_normals) == 0:
        return 0.0
    
    dot_products = np.sum(pred_normals * target_normals, axis=1)
    angles = np.arccos(np.clip(dot_products, -1.0, 1.0))
    accuracy = np.mean(angles < (np.pi / 6))
    
    return float(accuracy)