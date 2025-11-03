import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
from typing import Dict, List, Tuple

class SceneUnderstanding(nn.Module):
    def __init__(self):
        super().__init__()
        self.object_detector = ObjectDetector3D()
        self.semantic_segmenter = SemanticSegmenter3D()
        self.scene_graph_builder = SceneGraphBuilder()
        self.spatial_analyzer = SpatialRelationAnalyzer()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.object_detector(x)
    
    def to(self, device):
        super().to(device)
        self.object_detector.to(device)
        self.semantic_segmenter.to(device)
        return self
    
    def detect_objects(self, pointcloud: np.ndarray) -> List[Dict]:
        if isinstance(pointcloud, np.ndarray):
            pointcloud = torch.from_numpy(pointcloud).float()
        
        with torch.no_grad():
            objects = self.object_detector(pointcloud.unsqueeze(0))
        
        return objects
    
    def semantic_segmentation(self, pointcloud: np.ndarray) -> Dict[str, np.ndarray]:
        if isinstance(pointcloud, np.ndarray):
            pointcloud = torch.from_numpy(pointcloud).float()
        
        with torch.no_grad():
            segmentation = self.semantic_segmenter(pointcloud.unsqueeze(0))
        
        return {
            "labels": segmentation["labels"].squeeze(0).cpu().numpy(),
            "confidences": segmentation["confidences"].squeeze(0).cpu().numpy()
        }
    
    def build_scene_graph(self, pointcloud: np.ndarray) -> Dict:
        objects = self.detect_objects(pointcloud)
        spatial_relations = self.analyze_spatial_relations(pointcloud)
        
        scene_graph = {
            "objects": objects,
            "relations": spatial_relations,
            "global_features": self._extract_global_features(pointcloud)
        }
        
        return scene_graph
    
    def analyze_spatial_relations(self, pointcloud: np.ndarray) -> List[Dict]:
        objects = self.detect_objects(pointcloud)
        
        relations = []
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i != j:
                    relation = self._compute_spatial_relation(obj1, obj2)
                    relations.append(relation)
        
        return relations
    
    def analyze_structure(self, pointcloud: np.ndarray) -> Dict:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud)
        
        bbox = pcd.get_axis_aligned_bounding_box()
        convex_hull = pcd.compute_convex_hull()[0]
        
        return {
            "bounding_box": {
                "min": bbox.min_bound,
                "max": bbox.max_bound,
                "center": bbox.get_center(),
                "extent": bbox.get_extent()
            },
            "convex_hull_volume": convex_hull.get_volume(),
            "point_density": len(pointcloud) / bbox.volume(),
            "structural_complexity": self._compute_structural_complexity(pointcloud)
        }
    
    def _compute_spatial_relation(self, obj1: Dict, obj2: Dict) -> Dict:
        centroid1 = obj1.get("centroid", np.zeros(3))
        centroid2 = obj2.get("centroid", np.zeros(3))
        
        distance = np.linalg.norm(centroid1 - centroid2)
        direction = centroid2 - centroid1
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        
        relation_type = self._classify_spatial_relation(centroid1, centroid2, obj1, obj2)
        
        return {
            "object1": obj1.get("class", "unknown"),
            "object2": obj2.get("class", "unknown"),
            "relation_type": relation_type,
            "distance": distance,
            "direction": direction,
            "confidence": 0.8
        }
    
    def _classify_spatial_relation(self, centroid1: np.ndarray, centroid2: np.ndarray, 
                                 obj1: Dict, obj2: Dict) -> str:
        diff = centroid2 - centroid1
        
        if abs(diff[2]) > max(abs(diff[0]), abs(diff[1])):
            return "above" if diff[2] > 0 else "below"
        elif abs(diff[0]) > abs(diff[1]):
            return "right" if diff[0] > 0 else "left"
        else:
            return "behind" if diff[1] > 0 else "front"
    
    def _extract_global_features(self, pointcloud: np.ndarray) -> Dict[str, float]:
        if len(pointcloud) == 0:
            return {}
        
        centroid = np.mean(pointcloud, axis=0)
        covariance = np.cov(pointcloud.T)
        eigenvalues = np.linalg.eigvalsh(covariance)
        
        return {
            "centroid": centroid,
            "eigenvalues": eigenvalues,
            "spread": np.max(eigenvalues) - np.min(eigenvalues),
            "compactness": eigenvalues[2] / (eigenvalues[0] + 1e-8)
        }
    
    def _compute_structural_complexity(self, pointcloud: np.ndarray) -> float:
        if len(pointcloud) < 10:
            return 0.0
        
        from scipy.spatial import ConvexHull
        try:
            hull = ConvexHull(pointcloud)
            volume_ratio = hull.volume / (np.prod(np.ptp(pointcloud, axis=0)) + 1e-8)
            return 1.0 - volume_ratio
        except:
            return 0.5

class ObjectDetector3D(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        
        self.backbone = DGCNNFeatureExtractor(256)
        self.detection_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 7 + num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> List[Dict]:
        batch_size, num_points, _ = x.shape
        
        features = self.backbone(x)
        detections = self.detection_head(features)
        
        objects = []
        for b in range(batch_size):
            obj = {
                "class": int(torch.argmax(detections[b, 7:])),
                "confidence": float(torch.max(F.softmax(detections[b, 7:], dim=0))),
                "centroid": detections[b, :3].cpu().numpy(),
                "size": detections[b, 3:6].cpu().numpy(),
                "rotation": float(detections[b, 6])
            }
            objects.append(obj)
        
        return objects

class SemanticSegmenter3D(nn.Module):
    def __init__(self, num_classes: int = 8):
        super().__init__()
        self.num_classes = num_classes
        
        self.encoder = DGCNNFeatureExtractor(256)
        self.decoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, num_points, _ = x.shape
        
        point_features = self.encoder(x)
        
        expanded_features = point_features.unsqueeze(1).repeat(1, num_points, 1)
        
        logits = self.decoder(expanded_features)
        
        return {
            "labels": torch.argmax(logits, dim=-1),
            "confidences": F.softmax(logits, dim=-1).max(dim=-1)[0]
        }

class SceneGraphBuilder:
    def __init__(self):
        self.relation_classifier = RelationClassifier()
    
    def build(self, objects: List[Dict], pointcloud: np.ndarray) -> Dict:
        graph = {
            "nodes": [],
            "edges": [],
            "global_context": self._extract_global_context(pointcloud)
        }
        
        for i, obj in enumerate(objects):
            graph["nodes"].append({
                "id": i,
                "class": obj.get("class", "unknown"),
                "centroid": obj.get("centroid", np.zeros(3)),
                "features": obj.get("features", np.zeros(128))
            })
        
        for i in range(len(objects)):
            for j in range(len(objects)):
                if i != j:
                    relation = self.relation_classifier.classify(
                        objects[i], objects[j], pointcloud
                    )
                    graph["edges"].append({
                        "source": i,
                        "target": j,
                        "relation": relation
                    })
        
        return graph
    
    def _extract_global_context(self, pointcloud: np.ndarray) -> Dict:
        return {
            "scene_centroid": np.mean(pointcloud, axis=0),
            "scene_bounds": np.ptp(pointcloud, axis=0),
            "point_density": len(pointcloud) / np.prod(np.ptp(pointcloud, axis=0))
        }

class RelationClassifier:
    def classify(self, obj1: Dict, obj2: Dict, pointcloud: np.ndarray) -> str:
        centroid1 = obj1.get("centroid", np.zeros(3))
        centroid2 = obj2.get("centroid", np.zeros(3))
        
        distance = np.linalg.norm(centroid1 - centroid2)
        
        if distance < 0.5:
            return "close"
        elif distance < 2.0:
            return "medium"
        else:
            return "far"

class SpatialRelationAnalyzer:
    def analyze(self, pointcloud: np.ndarray, objects: List[Dict]) -> List[Dict]:
        relations = []
        
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i != j:
                    relation = self._compute_detailed_relation(obj1, obj2, pointcloud)
                    relations.append(relation)
        
        return relations
    
    def _compute_detailed_relation(self, obj1: Dict, obj2: Dict, pointcloud: np.ndarray) -> Dict:
        centroid1 = obj1.get("centroid", np.zeros(3))
        centroid2 = obj2.get("centroid", np.zeros(3))
        
        direction = centroid2 - centroid1
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            direction = direction / distance
        
        vertical_relation = self._classify_vertical_relation(centroid1, centroid2)
        horizontal_relation = self._classify_horizontal_relation(centroid1, centroid2)
        
        return {
            "objects": [obj1.get("class", "unknown"), obj2.get("class", "unknown")],
            "distance": distance,
            "vertical_relation": vertical_relation,
            "horizontal_relation": horizontal_relation,
            "direction_vector": direction,
            "spatial_context": self._get_spatial_context(centroid1, centroid2, pointcloud)
        }
    
    def _classify_vertical_relation(self, pos1: np.ndarray, pos2: np.ndarray) -> str:
        height_diff = pos2[2] - pos1[2]
        
        if abs(height_diff) < 0.1:
            return "same_level"
        elif height_diff > 0.5:
            return "significantly_above"
        elif height_diff > 0.1:
            return "slightly_above"
        elif height_diff < -0.5:
            return "significantly_below"
        else:
            return "slightly_below"
    
    def _classify_horizontal_relation(self, pos1: np.ndarray, pos2: np.ndarray) -> str:
        horizontal_diff = pos2[:2] - pos1[:2]
        distance_2d = np.linalg.norm(horizontal_diff)
        
        if distance_2d < 0.3:
            return "adjacent"
        elif distance_2d < 1.0:
            return "near"
        else:
            return "far"
    
    def _get_spatial_context(self, pos1: np.ndarray, pos2: np.ndarray, pointcloud: np.ndarray) -> Dict:
        midpoint = (pos1 + pos2) / 2
        distances = np.linalg.norm(pointcloud - midpoint, axis=1)
        nearby_points = pointcloud[distances < 1.0]
        
        return {
            "nearby_point_density": len(nearby_points),
            "midpoint_environment": self._classify_environment(nearby_points)
        }
    
    def _classify_environment(self, points: np.ndarray) -> str:
        if len(points) == 0:
            return "empty"
        
        height_var = np.var(points[:, 2])
        if height_var < 0.01:
            return "flat"
        elif height_var < 0.1:
            return "gentle"
        else:
            return "rugged"
