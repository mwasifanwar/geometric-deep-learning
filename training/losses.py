import torch
import torch.nn as nn
import torch.nn.functional as F

class ChamferLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.float()
        
        dist1, dist2 = self._chamfer_distance(pred, target)
        loss = torch.mean(dist1) + torch.mean(dist2)
        return loss
    
    def _chamfer_distance(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.unsqueeze(1)
        y = y.unsqueeze(2)
        
        dist = torch.sum((x - y) ** 2, 3)
        dist1, _ = torch.min(dist, 2)
        dist2, _ = torch.min(dist, 1)
        
        return dist1, dist2

class NormalConsistencyLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred_normals: torch.Tensor, target_normals: torch.Tensor) -> torch.Tensor:
        cosine_sim = F.cosine_similarity(pred_normals, target_normals, dim=-1)
        loss = 1 - torch.mean(cosine_sim)
        return loss

class SceneUnderstandingLoss(nn.Module):
    def __init__(self, 
                 detection_weight: float = 1.0,
                 segmentation_weight: float = 0.5,
                 relation_weight: float = 0.2):
        super().__init__()
        self.detection_weight = detection_weight
        self.segmentation_weight = segmentation_weight
        self.relation_weight = relation_weight
        
        self.detection_loss = nn.CrossEntropyLoss()
        self.segmentation_loss = nn.CrossEntropyLoss()
        self.relation_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, predictions: Dict, targets: Dict) -> torch.Tensor:
        detection_loss = self.detection_loss(
            predictions.get('detection_logits', torch.tensor(0.0)),
            targets.get('detection_labels', torch.tensor(0))
        )
        
        segmentation_loss = self.segmentation_loss(
            predictions.get('segmentation_logits', torch.tensor(0.0)),
            targets.get('segmentation_labels', torch.tensor(0))
        )
        
        relation_loss = self.relation_loss(
            predictions.get('relation_logits', torch.tensor(0.0)),
            targets.get('relation_labels', torch.tensor(0))
        )
        
        total_loss = (self.detection_weight * detection_loss +
                     self.segmentation_weight * segmentation_loss +
                     self.relation_weight * relation_loss)
        
        return total_loss