import torch
import torch.nn as nn
import torch.nn.functional as F

class MeshCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = MeshConv(3, 64)
        self.conv2 = MeshConv(64, 128)
        self.conv3 = MeshConv(128, 256)
        self.conv4 = MeshConv(256, 512)
        
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        self.pool1 = MeshPool()
        self.pool2 = MeshPool()
        self.pool3 = MeshPool()
        
    def forward(self, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        x = vertices
        
        x = F.relu(self.conv1(x, faces))
        x = self.pool1(x, faces)
        
        x = F.relu(self.conv2(x, faces))
        x = self.pool2(x, faces)
        
        x = F.relu(self.conv3(x, faces))
        x = self.pool3(x, faces)
        
        x = F.relu(self.conv4(x, faces))
        
        x = torch.max(x, 1)[0]
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)

class SpiralNet(nn.Module):
    def __init__(self, spiral_length: int = 9, in_dim: int = 3, hidden_dims: List[int] = [64, 128, 256]):
        super().__init__()
        self.spiral_length = spiral_length
        
        self.layers = nn.ModuleList()
        prev_dim = in_dim
        
        for hidden_dim in hidden_dims:
            self.layers.append(SpiralConv(prev_dim, hidden_dim, spiral_length))
            prev_dim = hidden_dim
        
        self.output_proj = nn.Linear(hidden_dims[-1], hidden_dims[-1])
    
    def forward(self, x: torch.Tensor, spirals: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, spirals)
        
        x = torch.max(x, 1)[0]
        return self.output_proj(x)

class SpiralConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, spiral_length: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.spiral_length = spiral_length
        
        self.conv = nn.Conv1d(in_dim * spiral_length, out_dim, 1)
        self.bn = nn.BatchNorm1d(out_dim)
    
    def forward(self, x: torch.Tensor, spirals: torch.Tensor) -> torch.Tensor:
        batch_size, num_vertices, _ = x.shape
        
        spiral_features = []
        for b in range(batch_size):
            batch_spirals = spirals[b]
            batch_features = []
            
            for v in range(num_vertices):
                spiral_indices = batch_spirals[v]
                spiral_feat = x[b, spiral_indices].view(-1)
                batch_features.append(spiral_feat)
            
            spiral_features.append(torch.stack(batch_features))
        
        spiral_features = torch.stack(spiral_features)
        spiral_features = spiral_features.transpose(1, 2)
        
        x = F.relu(self.bn(self.conv(spiral_features)))
        return x.transpose(1, 2)