import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphNeuralNetwork(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 64, output_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None) -> torch.Tensor:
        x = self.input_proj(x)
        
        for layer in self.gnn_layers:
            x = layer(x, edge_index, edge_attr)
        
        x = self.output_proj(x)
        return x

class GNNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.message_net = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        
        self.update_net = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None) -> torch.Tensor:
        row, col = edge_index
        
        source = x[row]
        target = x[col]
        
        if edge_attr is not None:
            message_input = torch.cat([source, target, edge_attr], dim=-1)
        else:
            message_input = torch.cat([source, target], dim=-1)
        
        messages = self.message_net(message_input)
        
        aggregated = torch.zeros_like(x)
        aggregated = aggregated.index_add_(0, col, messages)
        
        update_input = torch.cat([x, aggregated], dim=-1)
        updated = self.update_net(update_input)
        
        return F.relu(updated)

class DynamicGraphCNN(nn.Module):
    def __init__(self, k: int = 20, in_dim: int = 3, hidden_dims: List[int] = [64, 128, 256]):
        super().__init__()
        self.k = k
        self.in_dim = in_dim
        self.hidden_dims = hidden_dims
        
        self.layers = nn.ModuleList()
        prev_dim = in_dim
        
        for hidden_dim in hidden_dims:
            self.layers.append(EdgeConv(prev_dim, hidden_dim, k))
            prev_dim = hidden_dim
        
        self.output_proj = nn.Linear(hidden_dims[-1], hidden_dims[-1])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        
        x = torch.max(x, 1)[0]
        return self.output_proj(x)

class EdgeConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, k: int = 20):
        super().__init__()
        self.k = k
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.conv = nn.Conv2d(in_dim * 2, out_dim, 1)
        self.bn = nn.BatchNorm2d(out_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, _ = x.shape
        
        knn_indices = self._knn(x, self.k)
        edge_features = self._get_edge_features(x, knn_indices)
        
        x = F.relu(self.bn(self.conv(edge_features)))
        x = torch.max(x, -1)[0]
        
        return x
    
    def _knn(self, x: torch.Tensor, k: int) -> torch.Tensor:
        inner = -2 * torch.matmul(x, x.transpose(2, 1))
        xx = torch.sum(x**2, dim=2, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        
        idx = pairwise_distance.topk(k=k, dim=-1)[1]
        return idx
    
    def _get_edge_features(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, _ = x.shape
        k = idx.shape[-1]
        
        idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        
        x = x.view(batch_size * num_points, -1)
        neighbors = x[idx].view(batch_size, num_points, k, self.in_dim)
        x = x.view(batch_size, num_points, 1, self.in_dim).repeat(1, 1, k, 1)
        
        edge_features = torch.cat([x, neighbors - x], dim=-1)
        edge_features = edge_features.permute(0, 3, 1, 2)
        
        return edge_features