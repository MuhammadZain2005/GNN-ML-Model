import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initialize GNN model
        Args:
            input_dim: Number of node features
            hidden_dim: Size of hidden layers
            output_dim: Size of output (1 for energy prediction)
        """
        super(GNNModel, self).__init__()
        
        # GNN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Batch normalization layers
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn3 = torch.nn.BatchNorm1d(hidden_dim)
        
        # Output layers
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        
        # Dropout
        self.dropout = torch.nn.Dropout(p=0.2)
        
    def forward(self, data):
        """
        Forward pass
        Args:
            data: PyG Data object containing:
                - x: Node features
                - edge_index: Graph connectivity
                - batch: Batch indices for multiple graphs
        Returns:
            Graph-level prediction
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # If batch is None (single graph), create dummy batch
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # First GNN layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second GNN layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Third GNN layer
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # MLP layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def predict(self, data):
        """
        Make prediction for a single graph or batch
        Args:
            data: PyG Data object
        Returns:
            Predicted values
        """
        self.eval()
        with torch.no_grad():
            return self(data)
