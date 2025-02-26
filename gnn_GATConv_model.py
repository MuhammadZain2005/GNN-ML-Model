import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, BatchNorm

# This model uses GATConv
class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Improved GNN model using GATConv for attention-based learning.
        Args:
            input_dim: Number of node features
            hidden_dim: Size of hidden layers
            output_dim: Output size (1 for energy prediction)
        """
        super(GNNModel, self).__init__()

        # GAT Layers (Using multi-head attention)
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=True)
        self.conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True)
        self.conv3 = GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True)

        # Batch Normalization
        self.bn1 = BatchNorm(hidden_dim * 4)
        self.bn2 = BatchNorm(hidden_dim * 4)
        self.bn3 = BatchNorm(hidden_dim * 4)

        # Fully Connected Layers
        self.fc1 = torch.nn.Linear(hidden_dim * 4, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        node_types = data.node_types  # Custom feature for boron and carbon weights

        # Apply different attention scaling for boron and carbon nodes
        attention_weights = torch.where(node_types == 0, 1.5, 1.0)  # Example: Boron gets 1.5x attention
        x = self.conv1(x, edge_index) * attention_weights.unsqueeze(-1)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index) * attention_weights.unsqueeze(-1)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x, edge_index) * attention_weights.unsqueeze(-1)
        x = self.bn3(x)
        x = F.relu(x)

        x = global_mean_pool(x, data.batch)  # Global pooling
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
