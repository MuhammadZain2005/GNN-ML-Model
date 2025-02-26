import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, global_mean_pool, BatchNorm

# This model uses the GIN layer with a simple MLP as the neural network function.
# GINConv

class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Improved GNN model with residual connections and better pooling.
        Args:
            input_dim: Number of node features
            hidden_dim: Size of hidden layers
            output_dim: Output size (1 for energy prediction)
        """
        super(GNNModel, self).__init__()

        # GNN Layers (Using GIN for better feature extraction)
        self.conv1 = GINConv(torch.nn.Linear(input_dim, hidden_dim))
        self.conv2 = GINConv(torch.nn.Linear(hidden_dim, hidden_dim))
        self.conv3 = GINConv(torch.nn.Linear(hidden_dim, hidden_dim))

        # Batch Normalization
        self.bn1 = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)

        # Fully Connected Layers
        self.fc1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)

        # Dropout for Regularization
        self.dropout = torch.nn.Dropout(p=0.3)

    def forward(self, data):
        """
        Forward pass through the model.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device) if batch is None else batch

        # First GNN layer with residual connection
        x1 = F.relu(self.bn1(self.conv1(x, edge_index)))
        x1 = self.dropout(x1)

        # Second GNN layer with residual connection
        x2 = F.relu(self.bn2(self.conv2(x1, edge_index)))
        x2 = self.dropout(x2) + x1  # Residual connection

        # Third GNN layer with residual connection
        x3 = F.relu(self.bn3(self.conv3(x2, edge_index)))
        x3 = self.dropout(x3) + x2  # Residual connection

        # Global Mean Pooling
        x = global_mean_pool(x3, batch)

        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def predict(self, data):
        """
        Prediction mode.
        """
        self.eval()
        with torch.no_grad():
            return self(data)
