import torch
from torch.nn.functional import relu
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        ########################################
        # Previous model:
        # self.conv1 = GCNConv(in_channels, 16)
        # self.conv2 = GCNConv(16, 32)
        # self.conv3 = GCNConv(32, out_channels)
        ########################################
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, 16)
        self.conv4 = GCNConv(16, 8)
        self.conv5 = GCNConv(8, out_channels)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward method
        :param x: Node feature matrix [num_nodes, in_channels]
        :param edge_index: Graph connectivity in COO format [2, num_edges]
        """
        x = relu(self.conv1(x, edge_index))
        x = relu(self.conv2(x, edge_index))
        x = relu(self.conv3(x, edge_index))
        x = relu(self.conv4(x, edge_index))
        x = torch.sigmoid(self.conv5(x, edge_index))
        return x
