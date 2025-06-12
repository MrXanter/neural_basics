import torch.nn as nn
import torch
import torch.optim as optim
from model_loader import X, y, dataset, loader, N, input_dim, num_classes


class MySimpleModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)



model = MySimpleModel(input_dim, num_classes)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)