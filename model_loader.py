import torch
from torch.utils.data import TensorDataset, DataLoader



N = 100
input_dim = 10
num_classes = 2





X = torch.randn(N, input_dim)
y = torch.randint(0, num_classes, (N,))
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=16, shuffle=True)
