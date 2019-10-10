import torch
import torch.nn as nn

class Perceptron(nn.Module):
    def __init__(self, input_dim):
        super(Perceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1)

    def forward(self, x_in):
        """
        Forward propagate a Perceptron

        @param x_in (Tensor) tensor of shape (batch, num_features)
        """
        return torch.sigmoid(self.fc1(x_in)).squeeze(-1)
