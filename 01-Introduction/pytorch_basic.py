import torch

def describe(x):
    print(f'Type: {x.type()}')
    print(f'Shape: {x.shape}')
    print(f'Value: \n{x}')

x = torch.tensor([[1, 2, 3], [5, 6, 7]])
x = torch.unsqueeze(x, 0)
describe(x)
x = torch.squeeze(x, 0)
describe(x)
