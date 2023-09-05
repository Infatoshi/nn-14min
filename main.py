import torch
import torch.nn as nn
import torch.optim as optim

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)
    
    def forward(self, input):
        out = self.linear(input)
        return out
    
model = Model()
mse = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

inputs = torch.rand((5, 1), dtype=torch.float32)
factor = 2
targets = inputs * factor

for epoch in range(10000): 
    # Forward pass
    outputs = model.forward(inputs)
    loss = mse(outputs, targets)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'epoch {epoch} loss: {loss:.4f}')