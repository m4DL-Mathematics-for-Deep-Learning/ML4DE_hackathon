import torch
import torch.nn as nn
from torchdiffeq import odeint

# Base model class for all models
class BaseModel(nn.Module):
    def forward(self, t, x):
        raise NotImplementedError
    
    def get_loss(self, pred, target):
        """Default L2 loss between prediction and target"""
        return torch.mean((pred - target) ** 2)

# Neural ODE specific models
class BaseODEFunc(BaseModel):
    def forward(self, t, y):
        raise NotImplementedError

class SimpleODEFunc(BaseODEFunc):
    def __init__(self, hidden_dim=64):
        super(SimpleODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3)
        )
    
    def forward(self, t, y):
        return self.net(y)

class DeepODEFunc(BaseODEFunc):
    def __init__(self, hidden_dim=64):
        super(DeepODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3)
        )
    
    def forward(self, t, y):
        return self.net(y)

class ResODEFunc(BaseODEFunc):
    def __init__(self, hidden_dim=64):
        super(ResODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3)
        )
        self.residual = nn.Linear(3, 3)
    
    def forward(self, t, y):
        return self.net(y) + self.residual(y)

class NeuralODE(BaseModel):
    def __init__(self, func):
        super(NeuralODE, self).__init__()
        self.func = func
        
    def forward(self, t, x0):
        return odeint(self.func, x0, t)

# Example of a non-NODE model
class MLPModel(BaseModel):
    def __init__(self, hidden_dim=64):
        super(MLPModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_dim),  # 3 dims + time
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3)
        )
    
    def forward(self, t, x):
        # t is [T] and x is [batch_size, 3]
        t = t.unsqueeze(-1)
        tx = torch.cat([t, x], dim=-1)
        return self.net(tx)
    
    def get_loss(self, pred, target):
        """Simple MSE loss for MLP model"""
        return torch.mean((pred - target) ** 2)

def create_model(model_type, hidden_dim=64):
    """Create a model based on the specified type"""
    if model_type == 'node_simple':
        return NeuralODE(SimpleODEFunc(hidden_dim))
    elif model_type == 'node_deep':
        return NeuralODE(DeepODEFunc(hidden_dim))
    elif model_type == 'node_residual':
        return NeuralODE(ResODEFunc(hidden_dim))
    elif model_type == 'mlp':
        return MLPModel(hidden_dim)
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. " 
            "Available types: node_simple, node_deep, node_residual, mlp"
        )
