import torch
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchdiffeq import odeint
import argparse

from models import create_model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Lorenz system parameters
sigma = 10
rho = 28
beta = 8/3


def lorenz(t, state):
    """Original Lorenz system"""
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return torch.stack([dx, dy, dz])


def load_data(file_path):
    """Load training data from .npy file"""
    data = np.load(file_path)
    trajectories = torch.from_numpy(data).float()  # Shape: [T, 3]
    print(f"Loaded data shape: {trajectories.shape}")
    t = torch.linspace(0, 1, trajectories.shape[0])  # Shape: [T]
    # Add batch dimension if not present
    trajectories = trajectories.unsqueeze(0)  # Shape: [1, T, 3]
    return t, trajectories


class LorenzDataset(Dataset):
    def __init__(self, t, trajectories):
        self.t = t  # Shape: [T]
        self.trajectories = trajectories  # Shape: [N, T, 3]
        
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        # Return time vector and single trajectory
        # t: [T], trajectory: [T, 3]
        return self.t, self.trajectories[idx]


def plot_losses(losses):
    plt.figure(figsize=(10, 4))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Training Loss')
    plt.savefig('training_loss.png')
    plt.show()
    plt.close()
    plt.close()


def plot_3d_trajectories(*trajectories, labels=None, title="3D Trajectories", save_path="trajectories.png"):
    """Plot multiple trajectories in 3D.
    Args:
        *trajectories: Variable number of trajectories, each shape [T, 3]
        labels: List of labels for each trajectory
        title: Plot title
        save_path: Where to save the plot
    """
    if labels is None:
        labels = [f"Trajectory {i+1}" for i in range(len(trajectories))]
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a colormap for time
    T = trajectories[0].shape[0]
    time_points = np.linspace(0, 1, T)
    cmap = plt.cm.viridis
    
    for traj, label in zip(trajectories, labels):
        points = ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], 
                          c=time_points, cmap=cmap, label=label)
        # Add faint lines connecting points
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(points)
    cbar.set_label('Time')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    plt.title(title)
    plt.savefig(save_path)
    plt.show()
    plt.close()
    plt.close()

def plot_time_series(*trajectories, t=None, labels=None, title="Time Series", save_path="time_series.png"):
    """Plot time series of each coordinate.
    Args:
        *trajectories: Variable number of trajectories, each shape [T, 3]
        t: Time points [T]. If None, uses range(T)
        labels: List of labels for each trajectory
        title: Plot title
        save_path: Where to save the plot
    """
    if labels is None:
        labels = [f"Trajectory {i+1}" for i in range(len(trajectories))]
    if t is None:
        t = np.arange(trajectories[0].shape[0])
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    coords = ['x', 'y', 'z']
    
    for i, (ax, coord) in enumerate(zip(axes, coords)):
        for traj, label in zip(trajectories, labels):
            ax.plot(t, traj[:, i], label=label)
        ax.set_ylabel(coord)
        if i == 0:
            ax.legend()
    
    axes[-1].set_xlabel('Time')
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()


def plot_trajectory_stats(*trajectories, labels=None, title="Trajectory Statistics", save_path="trajectory_stats.png"):
    """Plot histograms of each coordinate for multiple trajectories.
    Args:
        *trajectories: Variable number of trajectories, each shape [T, 3]
        labels: List of labels for each trajectory
        title: Plot title
        save_path: Where to save the plot
    """
    if labels is None:
        labels = [f"Trajectory {i+1}" for i in range(len(trajectories))]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    coords = ['x', 'y', 'z']
    
    for i, ax in enumerate(axes):
        for traj, label in zip(trajectories, labels):
            ax.hist(traj[:, i], bins=30, alpha=0.5, label=label)
        ax.set_xlabel(coords[i])
        ax.set_ylabel('Count')
        ax.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train Neural ODE for Lorenz System')
    parser.add_argument('--data_path', type=str, 
                        default='/Users/jamesrowbottom/workspace/ML4DE_hackathon/data/lorenz_training.npy',
                        help='Path to training data')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension of the neural network')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate for optimizer')
    parser.add_argument('--n_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--model_type', type=str, default='mlp',
                        help='Type of model to use')
    parser.add_argument('--save_model', type=str, default='lorenz_node_model.pt',
                        help='Path to save the trained model')
    parser.add_argument('--load_model', type=str, default=None,
                        help='Path to load a pretrained model')
    return parser.parse_args()


def main():
    # Get command line arguments
    args = get_args()
    
    # Load training data
    t, trajectories = load_data(args.data_path)
    dataset = LorenzDataset(t, trajectories)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model and optimizer
    model = create_model(args.model_type, hidden_dim=args.hidden_dim).to(device)
    
    # Load pretrained model if specified
    if args.load_model:
        print(f"Loading model from {args.load_model}")
        model.load_state_dict(torch.load(args.load_model))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    losses = []

    print(f"Starting training with parameters:")
    print(f"Hidden dim: {args.hidden_dim}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Number of epochs: {args.n_epochs}")
    
    for epoch in range(args.n_epochs):
        epoch_loss = 0
        for batch_t, batch_y in dataloader:
            optimizer.zero_grad()
            
            batch_t = batch_t.to(device)  # Shape: [T]
            batch_y = batch_y.to(device)  # Shape: [B, T, 3]
            
            pred_y = model(batch_t, batch_y)            
            loss = model.get_loss(pred_y, batch_y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}, Loss: {avg_loss:.6f}')

    # Save model
    torch.save(model.state_dict(), args.save_model)
    
    # Plot results
    print("Plotting results...")
    plot_losses(losses)
    
    # Plot training data
    plot_3d_trajectories(
        trajectories[0].numpy(),
        labels=["Training Data"],
        title="Training Trajectory",
        save_path="training_data.png"
    )
    plot_trajectory_stats(
        trajectories[0].numpy(),
        labels=["Training Data"],
        title="Training Data Statistics",
        save_path="training_stats.png"
    )
    
    # Plot model prediction vs truth
    with torch.no_grad():
        model.eval()
        pred_traj = model(batch_t, batch_y)[0].cpu().numpy()
        
    plot_3d_trajectories(
        trajectories[0].numpy(),
        pred_traj,
        labels=["True", "Predicted"],
        title="True vs Predicted Trajectory",
        save_path="prediction.png"
    )
    plot_time_series(
        trajectories[0].numpy(),
        pred_traj,
        t=t.numpy(),
        labels=["True", "Predicted"],
        title="Coordinate Time Series",
        save_path="time_series.png"
    )
    plot_trajectory_stats(
        trajectories[0].numpy(),
        pred_traj,
        labels=["True", "Predicted"],
        title="Trajectory Statistics Comparison",
        save_path="prediction_stats.png"
    )
    
    print(f"Training complete! Model saved as {args.save_model}")
    print("Check the generated .png files for visualizations.")


if __name__ == "__main__":
    main()
