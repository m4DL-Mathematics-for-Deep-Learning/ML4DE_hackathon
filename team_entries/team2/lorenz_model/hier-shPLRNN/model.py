import torch
from torch import nn
from schemes.util import get_scheme_by_name
from plotter import Plotter
from evaluator import Evaluator
import numpy as np
from saving import Saver


def nll_loss(input, target, log_cov):
    md = 0.5 * torch.sum((input - target)**2 / torch.exp(log_cov), dim=-1)
    logdet = 0.5 * log_cov.sum(dim=-1)
    return torch.mean(logdet + md)


class shallowPLRNN(nn.Module):
    def __init__(self, args, dataset):
        super(shallowPLRNN, self).__init__()
        """Implementation of the shallow PLRNN model with hierarchical parameterization.
        Args:
            args: command line arguments
            dataset: dataset to train on, needed for plotting and evaluation
        """
        # save args
        self.dh = args.hidden_size
        self.dx = args.obs_size
        self.dz = args.latent_size if args.latent_size is not None else self.dx
        self.df = args.forcing_size if args.forcing_size is not None else self.dz
        if args.obs_model == 'identity':
            # if identity observation model, can only force dx dimensions
            self.df = self.dx
        self.num_subjects = dataset.num_subjects
        # handle teacher forcing stuff
        self.tf_alpha = args.tf_alpha_start
        self.tf_gamma = 1 if args.tf_alpha_end == args.tf_alpha_start else np.power(args.tf_alpha_end/args.tf_alpha_start, 1/args.num_epochs)
        # save plotter evaluator and saver
        self.plotter = Plotter(self, dataset)
        self.evaluator = Evaluator(self, args, dataset)
        self.saver = Saver(self, args, dataset)
        # initialize hierarchisation scheme, this handles the parameters
        self.hierarchisation_scheme = get_scheme_by_name(args.hierarchisation_scheme)(self, args)
        self.hierarchisation_scheme.init_parameters()
        self.latent_step = self.vanilla_step
        if args.clipped:
            self.latent_step = self.clipped_step
        self.device = args.device
        self.to(self.device)
    
    def encode(self, x):
        """'Encodes' the observations to the latent space."""
        pinv = torch.pinverse(self.obs_matrix)
        return x @ pinv

    def decode(self, z):
        """'Decodes' latent states back to the observation space."""
        return z @ self.obs_matrix
    
    def teacher_force(self, z_est, z_gt):
        """Generalized teacher forcing. The estimated latent state
        is linearly interpolated with the ground truth latent state.
        Forces only the first df dimensions of the latent state.
        Args:
            z_est: estimated latent state
            z_gt: ground truth latent state
        returns:
            z: teacher forced latent state
        """
        ans = z_est
        ans[..., :self.df] = self.tf_alpha * z_gt[..., :self.df] + (1 - self.tf_alpha) * z_est[..., :self.df]
        return ans
    
    def vanilla_step(self, z, A, W1, W2, h1, h2):
        """Vanilla shPLRNN state update step.
        Args:
            z: latent state
            ...: model parameters
        returns:
            z: next latent state"""
        return A * z + torch.einsum('bij,bj->bi', W1, torch.relu(torch.einsum('bij,bj->bi', W2, z) + h2)) + h1
    
    def clipped_step(self, z, A, W1, W2, h1, h2):
        """Clipped shPLRNN state update step.
        Args:
            z: latent state
            ...: model parameters
        returns:
            z: next latent state"""
        return A * z + torch.einsum('bij,bj->bi', W1, torch.relu(torch.einsum('bij,bj->bi', W2, z) + h2) - torch.relu(torch.einsum('bij,bj->bi', W2, z))) + h1

    def forward(self, x_gt, subject):
        """Forward pass of the shallow PLRNN model.
        Args:
            z_gt: ground truth latent states. Either from an encoder
                or simply equal to the observations. Expected to have
                shape (batch_size, timesteps, latent_size)
            subject: subject index for each sample in the batch"""
        # encode observations to forcing signals
        forcing_signals = self.encode(x_gt)
        # get model parameters
        params = self.hierarchisation_scheme.get_parameters(subject)
        # prepare output tensor
        B, T, _ = x_gt.shape
        Z = torch.empty(T, B, self.dz, device=self.device)
        # initialize latent state
        z = torch.zeros(B, self.dz, device=self.device)
        z[:,:self.df] = forcing_signals[:,0,:self.df]
        
        # iterate over timesteps
        for t in range(T):
            z = self.teacher_force(z, forcing_signals[:,t])
            z = self.latent_step(z, *params)
            Z[t] = z
        return self.decode(Z.permute(1,0,2))
    
    @torch.no_grad()
    def generate_free_trajectory(self, init, T, subject):
        """Generates a free trajectory.
        Args:
            init: initial condition
            T: length of the trajectory
            subject: subject to generate the trajectory for
                can be vectorized, then pass initial conditions 
                for each subject
        Returns:
            trajectory: generated trajectory
        """
        if not isinstance(subject, int):
            # assume vectorized
            assert len(init) == len(subject)
        if init is None:
            # random init
            init = torch.zeros(1, self.dz, device=self.device)
        else:
            # assume an observation state was passed
            if init.ndim == 1:
                # unsqueeze first dimension
                init = init.unsqueeze(0)
            init = self.encode(init)
        S, dz = init.shape
        # initialize trajectory
        trajectory = torch.zeros(T, S, dz, device=self.device)
        # set initial condition
        trajectory[0] = init
        # generate trajectory
        params = self.hierarchisation_scheme.get_parameters(subject)
        for t in range(1, T):
            trajectory[t] = self.latent_step(trajectory[t-1], *params)
        return self.decode(trajectory.permute(1,0,2))