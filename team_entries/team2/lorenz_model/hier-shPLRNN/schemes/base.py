import torch
from torch import nn
from matplotlib import pyplot as plt


class base_hierarchisation():
    def __init__(self, model, args):
        """Base class for hierarchisation schemes."""
        self.model = model
        self.num_subjects = model.num_subjects
        self.dh = model.dh
        self.dx = model.dx
        self.dz = model.dz
        self.df = model.df
        self.obs_model = args.obs_model
        self.learn_noise_cov = args.learn_noise_cov
        assert self.obs_model in ["identity", "linear"]

    def re_init(self):
        """Re-initializes all subject specific parameters for finetuning."""
        self.model.noise_cov = nn.Parameter(torch.zeros(size=(self.num_subjects, 1, self.dx)))
    
    def init_parameters(self):
        """Initializes the model parameters for the given hierarchisation scheme.
        Initializes the covariance of the gaussian noise model to all ones
        (Here: log cov -> to zeros)."""
        self.model.noise_cov = nn.Parameter(torch.zeros(size=(self.num_subjects, 1, self.dx)), requires_grad=self.learn_noise_cov)
        # obs model
        if self.obs_model == "identity":
            self.model.obs_matrix = nn.Parameter(torch.eye(self.dz, self.dx), requires_grad=False)
        else:
            self.model.obs_matrix = nn.Parameter(torch.randn(self.dz, self.dx))
    
    def get_parameters(self, subject):
        """Constructs and returns the model parameters.
        Args:
            subject: subject index for each sample in the batch
        returns:
            model parameters for the given subject(s)"""
        pass
    
    def grouped_parameters(self):
        """Returns a generator object for the individual and the shared 
        parameters respectively."""
        return [self.model.obs_matrix], [self.model.noise_cov]
    
    @torch.no_grad()
    def plot_stuff(self):
        """Function for viualizing interesting things. This is called by a plotter object
        to write it to tensorboard. Plots the noise covariance as an image"""
        fig, ax = plt.subplots(1, figsize=(self.num_subjects+1, self.dz))
        c = ax.imshow(self.model.noise_cov.squeeze(1).cpu().T)
        fig.colorbar(c, ax=ax)
        ax.set_ylabel(r"$\sigma^2_i$")
        ax.set_xlabel("subject")
        return [(fig, "noise_covariance")]
    
    def loss(self):
        """Defines any regularization losses on the parameters."""
        return .0