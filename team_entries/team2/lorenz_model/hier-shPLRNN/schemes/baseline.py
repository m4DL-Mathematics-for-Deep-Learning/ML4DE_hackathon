import torch
from torch import nn
from schemes.base import base_hierarchisation


class baseline_hierarchisation(base_hierarchisation):
    """'Hierarchisation' scheme for a vanilla/hierarchy free model."""
    def __init__(self, model, args):
        super().__init__(model, args)

    def init_parameters(self):
        """Initializes the model parameters for the projection hierarchisation scheme.
        Standard shPLRNN model parameters."""
        super().init_parameters() # init noise cov
        # init scheme specific params
        r1 = 1.0 / (self.dh ** 0.5)
        r2 = 1.0 / (self.dz ** 0.5)
        self.model.W1 = nn.Parameter(torch.empty(self.num_subjects, self.dz, self.dh).uniform_(-r1, r1))
        self.model.W2 = nn.Parameter(torch.empty(self.num_subjects, self.dh, self.dz).uniform_(-r2, r2))
        self.model.A = nn.Parameter(torch.empty(self.num_subjects, self.dz).uniform_(0.5, 0.9))
        self.model.h2 = nn.Parameter(torch.empty(self.num_subjects, self.dh).uniform_(-r1, r1))
        self.model.h1 = nn.Parameter(torch.zeros(self.num_subjects, self.dz))
    
    def get_parameters(self, subject):
        """Constructs and returns the model parameters.
        Args:
            subject: subject index for each sample in the batch
        returns:
            model parameters for the given subject(s)"""
        return self.model.A[[subject]], self.model.W1[[subject]], self.model.W2[[subject]], self.model.h1[[subject]], self.model.h2[[subject]]
    
    def grouped_parameters(self):
        """Returns a generator object for the individual and the shared 
        parameters respectively."""
        shared, individual = super().grouped_parameters()
        shared += [self.model.A, self.model.W1, self.model.W2, self.model.h1, self.model.h2]
        return shared, individual