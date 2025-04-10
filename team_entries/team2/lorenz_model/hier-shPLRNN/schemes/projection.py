import torch
from torch import nn
from matplotlib import pyplot as plt
from schemes.base import base_hierarchisation


class projection_hierarchisation(base_hierarchisation):
    def __init__(self, model, args):
        super().__init__(model, args)
        self.dp = args.num_individual_params
    
    def re_init(self):
        super().init_parameters()
        self.model.p_vector = nn.Parameter(torch.repeat_interleave(torch.empty(1, self.dp).uniform_(-1, 1), self.num_subjects, dim=0))

    def init_parameters(self):
        """Initializes the model parameters for the projection hierarchisation scheme.
        Each subject has an individual parameter vector, which is mapped onto the required
        model parameters by a shared projection matrix."""
        super().init_parameters() # init noise cov
        # init scheme specific params
        self.model.p_vector = nn.Parameter(torch.repeat_interleave(torch.empty(1, self.dp).uniform_(-1, 1), self.num_subjects, dim=0))
        self.model.p2A = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.dp, self.dz), gain=.1))
        self.model.p2W1 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.dp, self.dz, self.dh), gain=.1))
        self.model.p2W2 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.dp, self.dh, self.dz), gain=.1))
        self.model.p2h1 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.dp, self.dz), gain=.1))
        self.model.p2h2 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(self.dp, self.dh), gain=.1))
    
    def get_parameters(self, subject):
        """Constructs and returns the model parameters.
        Args:
            subject: subject index for each sample in the batch
        returns:
            model parameters for the given subject(s)"""
        # update p_vector with embedding if available
        p = self.model.p_vector[[subject]]
        A = torch.einsum('sp,pz->sz', p, self.model.p2A)
        W1 = torch.einsum('sp,pzh->szh', p, self.model.p2W1)
        W2 = torch.einsum('sp,phz->shz', p, self.model.p2W2)
        h1 = torch.einsum('sp,pz->sz', p, self.model.p2h1)
        h2 = torch.einsum('sp,ph->sh', p, self.model.p2h2)
        return A, W1, W2, h1, h2
    
    def grouped_parameters(self):
        """Returns a generator object for the individual and the shared 
        parameters respectively. Is only used for different learning rates."""
        shared, individual = super().grouped_parameters()
        shared += [self.model.p2A, self.model.p2W1, self.model.p2W2, self.model.p2h1, self.model.p2h2]
        individual += [self.model.p_vector]
        return shared, individual
    
    @torch.no_grad()
    def plot_stuff(self):
        """Function for viualizing interesting things. This is called by a plotter object
        to write it to tensorboard. Here the individual parameter vectors are visualized.
        Returns list of tuples containing figure and a name for the plot."""
        fig, ax = plt.subplots(1, figsize=(.6*self.dp, .6*self.num_subjects), layout='compressed')
        c = ax.imshow(self.model.p_vector.cpu(), origin="lower")
        fig.colorbar(c, ax=ax)
        ax.set_ylabel("subject")
        ax.set_xticks(range(self.dp))
        ax.set_xticklabels([r"$p_{" + f"{i+1}" + r"}$" for i in range(self.dp)])
        for s in range(self.num_subjects):
            for p in range(self.dp):
                ax.text(p, s, f"{self.model.p_vector[s, p].item():.1f}", ha="center", va="center")
        return [(fig, "param_vector")] + super().plot_stuff()
    
    def loss(self):
        """Defines any regularization losses on the parameters."""
        return torch.tensor(.0)