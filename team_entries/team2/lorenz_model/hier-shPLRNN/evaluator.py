import torch
import numpy as np
from eval.pse import compute_and_smooth_power_spectrum, power_spectrum_error
from eval.klx import state_space_divergence_binning, state_space_divergence_gmm
from eval.scyfi import metric as scyfi


class Evaluator(object):
    """Class for evaluating a model."""
    def __init__(self, model, args, dataset):
        """Initializes the evaluator with a model and data set.
        Args:
            model: The model to evaluate.
            args: commandline arguments.
            dataset: Data set containing test data."""
        self.model = model
        self.args = args
        self.test_data = dataset.get_test_data().to(args.device)
        self.eval_data = self.load_eval_data(args.device)
        if self.eval_data is None:
            # use the test data if no separate long eval trajectory is supplied
            self.eval_data = self.test_data
        self.smoothing = args.pse_smooth
        self.compute_gt_power_spectrum()
    
    def load_eval_data(self, device):
        """Loads the larger evaluation data set if specified. This is used
        for computing the state space divergence, since that needs the trajectories
        to fill as much of the attractor space as possible to be a sensible measure."""
        if self.args.eval_data_path is None:
            return None
        data = torch.load(self.args.eval_data_path, map_location=device)
        if data.ndim == 2: data = data[None]
        return data
    
    def compute_cheap(self, which, gen=None):
        """Computes only the cheap stuff."""
        S, T, _ = self.test_data.shape
        if gen is None:
            z0 = self.test_data[:,0]
            self.gen = self.model.generate_free_trajectory(z0, T, torch.arange(S))
            self.test = self.test_data
            gen = self.gen
        if 'pse' in which:
            self.compute_power_spectrum(gen[:, :T])
            self.compute_pse()
        elif 'power_spectrum' in which:
            self.compute_power_spectrum(gen[:, :T])
        if 'scyfi' in which:
            self.compute_scyfi()

    def compute_expensive(self, which):
        """Computes exepnsive and cheap stuff."""
        # generate long trajectory (saved for plotting)
        if self.eval_data.ndim == 2: self.eval_data.unsqueeze(0)
        S, T, _ = self.eval_data.shape
        z0 = self.eval_data[:,0]
        self.gen = self.model.generate_free_trajectory(z0, T, torch.arange(S))
        self.test = self.eval_data

        print("\nSaving freely-generated trajectories to results/predictions.pt...\n")
        ## Save the trajectories to 'predictions.pt'
        torch.save(self.gen, 'results/predictions.pt')
        torch.save(self.test, 'results/test_data.pt')

        # state space divergence
        if 'dstsp' in which:
            self.compute_state_space_divergence(self.gen)
        # compute cheap
        self.compute_cheap(which, self.gen)

    def compute_gt_power_spectrum(self):
        """Pre computes the ground truth power spectrum. This only needs to be done
        once since it doesn't change."""
        self.gt_power_spectrum = compute_and_smooth_power_spectrum(self.test_data, self.smoothing)
    
    def compute_power_spectrum(self, gen):
        """Generates a trajectory of same length as
        eval data and computes its power specturm."""
        self.gen_power_spectrum = compute_and_smooth_power_spectrum(gen, self.smoothing)
    
    def compute_pse(self):
        """Computes the average hellinger distance across
        latent dimensions of the gt and the gen power spectrum.
        This must be called after the generated power
        spectrum has been computed."""
        self.pse = power_spectrum_error(self.gt_power_spectrum, self.gen_power_spectrum)
    
    def compute_state_space_divergence(self, gen):
        """Computes the state space divergence between the eval data (test data if former
        has not been provided) and a generated trajectory of same length."""
        dstsp_fn = state_space_divergence_gmm if self.args.kl_bins == 0 else lambda x,y: state_space_divergence_binning(x, y, self.args.kl_bins)
        d = []
        # print("\n\nTest data shape: \n\n\n", self.test_data.shape)
        for s in range(self.test_data.shape[0]):
            d.append(dstsp_fn(gen[s], self.eval_data[s]))
        self.D_state_space = torch.tensor(d)
    
    def compute_scyfi(self):
        if self.args.latent_size > 3:
            return
        fps = []
        for s in range(self.args.num_subjects):
            A, W1, W2, h1, h2 = self.model.hierarchisation_scheme.get_parameters(s)
            A = np.diag(A.detach().squeeze().cpu().numpy())
            W1 = W1.detach().squeeze().cpu().numpy()
            W2 = W2.detach().squeeze().cpu().numpy()
            h1 = h1.detach().squeeze().cpu().numpy()
            h2 = h2.detach().squeeze().cpu().numpy()
            try:
                fps.append(scyfi(A, W1, W2, h1, h2)[0,:,0])
            except IndexError:
                fps.append(None)
                print('Something went wrong in the computation of the fixed points. (Probably did not find any).')
        self.fps = fps
    
    def get_power_spectrum(self):
        """Returns the previously saved power spectra without
        re-computing them."""
        return self.gen_power_spectrum, self.gt_power_spectrum
    
    def get_interval_of_interest(self):
        """Returns the interval of interest for the power spectrum plot.
        (A large portion of the power spectrum is vanishingly small, this makes
        sure that the plots are easier to evaluate by eye)."""
        # mask frequencies above some threshold
        threshold = 1e-3
        mask = np.logical_or(self.gt_power_spectrum > threshold, self.gen_power_spectrum > threshold)
        # get indices of first and last non-zero entry
        first = np.argmax(mask, axis=1)
        last = mask.shape[1] - np.argmax(np.flip(mask, axis=1), axis=1)
        # return interval
        return np.stack([first, last], axis=-1)

    def get_pse(self):
        return self.pse
    
    def get_state_space_divergence(self):
        """Returns the previously computed state space divergence."""
        return self.D_state_space
    
    def get_gen_data(self):
        """Returns the last generated trajectory. Used for plotting
        so that a trajectory is generated only once."""
        return self.gen

    def get_test_data(self):
        """Returns the trajectory of which the initial state has been used
        to last generate a trajectory. Either test data or eval data."""
        return self.test
    
    def get_scyfi(self):
        return self.fps
    
    def get_n_step_mse(self, n):
        return torch.nn.functional.mse_loss(self.gen[...,:n,:], self.test[...,:n,:])