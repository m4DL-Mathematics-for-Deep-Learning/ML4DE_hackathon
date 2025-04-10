from matplotlib import pyplot as plt
import numpy as np
import torch


class Plotter():
    """Class for plotting the results of the model."""
    def __init__(self, model, dataset):
        """Initializes the plotter.
        Args:
            model: model to plot the results for
            args: args from the command line
            dataset: dataset that contains the test set
        """
        self.model = model
        self.dataset = dataset
        # set model to eval mode
        self.model.eval()
    
    @torch.compiler.disable()
    def plot_fixed_points(self):
        generated = self.model.evaluator.get_gen_data()
        N, T, dx = generated.shape
        n = int(np.ceil(np.sqrt(N)))
        if dx == 3:
            fps = self.model.evaluator.get_scyfi()
            fig = plt.figure(figsize=(3*n, 3*n), layout='compressed')
            for i, gen in enumerate(generated):
                gen = gen.cpu().numpy()
                ax = fig.add_subplot(n, n, i+1, projection='3d')
                ax.plot(gen[:,0], gen[:,1], gen[:,2])
                if fps[i] is not None:
                    ax.scatter(fps[i][:,0], fps[i][:,1], fps[i][:,2], marker='x', color='black')
                ax.set_title("Subject {}".format(i))
            return fig
        else:
            return None
    
    @torch.compiler.disable()
    def plot_power_spectrum(self):
        """Plots the power spectrum of the test set and the generated
        trajectories.
        """
        # compute power spectrum for first time, they are then saved within evaluator
        gen_ps, gt_ps = self.model.evaluator.get_power_spectrum()
        N, f, dz = gen_ps.shape
        interval = self.model.evaluator.get_interval_of_interest()
        fig = plt.figure(figsize=(4*N, self.model.dz), layout='compressed')
        sfigs = fig.subfigures(1, N)
        for i, (gt, gen) in enumerate(zip(gt_ps, gen_ps)):
            sfig = sfigs[i] if N > 1 else sfigs
            for j in range(gt.shape[-1]):
                ax = sfig.add_subplot(gt.shape[-1], 1, j+1)
                ax.plot(gt[:,j], label="ground truth")
                ax.plot(np.linspace(0, len(gt), len(gen)), len(gen)/len(gt) * gen[:,j], label="generated")
                ax.set_xlim(interval[i,j])
            sfig.suptitle(f"Subject {i}")
        return fig
    
    @torch.compiler.disable()
    def plot_trajectory(self):
        """Plots a test trajectory and a generated trajectory
        of same length.
        """
        ground_truth = self.dataset.get_test_data()
        N, T, dz = ground_truth.shape
        ground_truth = self.model.evaluator.get_test_data()[:, :T]
        generated = self.model.evaluator.get_gen_data()[:,:T]
        fig = plt.figure(figsize=(4*N, self.model.dz), layout='compressed')
        sfigs = fig.subfigures(1, N)
        for i, (gt, gen) in enumerate(zip(ground_truth, generated)):
            gt = gt.cpu().numpy()
            gen = gen.cpu().numpy()
            sfig = sfigs[i] if N > 1 else sfigs
            for j in range(gt.shape[-1]):
                ax = sfig.add_subplot(gt.shape[-1], 1, j+1)
                ax.plot(gt[:,j], label="ground truth")
                ax.plot(gen[:,j], label="generated")
                ax.legend()
            sfig.suptitle(f"Subject {i}")
        return fig
    
    @torch.compiler.disable()
    def plot_3D_trajectory(self):
        """Plots a test trajectory and a generated trajectory
        of same length in 3D.
        """
        if self.model.dx != 3:
            return None
        ground_truth = self.dataset.get_test_data()
        N, T, dz = ground_truth.shape
        n = int(np.ceil(np.sqrt(N)))
        generated = self.model.evaluator.get_gen_data()[:,:T]
        fig = plt.figure(figsize=(3*n, 3*n), layout='compressed')
        for i, (gt, gen) in enumerate(zip(ground_truth, generated)):
            gt = gt.cpu().numpy()
            gen = gen.cpu().numpy()
            ax = fig.add_subplot(n, n, i+1, projection='3d')
            ax.plot(gt[:,0], gt[:,1], gt[:,2], label="ground truth")
            ax.plot(gen[:,0], gen[:,1], gen[:,2], label="generated")
            ax.set_title("Subject {}".format(i))
            ax.legend()
        return fig
    
    @torch.compiler.disable()
    def plot_hovmoller(self):
        """Plots a test trajectory and a generated trajectory
        of same length in a hovmoller diagram."""
        ground_truth = self.dataset.get_test_data()
        N, T, dz = ground_truth.shape
        generated = self.model.evaluator.get_gen_data()[:,:T]
        fig = plt.figure(figsize=(9*N, int(T/100)), layout='compressed')
        for i, (gt, gen) in enumerate(zip(ground_truth, generated)):
            gt = gt.cpu().numpy()
            gen = gen.cpu().numpy()
            ax = fig.add_subplot(N, 1, i+1)
            # concat trajectories along latent dim
            trajectory = np.concatenate([gt, gen], axis=-1)
            c = ax.imshow(trajectory.T, origin='lower', cmap='bwr', aspect='auto')
            ax.axhline(dz, color='black', linewidth=1)
            fig.colorbar(c, ax=ax)
            ax.set_title("Subject {}".format(i))
            ax.set_ylabel('T')
        return fig
    
    @torch.compiler.disable()
    def plot_trajectory_train(self):
        """Copy of plot_trajectory but uses a single train instance
        as ground truth
        """
        ground_truth = self.dataset[0][0][None]
        N, T, dz = ground_truth.shape
        generated = self.model.evaluator.get_gen_data()[:,:T,:]
        fig = plt.figure(figsize=(3*N, self.model.dz))
        sfigs = fig.subfigures(1, N, layout='compressed')
        for i, (gt, gen) in enumerate(zip(ground_truth, generated)):
            gt = gt.cpu().numpy()
            gen = gen.cpu().numpy()
            sfig = sfigs[i] if N > 1 else sfigs
            for j in range(gt.shape[-1]):
                ax = sfig.add_subplot(gt.shape[-1], 1, j+1)
                ax.plot(gt[:,j], label="ground truth")
                ax.plot(gen[:,j], label="generated")
                ax.legend()
            sfig.suptitle(f"Subject {i}")
        return fig

    def plot_hierarchisation_stuff(self):
        """Plots interesting things from the hierarchisation scheme."""
        return self.model.hierarchisation_scheme.plot_stuff()