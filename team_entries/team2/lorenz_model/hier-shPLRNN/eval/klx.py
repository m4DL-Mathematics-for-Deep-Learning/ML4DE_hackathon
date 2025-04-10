import torch


def calc_histogram(x, n_bins, min_, max_):
    dim_x = x.shape[1]  # number of dimensions

    coordinates = (n_bins * (x - min_) / (max_ - min_)).long()

    # discard outliers
    coord_bigger_zero = coordinates > 0
    coord_smaller_nbins = coordinates < n_bins
    inlier = coord_bigger_zero.all(1) * coord_smaller_nbins.all(1)
    coordinates = coordinates[inlier]

    size_ = tuple(n_bins for _ in range(dim_x))
    indices = torch.ones(coordinates.shape[0], device=coordinates.device)

    return torch.sparse_coo_tensor(indices=coordinates.t(), values=indices, size=size_, device=coordinates.device).to_dense()

def normalize_to_pdf_with_laplace_smoothing(histogram, n_bins, smoothing_alpha=10e-6):
    if histogram.sum() == 0:  # if no entries in the range
        pdf = None
    else:
        dim_x = len(histogram.shape)
        pdf = (histogram + smoothing_alpha) / (histogram.sum() + smoothing_alpha * n_bins ** dim_x)
    return pdf

def kullback_leibler_divergence(p1, p2):
    """
    Calculate the Kullback-Leibler divergence
    """
    if p1 is None or p2 is None:
        kl = torch.tensor([float('nan')])
    else:
        kl = (p1 * torch.log(p1 / p2)).sum()
    return kl

def state_space_divergence_binning(x_gen, x_true, n_bins=30):
    if not isinstance(x_true, torch.Tensor):
        x_true_ = torch.tensor(x_true)
        x_gen_ = torch.tensor(x_gen)
    else:
        x_true_ = x_true
        x_gen_ = x_gen
    # standardize
    x_true_ = (x_true_ - x_true_.mean(dim=0))/x_true_.std(dim=0)
    x_gen_ = (x_gen_ - x_gen_.mean(dim=0))/x_gen_.std(dim=0)
    # 
    min_, max_ = x_true_.min(0).values, x_true_.max(0).values
    hist_gen = calc_histogram(x_gen_, n_bins=n_bins, min_=min_, max_=max_)
    hist_true = calc_histogram(x_true_, n_bins=n_bins, min_=min_, max_=max_)

    p_gen = normalize_to_pdf_with_laplace_smoothing(histogram=hist_gen, n_bins=n_bins)
    p_true = normalize_to_pdf_with_laplace_smoothing(histogram=hist_true, n_bins=n_bins)
    return kullback_leibler_divergence(p_true, p_gen).item()

def clean_from_outliers(prior, posterior):
    nonzeros = (prior != 0)
    if any(prior == 0):
        prior = prior[nonzeros]
        posterior = posterior[nonzeros]
    outlier_ratio = (1 - nonzeros.float()).mean()
    return prior, posterior, outlier_ratio

def eval_likelihood_gmm_for_diagonal_cov(z, mu, std):
    T = mu.shape[0]
    mu = mu.reshape((1, T, -1))

    vec = z - mu  # calculate difference for every time step
    vec=vec.float()
    precision = 1 / (std ** 2)
    precision = torch.diag_embed(precision).float()

    prec_vec = torch.einsum('zij,azj->azi', precision, vec)
    exponent = torch.einsum('abc,abc->ab', vec, prec_vec)
    sqrt_det_of_cov = torch.prod(std, dim=1)
    likelihood = torch.exp(-0.5 * exponent) / sqrt_det_of_cov
    return likelihood.sum(dim=1) / T

def state_space_divergence_gmm(X_gen, X_true, scaling=1.0, max_used=10000):
    time_steps = min(X_true.shape[0], max_used)
    mu_true = X_true[:time_steps, :]
    mu_gen = X_gen[:time_steps, :]

    cov_true = torch.ones(X_true.shape[-1], device=X_true.device).repeat(time_steps, 1) * scaling
    cov_gen = torch.ones(X_gen.shape[-1], device=X_gen.device).repeat(time_steps, 1) * scaling

    mc_n = 1000
    t = torch.randint(0, mu_true.shape[0], (mc_n,))

    std_true = torch.sqrt(cov_true)
    std_gen = torch.sqrt(cov_gen)

    z_sample = (mu_true[t] + std_true[t] * torch.randn(mu_true[t].shape, device=std_true[t].device)).reshape((mc_n, 1, -1))

    prior = eval_likelihood_gmm_for_diagonal_cov(z_sample, mu_gen, std_gen)
    posterior = eval_likelihood_gmm_for_diagonal_cov(z_sample, mu_true, std_true)
    prior, posterior, outlier_ratio = clean_from_outliers(prior, posterior)
    kl_mc = torch.mean(torch.log(posterior + 1e-8) - torch.log(prior + 1e-8), dim=0)
    return kl_mc.item()