import numpy as np


def construct_relu_matrix(number_quadrant: int, dim: int):
    """
    Matrix describing the Relu function for different quadrants(subcompartments)
    """
    quadrant_index = format(number_quadrant, f'0{dim}b')[::-1]
    return np.diag(np.array([bool(int(bit)) for bit in quadrant_index]))

def construct_relu_matrix_list(dim: int, order: int):
    """
    Construct a list of relu matrices for a random sequence of quadrants
    """
    relu_matrix_list = np.empty((dim, dim, order))
    for i in range(order):
        n = int(np.floor(np.random.rand(1)[0] * (2 ** dim)))
        relu_matrix_list[:, :, i] = construct_relu_matrix(n, dim)
    return relu_matrix_list

def get_cycle_point_candidate(A, W1, W2, h1, h2, D_list, order):
    """
    get the candidate for a cycle point by solving the cycle equation
    """
    z_factor, h1_factor, h2_factor = get_factors(A, W1, W2, D_list, order)
    try:
        inverse_matrix = np.linalg.inv(np.eye(A.shape[0]) - z_factor)
        z_candidate = inverse_matrix.dot(h1_factor.dot(h1) + h2_factor.dot(h2))
        return z_candidate
    except np.linalg.LinAlgError:
        # Not invertible
        return None

def get_factors(A, W1, W2, D_list, order):
    """
    recursively applying map gives us the factors of the cycle equation
    """
    hidden_dim = W2.shape[0]
    latent_dim = W1.shape[0]
    factor_z = np.eye(A.shape[0])
    factor_h1 = np.eye(A.shape[0])
    factor_h2 = W1.dot(D_list[:, :, 0]).dot(np.eye(hidden_dim))
    for i in range(order - 1):
        factor_z = (A + W1.dot(D_list[:, :, i]).dot(W2)).dot(factor_z)
        factor_h1 = (A + W1.dot(D_list[:, :, i + 1]).dot(W2)).dot(factor_h1) + np.eye(A.shape[0])
        factor_h2 = (A + W1.dot(D_list[:, :, i + 1]).dot(W2)).dot(factor_h2) + W1.dot(D_list[:, :, i + 1])
    factor_z = (A + W1.dot(D_list[:, :, order-1]).dot(W2)).dot(factor_z)
    return factor_z, factor_h1, factor_h2

def get_latent_time_series(time_steps, A, W1, W2, h1, h2, dz, z_0=None):
    """
    Generate the time series by iteravely applying the PLRNN
    """
    if z_0 is None:
        z = np.random.randn(dz)
    else:
        z = z_0
    trajectory = [z]

    for t in range(1, time_steps):
        z = latent_step(z, A, W1, W2, h1, h2)
        trajectory.append(z)
    return trajectory

def latent_step(z, A, W1, W2, h1, h2):
    """
    PLRNN step
    """
    return A.dot(z) + W1.dot(np.maximum(W2.dot(z) + h2, 0)) + h1

def get_eigvals(A, W1, W2, D_list, order):
    """
    Get the eigenvalues for all the points along the trajectory to learn about the stability
    """
    e = np.eye(A.shape[0])
    for i in range(order):
        e = (np.diag(A) + W1.dot(D_list[:, :, i]).dot(W2)).dot(e)
    return np.linalg.eigvals(e)

def scy_fi(A, W1, W2, h1, h2, order, found_lower_orders, outer_loop_iterations=300, inner_loop_iterations=100):
    """
    heuristic algorithm for calculating FP/k-cycle
    """
    hidden_dim = h2.shape[0]
    latent_dim = h1.shape[0]
    cycles_found = []
    eigvals = []

    i = -1
    while i < outer_loop_iterations:
        i += 1
        relu_matrix_list = construct_relu_matrix_list(hidden_dim, order)
        difference_relu_matrices = 1
        c = 0
        while c < inner_loop_iterations:
            c += 1
            z_candidate = get_cycle_point_candidate(A, W1, W2, h1, h2, relu_matrix_list, order)
            if z_candidate is not None:
                trajectory = get_latent_time_series(order, A, W1, W2, h1, h2, latent_dim, z_0=z_candidate)
                trajectory_relu_matrix_list = np.empty((hidden_dim, hidden_dim, order))
                for j in range(order):
                    trajectory_relu_matrix_list[:, :, j] = np.diag((W2.dot(trajectory[j]) + h2) > 0)
                for j in range(order):
                    difference_relu_matrices = np.sum(np.abs(trajectory_relu_matrix_list[:, :, j] - relu_matrix_list[:, :, j]))
                    if difference_relu_matrices != 0:
                        break
                    if found_lower_orders:
                        if np.round(trajectory[0], decimals=2) in np.round(np.array(found_lower_orders).flatten(), decimals=2):
                            difference_relu_matrices = 1
                            break
                if difference_relu_matrices == 0:
                    if not np.any(np.isin(np.round(trajectory[0], 2), np.round(cycles_found, 2))):
                        e = get_eigvals(A, W1, W2, relu_matrix_list, order)
                        cycles_found.append(trajectory)
                        eigvals.append(e)
                        i = 0
                        c = 0
                if np.array_equal(relu_matrix_list, trajectory_relu_matrix_list):
                    relu_matrix_list = construct_relu_matrix_list(hidden_dim, order)
                else:
                    relu_matrix_list = trajectory_relu_matrix_list
            else:
                relu_matrix_list = construct_relu_matrix_list(hidden_dim, order)
    return cycles_found, eigvals

def main(A, W1, W2, h1, h2, order, outer_loop_iterations=None, inner_loop_iterations=None):
    found_lower_orders = []
    found_eigvals = []

    for i in range(1, order + 1):
        cycles_found, eigvals = scy_fi(A, W1, W2, h1, h2, i, found_lower_orders, outer_loop_iterations=outer_loop_iterations, inner_loop_iterations=inner_loop_iterations)

        found_lower_orders.append(cycles_found)
        found_eigvals.append(eigvals)

    return [found_lower_orders, found_eigvals]

def metric(A, W1, W2, h1, h2, order=1, outer_loop_iterations=100, inner_loop_iterations=30):
    objects, eig_vals = main(A, W1, W2, h1, h2, order, outer_loop_iterations, inner_loop_iterations)
    return np.array(objects)