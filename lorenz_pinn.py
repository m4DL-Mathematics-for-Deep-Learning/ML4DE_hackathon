import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def ode_system(x, y):
    """Lorenz system ODEs:
    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z
    """
    # Get the components of y
    x_coord, y_coord, z_coord = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    
    # Get time derivatives
    dx_t = dde.grad.jacobian(y, x, i=0)
    dy_t = dde.grad.jacobian(y, x, i=1)
    dz_t = dde.grad.jacobian(y, x, i=2)
    
    # Parameters
    sigma = 10.0
    rho = 28.0
    beta = 8.0/3.0
    
    # Lorenz equations
    return [
        dx_t - sigma * (y_coord - x_coord),
        dy_t - (x_coord * (rho - z_coord) - y_coord),
        dz_t - (x_coord * y_coord - beta * z_coord)
    ]


def boundary(_, on_initial):
    return on_initial


def func(x):
    """Solution to test against"""
    return np.hstack((np.sin(x), np.cos(x), np.zeros_like(x)))


def plot_solution(t, xyz):
    """Plot the Lorenz attractor solution"""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Lorenz Attractor')
    plt.savefig("lorenz_solution_pinn.png", dpi=300)
    plt.show(block=False)


def lorenz_pinn(params=None):
    if params is None:
        params = {
            'T': 10,
            'dt': 0.01,
        }
    
    geom = dde.geometry.TimeDomain(0, params['T'])
    ic1 = dde.icbc.IC(geom, lambda x: 0, boundary, component=0)
    ic2 = dde.icbc.IC(geom, lambda x: 1, boundary, component=1)
    ic3 = dde.icbc.IC(geom, lambda x: 1, boundary, component=2)

    data = dde.data.PDE(geom, ode_system, [ic1, ic2, ic3], 35, 3, solution=func, num_test=100)

    layer_size = [1] + [50] * 3 + [3]
    activation = "tanh"
    initializer = "Glorot uniform"
    net = dde.nn.FNN(layer_size, activation, initializer)

    model = dde.Model(data, net)
    
    # Create model directory for checkpoints
    os.makedirs("model", exist_ok=True)
    
    # Add callbacks
    checker = dde.callbacks.ModelCheckpoint(
        "model/model.ckpt", 
        save_better_only=True, 
        period=1000
    )

    model.compile("adam", lr=0.001, metrics=["l2 relative error"])
    
    return model, checker


if __name__ == "__main__":
    # Set parameters
    params = {
        'T': 10,
        'dt': 0.01,
        'num_steps': 10001  # Total steps for 0 to 100
    }
    
    # Create and train model
    model, checker = lorenz_pinn(params)
    losshistory, train_state = model.train(iterations=20000, callbacks=[checker])
    
    # Plot training history
    #dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    
    # Generate prediction points
    t_pred = np.linspace(0, params['T'], params['num_steps'])
    y_pred = model.predict(t_pred[5001:10001, None])
    
    # Plot the predicted solution
    plot_solution(t_pred, y_pred)
    
    # Save the prediction
    TEAM_FOLDER = "scoring/team3"
    os.makedirs(TEAM_FOLDER, exist_ok=True)
    
    PREDICTION_FILE = os.path.join(TEAM_FOLDER, "lorenz_prediction.npy")
    np.save(PREDICTION_FILE, y_pred)
    
    print(f"Saved prediction to: {PREDICTION_FILE}")
    print(f"Prediction shape: {y_pred.shape}")
    
    plt.show()