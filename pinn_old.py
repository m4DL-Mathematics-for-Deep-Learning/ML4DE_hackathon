import deepxde as dde
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

def load_training_data(params):
    """Load and process training data"""
    training_data = np.load('scoring/data/training.npy')
    t = np.linspace(0, params['T']/2, training_data.shape[0])  # First half of time domain
    x = np.linspace(0, params['L'], params['N'])
    
    # Create observation points
    X, T = np.meshgrid(x, t)
    X_train = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    y_train = training_data.flatten()[:, None]
    
    return X_train, y_train

def ks_pinn(params=None):
    if params is None:
        params = {
            'L': 32,
            'N': 128,
            'nu': 1.0,
            'dt': 0.5,
            'T': 100,
        }
    
    L, N, nu = params['L'], params['N'], params['nu']
    dt, T = params['dt'], params['T']
    
    def pde(x, y):
        dy_x = dde.grad.jacobian(y, x, i=0, j=0)
        dy_t = dde.grad.jacobian(y, x, i=0, j=1)
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        return dy_t + y * dy_x - 0.01 / np.pi * dy_xx

    space_domain = dde.geometry.Interval(0, L)  
    time_domain = dde.geometry.TimeDomain(0, T)
    geomtime = dde.geometry.GeometryXTime(space_domain, time_domain)

    def ic_func(x):
        return np.cos(x[:, 0:1]) + 0.1 * np.sin(2 * x[:, 0:1])

    ic = dde.IC(geomtime, ic_func, lambda _, on_initial: on_initial)

    def boundary_l(x, on_boundary):
        return on_boundary and np.isclose(x[0], 0)
    
    def boundary_r(x, on_boundary):
        return on_boundary and np.isclose(x[0], L)
    
    bc = dde.icbc.PeriodicBC(geomtime, 0, lambda x, _: np.isclose(x[0], 0) or np.isclose(x[0], L))

    # Load training data
    X_train, y_train = load_training_data(params)
    
    # Create the PDE problem with both physics and data
    observe_x = dde.PointSetBC(X_train, y_train, component=0)
    
    data = dde.data.TimePDE(
        geomtime, 
        pde, 
        [bc, ic, observe_x],  # Include observation points
        num_domain=2540,
        num_boundary=80,
        num_initial=160,
    )

    net = dde.nn.FNN([2] + [40] * 4 + [1], "tanh", "Glorot normal")  # Deeper network
    model = dde.Model(data, net)

    # Compile with weighted losses
    model.compile("adam", lr=1e-2, loss_weights=[1, 1, 1, 10])  # weights: [PDE residual, IC, BC, data]
    
    # Create model directory for checkpoints
    os.makedirs("model", exist_ok=True)
    
    # Add callbacks
    checker = dde.callbacks.ModelCheckpoint(
        "model/model.ckpt", 
        save_better_only=True, 
        period=1000
    )
    
    losshistory, train_state = model.train(
        epochs=4000,
        display_every=100,
        callbacks=[checker]
    )

    return model, losshistory, train_state

def plot_solution(x, t, u):
    plt.figure(figsize=(10, 6))
    X, T = np.meshgrid(x, t)
    plt.pcolormesh(X, T, u, shading='auto')
    plt.colorbar(label='u(x,t)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('Kuramoto-Sivashinsky Solution')
    plt.show(block=False)

if __name__ == "__main__":
    # Set up parameters matching the data generator
    params = {
        'L': 32,
        'N': 128,
        'nu': 1.0,
        'dt': 0.5,
        'T': 100,
        'num_steps': 201,  # Total steps for 0 to 100
    }
    
    model, losshistory, train_state = ks_pinn(params)
    
    dde.utils.plot_loss_history(losshistory)
    
    # Generate prediction points
    t = np.linspace(0, params['T'], 201)
    x = np.linspace(0, params['L'], params['N'])
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    
    u_pred = model.predict(X_star)
    u_pred = u_pred.reshape(len(t), len(x))
    
    plot_solution(x, t, u_pred)
    
    # Save the prediction
    TEAM_FOLDER = "scoring/team0"
    os.makedirs(TEAM_FOLDER, exist_ok=True)
    
    PREDICTION_FILE = os.path.join(TEAM_FOLDER, "prediction.npy")
    np.save(PREDICTION_FILE, u_pred[int((params['num_steps']-1)/2)+1:params['num_steps']])
    
    print(f"Saved prediction to: {PREDICTION_FILE}")
    print(f"Prediction shape: {u_pred[int((params['num_steps']-1)/2)+1:params['num_steps']].shape}")
    
    plt.show()