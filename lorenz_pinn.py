import deepxde as dde
import numpy as np
import matplotlib.pyplot as plt
import os

def load_training_data():
    """Load and process training data"""
    xyz_data = np.load("scoring/data/lorenz_training.npy")
    # Create time points (assuming uniform sampling from 0 to T/2)
    t = np.linspace(0, 50, xyz_data.shape[0])[:, None]  # First half of time domain
    # Stack time and xyz data for observations
    X_train = t
    y_train = xyz_data
    return X_train, y_train

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

def lorenz_pinn(params=None):
    if params is None:
        params = {
            'T': 100,
            'dt': 0.01,
        }
    
    # Create the domain
    geom = dde.geometry.TimeDomain(0, params['T'])
    
    # Load training data
    X_train, y_train = load_training_data()
    
    # Set initial conditions from first data point
    ic1 = dde.IC(geom, lambda X: y_train[0, 0], boundary, component=0)
    ic2 = dde.IC(geom, lambda X: y_train[0, 1], boundary, component=1)
    ic3 = dde.IC(geom, lambda X: y_train[0, 2], boundary, component=2)
    
    # Create observation for each component separately
    observe_x = dde.PointSetBC(X_train, y_train[:, 0:1], component=0)
    observe_y = dde.PointSetBC(X_train, y_train[:, 1:2], component=1)
    observe_z = dde.PointSetBC(X_train, y_train[:, 2:3], component=2)
    
    # Create the PDE problem with both physics and data
    data = dde.data.PDE(
        geom,
        ode_system,
        [ic1, ic2, ic3, observe_x, observe_y, observe_z],
        num_domain=300,
        num_boundary=2,
        anchors=X_train,
        num_test=3
    )

    layer_size = [1] + [50] * 4 + [3]
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

    # Compile with weighted losses - weights for [3 ODEs, 3 ICs, 3 observations]
    model.compile("adam", lr=0.005, loss_weights=[1, 1, 1, 0, 0, 0, 10, 10, 10])
    
    return model, checker


if __name__ == "__main__":
    # Create and train model
    model, checker = lorenz_pinn()
    
    # Train with display_every to monitor progress
    losshistory, train_state = model.train(
        iterations=3000,
        callbacks=[checker],
        display_every=300
    )
    
    # Predict solution
    t = np.linspace(0, 100, 5000)
    solution = model.predict(t[:, None])  # Add singleton dimension for time

    # Save prediction
    np.save("scoring/team3/lorenz_prediction.npy", solution)
    print("Saved prediction to: scoring/team3/lorenz_prediction.npy")
    print("Prediction shape:", solution.shape)

    # Plot the results
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    ax.plot3D(solution[:, 0], solution[:, 1], solution[:, 2], 'blue', label='PINN')
    
    # Load and plot training data
    _, y_train = load_training_data()
    ax.plot3D(y_train[:, 0], y_train[:, 1], y_train[:, 2], 'red', linestyle='dashed', label='Data')
    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    plt.show()