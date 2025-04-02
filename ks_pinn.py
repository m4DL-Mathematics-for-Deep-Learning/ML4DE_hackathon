import deepxde as dde
import torch
import numpy as np
import matplotlib.pyplot as plt

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

    data = dde.data.TimePDE(
        geomtime, pde, [bc, ic], num_domain=2540, num_boundary=80, num_initial=160
    )
    net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")
    model = dde.Model(data, net)

    model.compile("adam", lr=1e-3)
    
    checker = dde.callbacks.ModelCheckpoint(
        "model/model.ckpt", save_better_only=True, period=1000
    )
    
    losshistory, train_state = model.train(
        epochs=100000,
        display_every=1000
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
    import os

    params = {
        'L': 32,
        'N': 128,
        'nu': 1.0,
        'dt': 0.5,
        'T': 100,
    }
    
    model, losshistory, train_state = ks_pinn(params)
    
    dde.utils.plot_loss_history(losshistory)
    
    #dde.saveplot(losshistory, train_state, issave=True, isplot=True)

    t = np.linspace(0, params['T'], 201)
    x = np.linspace(0, params['L'], params['N'])
    X, T = np.meshgrid(x, t)
    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    
    u_pred = model.predict(X_star)
    u_pred = u_pred.reshape(len(t), len(x))
    
    plot_solution(x, t, u_pred)
    
    TEAM_FOLDER = "scoring/team0"
    os.makedirs(TEAM_FOLDER, exist_ok=True)
    
    PREDICTION_FILE = os.path.join(TEAM_FOLDER, "prediction.npy")
    np.save(PREDICTION_FILE, u_pred)
    
    print(f"Saved prediction to: {PREDICTION_FILE}")
    print(f"Prediction shape: {u_pred.shape}")
    
    plt.show()  