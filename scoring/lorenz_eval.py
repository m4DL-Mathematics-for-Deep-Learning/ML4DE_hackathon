import numpy as np
import matplotlib.pyplot as plt

k=20 # number of snapshots to test for short time 
modes = 1000

# SCORING FOR SHORT-TIME AND LONG-TIME FORECASTS OF ODES
def scoring_lorenz(truth, prediction, k, modes):
    '''produce long-time and short-time error scores.'''
    [m,n]=truth.shape
    Est = np.linalg.norm(truth[:,0:k]-prediction[:,0:k],2)/np.linalg.norm(truth[:,0:k],2)

    yt = truth[-modes:, :]
    M = np.arange(-20, 21, 1)
    M2 = np.arange(0, 51, 1)
    yhistxt, xhistx = np.histogram(yt[0, :], bins=M)
    yhistyt, xhisty = np.histogram(yt[1, :], bins=M)
    yhistzt, xhistz = np.histogram(yt[2, :], bins=M2)

    yp = prediction[-modes:, :]
    yhistxp, xhistx = np.histogram(yp[0, :], bins=M)
    yhistyp, xhisty = np.histogram(yp[1, :], bins=M)
    yhistzp, xhistz = np.histogram(yp[2, :], bins=M2)
    
    Eltx = np.linalg.norm(yhistxt-yhistxp,2)/np.linalg.norm(yhistxt,2)
    Elty = np.linalg.norm(yhistyt-yhistyp,2)/np.linalg.norm(yhistyt,2)
    Eltz = np.linalg.norm(yhistzt-yhistzp,2)/np.linalg.norm(yhistzt,2)
    
    Elt =  (Eltx+Elty+Eltz)/3
    
    E1 = 100*(1-Est)
    E2 = 100*(1-Elt)

    if np.isnan(E1):
        E1 = -np.inf
    if np.isnan(E2):
        E2 = -np.inf
    
    return E1, E2


if __name__ == "__main__":
    import os

    DATA_FOLDER = "scoring/data"
    TEAM_FOLDER = "scoring/team3"
    TRUTH_FILE = os.path.join(DATA_FOLDER, "lorenz_truth.npy")
    PREDICTION_FILE = os.path.join(TEAM_FOLDER, "lorenz_prediction.npy")

    truth = np.load(TRUTH_FILE)
    prediction = np.load(PREDICTION_FILE)
    E1, E2 = scoring_lorenz(truth, prediction, k, modes)
    print(E1)
    print(E2)