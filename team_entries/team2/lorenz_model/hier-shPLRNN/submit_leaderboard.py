#%% Plot the prediction vs the truth
import numpy as np
import torch

import matplotlib.pyplot as plt



LOAD_FOLDER = "results/"
SUBMIT_FOLDER = "../../"

print("\nLoading prediction and truth from ", LOAD_FOLDER)
pred = torch.load(LOAD_FOLDER+"predictions.pt").cpu().numpy()
truth = torch.load(LOAD_FOLDER+"test_data.pt").cpu().numpy()
# print(pred.shape)
# print(truth.shape)

# Plot the first 3 dimensions of the prediction and truth


## visualise
"""Plot the Lorenz attractor solution"""
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot(truth[..., 0], truth[..., 1], truth[..., 2], ".", lw=1, label="Truth")
ax.plot(pred[..., 0], pred[..., 1], pred[..., 2], label="Prediction")
plt.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('Lorenz Prediction vs Truth')
plt.savefig("lorenz_solution.png", dpi=300)
plt.show(block=False)


print("\nSaving predictions for submission...\n")

## Save the first trajectory in the predictions
if pred.shape[1] > 5000:
    np.save(SUBMIT_FOLDER+"lorenz_prediction.npy", pred[0, 1:, :])
else:
    np.save(SUBMIT_FOLDER+"lorenz_prediction.npy", pred[0, :, :])

