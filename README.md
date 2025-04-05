# ML4DE Hackathon

Welcome to the ML4DE Hackathon! This repository contains the infrastructure for the competition where teams will develop physics-informed neural networks (PINNs) for the Lorenz and Kuramoto-Sivashinsky (KS) systems.

## Team Structure

Each team should work within their own team folder (e.g., `team_entries/team1`, `team_entries/team2`, etc.). The structure should match the reference `team0` folder:

```
team_entries/teamX/
├── ks_model/
│   ├── ks_pinn.ipynb
│   └── ks_prediction.npy
└── lorenz_model/
    ├── lorenz_pinn.ipynb
    └── lorenz_prediction.npy
```

## Important Files

The two crucial prediction files that will be evaluated are:
- `ks_prediction.npy` - Contains predictions for the Kuramoto-Sivashinsky system
- `lorenz_prediction.npy` - Contains predictions for the Lorenz system

These files should be placed in their respective model directories as shown above.

## Real-time Results

You can view the current standings and scores in real-time at:

https://ml4de-hackathon.onrender.com/

The website will automatically update as teams submit their predictions.

## Important Notes

1. Only modify files within your team's folder (`team_entries/teamX`)
2. Ensure your prediction files have the correct names and are placed in the right directories
3. The evaluation system will automatically load your predictions and calculate scores
4. Scores are updated in real-time on the website

## Getting Started

1. Navigate to your team's folder
2. Develop your PINN models in the respective notebooks
3. Save your predictions as `ks_prediction.npy` and `lorenz_prediction.npy`
4. Check the website to see your team's current ranking

Good luck!
