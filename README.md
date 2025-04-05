# ML4DE Hackathon

Welcome to the ML4DE Hackathon! This repository contains the infrastructure for challenge 1 of the competition where teams will develop machine learning approaches for the Lorenz and Kuramoto-Sivashinsky (KS) systems.

## Evaluation

Each team should work within their own team folder (e.g., `team_entries/team1`, `team_entries/team2`, etc.).

The two crucial prediction files that will be evaluated are:
- `ks_prediction.npy` - Contains predictions for the Kuramoto-Sivashinsky system
- `lorenz_prediction.npy` - Contains predictions for the Lorenz system

These files should be placed in their respective team directories as shown on the `team0` folder.

## Real-time Results

You can view the current standings and scores in real-time at:

https://ml4de-hackathon.onrender.com/

The website will automatically update as teams submit their predictions.

## Team Folder Structure

While during the competition itself the important files are only the following:

```
team_entries/teamX/
├── ks_prediction.npy
└── lorenz_prediction.npy
```

At the end of the day, the top 3 teams are asked to submit their models for cross-checking to ensure the scores were obtained with valid ML approaches. For this we ask that throughout the competition you are mindful that you will have to submit a structure of the following form:

```
team_entries/teamX/
├── ks_prediction.npy
├── lorenz_prediction.npy
├── ks_model/
│   ├── model files
│   └── requirements.txt
└── lorenz_model/
    ├── model files
    └── requirements.txt
```

The model files should allow the competition administrator to run your model (with pretrained weights) on their local machine and obtain the same results as those submitted to the competition. If a top 3 team is unable to provide such a model folder, it will be disqualified from the final rankings.

## Important Notes

1. Only modify files within your team's folder (`team_entries/teamX`). Please DO NOT modify any other files on the repository! Any team pushing a change to a file outside their team folder will be immediately disqualified (1 grace strike for accidental commits applies).
2. Ensure your prediction files have the correct names and are placed in the right directories
3. The evaluation system will automatically load your predictions and calculate scores
4. Scores are updated in real-time on the website
5. Challenge details can be found at `challenge1_instructions.pdf`.

## Getting Started

1. Navigate to your team's folder
2. Check `team0/ks_model/ks_pinn.ipynb` and `team0/lorenz_pinn.ipynb` for the file format and array shape required for submitting entries
3. Develop your MLs models in the respective notebooks
4. Save your predictions as `ks_prediction.npy` and `lorenz_prediction.npy`
5. Check the website to see your team's current ranking

The Maths4DL team wishes you all best of luck for the competition!
