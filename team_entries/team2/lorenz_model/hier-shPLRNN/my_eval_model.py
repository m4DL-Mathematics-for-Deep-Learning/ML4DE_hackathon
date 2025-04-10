#%% 

model_path = "/home/gb21553/Projects/ML4DE-Hackathon/team_entries/team2/lorenz_model/hier-shPLRNN/trained_models/experiment/test/001/model_2.pt"

data_path = "/home/gb21553/Projects/ML4DE-Hackathon/team_entries/team2/lorenz_model/hier-shPLRNN/data/ml4de/lorenz_truth.pt"


# Load the model
import torch
from model import HierarchicalSHPLRNN
from utils import load_config

# Load the model
model = HierarchicalSHPLRNN()
model.load_state_dict(torch.load(model_path))
model.eval()
model.to("cpu")