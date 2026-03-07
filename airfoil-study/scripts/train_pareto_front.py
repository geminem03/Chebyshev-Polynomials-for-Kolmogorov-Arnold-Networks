import os
import sys

# Ensure the models directory is in sys.path to support the new file structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Global imports to avoid overhead during Optuna trials
from ChebyKANLayer import ChebyKANLayer
from fftKAN import NaiveFourierKANLayer
from MatrixKan import MatrixKAN

# ---------------------------------------------------------
# 1. Configuration & Model Selection
# ---------------------------------------------------------
# Choose the model type to optimize here: 'chebyshev', 'fourier', 'bspline', or 'mlp'
MODEL_TYPE = 'mlp' 

EPOCHS = 200
LR = 0.01

device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)

# ---------------------------------------------------------
# 2. EXACT Architectures (Hardware Aligned)
# ---------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

class Chebyshev(nn.Module):
    def __init__(self, layer_sizes, degree=3):
        super(Chebyshev, self).__init__()
        self.layers = nn.ModuleList([
            ChebyKANLayer(layer_sizes[i], layer_sizes[i+1], degree) 
            for i in range(len(layer_sizes) - 1)
        ])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Fourier(nn.Module):
    def __init__(self, layer_sizes, grid_size, layer_class):
        super(Fourier, self).__init__()
        self.layers = nn.ModuleList([
            layer_class(layer_sizes[i], layer_sizes[i+1], grid_size)
            for i in range(len(layer_sizes) - 1)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(layer_sizes[i+1], elementwise_affine=False) 
            for i in range(len(layer_sizes) - 2)
        ])
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.tanh(self.norms[i](x))
        return x

# ---------------------------------------------------------
# 3. Data Loading 
# ---------------------------------------------------------
class NASADataset:
    def __init__(self, csv_file, device):
        self.data = pd.read_csv(csv_file)
        X_all = self.data.iloc[:, :5].values
        y_all = self.data.iloc[:, 5].values
        
        X_tr, X_v, y_tr, y_v = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
        
        self.X_train = torch.tensor(X_tr, dtype=torch.float32).to(device)
        self.X_val = torch.tensor(X_v, dtype=torch.float32).to(device)
        
        # Optuna scale input features to [-1, 1]
        self.x_min = self.X_train.min(dim=0, keepdim=True)[0]
        self.x_max = self.X_train.max(dim=0, keepdim=True)[0]
        self.X_train = 2.0 * (self.X_train - self.x_min) / (self.x_max - self.x_min) - 1.0
        self.X_val = 2.0 * (self.X_val - self.x_min) / (self.x_max - self.x_min) - 1.0
        
        self.y_mean = y_tr.mean()
        self.y_std = y_tr.std()
        
        y_train_scaled = (y_tr - self.y_mean) / self.y_std
        y_val_scaled = (y_v - self.y_mean) / self.y_std
        
        self.y_train = torch.tensor(y_train_scaled, dtype=torch.float32).view(-1, 1).to(device)
        self.y_val = torch.tensor(y_val_scaled, dtype=torch.float32).view(-1, 1).to(device)
        self.y_val_orig = y_v 

# ---------------------------------------------------------
# Global Dataset Initialization
# ---------------------------------------------------------
# Initialize the dataset once before the Optuna study to save time per trial
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'nasa_airfoil_data.csv'))
print("Loading NASA dataset globally...")
dataset = NASADataset(DATA_PATH, device)

# ---------------------------------------------------------
# 4. Optuna Objective 
# ---------------------------------------------------------
def objective(trial):
    # KANs have 1 hidden layer (3 elements in layer_sizes), MLP can have up to 4 hidden layers
    if MODEL_TYPE in ['chebyshev', 'fourier', 'bspline']:
        hidden_dim = trial.suggest_int('hidden_dim', 4, 100)
        layer_sizes = [5, hidden_dim, 1]
    else: # mlp
        num_hidden_layers = trial.suggest_int('num_hidden_layers', 1, 4)
        layer_sizes = [5]
        for i in range(num_hidden_layers):
            layer_sizes.append(trial.suggest_int(f'mlp_hidden_dim_{i}', 10, 300))
        layer_sizes.append(1)

    if MODEL_TYPE == 'chebyshev':
        model = Chebyshev(layer_sizes, degree=3)
    elif MODEL_TYPE == 'fourier':
        model = Fourier(layer_sizes, grid_size=3, layer_class=NaiveFourierKANLayer)
    elif MODEL_TYPE == 'bspline':
        model = MatrixKAN(width=[5, hidden_dim, 1], grid=4, k=2, device=device_str, symbolic_enabled=False, auto_save=False, base_fun='zero')
    elif MODEL_TYPE == 'mlp':
        model = MLP(layer_sizes)

    # Convert the model's internal parameters to float32 to prevent Float vs Double errors
    model = model.float().to(device)
    
    # Calculate parameter count
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trial.set_user_attr('params', total_params)
    trial.set_user_attr('layer_sizes', str(layer_sizes))
    
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    best_acc = -float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        preds_train = model(dataset.X_train)
        loss = criterion(preds_train, dataset.y_train)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            preds_val_scaled = model(dataset.X_val)
            preds_val_real = (preds_val_scaled.cpu().numpy().flatten() * dataset.y_std) + dataset.y_mean
            
            r2 = r2_score(dataset.y_val_orig, preds_val_real)
            val_acc = max(0.0, r2 * 100)
        
        if val_acc > best_acc:
            best_acc = val_acc
            
    return best_acc

# ---------------------------------------------------------
# 5. Main Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    print(f"--- Starting Optuna Study for {MODEL_TYPE.upper()} ---")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50) # Adjust n_trials as needed
    
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
        
    # Extract pareto front
    trials_df = study.trials_dataframe()
    
    # Get successful trials
    success_df = trials_df[trials_df['state'] == 'COMPLETE'].copy()
    
    # Create the results list mapping
    results = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            results.append({
                'Model': MODEL_TYPE.capitalize(),
                'Best_Accuracy': t.value,
                'Params': t.user_attrs.get('params'),
                'FC_Dims': t.user_attrs.get('layer_sizes')
            })
            
    df = pd.DataFrame(results)
    
    # Sort by accuracy (descending) then parameters (ascending) to find the pareto front
    df = df.sort_values(by=['Best_Accuracy', 'Params'], ascending=[False, True])
    
    # Save to CSV in the new experiments directory
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'experiments', 'optuna_pareto_results'))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{MODEL_TYPE}_pareto_front.csv")
    
    df.to_csv(out_path, index=False)
    print(f"\nPareto front saved to {out_path}")