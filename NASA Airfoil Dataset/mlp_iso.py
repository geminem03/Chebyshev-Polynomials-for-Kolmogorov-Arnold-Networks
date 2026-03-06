import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import optuna
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# --- MASTER CONFIGURATION ---
TARGET_ACCURACY = 95.0  # The absolute minimum R^2 percentage allowed
N_TRIALS = 150 
EPOCHS = 200 

device_str = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_str)

# --- DATASET ---
class NASADataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        X_all = self.data.iloc[:, :5].values
        y_all = self.data.iloc[:, 5].values
        X_tr, X_v, y_tr, y_v = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
        
        self.X_train = torch.tensor(X_tr, dtype=torch.float32).to(device)
        self.X_val = torch.tensor(X_v, dtype=torch.float32).to(device)
        self.y_mean = y_tr.mean()
        self.y_std = y_tr.std()
        
        y_train_scaled = (y_tr - self.y_mean) / self.y_std
        y_val_scaled = (y_v - self.y_mean) / self.y_std
        
        self.y_train = torch.tensor(y_train_scaled, dtype=torch.float32).view(-1, 1).to(device)
        self.y_val = torch.tensor(y_val_scaled, dtype=torch.float32).view(-1, 1).to(device)
        self.y_val_orig = y_v 

# --- ARCHITECTURE ---
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

# --- TRAINING ---
def train_and_evaluate(model, dataset):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    val_accuracy_history = []
    
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
            acc_pct = max(0.0, r2 * 100) 
            val_accuracy_history.append(acc_pct)
            
    return np.mean(val_accuracy_history[-10:])

# --- OPTUNA OBJECTIVE ---
def objective(trial, dataset):
    n_hidden_layers = trial.suggest_int('n_hidden_layers', 1, 3)
    
    layer_sizes = [5] # Input 
    for i in range(n_hidden_layers):
        # Allow the MLP to get extremely wide if needed to hit the target
        width = trial.suggest_int(f'n_units_l{i}', 10, 2000)
        layer_sizes.append(width)
    layer_sizes.append(1) # Output 
    
    model = MLP(layer_sizes).to(device)
    params = sum(p.numel() for p in model.parameters())
    
    final_val_acc = train_and_evaluate(model, dataset)
    
    # Save the real accuracy and architecture for our records
    trial.set_user_attr("Architecture", str(layer_sizes))
    trial.set_user_attr("Real_Accuracy", final_val_acc)
    trial.set_user_attr("Real_Params", params)
    
    # --- THE ISO-ACCURACY ENFORCER ---
    if final_val_acc < TARGET_ACCURACY:
        # If it fails, report a massive fake parameter count. 
        # The worse the accuracy, the higher the penalty.
        penalty = 10_000_000 + ((TARGET_ACCURACY - final_val_acc) * 100_000)
        return penalty 
        
    # If it successfully hits 95.5% or higher, return the true parameter count to be minimized!
    return params

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    dataset = NASADataset('nasa_airfoil_data.csv')
    
    print(f"\n==================== COMMENCING ISO-ACCURACY NAS FOR MLP ====================")
    print(f"Goal: Find the absolute smallest MLP that achieves >= {TARGET_ACCURACY}%\n")
    
    # Single Objective: Minimize the returned parameter count
    study = optuna.create_study(direction="minimize", study_name="iso_mlp")
    study.optimize(lambda trial: objective(trial, dataset), n_trials=N_TRIALS)
    
    # Filter out the penalized trials to find the true winners
    successful_trials = [t for t in study.trials if t.user_attrs.get("Real_Accuracy", 0) >= TARGET_ACCURACY]
    
    if successful_trials:
        # Sort the successful trials by parameter count (lowest first)
        successful_trials.sort(key=lambda t: t.user_attrs["Real_Params"])
        best_trial = successful_trials[0]
        
        print("\nISO-ACCURACY CHAMPION FOUND! 🏆")
        print(f"Target Accuracy:     {TARGET_ACCURACY}%")
        print(f"Actual Accuracy:     {best_trial.user_attrs['Real_Accuracy']:.2f}%")
        print(f"Minimum Parameters:  {best_trial.user_attrs['Real_Params']:,}")
        print(f"Optimal MLP Dims:    {best_trial.user_attrs['Architecture']}")

        # Export the entire study history to a CSV
        df = study.trials_dataframe()
        os.makedirs("optuna_iso_results", exist_ok=True)   # <--- ADD THIS LINE
        csv_path = f"optuna_iso_results/iso_accuracy_mlp.csv"
        df.to_csv(csv_path, index=False)
            
    else:
        print(f"\nThe MLP could not hit {TARGET_ACCURACY}% in any of the {N_TRIALS} trials.")
        print("Try increasing the maximum width limit in the trial.suggest_int block.")