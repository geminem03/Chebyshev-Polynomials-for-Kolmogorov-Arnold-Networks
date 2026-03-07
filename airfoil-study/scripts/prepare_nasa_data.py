import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import requests

# 1. Download directly from UCI
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"
response = requests.get(url)
with open("airfoil_self_noise.dat", "wb") as f:
    f.write(response.content)

# 2. Load (Tab-separated, no headers)
columns = ['freq', 'angle', 'chord', 'velocity', 'thickness', 'noise']
df = pd.read_csv('airfoil_self_noise.dat', sep='\t', header=None, names=columns)

# 3. Scaling is CRITICAL for KAN stability
scaler = MinMaxScaler(feature_range=(-1, 1))
df[columns[:-1]] = scaler.fit_transform(df[columns[:-1]])

# 4. Save for your training scripts
df.to_csv('../data_test/nasa_airfoil_data.csv', index=False)
print("NASA Airfoil data preprocessed and saved to 'nasa_airfoil_data.csv'")