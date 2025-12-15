import pandas as pd
import glob
import os

base_path = r"C:\Users\suraa\OneDrive\Desktop\Adversarial-IDS\data\CICIDS2017\CSVfiles"

files = glob.glob(os.path.join(base_path, "*.csv"))


dfs = []

for file in files:
    df = pd.read_csv(file)
    dfs.append(df)

merged = pd.concat(dfs, ignore_index=True)

output_path = r"C:\Users\suraa\OneDrive\Desktop\Adversarial-IDS\data\CICIDS2017\merged.csv"
merged.to_csv(output_path, index=False)


