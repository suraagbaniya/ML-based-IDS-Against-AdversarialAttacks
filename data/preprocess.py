import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

df = pd.read_csv(r'D:\Adversarial-IDS\data\CICIDS2017\merged.csv')

print(df.head())
print(df.columns)

print("Merged Dataset shape: ", df.shape)

# 1. Remove Irrelevant Columns

drop_cols = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

print("After Removing irrelevant columns: ", df.shape)
# there was no irrelevant columns

# 2. Handle Missing Values

# convert infinite to NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# remove rows with missing values
df.dropna(inplace=True)

print("After removing Rows with missing values: ", df.shape)

# 3. Clean Column Names
# avoids crashes in ML
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")


# 4. Encode Target Labels

# this step is to convert benign(non-malicious) labels to 0 and
# malicious traffic like DDos, PortScan, Botnet, etc -> 1

df['label'] = df['label'].apply(lambda x: 0 if x == "BENIGN" else 1)

# 5. Remove Low-Variance Features

x = df.drop('label', axis=1)
y = df['label']

selector = VarianceThreshold(threshold=0.01)
x_selected = selector.fit_transform(x)

selected_features = x.columns[selector.get_support()]
x = pd.DataFrame(x_selected, columns=selected_features)

# 6. Train-Test Split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42, stratify=y
)

# 7. Feauture Scaling

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# 8. Handle Class Imbalance
# SMOTE Oversampling

sm = SMOTE(random_state=42)
x_train, y_train = sm.fit_resample(x_train, y_train)


# Save Cleaned Data

joblib.dump(x_train, "data/processed/x_train.pkl")
joblib.dump(x_test, "data/processed/x_test.pkl")
joblib.dump(y_train, "data/processed/y_train.pkl")
joblib.dump(y_test, "data/processed/y_test.pkl")
joblib.dump(scaler, "data/processed/scaler.pkl")
