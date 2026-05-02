# UPI Fraud Detection using Isolation Forest
# Run this in VS Code Jupyter Notebook

import pandas as pd
from sklearn.ensemble import IsolationForest

# Sample dataset
data = {
    "amount": [100, 200, 150, 5000, 120, 300, 7000],
    "time": [10, 12, 14, 2, 16, 18, 3]
}

df = pd.DataFrame(data)

# Model
model = IsolationForest(contamination=0.2)
df["anomaly"] = model.fit_predict(df)

# -1 = Fraud, 1 = Normal
df["fraud"] = df["anomaly"].apply(lambda x: 1 if x == -1 else 0)

print(df)

# Test new transaction
new_txn = [[6000, 2]]
prediction = model.predict(new_txn)

print("Fraud Prediction (1=Fraud, 0=Normal):", 1 if prediction[0] == -1 else 0)
