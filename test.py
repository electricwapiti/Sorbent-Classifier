from network import Network
from network import load
import numpy as np
import gzip
import pickle

with gzip.open("sorbents.pkl.gz", "rb") as f:
    (train_data, val_data, test_data) = pickle.load(f)

test_inputs, test_labels = test_data
test_inputs = [np.reshape(x, (3,1)) for x in test_inputs]
test_labels = [np.reshape(y, (1,1)) for y in test_labels]
net = load("sorbent_net.json")
preds = [net.feedforward(x)[0,0] for x in test_inputs]
actuals = [y[0,0] for y in test_labels]

print("Predicted\tActual")
for p, a in zip(preds[:10], actuals[:10]):
    print(f"{p:.3f}\t\t{a:.3f}")

mse = np.mean([(p - a)**2 for p,a in zip(preds, actuals)])
print(f"\nMean squared error on test set: {mse:.4f}")

# Compute R^2
actual_mean = np.mean(actuals)
ss_total = np.sum((np.array(actuals) - actual_mean)**2)
ss_res = np.sum((np.array(actuals) - np.array(preds))**2)
r2 = 1 - ss_res / ss_total

print(f"R^2 on test set: {r2:.4f}")