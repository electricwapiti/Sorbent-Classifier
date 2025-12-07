import numpy as np
import pickle
import gzip

N = 1000
molar_mass = np.random.uniform(50, 500, N)
amine_count = np.random.randint(1, 8, N)
heat_adsorption = np.random.uniform(50, 120, N)

inputs = np.vstack([molar_mass, amine_count, heat_adsorption]).T

# formulaic score
scores = 0.02 * amine_count + 0.01 * molar_mass - 0.03 * heat_adsorption

# Normalize scores for training
y_mean = np.mean(scores)
y_std  = np.std(scores)
scores_norm = (scores - y_mean) / y_std   # now roughly mean 0, std 1

# Normalize inputs
inputs[:,0] = (inputs[:,0] - np.mean(inputs[:,0])) / np.std(inputs[:,0])
inputs[:,1] = (inputs[:,1] - np.mean(inputs[:,1])) / np.std(inputs[:,1])
inputs[:,2] = (inputs[:,2] - np.mean(inputs[:,2])) / np.std(inputs[:,2])

train_split = int(0.8 * N)
val_split = int(0.9 * N)

train_inputs = inputs[:train_split]
train_labels = scores[:train_split]

val_inputs = inputs[train_split:val_split]
val_labels = scores[train_split:val_split]

test_inputs = inputs[val_split:]
test_labels = scores[val_split:]

dataset = (
    (train_inputs, train_labels),
    (val_inputs, val_labels),
    (test_inputs, test_labels)
)

with gzip.open("sorbents.pkl.gz", "wb") as f:
    pickle.dump(dataset, f)