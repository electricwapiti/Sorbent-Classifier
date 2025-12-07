import gzip
import pickle
import numpy as np

def load_sorbent_data(filename="sorbents.pkl.gz"):
    """
    Loads the sorbent dataset and converts it to lists of (x, y) pairs.
    Returns: (training_data, validation_data, test_data)
    Each of these is a list of tuples: (input_column_vector, output_column_vector)
    """
    with gzip.open(filename, "rb") as f:
        dataset = pickle.load(f)

    train_inputs, train_labels = dataset[0]
    val_inputs, val_labels = dataset[1]
    test_inputs, test_labels = dataset[2]

    # Convert to column vectors for network
    def to_column_vectors(inputs, labels):
        inputs_cv = [np.reshape(x, (len(x), 1)) for x in inputs]
        labels_cv = [np.reshape(y, (1, 1)) if np.isscalar(y) else np.reshape(y, (len(y), 1))
                     for y in labels]
        return list(zip(inputs_cv, labels_cv))

    training_data = to_column_vectors(train_inputs, train_labels)
    validation_data = to_column_vectors(val_inputs, val_labels)
    test_data = to_column_vectors(test_inputs, test_labels)

    return training_data, validation_data, test_data