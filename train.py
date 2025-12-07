from network import Network
from sorbent_loader import load_sorbent_data

train, val, test = load_sorbent_data()

print("Enter network architecture as comma-separated integers, e.g., 3,4,1")
arch_str = input("Network architecture: ")
sizes = [int(s) for s in arch_str.split(",")]
net = Network(sizes)

_epochs = int(input("Enter number of epochs, meaning go through training set this many times (e.g., 50): "))
_mini_batch_size = int(input("Enter mini-batch size, meaning update weights after every 10 samples (e.g., 10): "))
_eta = float(input("Enter learning rate, meaning step size in gradient descent (e.g., 0.1): "))

net.SGD(
    training_data=train,
    epochs=_epochs,
    mini_batch_size=_mini_batch_size,
    eta=_eta,
    evaluation_data=val,
    monitor_training_cost=True,
    monitor_evaluation_cost=True
)

net.save("sorbent_net.json")