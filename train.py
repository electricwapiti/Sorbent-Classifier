from network import Network
from sorbent_loader import load_sorbent_data

train, val, test = load_sorbent_data()

net = Network([3, 4, 1])

net.SGD(
    training_data=train,
    epochs=50,
    mini_batch_size=10,
    eta=0.1,
    evaluation_data=val,
    monitor_training_cost=True,
    monitor_evaluation_cost=True
)