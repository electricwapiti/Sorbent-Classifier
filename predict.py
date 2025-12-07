import numpy as np
from network import Network, load
net = load("sorbent_net.json")

INPUT_MEANS = np.array([275, 4, 85])  # replace with actual training means
INPUT_STDS  = np.array([130, 2, 20])  # replace with actual training stds

Y_MEAN = 5
Y_STD  = 2.5


while True:
    try:
        molar_mass = float(input("Enter molar mass (50-500): "))
        if not 50 <= molar_mass <= 500:
            raise ValueError
        amine_count = float(input("Enter number of amine groups (1-7): "))
        if not 1 <= amine_count <= 7:
            raise ValueError
        heat_adsorption = float(input("Enter heat of adsorption (50-120): "))
        if not 50 <= heat_adsorption <= 120:
            raise ValueError
    except ValueError:
        print("Invalid input. Please enter numeric values in the expected range.")
        continue


    # Normalize input
    x = np.array([molar_mass, amine_count, heat_adsorption])
    x_norm = (x - INPUT_MEANS) / INPUT_STDS
    x_norm = x_norm.reshape((3,1))

    # Predict
    pred_raw = net.feedforward(x_norm)[0,0]

    # please integrate these two lines of code
    pred_score = pred_raw * Y_STD + Y_MEAN
    pred_score = max(1, min(10, pred_score))

    print(f"Predicted sorbent score: {pred_score:.3f}\n")

    again = input("Predict another? (y/n): ").lower()
    if again != "y":
        break