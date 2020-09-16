import pandas as pd
import numpy as np
from sys import platform
import os
import matplotlib.pyplot as plt


if (platform == "darwin"):
    log_path = os.path.expanduser("~/gitRepos/mario_rl/trained_models_GPU_new_window_J_settings/training_log_GPU_new_window_J_settings.txt")
else:
    pass

file1 = open(log_path, 'r')
Lines = file1.readlines()

reward_data = np.empty(0)

for line in Lines:
    if line == '----------------------------------------------------------------------------\n':
        pass
    else:
        data = line.split(" ")
        reward = float(data[len(data)-1][:-1])
        reward_data = np.append(reward_data, reward)

plt.scatter(range(reward_data.shape[0]), reward_data)
