import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

data_directory = "/home/khinggan/my_research/ros_frl/ros1_ws/src/turtlebot3_machine_learning/turtlebot3_dqn/data/"
plot_directory = "/home/khinggan/my_research/ros_frl/ros1_ws/src/turtlebot3_machine_learning/turtlebot3_dqn/plots/"

# scan the csv files in `data_directory`
csv_files = glob.glob(os.path.join(data_directory, '*.csv'))
for file in csv_files:
    df = pd.read_csv(file)
    # Just plot the scores
    #  0   1  2    3  4    5   6   7   8         0  2     1      4    3       5   6     7   8
    # FRL_ep_10_round_5_client_1_stage_1.csv -> FRL 10 episodes, 5 rounds, client 1, stage 1
    fnl = os.path.basename(file).split('_')        # file name list
    if fnl[0] == 'FRL':
        label_name = fnl[0] + " " + fnl[2] + " episodes, " + fnl[4] + " rounds, client " + fnl[6] + ", stage " + fnl[-1].split(".")[0]
    elif fnl[0] == 'RL':
        label_name = fnl[0] + " " + fnl[2] + " episodes, client " + fnl[6] + ", stage " + fnl[-1].split(".")[0]
    else: 
        label_name = "ERROR"
    data = df.iloc[:, 0]
    plt.plot(data, label=f"{label_name}")
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title("Episode Reward Plot")
plt.legend()
plt.grid(True)

# If show, cannot save
# plt.show()

# generate save image name and save in `ros1_ws/src/turtlebot3_machine_learning/turtlebot3_dqn/plots`
plt.savefig(plot_directory + "plot.svg")
plt.close()