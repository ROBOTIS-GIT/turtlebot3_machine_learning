import pandas as pd
import matplotlib.pyplot as plt


# data = pd.read_csv('../frl_ep_6_round_10.csv')
# data = pd.read_csv('../single_ep_60.csv')
data = pd.read_csv('../data.csv')

plt.plot(data['data'])
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Stage 1 reward plot')
plt.grid(True)
plt.show()


