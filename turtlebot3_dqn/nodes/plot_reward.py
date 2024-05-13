import pandas as pd
import matplotlib.pyplot as plt

train_type = "frl"    # or 'frl'
ep = 10                  # or 10
round = 10               

data_file_name = '{}_ep_{}'.format(train_type, str(ep)+'_round_'+str(round) if train_type == 'frl' else str(ep))
title = 'Total Reward When {} Train {} Episodes {}'.format("DRL" if train_type == 'single' else "FRL", str(ep), "" if train_type == 'single' else "Round " + str(round))

data = pd.read_csv('../{}.csv'.format(data_file_name))
# data = pd.read_csv('../data.csv')

plt.plot(data['data'])
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title(title)
plt.grid(True)
plt.show()


