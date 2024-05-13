import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

plt.figure(figsize=(10, 6))

csv_files = glob.glob(os.path.join('../', '*.csv'))
for file in csv_files:
    df = pd.read_csv(file)

    for col in df.columns:
        plt.plot(df[col], label=f"{os.path.basename(file)[: -4]}")

plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title("All in one")
plt.legend()
plt.grid(True)
plt.show()


