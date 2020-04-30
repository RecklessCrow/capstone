import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

log_file = f'/home/cj/PycharmProjects/capstone/Project Files/logs/PPO/progress.csv'
plot_file = f'PPO_Results.png'

log = pd.read_csv(log_file)
log.plot(
    kind='line',
    x='misc/nupdates',
    y='eprewmean',
    title='Reward per episode'
)
plt.xlabel('Update')
plt.ylabel('Mean Reward')
plt.savefig(plot_file)
