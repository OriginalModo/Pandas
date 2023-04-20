import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.figsize': (9, 7), 'figure.dpi': 120})

# Importing data
df = pd.read_csv('Forecast/forecast_best2', names = ['value'], header = 0)


# The Genuine Series
fig, axes = plt.subplots(3, 2, sharex=True)
axes[0, 0].plot(df.value)
axes[0, 0].set_title('The Genuine Series')
plot_acf(df.value, ax=axes[0, 1])

# Order of Differencing: First
axes[1, 0].plot(df.value.diff());
axes[1, 0].set_title('Order of Differencing: First')
plot_acf(df.value.diff().dropna(), ax=axes[1, 1])

# Order of Differencing: Second
axes[2, 0].plot(df.value.diff().diff());
axes[2, 0].set_title('Order of Differencing: Second')
plot_acf(df.value.diff().diff().dropna(), ax=axes[2, 1])

plt.show()



