import numpy as np
import pandas as pd
from statsmodels.tsa.arima_model import ARIMA
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.figsize': (9, 3), 'figure.dpi': 120})

# importing data
mydata = pd.read_csv('forecast_best2', names=['value'], header=0)

# Creating ARIMA model
mymodel = ARIMA(mydata.value, order=(1, 1, 1))
# modelfit = mymodel.fit(disp=0)

# Actual vs Fitted
# modelfit.plot_predict(dynamic=False)
plt.show()