from pmdarima.arima.utils import ndiffs
import pandas as pd

df = pd.read_csv('forecast_best2', names=['value'], header=0)

X = df.value

# Augmented Dickey Fuller Test
adftest = ndiffs(X, test='adf')

# KPSS Test
kpsstest = ndiffs(X, test='kpss')

# PP Test
pptest = ndiffs(X, test='pp')

print("ADF Test =", adftest)
print("KPSS Test =", kpsstest)
print("PP Test =", pptest)