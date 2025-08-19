# TSA_EXP1B

CONVERSION OF NON STATIONARY TO STATIONARY DATA


# Date: 12.08.2025

### AIM:
To perform regular differncing,seasonal adjustment and log transformatio on Football goal scoring data


### ALGORITHM:
1. Import the required packages like pandas and numpy
2. Read the data using the pandas
3. Perform the data preprocessing if needed and apply regular differncing,seasonal adjustment,log transformation.
4. Plot the data according to need, before and after regular differncing,seasonal adjustment,log transformation.
5. Display the overall results.


### PROGRAM:

```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load and prepare data
data = pd.read_csv("results.csv")
data['date'] = pd.to_datetime(data['date'])
data = data.sort_values('date')

# Keep only numeric columns for resampling
data = data.set_index('date')
data_numeric = data[['home_score']].resample('YE').mean()

# Regular differencing
data_numeric['home_diff'] = data_numeric['home_score'] - data_numeric['home_score'].shift(1)

# Seasonal decomposition (period=10 years for example)
result = seasonal_decompose(data_numeric['home_score'], model='additive', period=10)
data_numeric['home_sea_diff'] = result.resid

# Log transformation
data_numeric['home_log'] = np.log1p(data_numeric['home_score'])
data_numeric['home_log_diff'] = data_numeric['home_log'] - data_numeric['home_log'].shift(1)

# Seasonal decomposition after log differencing
result_log = seasonal_decompose(data_numeric['home_log_diff'].dropna(), model='additive', period=10)
data_numeric['home_log_seasonal_diff'] = result_log.resid

# Plot
plt.figure(figsize=(16, 16))

plt.subplot(6, 1, 1)
plt.plot(data_numeric['home_score'], label='Original')
plt.legend(loc='best')
plt.title('Original Data')

plt.subplot(6, 1, 2)
plt.plot(data_numeric['home_diff'], label='Regular Differencing')
plt.legend(loc='best')
plt.title('Regular Differencing')

plt.subplot(6, 1, 3)
plt.plot(data_numeric['home_sea_diff'], label='Seasonal Adjustment')
plt.legend(loc='best')
plt.title('Seasonal Adjustment')

plt.subplot(6, 1, 4)
plt.plot(data_numeric['home_log'], label='Log Transformation')
plt.legend(loc='best')
plt.title('Log Transformation')

plt.subplot(6, 1, 5)
plt.plot(data_numeric['home_log_diff'], label='Log + Regular Differencing')
plt.legend(loc='best')
plt.title('Log + Regular Differencing')

plt.subplot(6, 1, 6)
plt.plot(data_numeric['home_log_seasonal_diff'], label='Log + Regular Differencing + Seasonal Differencing')
plt.legend(loc='best')
plt.title('Log + Regular Differencing + Seasonal Differencing')

plt.tight_layout()
plt.show()

```



### OUTPUT:

ORGINAL DATA:

<img width="950" height="155" alt="image" src="https://github.com/user-attachments/assets/8f5139e9-9bf4-49e5-99bd-1158ef08aba6" />

REGULAR DIFFERENCING:

<img width="958" height="163" alt="image" src="https://github.com/user-attachments/assets/ecdb5e3e-fde8-4b20-933d-434cb467cb6e" />

SEASONAL ADJUSTMENT:

<img width="951" height="163" alt="image" src="https://github.com/user-attachments/assets/e6dd79af-dd2d-4c53-8e2e-9750f165edf9" />

LOG TRANSFORMATION:

<img width="956" height="172" alt="image" src="https://github.com/user-attachments/assets/65307534-66bd-4ac3-927f-d3160f49dce4" />

LOG + REGULAR DIFFERENCING:

<img width="1095" height="191" alt="image" src="https://github.com/user-attachments/assets/84284dc7-aaf0-4be1-b24c-85d2e4093caa" />

LOG + REGULAR DIFFERENCING + SEASONAL DIFFERENCING:

<img width="1105" height="180" alt="image" src="https://github.com/user-attachments/assets/196734a8-4695-4ae5-a277-1a3c5a59ab0a" />


### RESULT:
Thus we have created the python code for the conversion of non stationary to stationary data on Football goal scoring
data.
