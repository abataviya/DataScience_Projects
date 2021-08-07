import numpy as np
import pandas as pd

# A dataframe with outliers
houses = pd.DataFrame()
houses['Price'] = [534433, 392333, 293222, 4322032]
houses['rooms'] = [2, 3, 2, 116]
houses['Square_Feet'] = [1500, 2500, 1500, 48000]
print(houses)

# Filtering outliers
houses_new= houses[houses['rooms']<20]
print()
print(houses_new)

# Mark them as outliers
houses['outliers']= np.where(houses['rooms']<20,0,1)
print()
print(houses)