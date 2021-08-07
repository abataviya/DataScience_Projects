import numpy as np
import pandas as pd
features= np.array([[1,2,3],[3,4,5],[6,7,8]])
print(features)
print()
def add_ten(x):
    return x+10

df= pd.DataFrame(features, columns =['feature1','feature2','feature3'])
print(df)
print()

df= df.apply(add_ten)
print(df)