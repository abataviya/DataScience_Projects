import pandas as pd
from sklearn import preprocessing

df= pd.DataFrame({
 "x" : [0, 1, 2, 3, 4],
 "y" : [25, 50, 75, 100, 125]})
print(df)
standard_sc= preprocessing.StandardScaler()
# df[['x','y']]= standard_sc.fit_transform(df[['x','y']])
# print()
# print(df)

# Single column transform
xy = standard_sc.fit_transform(df[['x']])
print(xy)
df[['x']]= xy
print()
print(df)

