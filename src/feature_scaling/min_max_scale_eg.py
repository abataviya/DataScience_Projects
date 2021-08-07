import pandas as pd
from sklearn import preprocessing

df= pd.DataFrame({'x': [0,1,2,3,4],'y':[25,50,75,100,125]})
# print(df)
# print()
# minmax_scale= preprocessing.MinMaxScaler(feature_range=(0,1))
# df[['x','y']]= minmax_scale.fit_transform(df[['x','y']])
# print(df)
# print()
# MinMaxScaler: A sing column
new_df= df[['x']].values.astype(float)
print(new_df)
print()
minmax_scale= preprocessing.MinMaxScaler(feature_range=(0,1))
scaled= minmax_scale.fit_transform(new_df)
print(scaled)


