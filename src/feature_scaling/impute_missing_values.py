import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

students = [[85, 'M', 'verygood'],
 [95, 'F', 'excellent'],
 [60, None, 'good'],
 [np.NaN, 'M', 'average'],
 [70, 'M', 'good'],
 [np.NaN, None, 'verygood'],
 [60, 'F', 'verygood'],
 [98, 'M', 'excellent']]

df= pd.DataFrame(students, columns= ['marks','gender','result'])
print(df)
print()
# Imputing missing numeric values with mean strategy
# impute= SimpleImputer(missing_values=np.NaN, strategy='mean')
# df.marks= impute.fit_transform(df['marks'].values.reshape(-1,1))[:,0]
# print(df)

# Imputing missing numeric values with median strategy
# impute= SimpleImputer(missing_values=np.NaN, strategy='median')
# df.marks= impute.fit_transform(df['marks'].values.reshape(-1,1))[:,0]
# print()
# print(df)

# Imputing missing numeric values with most_frequent strateg
# impute= SimpleImputer(missing_values=np.NaN, strategy='most_frequent')
# df.marks= impute.fit_transform(df['marks'].values.reshape(-1,1))[:,0]
# print(df)

# Imputing missing numeric values with constant strategy
impute= SimpleImputer(missing_values=np.NaN,strategy='constant',fill_value=80)
df['marks']= impute.fit_transform(df['marks'].values.reshape(-1,1))[:,0]
print(df)