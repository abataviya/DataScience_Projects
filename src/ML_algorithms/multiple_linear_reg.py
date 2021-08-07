"""
Machine learning – Multiple Linear Regression
Multiple Linear Regression explains the relationship between a single dependent
continuous variable and more than one independent variable

formula: y= mx+m1x1+m2x2+b
"""
import pandas as pd
from sklearn.linear_model import LinearRegression

df= pd.read_csv('../data/homeprices1.csv')
print(df.shape)
print(df)
print()
print(df.isna().sum())

df.bedrooms=df.bedrooms.fillna(df.bedrooms.median())
print()
print(df.isna().sum())
X= df.drop('price', axis='columns')
y=df.price

# Model training
reg= LinearRegression()
reg.fit(X,y)
print(reg.score(X,y))
print(reg.predict([[2700,3.0,14]]))
