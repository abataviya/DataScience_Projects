"""
Regression:-
1. Regression analysis is used to explain the relationship between a two variables.
2. Also called as it’s a relationship in between dependent variable and one or more
independent variables.
3. If two variables having relationship then if we draw this relationship in a two dimensional
then we get a straight line.
4. The goal of linear regression is to draw the best fitted line.
5. Best fitted line means that the line which passes as close as possible to these points.

Linear Regression:-
This is a technique and it explains the relationship between the dependent variable and
independent variables
 There are two types of linear regression
o Simple linear regression
o Multiple linear regression

1. Simple linear regression
 When you have only 1 independent variable and 1 dependent variable, it is called simple
linear regression.
2. Multiple linear regression
 When you have 2 or more independent variable and 1 dependent variable, it is called
multiple linear regression.

Formula : y=mx+c
"""


"""
Simple Linear Regression
Problem statement - 1
Assuming that we are planning to buy a new house and need to predict the price of a house
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df= pd.read_csv('../data/homeprices.csv')
print(df.shape)
print(df.head())

X= df.drop('price', axis= 'columns')
y= df.price
reg= LinearRegression()
reg.fit(X,y)
print(reg.coef_)
print()
print(reg.intercept_)
print()
print(reg.score(X,y))
print(reg.predict([[2600],[3000]]))
print()

# plt.xlabel("Area")
# plt.ylabel("Price")
# plt.title("Area vs Price")
# plt.scatter(df.area,df.price,color='red', marker= '*')
# plt.plot(df.area,reg.predict(df[['area']]),color='blue')
# plt.show()
area_df= pd.read_csv('../data/areas.csv')
print(area_df)
print(reg.predict(area_df))

