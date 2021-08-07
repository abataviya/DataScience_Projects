"""
When we plot the data, if a straight line is not fitting in between the data points then it might be the polynomial regression
 In simple words we can say that if data is not distributed linearly then it might be for polynomial regression.
 Polynomial Regression is a regression algorithm that explains the relationship between a dependent(y) and independent variable(x) as nth degree polynomial.

If we apply a linear model on a linear dataset, then it provides us a good result as we have seen in Simple Linear Regression.
 If we apply the same model without any modification on a non-linear dataset, then it will  produce a drastic output.
 Due to this,
o Loss function will increase
o The error rate will be high
o Accuracy will be decreased.

Formulas:
Simple Linear Regression equation
 y = b0+b1x

Multiple Linear Regression equation
 y= b0+b1x+ b2x2+ b3x3+....+ bnxn

Polynomial Regression equation
 y= b0+b1x+b2X^2+b3X^3+...

"""
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

df= pd.read_csv('../data/poly_dataset.csv')
print(df)
print(df.shape)
print(df.isna().sum())
X= df.iloc[:,1:2].values
print(X)
y= df.iloc[:,2].values
print(y)

# Model Training
reg= LinearRegression()
reg.fit(X,y)
print("Model got trained")

# plt.scatter(X, y, color="blue")
# plt.plot(X, reg.predict(X), color="red")
# plt.title("Linear Regression")
# plt.xlabel("Position Levels")
# plt.ylabel("Salary")
# plt.show()

poly_reg= PolynomialFeatures(degree=5)
X_poly=poly_reg.fit_transform(X)
lin_reg= LinearRegression()
lin_reg.fit(X_poly,y)
print("Fitting the Polynomial regression to the dataset ")

# Plotting Polynomial Regression
plt.scatter(X,y,color='blue')
plt.plot(X, lin_reg.predict(poly_reg.fit_transform(X)), color="red")
plt.title("Polynomial Regression")
plt.xlabel("Position Levels")
plt.ylabel("Salary")
plt.show()

poly_pred = lin_reg.predict(poly_reg.fit_transform([[6.5]]))
print(poly_pred)

print(reg.predict([[6.5]]))
