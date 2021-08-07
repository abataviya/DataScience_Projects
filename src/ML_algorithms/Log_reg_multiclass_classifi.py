"""
Logistic regression comes under supervised Learning.
 It is a technique that is used to solve for classification problems.
 It is used for predicting the categorical dependent variable using a given set of
independent variables
Types of logistic regression
 Binary classification
    o This is having two classes(0 or 1, Pass or Fail, Yes or No etc.)
 Multiclass classification
    o This is having more than two classes(Ok, good, best or Cat, dot, sheep etc)
"""

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Loading the dataset
digits= load_digits()
print(dir(digits))
print()
print(digits.data)
print(len(digits.data))

#
# plt.gray()
# for i in range(5):
#     plt.matshow(digits.images[i])
#     plt.show()

# print(digits.target[0:5])
X= digits.data
y= digits.target

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2)

print("model creation")
log_reg= LogisticRegression()
log_reg.fit(X_train,y_train)

print(log_reg.score(X_test, y_test))
print(log_reg.predict([digits.data[6]]))
print(log_reg.predict([digits.data[9]]))
print(log_reg.predict(digits.data[0:5]))