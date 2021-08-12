from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

digits= load_digits()
print("data set is loading")
print(dir(digits))
print(digits.feature_names)
X_train, X_test, y_train, y_test= train_test_split(digits.data, digits.target, test_size=0.3)
print(len(X_train))

print("Splitting the data into train and test")
lr = LogisticRegression()
lr.fit(X_train,y_train)
print(lr.score(X_test,y_test))
print()
# Applying SVM algorithm
from sklearn.svm import SVC
svm= SVC()
svm.fit(X_train,y_train)
print(svm.score(X_test,y_test))
print()
# Applying RandomForest algorithm
from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier(n_estimators=30)
rf.fit(X_train,y_train)
print(rf.score(X_test,y_test))
print()

from sklearn.model_selection import KFold
kf= KFold(n_splits=3)
print(kf)
for train_index, test_index in kf.split([1,2,3,4,5,6,7,8,9]):
    print(train_index, test_index)
print()

# Applying cross-val score
from sklearn.model_selection import cross_val_score
score1= cross_val_score(LogisticRegression(), digits.data, digits.target, cv=3)
print(score1)
print()
score2= cross_val_score(SVC(), digits.data, digits.target, cv=3)
print(score2)
print()
score3= cross_val_score(RandomForestClassifier(), digits.data, digits.target, cv=3)
print(score3)
print()
print(np.average(score1))
print()
print(np.average(score2))
print()
print(np.average(score3))
