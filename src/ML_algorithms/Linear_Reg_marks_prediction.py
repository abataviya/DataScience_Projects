import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset= pd.read_csv('../data/student_scores.csv')
print(dataset.shape)
print(dataset.head())
print(dataset.isna().sum())

# drawing graph
# plt.xlabel('Hours')
# plt.ylabel('Scores')
# plt.title('Hours Study vs Scores')
# plt.scatter(dataset.Hours, dataset.Scores, marker='*', color='blue')
# plt.plot(dataset.Hours, dataset.Scores, color='red')
# plt.show()

# Preparing data
X= dataset.iloc[:,:-1].values
y= dataset.iloc[:,1].values

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2, random_state=52)

# Creating model
reg= LinearRegression()
reg.fit(X_train,y_train)
print(reg.coef_)
print(reg.intercept_)
print(reg.score(X_train,y_train))
print()
print(reg.score(X_test,y_test))
print()
print(reg.predict([[5.5],[2.7],[6.1],[1.1],[3.5]]))
print(y_test)
print("=====================")
compare_df= pd.DataFrame({'Actual':y_test,'Predicted':reg.predict(X_test)})
print(compare_df)

