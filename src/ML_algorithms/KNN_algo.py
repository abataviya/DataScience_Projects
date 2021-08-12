from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris= load_iris()
print(dir(iris))
print(iris.feature_names)
print()
print(iris.target_names)
print(iris.data)
print()
print(len(iris.data))
df= pd.DataFrame(iris.data,columns=iris.feature_names)
print(df.head())
df['target']= iris.target
print(df.head())
print(df[df['target']==0])
print()
print(df[df['target']==1])
print()
print(df[df['target']==2])
df['Flower_name']= df.target.apply(lambda x: iris.target_names[x])
print(df.head())
setosa_50= df[:50]
print(setosa_50)
versicolor_50= df[50:100]
print(versicolor_50)
verginica_50= df[100:]
print(verginica_50)
print(df.head())
print(df.columns.values)

X= df.drop(['target','Flower_name'], axis=1)
y= df.Flower_name
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print("Splitting the data")

# Train Using K Neighbor classifier
classifier= KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train,y_train)
print('model got trained')
print(classifier.score(X_test,y_test))
print()
print(classifier.predict([[4.8, 3.0, 1.5, 0.3]]))

y_pred = classifier.predict(X_test)
print(y_pred)