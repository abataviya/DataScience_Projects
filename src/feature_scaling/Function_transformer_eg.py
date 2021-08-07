import numpy as np
from sklearn.preprocessing import FunctionTransformer

features= np.array([[2,3],
                   [4,5],
                   [6,7]])

def add_ten(x):
    return x+10
print(features)
add_ten_trans= FunctionTransformer(add_ten)
x= add_ten_trans.transform(features)
print()
print(x)

# single column
