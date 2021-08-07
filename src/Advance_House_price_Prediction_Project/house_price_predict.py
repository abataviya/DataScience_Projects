# All the life cycle in a data science project
    # Data analysis
    # Feature Engineering
    # Feature selection
    # Model Building
    # Model deployment
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Data analysis phase- main aim to understand about the data
dataset= pd.read_csv('../Advance_House_price_Prediction_Project/data/train.csv')
print(dataset.shape)
print()
# To display maximun columns
pd.pandas.set_option('display.max_columns',None)
print(dataset.head)
print()
print(dataset.columns)

# In data analysis we have to analyze to findout the below stuffs
# 1.Missing values
# 2.All the numericalvariables
# 3.Distribution of numerical variables\
# 4.Categorical variables
# 5. Cardinality of catagorical variable
# 6. Outliers
# 7. Relationships between dependent and independent variables(sales price)

# Finding missing values
features_with_na= [features for features in dataset.columns if dataset[features].isnull().sum() >1]
for feature in features_with_na:
    print(feature, np.round(dataset[feature].isnull().mean(),4), '% missing value')

# Since there are many missing values we need to findout the relationship between the missing value and sales price
for features in features_with_na:
    data= dataset.copy()

    data[features]= np.where(data[features].isnull(),1,0)
    data.groupby(features)['SalePrice'].median().plot.bar()
    # plt.title('feature')
    # plt.show()

print()
# Here with a relation bewween the missing value and dependent variable is clearly visible.So we need to replace these NaN values with something meaningful what we will handle in feature engineering.

# In the above dataset some of the feature like Id is not required, so we needs to remove that features in future
print("Id of houses {}".format(len(dataset['Id'])))

print()
# Findout numerical value features
features_numerical= [feature for feature in dataset.columns  if dataset[feature].dtype != 'O']
print("number of numerical values", len(features_numerical))
print()
print(dataset[features_numerical].head())

# Temporal variables(DateTime variables)
# From the dataset we hane 4 year variables. We have to extra information from DateTime vriables like no of years, no of days.
# One example from this specific scenario can difference in years between the year the house was build and the year the house was sold.
# We will be performing these analysis in feature engineering.
year_feature= [feature for feature in features_numerical if 'Yr' in feature or 'Year' in feature]
print(year_feature)
print()

# Explre the content of these year variable
for features in year_feature:
    print(features,dataset[features].unique())

# Lets analyze the temporal DateTime variable
# We will check whether there is a relationship between year of sold and sales price
dataset.groupby('YrSold')['SalePrice'].median().plot()
plt.xlabel('Year Sold')
plt.ylabel('Sales Price')
plt.title('Year Sold vs Sales Price')
# doubt about plotting


