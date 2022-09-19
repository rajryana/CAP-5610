#Import pandas
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import Series, DataFrame

import numpy as np # linear algebra
from sklearn.preprocessing import StandardScaler # preprocessing

# Visualization: matplotlib, seaborn
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# machine learning Models
import sklearn
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, cross_validate, GridSearchCV
from sklearn import feature_selection, model_selection, metrics

# get titanic & test csv files as a DataFrame
# get gender submission csv file as a DataFrame
train_df = pd.read_csv("../input/titanic/train.csv")
test_df    = pd.read_csv("../input/titanic/test.csv")
gender_sub = pd.read_csv("../input/titanic/gender_submission.csv")

# take a look at the train and test data to get an overview on the data type
print ("train data:\n")
print (train_df.head(5))
print("\n")

print("test data:\n")
print (test_df.head(5))

# check the features in train data
print(f"train data: {train_df.columns.values}")
# check the features in test data
# make sure test data does not have "survived" label
print(f"test data: {test_df.columns.values}")

# check the number of objects in train data
print(f"Total objects in train data = {len(train_df)}")

# check the number of objects in test data
print(f"Total objects in test data = {len(test_df)}")

# Look at the data type; number of objects corresponding to features
# This will give us information about the missing values in each feature
train_df.info()

test_df.info()
# This concludes that Age, Fare, cabin and Embarked features have missing data
# Passenger ID, PClass, Age, Sibsp, Parch, and Fare are Numerical data type
# Name, Sex, Ticket, Cabin, and Embarked are Categorical data type

# Now, lets have a look at the number of missing data in each attribute
# First for the Training data
train_df.isnull().sum()
# Training data has missing objects in Age (177), Cabin (687), and Embarked (2) out of 891 training data objects

# For the test data
test_df.isnull().sum()
# Test data has missing objects in Age (86), Cabin (687), and Fare (1) out of 418 test data objects

# In class lecture, it is best to drop feature with too much missing values
# Hence, drop "cabin" from both train as well as test datacombine data
train_df = train_df.drop(['Cabin'], axis=1)
test_df = test_df.drop(['Cabin'], axis=1)

# Now, Let's first deal with the missing values in the remaing data (Age, Fare, and Embarked)

# First fill the missing values in Numerical data type using median of the feature
# Median is choosen out of 'median, mean, and random numbers (between Std. Deviation & Median)

# Fill the missing values in the 'Age' attribute (Numerical Feature):
train_df['Age'].fillna((train_df['Age'].median()), inplace = True)
test_df['Age'].fillna((test_df['Age'].median()), inplace = True)

# Fill the missing values of the 'Fare' attribute (Numerical Feature):
train_df['Fare'].fillna((train_df['Fare'].median()), inplace = True)
test_df['Fare'].fillna((test_df['Fare'].median()), inplace = True)

# Fill the missing values in the Embarked feature with the mode of the column:
train_df['Embarked'].fillna((train_df['Embarked'].mode()[0]), inplace = True)
test_df['Embarked'].fillna((test_df['Embarked'].mode()[0]), inplace = True)

# Double check to make sure all the features do not have missing objects
train_df.isnull().sum()
test_df.isnull().sum()

# Now Analyse the data

# Analyze 'Name' with Survivality
Y_train = train_df["Survived"]
train_df_check = pd.concat([train_df.sort_index().iloc[:891], Y_train], axis = 1)
feature_Name = train_df_check.groupby(by = 'Name').size().sort_values(ascending = False)
feature_Name

# From data analysis on Name, all the names are different and have no correlation with Survival
# Hence, drop "Name" feature from train and test data
train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)

# Analyze 'PassengerID' (Nominal data) - Not going to give us much information
# Hence, drop "PassengerID" from training data
train_df = train_df.drop(['PassengerId'], axis=1)

# Focus on the preprocessing of Numerical data for further analysis

# First focus on the continous numerical data (Age & Fare)
# Lets look at the std. deviation, mean, percentile to do equal width clustering for continous data
train_df.describe()
test_df.describe()

# Visualize the Age feature relationship with Survivality

g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)

# box plot to see the percentile variation in Age
sns.boxplot(train_df['Age'])

train_df.describe()

# From Continuous numerical Age data type not much information is gained
# Make five clusters of Age Feature using std. deviations and percentiles calculated at 25%, 50%, 75% and 100%
# Convert them to the ordinal data type by assigning them labels

bins = [0, 15, 30, 60, 81]
labels = [0, 1, 2, 3]
train_df['AgeBand'] = pd.cut(train_df['Age'], bins=bins, labels=labels, right=False)
test_df['AgeBand'] = pd.cut(test_df['Age'], bins=bins, labels=labels, right=False)

# Hist plot to see the relation between Survival and AgeBand
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'AgeBand', bins=20)

# Visualize the Fare feature relationship with Survivality

g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Fare', bins=20)

# box plot to see the percentile variation in Fare
sns.boxplot(train_df['Fare'])

# From Continuous numerical Fare data type not much information is gained
# Make five clusters of Fare Feature using std. deviations and percentiles calculated at 25%, 50%, 75% and 100%
# Convert them to the ordinal data type by assigning them labels
bins = [0, 8, 15, 32, 513]
labels = [0, 1, 2, 3]
train_df['FareBand'] = pd.cut(train_df['Fare'], bins=bins, labels=labels, right=False)
test_df['FareBand'] = pd.cut(test_df['Fare'], bins=bins, labels=labels, right=False)

# Hist plot to see the relation between Survival and FareBand
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'FareBand', bins=20)

train_df = train_df.drop(['Age', 'Fare'], axis=1)
test_df = test_df.drop(['Age', 'Fare'], axis=1)

# Relationship between Survival, Sibsp and Parch

cols = ['Parch', 'SibSp']

fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (15, 5))

for col, subplot in zip(cols, ax.flatten()):
    sns.countplot(data = train_df, x = col,  hue = 'Survived', ax = subplot, palette = 'magma')
    subplot.legend(loc = 'upper right', title = 'Survived')

plt.show()

train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1

train_df['IsAlone'] = train_df['FamilySize'].map({1:1})
train_df['IsAlone'].fillna(0, inplace=True)

test_df['IsAlone'] = test_df['FamilySize'].map({1:1})
test_df['IsAlone'].fillna(0, inplace=True)

# drop the Parch, Sibsp, and FamilySize variables now
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)

# Similarly take a look at categorical attributes
train_df.describe(include=['O'])
test_df.describe(include=['O'])

train_df.describe()
test_df.describe()

# Drop the Ticket Feature, since it does not give much information (as per domain knowledge)
train_df = train_df.drop(['Ticket'], axis=1)
test_df = test_df.drop(['Ticket'], axis=1)

# Relationship between Survival and Sex
sns.factorplot('Sex','Survived', data=train_df,size=4,aspect=3)
# Relationship between Survival and Embarked
sns.factorplot('Embarked','Survived', data=train_df,size=4,aspect=3)

# Assign ordinal data type values to the categorical data for Sex
train_df['Sex'] = train_df['Sex'].map({'male':1, 'female':0})
test_df['Sex'] = test_df['Sex'].map({'male':1, 'female':0})

# Assign ordinal value to the categorical data for Sex
train_df['Embarked'] = train_df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} )
test_df['Embarked'] = test_df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} )

# Prepare the data for model evaluaion
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape

# Apply ML Model to the training dataset using Random Forest
std_scaler = StandardScaler()
X_train = std_scaler.fit_transform(X_train)
X_test = std_scaler.transform(X_test)
X_train.shape, Y_train.shape, X_test.shape

# Train the Model
rfc = RandomForestClassifier(criterion='gini',
                                           n_estimators=1100,
                                           max_depth=5,
                                           min_samples_split=4,
                                           min_samples_leaf=5,
                                           max_features='auto',
                                           oob_score=True,
                                           n_jobs=-1,
                                           verbose=1)
rfc.fit(X_train, Y_train)
# Predict the labels
Y_pred = rfc.predict(X_test)
acc_rfc = round(rfc.score(X_train, Y_train) * 100, 2)
acc_rfc

# Submit the test result to competition
submission = pd.read_csv('../input/titanic/gender_submission.csv')
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission['Survived'] = Y_pred
submission.head()

submission.to_csv('Submission.csv', index = False)
