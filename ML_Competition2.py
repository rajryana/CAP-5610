# Importing libraries
import numpy as np # linear algebra

#Import pandas
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Visualization: matplotlib, seaborn
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import warnings
warnings.filterwarnings("ignore")

# get spaceship-titanic train & test csv files as a DataFrame
train_df = pd.read_csv('../input/spaceship-titanic/train.csv', index_col='PassengerId')
test_df = pd.read_csv('../input/spaceship-titanic/test.csv', index_col='PassengerId')
print('train dataframe dimensions:', train_df.shape)
print('test dataframe dimensions:', test_df.shape)

# take a look at the train data to get an overview on the data type
train_df.info()
# The data has 13 features
# This concludes that all features have missing data
# Age, RoomService, FoodCourt, ShoppingMall, Spa, and VRDeck are Numerical data type
# HomePlanet, CryoSleep, Cabin, Destination, VIP and Name are Categorical data type

# check the features in train data
print(f"train data: {train_df.columns.values}")

# check the features in test data
#make sure test data does not have "survived" label
print(f"test data: {test_df.columns.values}")

train_df.isnull().sum()
test_df.isnull().sum()

# Exploratory Data Analysis on Categorical datas

# Figure size
plt.figure(figsize=(18,6))

# Pie plot to check the labels distribution on data
plt.subplot(2, 3, 1)
labels = ('False', 'True')
train_df['Transported'].value_counts().plot.pie(labels=labels,autopct='%1.1f%%', shadow=True,textprops={'fontsize':18}).set_title('Target distribution',fontsize=18 )
plt.subplots_adjust(left=1.25, bottom=1, right=2.5, top=3, wspace=0.5, hspace=0.4)
plt.show()

# Relationship between 'HomePlanet' and Transported
sns.factorplot('HomePlanet','Transported', data=train_df,size=4,aspect=3)

# Relationship between Destination and Transported
sns.factorplot('CryoSleep','Transported', data=train_df,size=4,aspect=3)

# Relationship between Destination and Transported
sns.factorplot('Destination','Transported', data=train_df,size=4,aspect=3)

# Relationship between VIP and Transported
sns.factorplot('VIP','Transported', data=train_df,size=4,aspect=3)

# Converting the True/Flase labels with the numerical values
train_df['Transported'].replace(False, 0, inplace=True)
train_df['Transported'].replace(True, 1, inplace=True)

# Define a function to get the information from cabin feature
# Partitioned cabin in three Deck, Number & side information
def cabin_process(train,test):
  train[['Deck','Num', 'Side']] = train['Cabin'].str.split('/', expand=True)
  test[['Deck','Num', 'Side']] = test['Cabin'].str.split('/', expand=True)

  return train,test

train_df , test_df = cabin_process(train_df,test_df)

# Drop Cabin and Name variables from data as Cabin information is already extracted
# Name variable is not giving much information
train_df.drop(['Cabin','Name'], axis=1, inplace=True)
test_df.drop(['Cabin','Name'], axis=1, inplace=True)

plt.figure(figsize=(15, 5))

# Colounm Deck plot
plt.subplot(2, 3, 1)
sns.countplot(data=train_df, x='Deck',hue='Transported')
plt.title('Deck',fontsize=18)
plt.xlabel('Deck', fontsize=15);
plt.ylabel('Counts', fontsize=16);
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend(title='Transported', fontsize=12)

plt.subplots_adjust(left=1.25, bottom=1, right=2.5, top=3, wspace=0.5, hspace=0.4)
plt.show()

# Relationship between Side and Transported
sns.factorplot('Side','Transported', data=train_df,size=4,aspect=3)

train_df.describe()

# box plot to see the percentile variation in Age
sns.boxplot(train_df['Age'])

# From Continuous numerical Age data type not much information is gained
# Make five clusters of Age Feature using std. deviations and percentiles calculated at 25%, 50%, 75% and 100%
# Convert them to the ordinal data type by assigning them labels

bins = [0, 6, 12, 18, 30, 42, 81]
labels = [0, 1, 2, 3, 4, 5]
train_df['AgeBand'] = pd.cut(train_df['Age'], bins=bins, labels=labels, right=False)
test_df['AgeBand'] = pd.cut(test_df['Age'], bins=bins, labels=labels, right=False)

# Hist plot to see the relation between Survival and AgeBand
g = sns.FacetGrid(train_df, col='Transported')
g.map(plt.hist, 'AgeBand', bins=20)

# Creating a new feature which has sum of all the expenses in train as well as test data
col_to_sum = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

train_df['SumSpends'] = train_df[col_to_sum].sum(axis=1)
test_df['SumSpends'] = test_df[col_to_sum].sum(axis=1)

#train_df['Expenditure']=train_df[col_to_sum].sum(axis=1)
g = sns.FacetGrid(train_df, col='Transported')
g.map(plt.hist, 'SumSpends', bins=20)

train_df = train_df.drop(['Num'], axis=1)
test_df = test_df.drop(['Num'], axis=1)

object_cols = [col for col in train_df.columns if train_df[col].dtype == 'object' or train_df[col].dtype == 'category']
numeric_cols = [col for col in train_df.columns if train_df[col].dtype == 'float64']

print(f'Object cols -- {object_cols}')
print(f'Numeric cols -- {numeric_cols}')

train_df[object_cols] = train_df[object_cols].astype('category')
test_df[object_cols] = test_df[object_cols].astype('category')

null_value=train_df.isnull().sum()
null_value

null_cols = train_df.isnull().sum().sort_values(ascending=False)
null_cols = list(null_cols[null_cols>1].index)
null_cols

print(f'Train DF shape: {train_df.shape}')
print(f'Test DF shape: {test_df.shape}')

plt.figure(figsize=(12,12))
sns.heatmap(train_df[['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'AgeBand', 'Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'SumSpends']].corr(), annot = True) #overall correlation between the various columns present in our data
plt.title('Overall relation between Numerical columns of the Dataset', fontsize = 20)
plt.show()

# Using OrdinalEncoder to transform categorical values as an integer array
# The features are converted to ordinal integers
from sklearn.preprocessing import OrdinalEncoder

oc = OrdinalEncoder()

# Merge train and test data to make a temporary data to apply Ordinal Encoder
# simulataneously on train and test data
df_for_encode = pd.concat([train_df, test_df])

df_for_encode[object_cols] = df_for_encode[object_cols].astype('category')

# Fit to data, then transform it
df_for_encode[object_cols] = oc.fit_transform(df_for_encode[object_cols])

del train_df, test_df

train_df = df_for_encode.iloc[:8693, :]
test_df = df_for_encode.iloc[8693: , :]

# delete the temporary data mde for encoding
del df_for_encode

test_df.drop('Transported', inplace=True, axis=1)

print(f'Train DF shape: {train_df.shape}')
print(f'Test DF shape: {test_df.shape}')

# Replace missing values using a descriptive statistic (e.g. mean, median, or most frequent)
# along each column, or using a constant value.
from sklearn.impute import SimpleImputer

# Applies transformers to columns of an array or pandas DataFrame.
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([("imp", SimpleImputer(strategy='mean'), null_cols)])

# Fit to data, then transform it
train_df[null_cols] = ct.fit_transform(train_df[null_cols])
test_df[null_cols] = ct.fit_transform(test_df[null_cols])

null_value=train_df.isnull().sum()
null_value

train_df = train_df.drop(['Age'], axis=1)
test_df = test_df.drop(['Age'], axis=1)

X = train_df.copy()
y = X.pop('Transported')

from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, KFold
x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=23, test_size=0.3)
kfold = KFold(n_splits=5, shuffle=True, random_state=10)

# Importing machine learning Model libraries
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score

model_factory = [CatBoostClassifier(eval_metric = 'Accuracy', verbose=0, rsm = 0.82, iterations = 700),
                 XGBClassifier(max_depth = 4, subsample = 0.75, n_estimators = 550, learning_rate = 0.03, min_child_weight = 0.9, random_state = 1),
                 LGBMClassifier(min_child_weight=0.8, random_state=1, n_estimators=600, learning_rate = 0.01, subsample=0.7, subsample_freq=1, colsample_bytree = 0.85)]

val = []
model_name = []

for model in model_factory:
    mf = model.fit(x_train, y_train)
    Pred = mf.predict(x_test)
    scores=cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    print(model.__class__.__name__, " : Train Accuracy: ", accuracy_score(y_test, Pred), " : Validation Accuracy : ", np.mean(scores))
    model_name.append(model.__class__.__name__)
    val.append(np.mean(scores).item())

# CatBoost Classifier is giving highest accuracy on the test data.
# Therefore, CatBoost ML Model is further tuned with Hyperpraram
from sklearn.feature_selection import SequentialFeatureSelector
model_fs = CatBoostClassifier(eval_metric = 'Accuracy', verbose=0, rsm = 0.82, iterations = 700)
sf = SequentialFeatureSelector(model_fs, scoring='accuracy', direction = 'backward')
sf.fit(X, y)

# Extracting the best features/attributes for CatBoost ML Model
best_features_cat = list(sf.get_feature_names_out())
best_features_cat

# Performing optimization for various parameters like number of iterations, depth of trees, learning rate etc.
params = {'iterations': range(200,2000,200), 'eval_metric': ['Accuracy'], 'verbose':[0],'depth':[4,6,8],
          'learning_rate':[0.03,0.1], 'l2_leaf_reg': [1, 3, 5]}
model_cat = GridSearchCV(CatBoostClassifier(verbose=False), param_grid=params, scoring='accuracy', cv=kfold, n_jobs=-1)
model_cat.fit(X[best_features_cat],y)

# Printing out the best paramteres after optimization
print(model_cat.best_params_)

# Printing accuracy of ML Model on test data with optimized parameters
cat_pred=model_cat.predict(x_test)
print("accuracy_score = ",accuracy_score(y_test.values,cat_pred))

from sklearn.metrics import classification_report
print(classification_report(y_test, cat_pred))

prediction = model_cat.predict(test_df)
prediction

final = pd.DataFrame()
final.index = test_df.index
final['Transported'] = prediction
final['Transported'].replace(0, False, inplace=True)
final['Transported'].replace(1, True, inplace=True)
final.to_csv('submission.csv')


