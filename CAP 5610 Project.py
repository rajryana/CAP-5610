#general packages for data manipulation
import os
import pandas as pd
import numpy as np
#visualizations
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#handle the warnings in the code
import warnings
warnings.filterwarnings(action='ignore',category=DeprecationWarning)
warnings.filterwarnings(action='ignore',category=FutureWarning)

# Import text preprocessing libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import re

# Display pandas dataframe columns
pd.options.display.max_columns = None

# Import the machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

# Import ML models post processing modules
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

# Load the data
#load the csv file as a pandas dataframe
#ISO-8859-1
tweet = pd.read_csv('/kaggle/input/twitter-hate-speech/TwitterHate.csv',delimiter=',',engine='python',encoding='utf-8-sig')
tweet.head()

# Data visualization
# Check the report of data
from pandas_profiling import ProfileReport
profile = ProfileReport(tweet, title="Profiling Report")
profile

# Data preprocessing
#get rid of the identifier number of the tweet
# Drop ID feature from data
tweet.drop('id',axis=1,inplace=True)
sns.countplot('label',data=tweet)
#create a copy of the original data to work with
df = tweet.copy()
# Create a simplify function to handle diacrtics (ex:, ', ~, " etc.) from the data
def simplify(text):
    '''Function to handle the diacritics in the text'''
    import unicodedata
    try:
        text = unicode(text, 'utf-8')
    except NameError:
        pass
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
    return str(text)
# Apply simplify function on the data
df['tweet'] = df['tweet'].apply(simplify)
#remove all the user handles --> strings starting with @ (ex: @user1, @user2, etc.)
df['tweet'].replace(r'@\w+','',regex=True,inplace=True)
# Remove urls from the data set
df['tweet'].replace(r'http\S+','',regex=True,inplace=True)
#tokenize the tweets in the dataframe using TweetTokenizer
# tokenizer breaks down sentences into each word (breaks them by leveraging spaces in sentences)
tokenizer = TweetTokenizer(preserve_case=True)
df['tweet'] = df['tweet'].apply(tokenizer.tokenize)
#Review the tokenized tweets
df.head(5)
# General stop words are a, an, the, he, she, is, in, be, by etc.)
stop_words = stopwords.words('english')

#add additional stop words to be removed from the text
add_list = ['amp','rt','u',"can't",'ur']

for words in add_list:
    stop_words.append(words)
# Checking the last 10 stop words among all stop words
stop_words[-10:]
#define function to remove stop words from data set
def remove_stopwords(text):
    '''Function to remove the stop words from the text corpus'''
    clean_text = [word for word in text if not word in stop_words]
    return clean_text
#remove the stop words from the tweets
df['tweet'] = df['tweet'].apply(remove_stopwords)
df['tweet'].head()
#apply spelling correction on a sample text
from textblob import TextBlob
sample = 'amazng man you did it finallyy'
txtblob = TextBlob(sample)
corrected_text = txtblob.correct()
print(corrected_text)
#apply auto correct on the twitter data set
#textblob expect a string to be passed and not a list of strings
from textblob import TextBlob

def spell_check(text):
    '''Function to do spelling correction using '''
    txtblob = TextBlob(text)
    corrected_text = txtblob.correct()
    return corrected_text
#Defining function to remove hash symbols from data set
def remove_hashsymbols(text):
    '''Function to remove the hashtag symbol from the text'''
    pattern = re.compile(r'#')
    text = ' '.join(text)
    clean_text = re.sub(pattern,'',text)
    return tokenizer.tokenize(clean_text)
df['tweet'] = df['tweet'].apply(remove_hashsymbols)
# Define function to remove words containing only 1 or 2 chacters like 'a', 'ig' etc.
def rem_shortwords(text):
    '''Function to remove the short words of length 1 and 2 characters'''
    '''Arguments: 
       text: string
       returns: string without containing words of length 1 and 2'''
    lengths = [1,2]
    new_text = ' '.join(text)
    for word in text:
        text = [word for word in tokenizer.tokenize(new_text) if not len(word) in lengths]

    return new_text
# Remove the short words
df['tweet'] = df['tweet'].apply(rem_shortwords)
# Again apply tokenizer to break the sentences into words
df['tweet'] = df['tweet'].apply(tokenizer.tokenize)
# Defining a function to remove digits from the dataset
def rem_digits(text):
    '''Function to remove the digits from the list of strings'''
    no_digits = []
    for word in text:
        no_digits.append(re.sub(r'\d','',word))
    return ' '.join(no_digits)
df['tweet'] = df['tweet'].apply(rem_digits)
df['tweet'] = df['tweet'].apply(tokenizer.tokenize)
# Defining a function to remove special charcters from the dataset
def rem_nonalpha(text):
    '''Function to remove the non-alphanumeric characters from the text'''
    text = [word for word in text if word.isalpha()]
    return text
#remove the non alpha numeric characters from the tweet tokens
df['tweet'] = df['tweet'].apply(rem_nonalpha)
# Data Visualization for hate and non hate tweets
sns.countplot(df['label'])
plt.title('Offensive vs Nonoffensive Tweets')
plt.grid()
plt.show()
# Data visualization on the most used terms in tweets
from collections import Counter
results = Counter()
df['tweet'].apply(results.update)
#print the top 10 most common terms in the tweet
print(results.most_common(15))
#plot the frequency of the 15 mostly used terms
frequency = nltk.FreqDist(results)
plt.title('Top 15 Mostly used Terms')
frequency.plot(15,cumulative=True)
plt.show()
# A final check on the dataset before applying the ML models
df.head()
# Add tokens to make the preprocessed tweets
df['tweet'] = df['tweet'].apply(lambda x: ' '.join(x))

# ML Model Application
#split the data into input X and output y
X = df['tweet']
y = df['label']
#split the dataset into train and test data
test_size = 0.25 #splitting the train and test data in 75%:25% ratio
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=23,stratify=df['label'])
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
#import tfidf vectorizer
#To convert texts to a matrix of TF-IDF features.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# Defining the maximum number of features by term frequency to be considered for ML Models
vectorizer = TfidfVectorizer(max_features=4500)
#fitting the tdif vectorizer on the training dataset
X_train = vectorizer.fit_transform(X_train)
#transform the test data and apply the tdif vectorizer
X_test = vectorizer.transform(X_test)
#check the shape of data
X_train.shape, X_test.shape
from sklearn.model_selection import StratifiedKFold
#define the number of folds
seed = 51
folds = StratifiedKFold(n_splits=4,shuffle=True, random_state=seed)
model_factory = [LogisticRegression(),
                 LGBMClassifier(objective='binary'),
                 RandomForestClassifier(n_jobs=-1),
                 MultinomialNB(),
                 DecisionTreeClassifier(),
                 SVC(C=100.0),
                 XGBClassifier(objective='binary:logistic')]

val = []
model_name = []

for model in model_factory:
    mf = model.fit(X_train, y_train)
    Pred = mf.predict(X_test)
    scores=cross_val_score(model, X_train, y_train, cv=folds, scoring='accuracy')
    print(model.__class__.__name__, " : Train Accuracy: ", accuracy_score(y_test, Pred), " : Validation Accuracy : ", np.mean(scores))
    model_name.append(model.__class__.__name__)
    val.append(np.mean(scores).item())

fig = plt.figure(figsize = (60,30))
plt.bar(model_name, val, color=['black', 'red', 'green', 'blue', 'cyan', 'yellow', 'violet'], width=0.75)
plt.title('Cross Validation Accuracy of Models',fontsize=70)
plt.tick_params(axis='both', which='major',labelsize=40)
plt.show()


# SVC is giving the highest cross validation accuracy of 0.9605356157060853
# Hence, tuning the SVC ML model using hyper params
# Hyperparameter Optimization using GridSearch CV
# declare parameters for hyperparameter tuning
parameters = [ {'C':[1, 10, 100], 'kernel':['linear']},
               {'C':[1, 10, 100], 'kernel':['rbf'], 'gamma':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
               {'C':[1, 10, 100], 'kernel':['poly'], 'degree': [2,3] ,'gamma':[0.01,0.02,0.03]}
              ]

grid_search = GridSearchCV(estimator = SVC(),
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = folds,
                           verbose=0)


grid_search.fit(X_train, y_train)

# best score achieved during the GridSearchCV
print('GridSearch CV best score : {:.4f}\n\n'.format(grid_search.best_score_))

# print parameters that give the best results
print('Parameters that give the best results :','\n\n', (grid_search.best_params_))

# print estimator that was chosen by the GridSearch
print('\n\nEstimator that was chosen by the search :','\n\n', (grid_search.best_estimator_))

# calculate GridSearch CV score on test set

print('GridSearch CV score on test set: {0:0.4f}'.format(grid_search.score(X_test, y_test)))

