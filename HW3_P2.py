#Homework 3: Machine Learning with Matrix Data for Recommender Systems

# Import Libraries for data wrangling and analysis
import pandas as pd
import numpy as np

# visualization
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import init_notebook_mode, plot, iplot
import plotly.graph_objs as go
from IPython.display import Image
init_notebook_mode(connected=True)
%matplotlib inline

# machine learning
import surprise
from surprise import KNNBasic
from surprise.model_selection import GridSearchCV
from surprise import Reader
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import KNNWithMeans

#(a) Read the data from ratings.csv with line format: userID, movieID, rating, timestamp.
# load the movies rating data (small)
rating_df = pd.read_csv("../input/ratings-small/ratings_small.csv")

# check the raw data
rating_df.head()
print("Dimension of the data: ", rating_df.shape)
# summary of the data
rating_df.describe()

# load the data into surprise specific data-structure
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(rating_df[['userId', 'movieId', 'rating']], reader)

# (b & c)Compute the average MAE and RMSE of the Probabilistic Matrix Factorization
# (PMF), User based Collaborative Filtering, Item based Collaborative Filtering,
# under the 5-folds cross-validation (10 points)

# default setting: distance MSD, k=10
benchmark = []

# iterate over all algorithms
for algorithm in [SVD(biased=False), KNNBasic(sim_options = {'user_based': True }), KNNBasic(sim_options = {'user_based': False})]:
    # perform cross validation
    results = cross_validate(algorithm, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    # get results & append algorithm name
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))

    # store data
    benchmark.append(tmp)

# (d) Compare the average (mean) performances of User-based collaborative filtering,
# item-based collaborative filtering, PMF with respect to RMSE and MAE. Which
# ML model is the best in the movie rating data? (10 points)
benchmark = pd.DataFrame(benchmark)

# udpate algrithm names
new_algorithms = ['PMF','UserCF','ItemCF']
benchmark['Algorithm'] = new_algorithms

benchmark

# Examine how the cosine, MSD (Mean Squared Difference), and Pearson
# similarities impact the performances of User based Collaborative Filtering and
# Item based Collaborative Filtering. Plot your results. Is the impact of the three
# metrics on User based Collaborative Filtering consistent with the impact of the
# three metrics on Item based Collaborative Filtering? (10 points)


benchmark2 = []

# iterate over all algorithms
for algorithm in [KNNBasic(sim_options = {'name':'cosine','user_based': True}), KNNBasic(sim_options = {'name':'MSD', 'user_based':True }),
                 KNNBasic(sim_options = {'name':'pearson','user_based': True}),
                 KNNBasic(sim_options = {'name':'cosine', 'user_based':False }),KNNBasic(sim_options = {'name':'MSD', 'user_based':False }),
                 KNNBasic(sim_options = {'name':'pearson','user_based': False})
                 ]:
    # perform cross validation
    results = cross_validate(algorithm, data, measures=['RMSE', 'MAE'], cv=5, verbose=False)

    # get results & append algorithm names
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)
    tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))

    # store data
    benchmark2.append(tmp)

benchmark2 = pd.DataFrame(benchmark2)

# udpate algorithm names
new_algorithms2 = ['Cosine-UserCF','MSD-UserCF','Pearson-UserCF','Cosine-ItemCF','MSD-ItemCF','Pearson-ItemCF']
benchmark2['Algorithm'] = new_algorithms2

# store results
results2 = benchmark2.set_index('Algorithm').sort_values('test_rmse', ascending=False)
results2

# plotting the results

# prepare the data for plotting
data = results2[['test_rmse', 'test_mae']]
grid = data.values

# create axis labels
x_axis = [label.split('_')[1].upper() for label in data.columns.tolist()]
y_axis = data.index.tolist()

x_label = 'Function'
y_label = 'Algorithm'


# get annotations and hovertext
hovertexts = []
annotations = []

for i, y_value in enumerate(y_axis):
    row = []
    for j, x_value in enumerate(x_axis):
        annotation = grid[i, j]
        row.append('Error: {:.4f}<br>{}: {}<br>{}: {}<br>Fit Time: {:.3f}s<br>Test Time: {:.3f}s'.format(annotation, y_label, y_value ,x_label, x_value,
                                                                                                         results2.loc[y_value]['fit_time'],
                                                                                                         results2.loc[y_value]['test_time']))
        annotations.append(dict(x=x_value, y=y_value, text='{:.4f}'.format(annotation), ax=0, ay=0, font=dict(color='#000000')))
    hovertexts.append(row)

# create trace
trace = go.Heatmap(x = x_axis,
                   y = y_axis,
                   z = data.values,
                   text = hovertexts,
                   hoverinfo = 'text',
                   colorscale = 'Picnic',
                   colorbar = dict(title = 'Error'))

# Create layout
layout = go.Layout(title = 'Cross-validated Comparison of Algorithms',
                   xaxis = dict(title = x_label),
                   yaxis = dict(title = y_label,
                                tickangle = -40),
                   annotations = annotations)

# Create plot
fig = go.Figure(data=[trace], layout=layout)
fig.show()

# (f) Examine how the number of neighbors impacts the performances of User based
# Collaborative Filtering and Item based Collaborative Filtering? Plot your results.
# (10 points)

# load the data into surprise specific data-structure format
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(rating_df[['userId', 'movieId', 'rating']], reader)

# User-based Collaborative Filtering: optimal k
benchmark_ucf = []

for i in range(1,30):
    # perform cross validation
    algorithm =KNNBasic(k=i, sim_options = {'name':'MSD','user_based': True})
    results = cross_validate(algorithm, data, measures=['RMSE', 'MAE'], cv=3, verbose=False)

    # get results & append algorithm names
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)

    # Store data
    benchmark_ucf.append(tmp)

# Item-based Collaborative Filtering: optimal k
benchmark_icf = []

for i in range(1,30):
    # perform cross validation
    algorithm = KNNBasic(k=i, sim_options = {'name':'MSD','user_based': False})
    results = cross_validate(algorithm, data, measures=['RMSE','MAE'], cv=3, verbose=False)

    # get results & append algorithm names
    tmp = pd.DataFrame.from_dict(results).mean(axis=0)

    # Store data
    benchmark_icf.append(tmp)

benchmark_ucf = pd.DataFrame(benchmark_ucf)
benchmark_icf = pd.DataFrame(benchmark_icf)

acc_userCF1 = benchmark_ucf['test_rmse']
acc_itemCF1 = benchmark_icf['test_rmse']

acc_userCF2 = benchmark_ucf['test_mae']
acc_itemCF2 = benchmark_icf['test_mae']

acc_itemCF1
acc_userCF1

# plotting the results (RMSE)

plt.figure(figsize=(12,8))
plt.plot(range(1,30), acc_userCF1, label = "User-based CF")
plt.plot(range(1,30), acc_itemCF1, label = "Item-based CF")
plt.title('')
plt.xlabel('Number of neighbors (K)', fontsize=12)
plt.ylabel('RMSE', fontsize=12)

plt.title('K Neighbors vs. RMSE (User-based CF and Item-based CF)', fontsize=18, y=1.03)
plt.legend(loc='best')
plt.grid(ls='dotted')

plt.savefig("plot_f (RMSE).png", dpi=300)

plt.show()

# plotting the results (MAE)

plt.figure(figsize=(12,8))
plt.plot(range(1,30), acc_userCF2, label = "User-based CF")
plt.plot(range(1,30), acc_itemCF2, label = "Item-based CF")
plt.title('')
plt.xlabel('Number of neighbors (K)', fontsize=12)
plt.ylabel('MAE', fontsize=12)

plt.title('K Neighbors vs. MAE (User-based CF and Item-based CF)', fontsize=18, y=1.03)
plt.legend(loc='best')
plt.grid(ls='dotted')

plt.savefig("plot_f (MAE).png", dpi=300)

plt.show()

# (g) Identify the best number of neighbor (denoted by K) for User/Item based
# collaborative filtering in terms of RMSE. Is the best K of User based collaborative
# filtering the same with the best K of Item based collaborative filtering? (10 points)

# find out the best number of neighbor (K) for User/Item based collaborative filtering in terms of RMSE
print("For User-based CF, the best number of neighbor (K) is at K =" , acc_userCF1.idxmin()+1, "with minimum RMSE:", min(acc_userCF1))
print("For Item-based CF, the best number of neighbor (K) is at K =" , acc_itemCF1.idxmin()+1 , "with minimum RMSE:", min(acc_itemCF1))

# find out the best number of neighbor (K) for User/Item based collaborative filtering in terms of MAE
print("For User-based CF, the best number of neighbor (K) is at K =" , acc_userCF2.idxmin()+1, "with minimum MAE:", min(acc_userCF2))
print("For Item-based CF, the best number of neighbor (K) is at K =" , acc_itemCF2.idxmin()+1 , "with minimum MAE:", min(acc_itemCF2))

