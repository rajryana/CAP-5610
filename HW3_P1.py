# KMeans file (kmeans.py) used as utility script
import kmeans as km
import numpy as np
import pandas as pd
from time import time
# Number of labels
k = 10
print(k)
# Reading data
dataset2 = pd.read_csv('../input/kmeans-data/data.csv')
label2 = pd.read_csv('../input/kmeans-data/label.csv')
dataset2 = pd.concat([label2, dataset2], axis=1)
dataset2
print(dataset2.shape)
df_data=pd.DataFrame(dataset2)
dataset2=np.array(df_data.iloc[0:])
dataset2
print(dataset2.shape)
dataset2 = np.array([(x[0:785]) for x in dataset2])
dataset2 = dataset2.tolist()

#Q1 Run K-means clustering with Euclidean, Cosine and Jarcard similarity. Specify K= the
# number of categorical values of y (the number of classifications). Compare the SSEs of
# Euclidean-K-means, Cosine-K-means, Jarcard-K-means. Which method is better?

euclidian_start1 = time()
clustering_euclidian = km.kmeans(dataset2,k,dist_type='euclidian')
euclidian_time1 = time() - euclidian_start1
print('Euclidian SSE: ', clustering_euclidian['withinss'])


cosine_start1 = time()
clustering_cosine = km.kmeans(dataset2,k,dist_type='cosine')
cosine_time1 = time() - cosine_start1
print('Cosine SSE: ', clustering_cosine['withinss'])


jaccard_start1 = time()
clustering_jaccard = km.kmeans(dataset2,k,dist_type='jaccard')
jaccard_time1 = time() - jaccard_start1
print('Jaccard SSE: ', clustering_jaccard['withinss'])

# Q2: Compare the accuracies of Euclidean-K-means Cosine-K-means, Jarcard-K-means. First,
# label each cluster using the majority vote label of the data points in that cluster. Later, compute
# the predictive accuracy of Euclidean-K-means, Cosine-K-means, Jarcard-K-means. Which metric
# is better? (10 points)

def accuracy(cluster):
    label_accuracy = list()
    for cluster in cluster['clusters']:
        labels = dict()
        for item in cluster:
            labels.setdefault(item[0],0)
            labels[item[0]] += 1
        vals = np.array(list(labels.values()))
        vals.sort()
        if len(vals) == 0:
            label_accuracy.append(0)
        else:
            label_accuracy.append(vals[-1]/sum(vals))
    label_accuracy = np.array(label_accuracy)
    return label_accuracy.mean()

print('Euclidian Accuracy: ', accuracy(clustering_euclidian))
print('Cosine Accuracy: ', accuracy(clustering_cosine))
print('Jaccard Accuracy: ', accuracy(clustering_jaccard))

# Q3: Set up the same stop criteria: “when there is no change in centroid position OR when the
# SSE value increases in the next iteration OR when the maximum preset value (e.g., 500, you
# can set the preset value by yourself) of iteration is complete”, for Euclidean-K-means, Cosine-Kmeans,
# Jarcard-K-means. Which method requires more iterations and times to converge? (10
# points)

print('Euclidian Iterations: ', clustering_euclidian['iterations'])
print('Cosine Iterations: ', clustering_cosine['iterations'])
print('Jaccard Iterations: ', clustering_jaccard['iterations'])

print("Euclidian \t Time: {} \t Iterations: {}".format(euclidian_time1, clustering_euclidian['iterations']))
print("Cosine \t Time: {} \t Iterations: {}".format(cosine_time1, clustering_cosine['iterations']))
print("Jaccard \t Time: {} \t Iterations: {}".format(jaccard_time1, clustering_jaccard['iterations']))

# Q4: Compare the SSEs of Euclidean-K-means Cosine-K-means, Jarcard-K-means with respect to
# the following three terminating conditions: (10 points)

def run_condition(condition, dataset,k):

    euclidian_start = time()
    clustering_euclidian = km.kmeans(dataset,k,dist_type='euclidian',condition=condition)
    euclidian_time = time() - euclidian_start

    print("Euclidian \t Time: {} \t Iterations: {}".format(euclidian_time, clustering_euclidian['iterations']))
    print('Euclidian SSE: ', clustering_euclidian['withinss'])

    cosine_start = time()
    clustering_cosine = km.kmeans(dataset,k,dist_type='cosine',condition=condition)
    cosine_time = time() - cosine_start

    print("Cosine \t\t Time: {} \t Iterations: {}".format(cosine_time, clustering_cosine['iterations']))
    print('Cosine SSE: ', clustering_cosine['withinss'])

    jaccard_start = time()
    clustering_jaccard = km.kmeans(dataset,k,dist_type='jaccard',condition=condition)
    jaccard_time = time() - jaccard_start

    print("Jaccard \t Time: {} \t Iterations: {}".format(jaccard_time, clustering_jaccard['iterations']))
    print('Jaccard SSE: ', clustering_jaccard['withinss'])

# when there is no change in centroid position
run_condition('centroid',dataset2,k)

# when the SSE value increases in the next iteration
run_condition('sse',dataset2,k)

# when the maximum preset value (e.g., 100) of iteration is complete
run_condition('iteration',dataset2,k)
