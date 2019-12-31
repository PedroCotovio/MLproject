# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import seaborn as sns; sns.set()

# All functions used in the Second Task of the ML project

# Creates confusion matrix
def get_cm(tar, res):
    """
    Compare unsupervised learning results against known labels
    Homologous of comp_val in EDA_Functions

    :param tar: list, known labels
    :param res: list, results
    :return: dataframe, confusion matrix
    """
    cl = set(tar)
    cl = set(tar)
    result = [{x: 0 for x in cl} for i in range(max(res) + 1)]
    for i, c in enumerate(res):
        result[c][tar[i]] += 1
    return pd.DataFrame(result)

# Matrix distribution
def checkcm_dist(cm, th=0.75):
    """
    Checks matrix distribution

    :param cm: dataframe, confusion matrix
    :param th: float, threshold
    :return: bool
    """
    counts = []
    for line in cm.values:
        counts.append(sum(line))
    counts = counts / sum(counts)
    for count in counts:
        if count > th: return False
    return True

#Calculate distances
def get_distances(X,model,mode='l2'):
    """
    Calculates distance between clusters

    :param X: dataframe, features
    :param model: object, clustering algorithm
    :param mode: str, metric
    :return: lists of distances and weights
    """
    distances = []
    weights = []
    children=model.children_
    dims = (X.shape[1],1)
    distCache = {}
    weightCache = {}
    for childs in children:
        c1 = X[childs[0]].reshape(dims)
        c2 = X[childs[1]].reshape(dims)
        c1Dist = 0
        c1W = 1
        c2Dist = 0
        c2W = 1
        if childs[0] in distCache.keys():
            c1Dist = distCache[childs[0]]
            c1W = weightCache[childs[0]]
        if childs[1] in distCache.keys():
            c2Dist = distCache[childs[1]]
            c2W = weightCache[childs[1]]
        d = np.linalg.norm(c1-c2)
        cc = ((c1W*c1)+(c2W*c2))/(c1W+c2W)

        X = np.vstack((X,cc.T))

        newChild_id = X.shape[0]-1

        # How to deal with a higher level cluster merge with lower distance:
        if mode=='l2':  # Increase the higher level cluster size suing an l2 norm
            added_dist = (c1Dist**2+c2Dist**2)**0.5
            dNew = (d**2 + added_dist**2)**0.5
        elif mode == 'max':  # If the previrous clusters had higher distance, use that one
            dNew = max(d,c1Dist,c2Dist)
        elif mode == 'actual':  # Plot the actual distance.
            dNew = d


        wNew = (c1W + c2W)
        distCache[newChild_id] = dNew
        weightCache[newChild_id] = wNew

        distances.append(dNew)
        weights.append( wNew)
    return distances, weights

#Plot dendogram
def plot_dendo(model, X, **kwargs):
    """
    Plot hierarchical clustering dendogram
    :param X: dataframe, features
    :param model: object, classifier
    """
    # Create linkage matrix and then plot the dendrogram

    distance, weight = get_distances(X, model)
    linkage_matrix = np.column_stack([model.children_, distance, weight]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

#AC parameter grid search
def ACparm(X, y, n):
    """
    Grid search for Agglomerative clustering algorithm

    :param X: dataframe, features
    :param y: array, target
    :param n: number of grid search iterations
    :return: model: object, best classifier
    best: float, fowlkes mallows score
    cm: dataframe, confusion matrix
    """

    linkages = ['ward', 'complete', 'average', 'single']
    affinitys = ['euclidean', 'l1', 'l2', 'manhattan']
    best = 0
    for linkage in linkages:
        for affinity in affinitys:
            if linkage == 'ward' and affinity != 'euclidean':
                continue
            else:
                cluster = AgglomerativeClustering(n_clusters=n, linkage=linkage, affinity=affinity)
                cluster.fit(X)
                tcm = get_cm(y, cluster.labels_)

                if metrics.fowlkes_mallows_score(y, cluster.labels_) > best \
                        and checkcm_dist(tcm) is True:
                    model = cluster
                    best = metrics.fowlkes_mallows_score(y, cluster.labels_)

    cm = get_cm(y, model.labels_)

    return model, best, cm

def model_metrics2(y, classifier):
    """
    Get unsupervised model metrics

    :param y: array, target
    :param classifier: object, clustering algorithm
    :return: Confusion matrix and fowlkes mallows score
    """

    np.set_printoptions(precision=2)

    return get_cm(y, classifier.labels_), \
           metrics.fowlkes_mallows_score(y, classifier.labels_)

# binary dataset
def create_bin(X,y):
    """
    Create transaction based binary dataset

    :param X: dataframe, features
    :param y: array, target
    :return: dataframe, binary dataset
    """
    # check if feature is binary and create transactions
    bindf = pd.DataFrame()
    X['Target'] = y
    transactions = [[] for i in range(len(X))]
    for col in X:
        if len(set(X[col])) == 2:
            bindf[col] = X[col].ge(1)
        else:
            for r, row in enumerate(X[col]):
                transactions[r].append(col + str(row))

    # Create binary dataset
    TxE = TransactionEncoder()
    te_ary = TxE.fit(transactions).transform(transactions)
    binary_database = pd.DataFrame(te_ary, columns=TxE.columns_)
    binary_database = pd.concat([binary_database, bindf], axis=1)
    return binary_database

def get_rules(binary_database):
    """
    Apply association rules algorithm, to check for patterns

    :param binary_database: dataframe, transaction based binary dataset
    :return: Rules
    """
    # Compute itemsets
    frequent_itemsets = apriori(binary_database, min_support=0.3, use_colnames=True)
    # Compute association rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    # Get every target significant rule with lift not 1
    Brules = []
    for i, rule in enumerate(rules.consequents):
        if 'Target' in str(rule) and rules.lift.values[i] != 1 and len(rule) == 1:
            Brules.append(i)
    rules = rules.iloc[Brules]
    if len(rules) == 0:
        return 'No rules found'
    return rules

def plot_dendogram(samples, method='ward', distance='euclidean', p=30):
    """
    Plot dendogram (without specific model)

    :param samples: dataframe, features
    :param method: str, method to calculate distances
    :param distance: str, type of distances
    :param p: int, number of branches to plot
    :return: plot
    """
    mergings = linkage(samples, method=method, metric=distance)

    dendrogram(mergings, p =p, truncate_mode='level',
               leaf_rotation=90,
               leaf_font_size=6,
               )

    return plt