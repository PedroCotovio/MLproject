import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

#from https://www.kaggle.com/aroraaman/quadratic-kappa-metric-explained-in-5-simple-steps
def quadratic_kappa(actuals, preds, N=5):
    """This function calculates the Quadratic Kappa Metric used for Evaluation in the PetFinder competition
    at Kaggle. It returns the Quadratic Weighted Kappa metric score between the actual and the predicted values
    of adoption rating."""
    w = np.zeros((N, N))
    O = confusion_matrix(actuals, preds)
    for i in range(len(w)):
        for j in range(len(w)):
            w[i][j] = float(((i - j) ** 2) / (N - 1) ** 2)

    act_hist = np.zeros([N])
    for item in actuals:
        act_hist[item] += 1

    pred_hist = np.zeros([N])
    for item in preds:
        pred_hist[item] += 1

    E = np.outer(act_hist, pred_hist);
    E = E / E.sum();
    O = O / O.sum();

    num = 0
    den = 0
    for i in range(len(w)):
        for j in range(len(w)):
            num += w[i][j] * O[i][j]
            den += w[i][j] * E[i][j]
    return (1 - (num / den))

def cv_metrics(X, y, classifier, class_names, n_folds):
    folds = StratifiedKFold(10)
    best_ = 0
    for train_index, test_index in folds.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # Fit Model
        classifier.fit(X_train, y_train)
        # Get Score
        temp = classifier.score(X_test, y_test)
        if temp > best_:
            best_ = temp
            estimator_ = classifier
            test = [X_test, y_test]

    plt.figure(figsize=(18, 8));
    disp = plot_confusion_matrix(estimator_, test[0], test[1],
                                 display_labels=[0, 1, 2, 3],
                                 cmap=plt.cm.Blues,
                                 normalize='true')
    disp.ax_.set_title("Normalized confusion matrix")
    # Metrics
    y_pred = estimator_.predict(test[0])
    metrics = classification_report(test[1], y_pred)
    quad_kappa = quadratic_kappa(test[1], y_pred)

    return plt, estimator_, metrics, quad_kappa

def model_metrics(X, y, classifier, class_names):

    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    #Fit classifier
    classifier.fit(X_train,y_train)

    np.set_printoptions(precision=2)
    # Plot
    plt.figure(figsize=(18, 8));
    disp = plot_confusion_matrix(classifier, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
    disp.ax_.set_title("Normalized confusion matrix")
    # Metrics
    y_pred = classifier.predict(X_test)
    metrics = classification_report(y_test, y_pred)

    return plt, metrics


def svc_param_selection(X, y, nfolds, n):
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    degrees = [1, 2, 3, 4, 5, 6, 7, 8]
    param_grid = {'C': Cs, 'gamma': gammas, 'degree': degrees}
    r_search = RandomizedSearchCV(OneVsRestClassifier(svm.SVC(kernel='poly')), param_grid, cv=nfolds,
                                  n_jobs=nfolds, verbose=2, n_iter=n)
    r_search.fit(X, y)
    return r_search.best_params_

def calculate_vif_(X, thresh=5.0):
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    print('Remaining variables:')
    print(X.columns[variables])
    return X.iloc[:, variables]

def randS (clf, X, y, base_accuracy, n_iter, k_folds, grid):

    # Use the random grid to search for best hyper-parameters
    rf_random = RandomizedSearchCV(estimator=clf, param_distributions=grid,
                                   n_iter=n_iter, cv=k_folds, verbose=2, random_state=42, n_jobs=-1)
    # Fit the random search model
    rf_random.fit(X, y)
    #Evaluate improvements

    improv = (100 * (rf_random.best_score_ - base_accuracy) / base_accuracy)

    return rf_random, improv


def feature_selector(X, y):

    #Remove colinear features
    vif = calculate_vif_(X)
    #Remove not important features
    clf = ExtraTreesClassifier(n_estimators=100, random_state=0).fit(vif, y)
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(vif)
    return X_new

def et_Srandom(clf, X, y, base_accuracy, n_iter=100, k_folds=3):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=100, stop=2000, num=10)]
    # Type of criterion to consider
    criterion = ['gini', 'entropy']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'criterion': criterion,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    #Search Function
    rf_random, improv = randS(clf, X, y, base_accuracy, n_iter, k_folds, random_grid)

    return rf_random, improv


def knn_Srandom (clf, X, y, base_accuracy, n_iter, k_folds=3):

    # Number of Neighbors
    neighbors = list(range(1, 10)) + list(range(11, 30, 3)) + list(range(2, 8))

    # leaf Size
    leafs = list(range(10, 50))

    # Weights metric
    weight = ['uniform', 'distance']

    # Define Metric
    ps = [1,2,3]

    random_grid = {'n_neighbors': neighbors,
                   'leaf_size': leafs,
                   'weights': weight,
                   'p': ps}

    # Search Function
    rf_random, improv = randS(clf, X, y, base_accuracy, n_iter, k_folds, random_grid)

    return rf_random, improv