import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel


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

def feature_selector(X, y):

    #Remove colinear features
    vif = calculate_vif_(X)
    #Remove not important features
    clf = ExtraTreesClassifier(n_estimators=100, random_state=0).fit(vif, y)
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(vif)
    return X_new

def et_Srandom(clf, X, y, base_accuracy, n_iter, k_folds):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=100, stop=2000, num=10)]
    # Type of criterion to consider
    criterions = ['gini', 'entropy']
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
                   'criterion': criterions,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
    # Use the random grid to search for best hyperparameters
    rf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid, n_iter=n_iter, cv=k_folds, verbose=2,
                                   random_state=42, n_jobs=-1)
    # Fit the random search model
    rf_random.fit(X, y)
    #Evaluate improvements

    improv = (100 * (rf_random.best_score_ - base_accuracy) / base_accuracy)

    return rf_random, improv

