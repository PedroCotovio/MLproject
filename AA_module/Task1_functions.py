# Imports
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix, confusion_matrix, classification_report, auc, plot_roc_curve
from sklearn.model_selection import RandomizedSearchCV, train_test_split, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import calibration_curve

# All functions used in the First Task of the ML project

def quadratic_kappa(actuals, preds, N=5):
    """
    This function calculates the Quadratic Kappa Metric used for Evaluation in the PetFinder competition
    at Kaggle. It returns the Quadratic Weighted Kappa metric score between the actual and the predicted values
    of adoption rating.

    :param actuals: array, real labels
    :param preds: array, predicted labels
    :param N: int, number of classes

    #from https://www.kaggle.com/aroraaman/quadratic-kappa-metric-explained-in-5-simple-steps
    """

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

def cv_metrics(X, y, classifier, class_names, n_folds, quad=True):
    """
    Create supervised learning metrics with cross-validation

    :param X: dataframe, features
    :param y: array, labels
    :param classifier: object, unfitted classifier
    :param class_names: list, strings of names to use in confusion matrix
    :param n_folds: int, number of folds in cross-validation
    :param quad: bool, if quadratic metric should be calculated
    :return: all metrics
    plt, confusion matrix plot
    estimator_, best classifier
    metrics, table of metrics
    quad_kappa, quadratic metric if calculated
    """
    folds = StratifiedKFold(n_folds)
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
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize='true')
    disp.ax_.set_title("Normalized confusion matrix")
    # Metrics
    y_pred = estimator_.predict(test[0])
    metrics = classification_report(test[1], y_pred)
    if quad is True:

        quad_kappa = quadratic_kappa(test[1], y_pred)
        return plt, estimator_, metrics, quad_kappa

    else:

        return plt, estimator_, metrics

def model_metrics(X, y, classifier, class_names, plot=True):
    """
    Create supervised learning metrics

    :param X: dataframe, features
    :param y: array, labels
    :param classifier: object, unfitted classifier
    :param class_names: list, strings of names to use in confusion matrix
    :return: all metrics
    plt, confusion matrix plot
    metrics, table of metrics
    """
    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    #Fit classifier
    classifier.fit(X_train, y_train)

    np.set_printoptions(precision=2)

    # Metrics
    y_pred = classifier.predict(X_test)
    metrics = classification_report(y_test, y_pred)

    if plot is True:
        # Plot
        plt.figure(figsize=(18, 8));
        disp = plot_confusion_matrix(classifier, X_test, y_test,
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues,
                                     normalize='true')
        disp.ax_.set_title("Normalized confusion matrix")

        return plt, metrics

    else:
        return metrics


def svc_param_selection(X, y, nfolds, n):

    """
    Search for best hyper-parameters for the svc classifier

    :param X: dataframe, features
    :param y: array, labels
    :param nfolds: int, number of folds in cross-validation
    :param n: int, number of iterations in random search
    :return: object, classifier object with best parameters found
    """
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
    degrees = [1, 2, 3, 4, 5, 6, 7, 8]
    param_grid = {'C': Cs, 'gamma': gammas, 'degree': degrees}
    r_search = RandomizedSearchCV(OneVsRestClassifier(svm.SVC(kernel='poly')), param_grid, cv=nfolds,
                                  n_jobs=nfolds, verbose=2, n_iter=n)
    r_search.fit(X, y)
    return r_search.best_params_

def calculate_vif_(X, thresh=5.0):
    """
    remove features that have high VIF

    :param X: dataframe, features
    :param thresh: float, vif threshold
    :return: dataframe, clean dataset
    """
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

    """
    Randomized search

    :param clf: object, classifier
    :param X: dataframe, features
    :param y: array, target
    :param base_accuracy: float, accuracy with default parameters
    :param n_iter: int, number of iterations in random search
    :param k_folds: int, number of folds in cross-validation
    :param grid: dict, grid of parameters to search from
    :return: rf_random: object, best classifier found
    improv: float, percentage of improvement
    """

    # Use the random grid to search for best hyper-parameters
    rf_random = RandomizedSearchCV(estimator=clf, param_distributions=grid,
                                   n_iter=n_iter, cv=k_folds, verbose=2, random_state=42, n_jobs=-1)
    # Fit the random search model
    rf_random.fit(X, y)
    #Evaluate improvements

    improv = (100 * (rf_random.best_score_ - base_accuracy) / base_accuracy)

    return rf_random, improv


def feature_selector(X, y):
    """
    Clean dataset of not needed features

    :param X: dataframe, features
    :param y: array, target
    :return: dataframe, clean dataset
    """

    #Remove colinear features
    vif = calculate_vif_(X)
    #Remove not important features
    clf = ExtraTreesClassifier(n_estimators=100, random_state=0).fit(vif, y)
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(vif)
    return X_new

def et_Srandom(clf, X, y, base_accuracy, n_iter=100, k_folds=3):
    """
    Random search for extra tree classifier

    :param clf: object, classifier
    :param X: dataframe, features
    :param y: array, target
    :param base_accuracy: float, accuracy with default parameters
    :param n_iter: int, number of iterations in random search
    :param k_folds: int, number of folds in cross-validation
    :return: rf_random: object, best classifier found
    improv: float, percentage of improvement
    """
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
    """
    Random search for Kneighbors classifier

    :param clf: object, classifier
    :param X: dataframe, features
    :param y: array, target
    :param base_accuracy: float, accuracy with default parameters
    :param n_iter: int, number of iterations in random search
    :param k_folds: int, number of folds in cross-validation
    :return: rf_random: object, best classifier found
    improv: float, percentage of improvement
    """

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

# Calibration analysis
def plot_cal_curve(X, y, clfs):
    """
    Plot calibration curve for list of classifiers

    :param X: dataframe, features
    :param y: array, target
    :param clfs: list, classifiers
    :return: plot, calibration curve
    """
    # Split the data into a training set and a test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # Plot calibration plots

    plt.figure(figsize=(18, 15));
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in clfs:
        clf.fit(X_train, y_train)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s" % (name,))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives")
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)')

    ax2.set_xlabel("Mean predicted value")
    ax2.set_ylabel("Count")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    return plt

# Classification and ROC analysis
def roc_metrics(X, y, classifier, n_splits):
    """
    Plot cross validated roc curve

    :param X: dataframe, features
    :param y: array, target
    :param classifier: object, classifier
    :param n_splits: int, number of folds in cross-validation
    :return: plot, Roc Curve
    """
    # Run classifier with cross-validation and plot ROC curves
    cv = StratifiedKFold(n_splits=n_splits)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X.iloc[train], y[train])
        viz = plot_roc_curve(classifier, X.iloc[test], y[test],
                             name='ROC fold {}'.format(i),
                             alpha=0.3, lw=1, ax=ax)
        interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label=r'$\pm$ 1 std. dev.')

    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           title="Receiver operating characteristic example")
    ax.legend(loc="lower right")
    return plt
