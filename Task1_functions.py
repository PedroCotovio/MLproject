import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix, classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm


def model_metrics(X_test, y_test, classifier, class_names):
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
    r_search = RandomizedSearchCV(OneVsRestClassifier(svm.SVC(kernel='poly')), param_grid, cv=nfolds, n_jobs=nfolds, n_iter=n)
    r_search.fit(X, y)
    return r_search.best_params_