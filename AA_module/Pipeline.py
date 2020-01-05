from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from imblearn.combine import SMOTETomek
import numpy as np

class P_Classifier:
    def __init__(self, zeros=4, balanced=True):

        #Set basic vars
        self.z = zeros
        self.balanced = balanced
        self.train = False

        #Set classifiers

        #Multiclass
        # Define Tree model
        clf1 = ExtraTreesClassifier(bootstrap=False, ccp_alpha=0.0, class_weight=None,
                                    criterion='entropy', max_depth=40, max_features='auto',
                                    max_leaf_nodes=None, max_samples=None,
                                    min_impurity_decrease=0.0, min_impurity_split=None,
                                    min_samples_leaf=1, min_samples_split=5,
                                    min_weight_fraction_leaf=0.0, n_estimators=1366,
                                    n_jobs=None, oob_score=False, random_state=None, verbose=0,
                                    warm_start=False)
        # Define KNN model
        clf2 = KNeighborsClassifier(algorithm='auto', leaf_size=49, metric='minkowski',
                                    metric_params=None, n_jobs=None, n_neighbors=8, p=1,
                                    weights='distance')
        # Create model list
        clf_l = [('ET', clf1), ('KNN', clf2)]
        # Define Voting Classifier
        vclf = VotingClassifier(estimators=clf_l, voting='soft')

        #Binary class
        clf = ExtraTreesClassifier(n_estimators=100, random_state=0)

        #set self's
        self.cat_bin = clf
        self.dog_bin = clf
        self.cat_mc = vclf
        self.dog_mc = vclf

    def balance(self, X, y):
        smt = SMOTETomek()
        X_smt, y_smt = smt.fit_sample(X, y)

        return X_smt, y_smt

    def fit(self, X, y):

        if self.train is False:

            # Create binary target

            by = []
            for i in y:
                if i == self.z:
                    by.append(0)
                else:
                    by.append(1)

            # Create Multi Class dataset

            X['temp'] = by
            X['temp2'] = y
            MC_X = X[X.temp == 1]

            # Separate dogs from cats

            # MultiClass Datasets
            # Dogs
            Dd = MC_X[MC_X.Type == 1].reset_index()
            dy = Dd['temp2']
            # Cats
            Cd = MC_X[MC_X.Type == 2].reset_index()
            cy = Cd['temp2']

            # Drop y
            Dd = Dd.drop(['Type', 'index', 'temp', 'temp2'], axis=1)
            Cd = Cd.drop(['Type', 'index', 'temp', 'temp2'], axis=1)

            # Binary Datasets
            # Dogs
            Ddb = X[X.Type == 1].reset_index()
            dyb = Ddb['temp']
            # Cats
            Cdb = X[X.Type == 2].reset_index()
            cyb = Cdb['temp']

            # Drop y
            Ddb = Ddb.drop(['Type', 'index', 'temp', 'temp2'], axis=1)
            Cdb = Cdb.drop(['Type', 'index', 'temp', 'temp2'], axis=1)

            # Balance

            if self.balanced is True:

                Dd, dy = self.balance(Dd, dy)
                Cd, cy = self.balance(Cd, cy)
                Ddb, dyb = self.balance(Ddb, dyb)
                Cdb, cyb = self.balance(Cdb, cyb)

            # fitting

            self.cat_bin.fit(Cdb, cyb)
            self.dog_bin.fit(Ddb, dyb)
            self.cat_mc.fit(Cd, cy)
            self.dog_mc.fit(Dd, dy)

            print('Classifier fitted')

        else:
            print('Classifier already fitted')

    def predict(self, X, animal=0):

        if 'DataFrame' in str(type(X)):
            X = X.values

        elif 'numpy.ndarray' not in str(type(X[0])) or 'list' not in str(type(X[0])):
            X = [X]

        predictions = []

        for line in X:

            if line[animal] == 2:
                line = np.delete(line, animal)
                line = line.reshape(1, -1)
                temp = self.cat_bin.predict(line)

                if temp == 0:
                    predictions.append(self.z)
                else:
                    temp = self.cat_mc.predict(line)
                    predictions.append(temp[0])

            elif line[animal] == 1:
                line = np.delete(line, animal)
                line = line.reshape(1, -1)
                temp = self.dog_bin.predict(line)

                if temp == 0:
                    predictions.append(self.z)
                else:
                    temp = self.dog_mc.predict(line)
                    predictions.append(temp[0])

        return predictions

    def score(self, X, y, animal=0):

        predictions = self.predict(X, animal)

        return accuracy_score(y, predictions)













