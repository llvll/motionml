# Copyright (c) 2015, Oleg Puzanov
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Helper classes for the basic classification tasks with Scikit-Learn and Pandas."""

import logging
import numpy as np
from fastdtw import fastdtw
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import (ExtraTreesClassifier,
                              RandomForestClassifier)
from sklearn.svm import SVC
from sklearn.linear_model import (LogisticRegression,
                                  SGDClassifier)


class FeatureReducer(object):
    """ Removes the features (columns) from the supplied DataFrame according to the
        function 'reduce_func'.

        The default use case is about removing the features, which have a very small weight
        and won't be useful for classification tasks.

        Feature weighting is implemented using ExtraTreesClassifier.
    """
    def __init__(self, df_features, df_targets, reduce_func=None):
        self.df_features = df_features
        self.df_targets = df_targets
        self.reduce_func = reduce_func
        self.dropped_cols = []

    def reduce(self, n_estimators=10):
        total_dropped = 0
        self.dropped_cols = []

        if self.reduce_func is not None:
            clf = ExtraTreesClassifier(n_estimators)
            clf.fit(self.df_features, self.df_targets).transform(self.df_features)

            for i in range(len(clf.feature_importances_)):
                if self.reduce_func(clf.feature_importances_[i]):
                    total_dropped += 1
                    logging.info("FeatureReducer: dropping column \'" +
                                 self.df_features.columns.values[i] + "\'")
                    self.dropped_cols.append(self.df_features.columns[i])

            [self.df_features.drop(c, axis=1, inplace=True) for c in self.dropped_cols]
        return total_dropped

    def print_weights(self, n_estimators=10):
        clf = ExtraTreesClassifier(n_estimators)
        clf.fit(self.df_features, self.df_targets).transform(self.df_features)
        [print("Feature \'" + self.df_features.columns.values[i] + " has weight " +
               clf.feature_importances_[i]) for i in range(len(clf.feature_importances_))]


class CrossValidator(object):
    """ Thin wrapper around 'cross_val_score' method of Scikit-Learn.
    """
    def __init__(self, estimator, df_features, df_targets, cv=5):
        self.scores = np.empty
        self.estimator = estimator
        self.df_features = df_features
        self.df_targets = df_targets
        self.cv = cv

    def cross_validate(self):
        self.scores = cross_val_score(self.estimator, self.df_features, self.df_targets, cv=self.cv)
        return self.scores

    def print_summary(self):
        if self.scores.size == 0:
            print("No data, please execute 'cross_validate' at first.")
        else:
            print("Cross-validation summary for " + self.estimator.__class__.__name__)
            print("Mean score: %0.2f (+/- %0.2f)" % (self.scores.mean(), self.scores.std() * 2))
            [print("Score #" + i + ": %0.2f", self.scores[i]) for i in range(len(self.scores))]


class CvEstimatorSelector(object):
    """Executes the cross-validation procedures to discover the best performing estimator
       from the supplied ones.

       The best estimator is selected according to the highest mean score.
    """
    def __init__(self, df_features, df_targets, cv=5):
        self.scores = {}
        self.estimators = {}
        self.df_features = df_features
        self.df_targets = df_targets
        self.cv = cv
        self.selected_name = None

    def add_estimator(self, name, instance):
        self.estimators[name] = instance

    def select_estimator(self):
        self.selected_name = None
        largest_val = 0

        for name in self.estimators:
            c_val = CrossValidator(self.estimators[name], self.df_features, self.df_targets, self.cv)
            self.scores[name] = c_val.cross_validate().mean()
            logging.info("Mean score for \'" + name + "\' estimator is " + str(self.scores[name]))
            if largest_val < self.scores[name]:
                largest_val = self.scores[name]
                self.selected_name = name

        return self.selected_name

    def print_summary(self):
        if self.selected_name is None:
            print("No data, please execute 'select_estimator' at first.")
        else:
            print("Selection summary based on the cross-validation of " +
                  str(len(self.estimators)) + " estimators.")
            print("Selected estimator \'" + self.selected_name +
                  "\' with " + str(self.scores[self.selected_name]) + " mean score.")
            print("Other scores ...")
            [print("Estimator \'" + n + " \' has mean score " +
                   str(self.scores[n])) for n in self.estimators if (n != self.selected_name)]


class GridSearchEstimatorSelector(object):
    """Thin wrapper around GridSearchCV class of Scikit-Learn for discovering
       the best performing estimator.
    """
    def __init__(self, df_features, df_targets, cv=5):
        self.scores = {}
        self.estimators = {}
        self.df_features = df_features
        self.df_targets = df_targets
        self.cv = cv
        self.selected_name = None
        self.best_estimator = None

    def add_estimator(self, name, instance, params):
        self.estimators[name] = {'instance': instance, 'params': params}

    def select_estimator(self):
        self.selected_name = None
        largest_val = 0

        for name in self.estimators:
            est = self.estimators[name]
            clf = GridSearchCV(est['instance'], est['params'], cv=self.cv)
            clf.fit(self.df_features, self.df_targets)
            self.scores[name] = clf.best_score_
            logging.info("Best score for \'" + name + "\' estimator is " + str(clf.best_score_))
            if largest_val < self.scores[name]:
                largest_val = self.scores[name]
                self.selected_name = name
                self.best_estimator = clf.best_estimator_

        return self.selected_name

    def print_summary(self):
        if self.selected_name is None:
            print("No data, please execute 'select_estimator' at first.")
        else:
            print("Selection summary based on GridSearchCV and " +
                  str(len(self.estimators)) + " estimators.")
            print("Selected estimator \'" + self.selected_name +
                  "\' with " + str(self.scores[self.selected_name]) + " mean score.")
            print(self.best_estimator)
            print("\nOther scores ...")
            [print("Estimator \'" + n + "\' has mean score " +
                   str(self.scores[n])) for n in self.estimators.keys() if (n != self.selected_name)]


class KnnDtwClassifier(BaseEstimator, ClassifierMixin):
    """Custom classifier implementation for Scikit-Learn using Dynamic Time Warping (DTW)
       and KNN (K-Nearest Neighbors) algorithms.

       This classifier can be used for labeling the varying-length sequences, like time series
       or motion data.

       FastDTW library is used for faster DTW calculations - linear instead of quadratic complexity.
    """
    def __init__(self, n_neighbors=1):
        self.n_neighbors = n_neighbors
        self.features = []
        self.labels = []

    def get_distance(self, x, y):
        return fastdtw(x, y)[0]

    def fit(self, X, y=None):
        for index, l in enumerate(y):
            self.features.append(X[index])
            self.labels.append(l)
        return self

    def predict(self, X):
        dist = np.array([self.get_distance(X, seq) for seq in self.features])
        indices = dist.argsort()[:self.n_neighbors]
        return np.array(self.labels)[indices]

    def predict_ext(self, X):
        dist = np.array([self.get_distance(X, seq) for seq in self.features])
        indices = dist.argsort()[:self.n_neighbors]
        return (dist[indices],
                indices)


class CommonClassifier(object):
    """Helper class to execute the common classification workflow - from training to prediction
       to metrics reporting with the popular ML algorithms, like SVM or Random Forest.

       Includes the default list of estimators with instances and parameters, which have been
       proven to work well.
    """
    def __init__(self, default=True, cv=5, reduce_func=None):
        self.cv = cv
        self.default = default
        self.reduce_func = reduce_func
        self.reducer = None
        self.grid_search = None

    def add_estimator(self, name, instance, params):
        self.grid_search.add_estimator(name, instance, params)

    def fit(self, X, y=None):
        if self.default:
            self.grid_search = GridSearchEstimatorSelector(X, y, self.cv)
            self.grid_search.add_estimator('SVC', SVC(), {'kernel': ["linear", "rbf"],
                                                          'C': [1, 5, 10, 50],
                                                          'gamma': [0.0, 0.001, 0.0001]})
            self.grid_search.add_estimator('RandomForestClassifier', RandomForestClassifier(),
                                       {'n_estimators': [5, 10, 20, 50]})
            self.grid_search.add_estimator('ExtraTreeClassifier', ExtraTreesClassifier(),
                                       {'n_estimators': [5, 10, 20, 50]})
            self.grid_search.add_estimator('LogisticRegression', LogisticRegression(),
                                       {'C': [1, 5, 10, 50], 'solver': ["lbfgs", "liblinear"]})
            self.grid_search.add_estimator('SGDClassifier', SGDClassifier(),
                                       {'n_iter': [5, 10, 20, 50], 'alpha': [0.0001, 0.001],
                                        'loss': ["hinge", "modified_huber",
                                                 "huber", "squared_hinge", "perceptron"]})

        if self.reduce_func is not None:
            self.reducer = FeatureReducer(X, y, self.reduce_func)
            self.reducer.reduce(10)

        return self.grid_search.select_estimator()

    def print_fit_summary(self):
        return self.grid_search.print_summary()

    def predict(self, X):
        if self.grid_search.selected_name is not None:
            if self.reduce_func is not None and len(self.reducer.dropped_cols) > 0:
                X.drop(self.reducer.dropped_cols, axis=1, inplace=True)
            return self.grid_search.best_estimator.predict(X)
        else:
            return None
