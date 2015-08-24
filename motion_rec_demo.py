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

"""
Demo script for the motion patterns recognition based on the accelerometer data from UCI:
https://archive.ics.uci.edu/ml/datasets/Dataset+for+ADL+Recognition+with+Wrist-worn+Accelerometer

Dynamic Time Warping (DTW) and K-Nearest Neighbors (KNN) algorithms for machine learning are used
to demonstrate labeling of the varying-length sequences. It can be applied to time series classification or
other cases, which require matching / training sequences with unequal lengths.

Scikit-Learn doesn't have any DTW implementations, so a custom class has been implemented (KnnDtwClassifier)
as a part of TinyLearn module.

DTW is slow by default, taking into account its quadratic complexity, that's why we're speeding up the classification
using an alternative approach with histograms and CommonClassifier from TinyLearn.
"""

from tinylearn import KnnDtwClassifier
from tinylearn import CommonClassifier
import pandas as pd
import numpy as np
import os

train_labels = []
test_labels = []
train_data_raw = []
train_data_hist = []
test_data_raw = []
test_data_hist = []

# Utility function for normalizing numpy arrays
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

# Loading all data for training and testing from TXT files
def load_data():
    for d in os.listdir("data"):
        for f in os.listdir(os.path.join("data", d)):
            if f.startswith("TRAIN"):
                train_labels.append(d)
                tr = normalize(np.ravel(pd.read_csv(os.path.join("data", d, f),
                                                    delim_whitespace=True,
                                                    header=None)))
                train_data_raw.append(tr)
                train_data_hist.append(np.histogram(tr, bins=20)[0])
            else:
                test_labels.append(d)
                td = normalize(np.ravel(pd.read_csv(os.path.join("data", d, f),
                                                delim_whitespace=True,
                                                header=None)))
                test_data_raw.append(td)
                test_data_hist.append(np.histogram(td, bins=20)[0])

# Demonstration of KnnDtwClassifier and CommonClassifier for motion pattern recognition
def demo_classifiers():
    # Raw sequence labeling with KnnDtwClassifier and KNN=1
    clf1 = KnnDtwClassifier(1)
    clf1.fit(train_data_raw, train_labels)

    for index, t in enumerate(test_data_raw):
        print("KnnDtwClassifier prediction for " +
              str(test_labels[index]) + " = " + str(clf1.predict(t)))

    # Let's do an extended prediction to get the distances to 3 nearest neighbors
    print("\n")
    clf2 = KnnDtwClassifier(3)
    clf2.fit(train_data_raw, train_labels)

    for index, t in enumerate(test_data_raw):
        res = clf2.predict_ext(t)
        nghs = np.array(train_labels)[res[1]]
        print("KnnDtwClassifier neighbors for " + str(test_labels[index]) + " = " + str(nghs))
        print("KnnDtwClassifier distances to " + str(nghs) + " = " + str(res[0]))

    # Now let's use CommonClassifier with the histogram data for faster prediction
    print("\n")
    clf3 = CommonClassifier(default=True)
    clf3.fit(train_data_hist, train_labels)

    clf3.print_fit_summary()
    print("\n")

    for index, t in enumerate(test_data_hist):
        print("CommonClassifier prediction for " + str(test_labels[index]) +
              " = " + str(clf3.predict(t)))

if __name__ == "__main__":
    load_data()
    demo_classifiers()
