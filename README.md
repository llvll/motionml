# motionml
### Motion pattern recognition using KNN-DTW and classifiers from TinyLearn

This is a domain-specific example of using TinyLearn module for recognizing (classifying) the motion patterns according to the supplied accelerometer data. 

The following motion patterns are included into this demo:

* Walking
* Sitting down on a chair
* Getting up from a bed
* Drinking a glass
* Descending stairs
* Combing hair
* Brushing teeth

The accelerometer data is based on the following public dataset from UCI: https://archive.ics.uci.edu/ml/datasets/Dataset+for+ADL+Recognition+with+Wrist-worn+Accelerometer

Dynamic Time Warping (DTW) and K-Nearest Neighbors (KNN) algorithms for machine learning are used
to demonstrate labeling of the varying-length sequences with accelerometer data. Such algorithms can be applied to time series classification or other cases, which require matching / training sequences with unequal lengths.

Scikit-Learn doesn't have any DTW implementations, so a custom class has been implemented (KnnDtwClassifier)
as a part of TinyLearn module.

DTW is slow by default, taking into account its quadratic complexity, that's why we're speeding up the classification
using an alternative approach with histograms and CommonClassifier from TinyLearn.

IPython Notebook (.ipynb file) is included for step-by-step execution of the demo application with extra comments.

This code is using TinyLearn framework, which simplifies the classification tasks with Python and the following modules:

* Scikit-Learn
* Pandas
* OpenCV
* Other libraries, like FastDTW 

TinyLearn framework is still at an early development stage. Please use with caution and feel free to ask any questions: oleg.v.puzanov@gmail.com
