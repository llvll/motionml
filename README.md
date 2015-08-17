# stclassify
### Spatio-temporal classification framework based on KNN-DTW and HMM

This is a domain-specific example of using TinyLearn module for matching (classifying) time-series data, which includes both spatial and temporal components. 

The following ML algorithms are used according to the best practices for time-series classification tasks:

1. KNN-DTW: K-Nearest Neighbors with Dynamic Time Warping
2. HMM: Hidden Markov Model

The project includes a demo application for human action recognition by using the supervised machine learning and pattern matching. The following spatio-temporal patterns (labels) have been provided for the demonstration purposes:

* Walking
* Running
* Staying
* Sleeping
* Eating (inside kitchen)
* Smoking (inside smoking room)

Classification is done according to the test data with location, acceleration and timestamp components. Such data can be collected using any smartphone device with GPS and accelerometer/gyroscope on board. 

IPython Notebook (.ipynb file) is included for step-by-step execution of the demo application with extra comments.

This code is using TinyLearn framework, which simplifies the classification tasks with Python and the following modules:

* Scikit-Learn
* Pandas
* Seqlearn
* OpenCV
* Mlpy

TinyLearn framework is still at an early development stage. Please use with caution and feel free to ask any questions: oleg.v.puzanov@gmail.com
