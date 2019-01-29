# ISwitch Application Identificaiton

This part of the code corresponds to the sixth part of our paper:**IN-NETWORK INTELLIGENT APPLICATION IDENTIFICATION**. In-network application identification is beneficial for malware detection, content cache, application-specific QoS, traffic control,etc. This section implements specific application identification methods into iSwitch.


# Environmen

 

 - Ubuntu 16.04
 - Python 3.6
 - Pytorch 0.4.0

## Usage

**Xgboost**
XGBoost is an optimized distributed gradient boosting library designed to be highly _**efficient**_, _**flexible**_ and _**portable**_. It implements machine learning algorithms under the [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting) framework. XGBoost provides a parallel tree boosting (also known as GBDT, GBM) that solve many data science problems in a fast and accurate way.
The testing dataset is 

> iswitch-app_dataset.csv


Run the following command to train the model:

> `'python train_app.py '`

  
Run the following command to test the model:

> `'python load_app_model.py '`

To install xgboost, please see :

> £¨https://xgboost.readthedocs.io/en/latest/£©

**ThunderSvm**
The mission of ThunderSVM is to help users easily and efficiently apply SVMs to solve problems. ThunderSVM exploits GPUs and multi-core CPUs to achieve high efficiency. Key features of ThunderSVM are as follows.

-   Support all functionalities of LibSVM such as one-class SVMs, SVC, SVR and probabilistic SVMs.
-   Use same command line options as LibSVM.
-   Support Python, R and Matlab interfaces.
-   Supported Operating Systems: Linux, Windows and MacOS.
The training/testing dataset is 

> svm_app_train.txt/svm_app_test.txt


Run the following command to train the model:

> `'python train_app.py '`

  
The trained model is saved as:

> `'svm_app.model '`

To install thundersvm, please see :

> £¨https://github.com/Xtra-Computing/thundersvm£©


**DNN**
The training/testing dataset is 

>iswitch_app_dataset.csv

The deep learning network model is:

> app_class.py

Run the following code to train the DNN model:

> python app-DNN.py

**CNN**

Run the following code to train the CNN model:

> python traffic_cnn.py
## Parameters Setting and Results
| Algorithm | Parameter | Accuracy |
|--|--|--
| XGBoost  | tree-method = gpu-hist, max-depth = 6 |0.99
| ThunderSVM | kernel = rbf,cost parameter c = 10, $\gamma$ =0.025 |0.93
| DNN | 6 fully-connected layer+3 batch normalize layer|0.89
| CNN |2 convolution layer+2 fully connected layer |0.97



```