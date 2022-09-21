===========================
List of Availble Algorithms
===========================

Notation
-----------

To avoid confusion of terms and notations, we make the following basic definitions:

- :math:`X`: feature matrix

.. math::

   X = \begin{pmatrix}
    x_{11} & x_{12} & \dots & x_{1m} \\
    x_{21} & x_{22} & \dots &  x_{2m} \\
    \vdots & \vdots & \dots &  \vdots \\
    v_{n1} & x_{n2} & \dots &  x_{nm} \\
    \end{pmatrix}


Here, each row denotes a sample (or an observation) :math:`x_i (i=1, \dots, n)` or :math:`X_{i.}` . 
Each column denotes a feature, :math:`X_{.j} (j = 1, \dots , p)`

- :math:`Y`: the label in supervised learning

.. math::
   
   Y = \begin{pmatrix}
   y_1\\
   y_2\\
   \vdots \\
   y_n \\
   \end{pmatrix}

We have :math:`y_i \in \mathcal{R}` for regression problem, and :math:`y_i \in \mathcal{Z}` for classification

- Training/Validation dataset: in XFL, we mainly use two types of dataset, one for training and the other for validation.
We use the superscript "train", "val" to identify them.
For example, :math:`X^{train}`, :math:`X^{val}` denote training dataset, validation dataset respectively.


List of Algorithms
-------------------

.. csv-table::
   :header: "Algorithm", "Module", "Description"
   
   ":doc:`Local Normalization <./algos/LocalNormalization>`", "local/normalization", "normalize data"
   ":doc:`Local Standard Scaler <./algos/LocalStandardScaler>`", "local/standard_scaler", "standardize data"
   ":doc:`Local Data Split <./algos/LocalDataSplit>`", "local/data_split", "split data into train and validation set"
   ":doc:`Local Feature Preprocess <./algos/LocalFeaturePreprocess>`", "local/feature_preprocess", "preprocess feature"
   ":doc:`Horizontal Linear Regression <./algos/HorizontalLinearRegression>`", "horizontal/linear_regression", "two-party or multi-party horizontal linear regression"
   ":doc:`Horizontal Logistic Regression <./algos/HorizontalLogisticRegression>`", "horizontal/logistic_regression", "two-party or multi-party horizontal logistic regression"
   ":doc:`Horizontal ResNet <./algos/HorizontalResNet>`", "horizontal/Resnet", "two-party or multi-party horizontal ResNet"
   ":doc:`Horizontal Bert <./algos/HorizontalBert>`", "horizontal/Bert", "two-party or multi-party horizontal Bert"
   ":doc:`Vertical Feature Binning <./algos/VerticalBinningWoeIV>`", "vertical/binning_woe_iv", "calulate WoE and IV using equal-frequency binning or equal-width binning"
   ":doc:`Vertical Pearson <./algos/VerticalPearson>`", "vertical/pearson", "two-party or multi-party vertical Pearson correlation coefficient"
   ":doc:`Vertical Feature Selection <./algos/VerticalFeatureSelection>`", "vertical/feature_selection", "two-party or multi-party vertical feature selection"
   ":doc:`Vertical Logistic Regression <./algos/VerticalLogisticRegression>`", "vertical/logistic_regression", "two-party or multi-party vertical logistic regression"
   ":doc:`Vertical XGBoost <./algos/VerticalXgboost>`", "vertical/xgboost", "two-party or multi-party vertical xgboost"
   ":doc:`Vertical Kmeans <./algos/VerticalKMeans>` ", "vertical/kmeans", "two-party or multi-party vertical kmeans"
   ":doc:`Vertical Sampler <./algos/VerticalSampler>` ", "vertical/sampler", "two-party or multi-party vertical sampler"