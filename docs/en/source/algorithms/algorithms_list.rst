===========================
List of Availble Algorithms
===========================

Notation
-----------

For the sake of clarity and consistency, we use the following notations.

- :math:`X`：feature matrix

.. math::

   X = \begin{pmatrix}
    x_{11} & x_{12} & \dots & x_{1m} \\
    x_{21} & x_{22} & \dots &  x_{2m} \\
    \vdots & \vdots & \dots &  \vdots \\
    v_{n1} & x_{n2} & \dots &  x_{nm} \\
    \end{pmatrix}


Here, each row denotes a sample (or an observation) :math:`x_i (i=1, \dots, n)` or :math:`X_{i.}` . Each column denotes a feature, :math:`X_{.j} (j = 1, \dots , p)`

- :math:`Y`: label in supervised learning

.. math::
   
   Y = \begin{pmatrix}
   y_1\\
   y_2\\
   \vdots \\
   y_n \\
   \end{pmatrix}

For regression, :math:`y_i \in \mathcal{R}`. For classification, :math:`y_i \in [1, 2, \dots, K]`, where :math:`K` is the number of classes.

- Training set, validation set, and test set: for machine learning, data are usually split into training set, validation set, and test set. We use the superscript train, val, and test to distinguish them。For example, :math:`X^{train}`, :math:`X^{val}`, and :math:`X^{test}` denote training set features, validation set features, and test set features respectively.



List of Algorithms
-------------------

.. csv-table::
   :header: "Algorithm", "Module", "Description"
   
   ":doc:`Local Normalization <./algos/LocalNormalization>`", "local/normalization", "normalize data"
   ":doc:`Local Standard Scaler <./algos/LocalStandardScaler>`", "local/standard_scaler", "standardize data"
   ":doc:`Horizontal Linear Regression <./algos/HorizontalLinearRegression>`", "horizontal/linear_regression", "two-party or multi-party horizontal linear regression"
   ":doc:`Horizontal Logistic Regression <./algos/HorizontalLogisticRegression>`", "horizontal/logistic_regression", "two-party or multi-party horizontal logistic regression"
   ":doc:`Horizontal ResNet <./algos/HorizontalResNet>`", "horizontal/Resnet", "two-party or multi-party horizontal horizontal ResNet"
   ":doc:`Vertical Feature Binning <./algos/VerticalBinningWoeIV>`", "vertical/binning_woe_iv", "calulate WoE and IV using equal-frequency binning or equal-width binning"
   ":doc:`Vertical Pearson <./algos/VerticalPearson>`", "vertical/pearson", "two-party or multi-party vertical Pearson correlation coefficient"
   ":doc:`Vertical Feature Selection <./algos/VerticalFeatureSelection>`", "vertical/feature_selection", "two-party or multi-party vertical feature selection"
   ":doc:`Vertical Logistic Regression <./algos/VerticalLogisticRegression>`", "vertical/logistic_regression", "two-party or multi-party vertical logistic regression"
   ":doc:`Vertical XGBoost <./algos/VerticalXgboost>`", "vertical/xgboost", "two-party or multi-party vertical xgboost"
   ":doc:`Vertical Kmeans <./algos/VerticalKMeans>` ", "vertical/kmeans", "two-party or multi-party vertical kmeans"