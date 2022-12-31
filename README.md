Random Forest using Scikit-Learn and LIME for Regression

- random forest
- regression
- lime
- black box
- bagging
- ensemble
- local explanations
- xai
- python
- feature engine
- scikit optimize
- flask
- nginx
- gunicorn
- docker

This is an explainable version of Random Forest- black box model using LIME (Local interpretable Model-agnostic Explanations).

A Random Forest algorithm fits a number of decision trees on various samples of the dataset and uses mean of all outputs to improve the predictive accuracy and controls over-fitting.

It can be categorized as a black box model since it is complex and not straightforwardly interpretable to humans.

Model explainability is provided using LIME. Local explanations are provided here. Explanations at each instance can be understood using LIME. These explanations can be viewed by means of various plots.

The data preprocessing step includes:

- for categorical variables
  - Handle missing values in categorical:
    - When missing values are frequent, then impute with 'missing' label
    - When missing values are rare, then impute with most frequent
- Group rare labels to reduce number of categories
- One hot encode categorical variables

- for numerical variables

  - Add binary column to represent 'missing' flag for missing values
  - Impute missing values with mean of non-missing
  - MinMax scale variables prior to yeo-johnson transformation
  - Use Yeo-Johnson transformation to get (close to) gaussian dist.
  - Standard scale data after yeo-johnson

- for target variable
  - Use Yeo-Johnson transformation to get (close to) gaussian dist.
  - Standard scale target data after yeo-johnson

HPT based on Bayesian optimization is included for tuning Random Forest hyper-parameters.

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as abalone, auto_prices, computer_activity, heart_disease, white_wine, and ailerons.

The main programming language is Python. Other tools include Scikit-Learn for main algorithm, interpret package for model explainability, feature-engine for preprocessing, Scikit-Optimize for HPT, Flask + Nginx + gunicorn for web service. The web service provides three endpoints- /ping for health check, /infer for predictions in real time and /explain to generate local explanations.
