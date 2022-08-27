Random Forest using Scikit-Learn and LIME for Regression

- random forest
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

Preprocessing includes missing data imputation, standardization, one-hot encoding etc. For numerical variables, missing values are imputed with the mean and a binary column is added to represent 'missing' flag for missing values. For categorical variable missing values are handled using two ways: when missing values are frequent, impute them with 'missing' label and when missing values are rare, impute them with the most frequent.

HPT based on Bayesian optimization is included for tuning Random Forest hyper-parameters.

The main programming language is Python. Other tools include Scikit-Learn for main algorithm, interpret package for model explainability, feature-engine for preprocessing, Scikit-Optimize for HPT, Flask + Nginx + gunicorn for web service. The web service provides three endpoints- /ping for health check, /infer for predictions in real time and /explain to generate local explanations.
