Random Forest using Scikit-Learn and LIME for regression

* randomm forest
* lime
* blackbox 
* bagging
* local explanations



This is an explainable version of Random Forest- blackbox model using LIME (Local interpretabl Model-agnostic Explanations). \

Global as well as Local explanations are provided here. Explanations at each instance can be understood using LIME.  \

These explanations can be viewed by means of various plots.\


Preprocessing includes missing data imputation, standardization, one-hot encoding etc. \

HPT based on Bayesian optimization is included for tuning Random Forest hyper-parameters. \


The main programming language is Python. Other tools include Scikit-Learn for main algorithm, LIME for model explanability, feature-engine for preprocessing, Scikit-Optimize for HPT, Flask + Nginx + gunicorn for web service.



