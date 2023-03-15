# Chapter: 5 -> Model Tuning

# --------Tuning a CART's Hyperparameters--------

#Tree hyperparameters
# In the following exercises you'll revisit the Indian Liver Patient dataset which was introduced in a previous chapter.

# Your task is to tune the hyperparameters of a classification tree. Given that this dataset is imbalanced, you'll be using the ROC AUC score as a metric instead of accuracy.

# We have instantiated a DecisionTreeClassifier and assigned to dt with sklearn's default hyperparameters. You can inspect the hyperparameters of dt in your console.

# Which of the following is not a hyperparameter of dt?

# Answer: min_features

# -----

# Set the tree's hyperparameter grid

# Define params_dt
params_dt = {'max_depth' : [2, 3, 4],
            'min_samples_leaf' : [0.12, 0.14, 0.16, 0.18]}

#-----

# Search for the optimal tree

# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Instantiate grid_dt
grid_dt = GridSearchCV(estimator = dt, 
                        param_grid = params_dt,
                        scoring = 'roc_auc',
                        cv = 5,
                        n_jobs = - 1)
#------

# Evaluate the optimal tree

# Import roc_auc_score from sklearn.metrics
from sklearn.metrics import roc_auc_score

# Extract the best estimator
best_model = grid_dt.best_estimator_

# Predict the test set probabilities of the positive class
y_pred_proba = best_model.predict_proba(X_test)[:,1]

# Compute test_roc_auc
test_roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print test_roc_auc
print('Test set ROC AUC score: {:.3f}'.format(test_roc_auc))

#------




# --------Tuning a RF's Hyperparameters--------

# Random forests hyperparameters

# In the following exercises, you'll be revisiting the Bike Sharing Demand dataset that was introduced in a previous chapter. Recall that your task is to predict the bike rental demand using historical weather data from the Capital Bikeshare program in Washington, D.C.. For this purpose, you'll be tuning the hyperparameters of a Random Forests regressor.

# We have instantiated a RandomForestRegressor called rf using sklearn's default hyperparameters. You can inspect the hyperparameters of rf in your console.

# Which of the following is not a hyperparameter of rf?

# Answer : learning_rate

#-----

# Set the hyperparameter grid of RF

# Define the dictionary 'params_rf'
params_rf = {
                'n_estimators': [100, 350, 500],
                'min_samples_leaf': [2, 10, 30],
                'max_features': ['log2', 'auto', 'sqrt']
            }

#-----

# Search for the optimal forest

# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Instantiate grid_rf
grid_rf = GridSearchCV(estimator = rf,
                    param_grid = params_rf, 
                    cv = 3, 
                    scoring = 'neg_mean_squared_error', 
                    verbose = 1, 
                    n_jobs = -1)
#-----

# Evaluate the optimal forest

# Import mean_squared_error from sklearn.metrics as MSE 
from sklearn.metrics import mean_squared_error as MSE

# Extract the best estimator
best_model = grid_rf.best_estimator_

# Predict test set labels
y_pred = best_model.predict(X_test)

# Compute rmse_test
rmse_test = MSE(y_test, y_pred) ** (1/2)

# Print rmse_test
print('Test RMSE of best model: {:.3f}'.format(rmse_test)) 

#-------
