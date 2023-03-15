# Chapter: 2 -> The Bias-Variance Tradeoff


# --------Generalization Error--------

#Complexity, bias and variance

# In the video, you saw how the complexity of a model labeled f
# influences the bias and variance terms of its generalization error.
# Which of the following correctly describes the relationship between
# f's complexity and
# f's bias and variance terms?

# Answer: Option 4 (As the complexity of f increases, the bias term decreases while the variance term increases.) 

#-----

#Overfitting and underfitting

# In this exercise, you'll visually diagnose whether a model is overfitting or underfitting the training set.

# For this purpose, we have trained two different models A and B
# on the auto dataset to predict the mpg consumption of a car using only the car's displacement (displ) as a feature.

# The following figure shows you scatterplots of mpg versus displ along with lines corresponding to the training set predictions of models A and B in red.

#Answer : (option 3) B suffers from high bias and underfits the training set.

#-----

# --------Diagnose bias and variance problems--------

# Instantiate the model

from sklearn.tree import DecisionTreeRegressor
# Import train_test_split from sklearn.model_selection
from sklearn.model_selection import train_test_split
# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_val_score

##                  Instantiate The Model                  ##

# Set SEED for reproducibility
SEED = 1

# Split the data into 70% train and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=SEED)

# Instantiate a DecisionTreeRegressor dt
dt = DecisionTreeRegressor(max_depth = 4, min_samples_leaf = 0.26, random_state=SEED)

#-----
# Evaluate the 10-fold CV error
# Compute the array containing the 10-folds CV MSEs
# Set n_jobs to -1 to exploit all available CPUs in computation. 
MSE_CV_scores = - cross_val_score(dt, X_train, y_train, cv = 10, 
                       scoring = 'neg_mean_squared_error',
                       n_jobs = -1)

# Compute the 10-folds CV RMSE
RMSE_CV = (MSE_CV_scores.mean())**(1/2)

# Print RMSE_CV
print('CV RMSE: {:.2f}'.format(RMSE_CV))

#-----

# Evaluate the training error

from sklearn.tree import DecisionTreeRegressor
# Import train_test_split from sklearn.model_selection
from sklearn.model_selection import train_test_split
# Import mean_squared_error from sklearn.metrics as MSE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_val_score
# Fit dt to the training set
dt.fit(X_train, y_train)

# Predict the labels of the training set
y_pred_train = dt.predict(X_train)

# Evaluate the training set RMSE of dt
RMSE_train = (MSE(y_train, y_pred_train))**(1/2)

# Print RMSE_train
print('Train RMSE: {:.2f}'.format(RMSE_train))

# -----

# High bias or high variance?

# In this exercise you'll diagnose whether the regression tree dt you trained in the previous exercise suffers from a bias or a variance problem.

# The training set RMSE (RMSE_train) and the CV RMSE (RMSE_CV) achieved by dt are available in your workspace. In addition, we have also loaded a variable called baseline_RMSE which corresponds to the root mean-squared error achieved by the regression-tree trained with the disp feature only (it is the RMSE achieved by the regression tree trained in chapter 1, lesson 3). Here baseline_RMSE serves as the baseline RMSE above which a model is considered to be underfitting and below which the model is considered 'good enough'.

# Does dt suffer from a high bias or a high variance problem?


# Answer: (Option 2) dt suffers from high bias because RMSE_CV RMSE_train and both scores are greater than baseline_RMSE.
# -----

#--------Ensemble Learning--------

# Define the ensemble

# Set seed for reproducibility
SEED=1

# Instantiate lr
lr = LogisticRegression(random_state = SEED)

# Instantiate knn
knn = KNN(n_neighbors = 27)

# Instantiate dt
dt = DecisionTreeClassifier(min_samples_leaf = 0.13, random_state = SEED)

# Define the list classifiers
classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn), ('Classification Tree', dt)]

#-----

# Evaluate individual classifiers

for clf_name, clf in classifiers:    
 
    # Fit clf to the training set
    clf.fit(X_train, y_train)    
   
    # Predict y_pred
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred) 
   
    # Evaluate clf's accuracy on the test set
    print('{:s} : {:.3f}'.format(clf_name, accuracy))

#-----

# Better performance with a Voting Classifier

# Import VotingClassifier from sklearn.ensemble
from sklearn.ensemble import VotingClassifier

# Instantiate a VotingClassifier vc
vc = VotingClassifier(estimators = classifiers)     

# Fit vc to the training set
vc.fit(X_train, y_train)   

# Evaluate the test set predictions
y_pred = vc.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print('Voting Classifier: {:.3f}'.format(accuracy))

#-----



