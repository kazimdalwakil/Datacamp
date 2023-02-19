
#---------------------------- Part 1: Data Distributions ------------------------------------------------
# >> What does your data look like? (I)

# Create a histogram
so_numeric_df.hist()
plt.show()

#--------------------
# Create a boxplot of two columns
so_numeric_df[['Age', 'Years Experience']].boxplot()
plt.show()
#--------------------
# Create a boxplot of ConvertedSalary

so_numeric_df[['ConvertedSalary']].boxplot()
plt.show()

# >> What does your data look like? (II)
# Import packages
import matplotlib.pyplot as plt
import seaborn as sns

# Plot pairwise relationships
sns.pairplot(so_numeric_df)

# Show plot
plt.show()

#--------------------------
# Print summary statistics
print(so_numeric_df.describe())
#-------------------------
# >> When don't you have to transform your data?
# While making sure that all of your data is on the same scale is advisable for most analyses, 
# for which of the following machine learning models is normalizing data not always necessary?
# Answer: Decision Trees
# As decision trees split along a singular point, they do not require all the columns to be on the same scale.


#---------------------------- Part 2: Scaling and Transfromations ------------------------------------------------
# >>  Normalization

# Import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
# Instantiate MinMaxScaler
MM_scaler = MinMaxScaler()

# Fit MM_scaler to the data
MM_scaler.fit(so_numeric_df[['Age']])

# Transform the data using the fitted scaler
so_numeric_df['Age_MM'] = MM_scaler.transform(so_numeric_df[['Age']])

# Compare the origional and transformed column
print(so_numeric_df[['Age_MM', 'Age']].head())


#----------------------
# >>  Standardization
# Import StandardScaler
from sklearn.preprocessing import StandardScaler
# Instantiate StandardScaler
SS_scaler = StandardScaler()

# Fit SS_scaler to the data
SS_scaler.fit(so_numeric_df[['Age']])

# Transform the data using the fitted scaler
so_numeric_df['Age_SS'] = SS_scaler.transform(so_numeric_df[['Age']])

# Compare the origional and transformed column
print(so_numeric_df[['Age_SS', 'Age']].head())


#----------------------
# >> Log transformation
# Import PowerTransformer
from sklearn.preprocessing import PowerTransformer

# Instantiate PowerTransformer
pow_trans = PowerTransformer()

# Train the transform on the data
pow_trans.fit(so_numeric_df[['ConvertedSalary']])

# Apply the power transform to the data
so_numeric_df['ConvertedSalary_LG'] = pow_trans.transform(so_numeric_df[['ConvertedSalary']])

# Plot the data before and after the transformation
so_numeric_df[['ConvertedSalary', 'ConvertedSalary_LG']].hist()
plt.show()

#----------------------

#When can you use normalization?

# When could you use normalization (MinMaxScaler) when working with a dataset?

# Answer: When you know the the data has a strict upper and lower bound.
# Normalization scales all points linearly between the upper and lower bound.

#---------------------------- Part 3: Removing Outliers ------------------------------------------------
# >> Percentage based outlier removal


# Find the 95th quantile
quantile = so_numeric_df['ConvertedSalary'].quantile(0.95)

# Trim the outliers
trimmed_df = so_numeric_df[so_numeric_df['ConvertedSalary'] < quantile]

# The original histogram
so_numeric_df[['ConvertedSalary']].hist()
plt.show()
plt.clf()

# The trimmed histogram
trimmed_df[['ConvertedSalary']].hist()
plt.show()

#----------------

# >> Statistical outlier removal


# Find the mean and standard dev
std = so_numeric_df['ConvertedSalary'].std()
mean = so_numeric_df['ConvertedSalary'].mean()

# Calculate the cutoff
cut_off = std * 3
lower, upper = mean - cut_off, mean + cut_off

# Trim the outliers
trimmed_df = so_numeric_df[(so_numeric_df['ConvertedSalary'] < upper) & (so_numeric_df['ConvertedSalary'] > lower)]

# The trimmed box plot
trimmed_df[['ConvertedSalary']].boxplot()
plt.show()

#---------------------------- Part 4: Scaling and Transforming new Data ------------------------------------------------
# >> Theory 

# Reuse training scalers
scaler = StandaedScaler()

scaler.fit(train[['col']])

train['scaled_col'] = scaler.tranform(train[['col']])

# Fit some model
# ......

test = pd.read_csv('test_csv')

test['scaled_col'] = scaler.transform(test[['col']])



# Training transformations for reuse
train_std = train['col'].std()
train_mean = train['col'].mean()

# Calculate the cutoff
cut_off = train_std * 3
train_lower, train_upper = train_mean - cut_off, train_mean + cut_off

# Subset train data
test = pd.read_csv('test_csv')

# Subset test data
test = test[(test['col'] < train_upper) & (test['col'] > train_lower)]

#---------------------------------------

# >> Train and testing transformations (I)
# Import StandardScaler
from sklearn.preprocessing import StandardScaler

# Apply a standard scaler to the data
SS_scaler = StandardScaler()

# Fit the standard scaler to the data
SS_scaler.fit(so_train_numeric[['Age']])

# Transform the test data using the fitted scaler
so_test_numeric['Age_ss'] = SS_scaler.transform(so_test_numeric[['Age']])
print(so_test_numeric[['Age', 'Age_ss']].head())

# >> Train and testing transformations (II)

train_std = so_train_numeric['ConvertedSalary'].std()
train_mean = so_train_numeric['ConvertedSalary'].mean()

cut_off = train_std * 3
train_lower, train_upper = train_mean - cut_off, train_mean + cut_off

# Trim the test DataFrame
trimmed_df = so_test_numeric[(so_test_numeric['ConvertedSalary'] < train_upper) \
                             & (so_test_numeric['ConvertedSalary'] > train_lower)]