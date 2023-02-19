#---------------------------- Part 1: Why do Missing Values Exist ------------------------------------------------
# >> How sparse is my data?

# Subset the DataFrame
sub_df = so_survey_df[['Age', 'Gender']]

# Print the number of non-missing values
print(sub_df.info())

# Answer: 693
#------------------------------------------
# >> Finding the missing values

# Print the top 10 entries of the DataFrame
print(sub_df.head(10))

# Print the locations of the non-missing values
print(sub_df.head(10).isnull())

# Print the locations of the non-missing values
print(sub_df.head(10).notnull())

#---------------------------- Part 2: Dealing with missing values (I) ------------------------------------------------
# >> Listwise deletion
# Print the number of rows and columns
print(so_survey_df.shape)

# Create a new DataFrame dropping all incomplete rows
no_missing_values_rows = so_survey_df.dropna(how='any')

# Print the shape of the new DataFrame
print(no_missing_values_rows.shape)

# Create a new DataFrame dropping all columns with incomplete rows
no_missing_values_cols =so_survey_df.dropna(how='any', axis=1)

# Print the shape of the new DataFrame
print(no_missing_values_cols.shape)

# Drop all rows where Gender is missing
no_gender = so_survey_df.dropna(subset=['Gender'])

# Print the shape of the new DataFrame
print(no_gender.shape)

# >> Replacing missing values with constants

# Print the count of occurrences
print(so_survey_df['Gender'].value_counts())

# Replace missing values
so_survey_df['Gender'].fillna(value='Not Given', inplace = True)

# Print the count of each value
print(so_survey_df['Gender'].value_counts())


#---------------------------- Part 3: Dealing with missing values (II) ------------------------------------------------
# >> Filling continuous missing values

# Print the first five rows of StackOverflowJobsRecommend column
print(so_survey_df.StackOverflowJobsRecommend.head(5))

# Fill missing values with the mean
so_survey_df['StackOverflowJobsRecommend'].fillna(so_survey_df['StackOverflowJobsRecommend'].mean(), inplace=True)

# Round the StackOverflowJobsRecommend values
so_survey_df['StackOverflowJobsRecommend'] = round(so_survey_df['StackOverflowJobsRecommend'])

# Print the top 5 rows
print(so_survey_df['StackOverflowJobsRecommend'].head())
#---------------------------------------------------------------

# Imputing values in predictive models

# Ques:
# When working with predictive models you will often have a separate train and test DataFrames.
# In these cases you want to ensure no information from your test set leaks into your train set.
# When filling missing values in data to be used in these situations 
# how should approach the two datasets?

# Answer: Apply the measures of central tendency (mean/median etc.) calculated on the train set to both the train and test sets.

#---------------------------- Part 4: Dealing with other data issues ------------------------------------------------
# >> Dealing with stray characters (I)
# Remove the commas in the column
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].str.replace(',', '')

# Remove the Dollar Sign in the column
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].str.replace('$', '')

#-------------------------------------------

# >> Dealing with stray characters (II)

# Attempt to convert the column to numeric values
numeric_vals = pd.to_numeric(so_survey_df['RawSalary'], errors='coerce')

# Find the indexes of missing values
idx = numeric_vals.isna()

# Print the relevant rows
print(so_survey_df['RawSalary'][idx])

#----------------------
# Replace the offending characters
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].str.replace('£', '')

# Convert the column to float
so_survey_df['RawSalary'] = so_survey_df['RawSalary'].astype('float')

# Print the column
print(so_survey_df['RawSalary'])


# >> Method chaining
# !!! Theory 
# Method chaining
df['column'] = df['column'].method1().method2().method3()

# Same as 
df['column'] = df['column'].method1()
df['column'] = df['column'].method2()
df['column'] = df['column'].method3()

# Use method chaining
so_survey_df['RawSalary'] = so_survey_df['RawSalary']\
                              .str.replace(',', '')\
                              .str.replace('$', '')\
                              .str.replace('£', '')\
                              .astype('float')
 
# Print the RawSalary column
print(so_survey_df['RawSalary'])

#----------------------------------------------------------------------





