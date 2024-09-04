#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import random


# # Data Loading
# 
# Loading `client_df` and `price_df` from inital .csv files into pandas dataframe.

# In[2]:


client_df = pd.read_csv('data/t2-client_data.csv')
price_df = pd.read_csv('data/t2-price_data.csv')


# # Data Preprocessing
# 
# ## Transform all date columns from object to datetime.

# In[5]:


# CLIENT dataset
column_dates = ['date_activ','date_end','date_modif_prod','date_renewal']
for item in column_dates:
      client_df[item] = pd.to_datetime(client_df[item])
# PRICE dataset
column_dates = ['price_date']
price_df[column_dates[0]] = pd.to_datetime(price_df[column_dates[0]])


# ## Adding new columns to CLIENT dataframe

# In[15]:


columns_to_set_zero = [
    'VARIABLE_price_off_peak',
    'VARIABLE_price_peak',
    'VARIABLE_price_mid_peak',
    'FIXED_price_off_peak',
    'FIXED_price_peak',
    'FIXED_price_mid_peak',
    'AVG_VARIABLE',
    'AVG_FIXED'
]
client_df[columns_to_set_zero] = 0.0


# In[16]:


client_df.info()


# ### Populate those columns with average and percentage of price in the last 6 months.

# In[17]:


import os
import multiprocessing
import threading

def get_cpu_info():
    num_cpu_cores = os.cpu_count()
    max_threads = multiprocessing.cpu_count()
    return num_cpu_cores, max_threads

num_cpu_cores, max_threads = get_cpu_info()


CountID = num_cpu_cores
thread_lists = [[] for _ in range(CountID)]
get_ids_list = client_df['id'].tolist()
num_ids_per_list = len(get_ids_list) // CountID
for i in range(CountID):
    start_idx = i * num_ids_per_list
    end_idx = (i + 1) * num_ids_per_list if i < 7 else None
    thread_lists[i] = get_ids_list[start_idx:end_idx]


# In[18]:


CLUSTER = range(max_threads)
for id in CLUSTER:
    print(thread_lists[id])


# In[19]:


# POPULATION functions

def extract_first_non_zero(series, start_idx, end_idx):
    """
    Extracts the first non-zero value in a given series within a specified index range.

    Args:
    series (pandas.Series): The series to search for non-zero values.
    start_idx (int): The starting index to search from (inclusive).
    end_idx (int): The ending index to search until (exclusive).

    Returns:
    float: The first non-zero value found in the specified range, or 0.0 if no non-zero values are found.
    """
    reversed_series = series.iloc[start_idx:end_idx][::-1]
    for price in reversed_series:
        if price != 0.0:
            return price
    return 0.0


def extract_diff_avg(series):
    """
    Calculate the percentage increase, difference, and average of two non-zero prices in a series.

    Args:
    series (pandas.Series): The series containing prices over time.

    Returns:
    list: A list containing the percentage increase, difference, and average of two non-zero prices.
          If no two non-zero prices are found, it returns [0, 0, 0].
    """
    vp_6_0 = extract_first_non_zero(series, 0, 6)
    vp_12_6 = extract_first_non_zero(series, 6, 12)

    if vp_6_0 == 0 or vp_12_6 == 0:
        return [0, 0]
    else:
        difference = vp_12_6 - vp_6_0
        avg = (vp_12_6 + vp_6_0) / 2
        return [difference, avg]


def update_client_df(e_uid, name, value_list):
    """
    Update a specific column and its corresponding percentage column in the client DataFrame.

    Args:
    e_uid (str): The unique identifier for the client.
    name (str): The name of the column to update.
    value_list (list): A list containing two values: the percentage value and the new value for the column.

    Returns:
    None
    """
    client_df.loc[client_df['id'] == e_uid, name] = value_list[0]


def calculate_weighted_average(e_uid, col, num1, num2, num3):
    """
    Calculate the weighted average of three numbers and update a specific column in the client DataFrame.

    Args:
    e_uid (str): The unique identifier for the client.
    col (str): The name of the column to update.
    num1 (float): The first number.
    num2 (float): The second number.
    num3 (float): The third number.

    Returns:
    None
    """
    average = 0
    total = 0
    divider = 0

    if num1:
        total += num1
        divider += 1
    if num2:
        total += num2
        divider += 1
    if num3:
        total += num3
        divider += 1
    
    if divider:
        average = total / divider
    
    client_df.loc[client_df['id'] == e_uid, col] = average


def function_clu(ext_id):
    print(ext_id)

    grouped_price_df = price_df.groupby('id')
    group = grouped_price_df.get_group(ext_id)

    v_p_o_p = extract_diff_avg(group['price_off_peak_var'])
    v_p_p = extract_diff_avg(group['price_peak_var'])
    v_p_m_p = extract_diff_avg(group['price_mid_peak_var'])

    f_p_o_p = extract_diff_avg(group['price_off_peak_fix'])
    f_p_p = extract_diff_avg(group['price_peak_fix'])
    f_p_m_p = extract_diff_avg(group['price_mid_peak_fix'])

    update_client_df(ext_id, 'VARIABLE_price_off_peak', v_p_o_p)
    update_client_df(ext_id, 'VARIABLE_price_peak', v_p_p)
    update_client_df(ext_id, 'VARIABLE_price_mid_peak', v_p_m_p)

    update_client_df(ext_id, 'FIXED_price_off_peak', f_p_o_p)
    update_client_df(ext_id, 'FIXED_price_peak', f_p_p)
    update_client_df(ext_id, 'FIXED_price_mid_peak', f_p_m_p)

    calculate_weighted_average(ext_id, 'AVG_VARIABLE',v_p_o_p[1],v_p_p[1],v_p_m_p[1])
    calculate_weighted_average(ext_id, 'AVG_FIXED',f_p_o_p[1],f_p_p[1],f_p_m_p[1])


# In[20]:


def process_list(ident, id_list):
    for uuid in id_list:
       function_clu(uuid)


# Create a list to store the thread objects
threads = []


for i, id_list in enumerate(thread_lists):
    # Create a thread, set the target to 'process_list' function,
    # and pass 'i' as the index and 'id_list' as the argument
    thread = threading.Thread(target=process_list, args=(i, id_list))
    # Append the thread to the 'threads' list
    threads.append(thread)
    # Start the thread to begin execution of the 'process_list' function
    thread.start()
    

# Wait for all threads to finish
for thread in threads:
    thread.join()


# In[21]:


client_df.to_csv('data/t4-sit.csv', sep=",", index=False) 


# ## Reading Dataset

# ### Mantain only necessary columns

# In[31]:


tdf = pd.read_csv('data/t4-sit.csv')

columns_to_keep = ['id','channel_sales', 'cons_12m', 'cons_gas_12m', 'cons_last_month', 'forecast_discount_energy', 'margin_gross_pow_ele', 'margin_net_pow_ele', 'nb_prod_act', 'num_years_antig', 'origin_up', 'pow_max', 'AVG_VARIABLE', 'AVG_FIXED', 'churn']
tdf = tdf[columns_to_keep]
#tdf.info()


# ### Hot-Encoding classes with low volume of UniqueIDs

# In[32]:


# One-hot encode the 'origin_up' column
tdf = pd.get_dummies(tdf, columns=['origin_up'], prefix='origin_up')
# One-hot encode the 'channel_sales' column
tdf = pd.get_dummies(tdf, columns=['channel_sales'], prefix='channel_sales')
#tdf.info()


# ### Converting Bools to Int

# In[33]:


# List of boolean columns
bool_columns = tdf.select_dtypes(include='bool').columns
# Convert boolean columns to integers
tdf[bool_columns] = tdf[bool_columns].astype(int)
#tdf.info()


# In[34]:


tdf.info()


# In[30]:


# Group off-peak prices by companies and month
monthly_price_by_id = price_df.groupby(['id', 'price_date']).agg({'price_off_peak_var': 'mean', 'price_off_peak_fix': 'mean'}).reset_index()

# Get january and december prices
jan_prices = monthly_price_by_id.groupby('id').first().reset_index()
dec_prices = monthly_price_by_id.groupby('id').last().reset_index()

# Calculate the difference
diff = pd.merge(dec_prices.rename(columns={'price_off_peak_var': 'dec_1', 'price_off_peak_fix': 'dec_2'}), jan_prices.drop(columns='price_date'), on='id')
diff['offpeak_diff_dec_january_energy'] = diff['dec_1'] - diff['price_off_peak_var']
diff['offpeak_diff_dec_january_power'] = diff['dec_2'] - diff['price_off_peak_fix']
diff = diff[['id', 'offpeak_diff_dec_january_energy','offpeak_diff_dec_january_power']]
diff.head()


# In[36]:


merged_df = pd.merge(tdf, diff, on='id')


# In[37]:


# Get the 'churn' column
churn_column = merged_df.pop('churn')
# Add the 'churn' column back as the last column
merged_df['churn'] = churn_column

merged_df.info()


# In[38]:


merged_df.to_csv('data/t4-premodel.csv', sep=",", index=False) 


# ---
# 
# ## ML model

# In[46]:


tdf = pd.read_csv('data/t4-premodel.csv')
tdf = tdf.dropna(subset=['churn'])


# ## Defining feature columns (X) and target column (y)

# In[47]:


from sklearn.model_selection import train_test_split

# Features: Exclude 'churn' (label) and 'id' (composed of UUIDs) from the feature set.
X = tdf.drop(columns=['churn', 'id'])

# Target label: The column to predict is 'churn'.
y = tdf['churn']

# Split the data into a training set (80%) and a test set (20%).
# I used 42 as the random state for reproducibility and consistency.
# The choice of 42 as the seed is arbitrary; any number can be used. 🌌
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## Train a Random Forest classifier

# In[48]:


from sklearn.ensemble import RandomForestClassifier

# Binary Classifier
# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)


# In[49]:


y_pred = rf_classifier.predict(X_test)


# ## Model Evaluation

# In[50]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("Confusion Matrix:\n", conf_matrix)


# #### Explaination of results
# 
# 
# 1. **Accuracy (0.8983)**: This is the overall proportion of correct predictions. 
# 
# The model is correct approximately 89.83% of the time, which indicates good overall performance.
# 
# 2. **Precision (0.6897)**: Precision measures the model's ability to correctly predict positive cases. 
# 
# In this case, the model correctly predicts around 68.97% of clients as churned out of the total predicted as churned. 
# 
# It's a metric to assess how reliable the positive predictions are.
# 
# 3. **Recall (0.0649)**: Recall, also known as sensitivity or true positive rate, measures the ability to find all positive cases. 
# 
# The model only identifies about 6.49% of the actual churn cases, which means it misses many clients who actually churned.
# 
# 4. **F1-Score (0.1187)**: The F1-Score is the harmonic mean of precision and recall. 
# 
# It balances the trade-off between precision and recall. In this case, it's relatively low at 11.87%, indicating an imbalance between precision and recall.
# 
# 5. **Confusion Matrix**: This matrix provides a more detailed breakdown of the model's predictions. It shows the following:
#    - **True Negatives (TN) [2604]**: Clients correctly predicted as not churned.
#    - **False Positives (FP) [9]**: Clients incorrectly predicted as churned (Type I error).
#    - **False Negatives (FN) [288]**: Clients incorrectly predicted as not churned (Type II error).
#    - **True Positives (TP) [20]**: Clients correctly predicted as churned.
# 
# Confusion Matrix structure:
# 
# [[ TN FP ]
# 
#  [ FN TP ]]
# 
# ---
# 
# #### Summary
# 
# In summary, the model has good overall accuracy but suffers from low recall, meaning it doesn't identify many of the actual churn cases. 
# 
# This indicates a potential need for improving the model's ability to detect clients who are likely to churn.
# 
# ---
# 
# #### Where the model underperforms:
# 
# 1. The recall is quite low, indicating that the model is not effectively identifying actual churn cases. 
# 
# It misses a significant number of clients who are actually churning (high FN rate).
# 
# 2. The precision is moderate, but it could be improved. 
# 
# The model is making some false positive predictions, classifying some non-churning clients as churning (FP rate).
# 
# ---
# 
# ####  How to improve results:
# 
# 1. Addressing class imbalance: There is a significant class imbalance (many more non-churning clients than churning ones),
# 
# I might consider techniques like oversampling, undersampling, or using different class weights to balance the dataset.
# 
# 2. Feature engineering: Carefully selecting and engineering new relevant features that can improve model performance.
# 
# 3. Hyperparameter tuning: Experiment with different hyperparameters of the Random Forest model, such as the number of trees, tree depth, and feature selection methods, to find the best combination for my dataset.
# 
# 4. Try different algorithms: Consider trying other classification algorithms, as Random Forest may not be the best fit for all datasets. 
# 
# Algorithms like Gradient Boosting or Support Vector Machines might perform better.

# ---
# 
# ## Fine-Tuning ML Model

# In[51]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Define a grid of hyperparameters to search
param_grid = {
    'n_estimators': [100, 200, 300],   # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],   # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],   # Minimum samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]      # Minimum samples required for a leaf node
}

# Create a GridSearchCV object with the classifier and parameter grid
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit the GridSearchCV object to my data
grid_search.fit(X_train, y_train)

# Get the best parameters and estimator from the search
best_params = grid_search.best_params_
best_rf_classifier = grid_search.best_estimator_

# Use the best estimator for prediction
y_pred = best_rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Best Parameters:", best_params)
print("Test Accuracy:", accuracy)


# #### Explaining finetuning results:
# 
# In this code, I have performed hyperparameter tuning for a Random Forest classifier using GridSearchCV. 
# 
# The results include the best hyperparameters found for the model: a maximum tree depth of 30, minimum samples per leaf set to 1, minimum samples to split an internal node set to 2, and 200 trees in the forest. 
# 
# The test accuracy, which measures the model's correctness in predicting the test data, is approximately 0.899 (or 89.9%).
# 
# These best hyperparameters can be used to configure the Random Forest classifier for improved performance on my dataset, resulting in more accurate predictions.

# ### Testing Tuned Model

# In[52]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Define a Random Forest classifier with the best parameters
best_rf_classifier = RandomForestClassifier(
    n_estimators=200,
    max_depth=30,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

# Fit the classifier to my training data
best_rf_classifier.fit(X_train, y_train)

# Use the classifier for prediction
y_pred = best_rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("Confusion Matrix:\n", conf_matrix)


# ---
# 
# ### Final results:
# 
# In the initial model:
# 
# - Accuracy is 0.8983, which means 89.83% of the predictions were correct.
# - Precision is 0.6897, indicating that out of all positive predictions, 68.97% were true positives.
# - Recall is 0.0649, suggesting that the model correctly identified only 6.49% of actual positive cases.
# - F1-Score is 0.1187, which balances precision and recall into a single metric.
# 
# In the tuned model:
# 
# - Accuracy improved slightly to 0.8990, indicating a 0.10% increase in correct predictions.
# - Precision increased significantly to 0.7826, which means the model improved in correctly identifying true positives among positive predictions.
# - Recall slightly decreased to 0.0584, indicating that the model still struggles to capture actual positive cases.
# - F1-Score is 0.1088, showing that the balance between precision and recall remains similar.
# 
# ---
# 
# In both models, the recall is quite low, which means they have difficulty identifying actual positive cases.
# 
# The precision improved in the tuned model, indicating a better positive prediction accuracy. 
# 
# However, the overall performance still leaves room for improvement.
# 
# To further improve the model, I may consider exploring different classification algorithms, feature engineering, collecting more data, or addressing class imbalance.
