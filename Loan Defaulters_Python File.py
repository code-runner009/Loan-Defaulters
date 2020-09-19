#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[2]:


data = pd.read_csv("E:\Imarticus\Group Project\XYZCorp_LendingData.txt", 
                   encoding = 'utf-8', sep = '\t', low_memory=False)

data.shape


# In[3]:


var_null_pc = data.isnull().sum(axis=0).sort_values( ascending=False)/float(len(data) )
var_null_pc[ var_null_pc > 0.75 ]


# In[4]:


data.drop( var_null_pc[ var_null_pc > 0.75 ].index, axis = 1, inplace = True ) 
data.dropna( axis = 0, thresh = 30, inplace = True )


# In[25]:


#One of the best approach is to save specific columns or values in one variable and we can then make operations on that variable
#Operations might be removing those columns, make them categorical etc etc..

vars_to_be_removed = ['policy_code', 'pymnt_plan', 'id', 'member_id', 'application_type', 
                      'acc_now_delinq','emp_title', 'zip_code','title']

data.drop( vars_to_be_removed , axis = 1, inplace = True )


# In[ ]:


data['term'] = data['term'].str.split(' ').str[1]

# extract numbers from emp_length and fill missing values with the median
data['emp_length'] = data['emp_length'].str.extract('(\d+)').astype(float)
data['emp_length'] = data['emp_length'].fillna(data.emp_length.median())

col_dates = data.dtypes[data.dtypes == 'datetime64[ns]'].index
for d in col_dates:
    data[d] = data[d].dt.to_period('M')


# In[28]:


#While building models on banking data, we MUST follow(research) the description of the data provided by the bank or client.
#Not only bank data, data with any domain, we must do some research like below, so that we can make special modifications with
#that data.

data['amt_difference'] = 'eq'
#created new column named as 'amt-difference and filled the default values as eq'

data.loc[ ( data['funded_amnt'] - data['funded_amnt_inv']) > 0, 'amt_difference' ] = 'less'
#data.loc = replace all values with "less", where difference between 'funded_amnt' and 'funded_amnt_inv' is greater than 0
#We are using two column values to put it in single column, so that we can easily categories them.

# Make categorical

data[ 'delinq_2yrs_cat' ] = 'no'
data.loc[ data [ 'delinq_2yrs' ] > 0, 'delinq_2yrs_cat' ] = 'yes'

data[ 'inq_last_6mths_cat' ] = 'no'
data.loc[ data['inq_last_6mths' ] > 0, 'inq_last_6mths_cat' ] = 'yes'

data[ 'pub_rec_cat' ] = 'no'
data.loc[ data['pub_rec'] > 0,'pub_rec_cat' ] = 'yes'

# Create new metric
data['acc_ratio'] = data.open_acc / data.total_acc


# In[29]:


features = [
            'loan_amnt', 'amt_difference', 'term', 
            'installment', 'grade','emp_length',
            'home_ownership', 'annual_inc','verification_status',
            'purpose', 'dti', 'delinq_2yrs_cat', 'inq_last_6mths_cat', 
            'open_acc', 'pub_rec', 'pub_rec_cat', 'acc_ratio', 'initial_list_status',
            'issue_d','default_ind'
           ]

data = data[features]

# Drop any residual missing values
data.dropna( axis=0, how = 'any', inplace = True )


# In[30]:


oot_test_months = ['Jun-2015', 'Jul-2015', 'Aug-2015', 'Sep-2015', 'Oct-2015', 'Nov-2015', 'Dec-2015']

train = data.loc [ -data.issue_d.isin(oot_test_months) ]
oot_test = data.loc [ data.issue_d.isin(oot_test_months) ]


# In[31]:


#Selecting Categorical columns so that we can easily use them afterwards.
categorical_features = ['term','amt_difference', 'grade', 'home_ownership', 'verification_status', 
                            'purpose', 'delinq_2yrs_cat', 'inq_last_6mths_cat', 'initial_list_status', 'pub_rec_cat']


# In[32]:


train[train.columns[:-2]].head()
#train.collumns[:] = Will give you only list of columns
#train.collumns[:-2] = Will give you only list of columns, excluuding last two  columns
#train[train.collumns[:]] or train[train.collumns[:]].head()  = Will give you every value from  the  column


# # Below code is for one Hot encoding, which is used to make the columns categorical

# In[33]:


#Using the function is straightforward - you specify which columns you want encoded and get a dataframe with original columns 
#replaced with one-hot encodings.
#syntax used :-  df_with_dummies = pd.get_dummies( df, columns = cols_to_transform )

X_model_train = pd.get_dummies(train[train.columns[:-2]], columns=categorical_features).astype(float)
y_model_train = train['default_ind']

X_oot_test = pd.get_dummies(oot_test[oot_test.columns[:-2]], columns=categorical_features).astype(float)
y_oot_test = oot_test['default_ind']

print(X_model_train.shape, X_oot_test.shape)


# In[34]:


X_model_train.head()


# In[35]:


X_oot_test.head()


# In[ ]:


vars_not_in_oot_test = ['home_ownership_OWN','home_ownership_RENT','purpose_educational']

X_model_train.drop( vars_not_in_oot_test , axis = 1, inplace = True )

print(X_model_train.shape)


# In[17]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.datasets import make_imbalance
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import make_pipeline
from imblearn.metrics import classification_report_imbalanced
from imblearn.over_sampling import SMOTE


# In[18]:


X_scaled_model_train =  preprocessing.scale(X_model_train)
X_scaled_oot_test = preprocessing.scale(X_oot_test)


# In[19]:


print(X_scaled_model_train.shape, X_scaled_oot_test.shape)


# In[ ]:


def run_models(X_train, y_train, X_test, y_test, model_type = 'Non-balanced'):
    
    clfs = {'GradientBoosting': GradientBoostingClassifier(max_depth= 6, n_estimators=100, max_features = 0.3),
            'LogisticRegression' : LogisticRegression(),
            'RandomForestClassifier': RandomForestClassifier(n_estimators=10)
            }
    cols = ['model', 'roc_auc_score', 'precision_score', 'recall_score','f1_score']

    models_report = pd.DataFrame(columns = cols)
    conf_matrix = dict()

    for clf, clf_name in zip(clfs.values(), clfs.keys()): #The zip() function returns an iterator of tuples based on the iterable object. If a single iterable is passed, zip() returns an iterator of 1-tuples. Meaning, the number of elements in each tuple is 1.

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_score = clf.predict_proba(X_test)[:,1]

        print('computing {} - {} '.format(clf_name, model_type))

        tmp = pd.Series({'model_type': model_type,
                         'model': clf_name,
                         'roc_auc_score' : metrics.roc_auc_score(y_test, y_score),
                         'precision_score': metrics.precision_score(y_test, y_pred),
                         'recall_score': metrics.recall_score(y_test, y_pred),
                         'f1_score': metrics.f1_score(y_test, y_pred)})

        models_report = models_report.append(tmp, ignore_index = True)
        conf_matrix[clf_name] = pd.crosstab(y_test, y_pred, rownames=['True'], colnames= ['Predicted'], margins=False)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score, drop_intermediate = False, pos_label = 1)

        plt.figure(1, figsize=(6,6))
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.title('ROC curve - {}'.format(model_type))
        plt.plot(fpr, tpr, label = clf_name )
        plt.legend(loc=2, prop={'size':11})
    plt.plot([0,1],[0,1], color = 'black')
    
    return models_report, conf_matrix

