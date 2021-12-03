#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# #############################################
# # FEATURE ENGINEERING & DATA PRE-PROCESSING
# #############################################
# 
# # 1. OUTLIERS
# #    - Capture Outliers: boxplot, outlier_thresholds, check_outlier, grab_outliers
# #    - Solving Outlier Problem: Deletion, re-assignment with thresholds, Local Outlier Factor
# 
# # 2. MISSING VALUES
# #    - Capture Missing Values
# #    - Solving Missing Value Problem: Delete, fill with Lambda and Apply, fill in according to categorical variables
# #    - Advanced Analytics: Structure and Randomness Analysis, missing_vs_target
# 
# # 3. LABEL ENCODING
# 
# # 4. ONE-HOT ENCODING
# 
# # 5. RARE ENCODING
# 
# # 6. STANDARDIZATION: StandardScaler, RobustScaler, MinMaxScaler, Log, Numeric to Categorical
# 
# # 7. FEATURE EXTRACTION: NA_FLAG, BOOL, BINARY, LETTER COUNT, WORD COUNT, SPECIAL_CHAR
# 
# # 8. INTERACTIONS: Addition, multiplication, combination, weighting
# 
# # 9. END TO END APPLICATION

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
import os
from sklearn.metrics import accuracy_score
from sklearn.neighbors import LocalOutlierFactor

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# In[3]:


df = pd.read_csv('../input/d/atilaysamiloglu/titanic/titanic.csv')
df.head(5)


# # OUTLIERS

# In[4]:


# How to catch outliers ?

q1 = df['Age'].quantile(0.25)
q3 = df['Age'].quantile(0.75)
iqr = q3 - q1  # iqr = Interquartile range

up = q3 + 1.5 * iqr
low = q1 -1.5 * iqr

print(low, up)


# In[5]:


# Is there any outlier or not ?

df[(df["Age"] > up) | (df["Age"] < low)].any(axis=None)


# ## Let's Functionalize

# In[6]:


def outlier_thresholds(dataframe, col_name):
        quartile1 = dataframe[col_name].quantile(0.25)
        quartile3 = dataframe[col_name].quantile(0.75)
        interquantile_range = quartile3 - quartile1
        up_limit = quartile3 + 1.5 * interquantile_range
        low_limit = quartile1 - 1.5 * interquantile_range
        return low_limit, up_limit

outlier_thresholds(df, "Fare")

low, up = outlier_thresholds(df, "Fare")
print(low, up)


# In[7]:


# Outliers are only valied for numerical variables

num_cols = [col for col in df.columns if df[col].dtypes != 'O' and col != 'PassengerId']
print(num_cols)


# In[8]:


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for i in num_cols:
    print(i, check_outlier(df, i))


# ## Catching Outliers

# In[9]:


def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

grab_outliers(df, "Age")


# In[10]:


grab_outliers(df, "Age", True)


# In[11]:


# RECAP


# outlier_thresholds(df, "Age")
# check_outlier(df, "Age")
# grab_outliers(df, "Age", True)


# ## Solving Outliers Problem

# ## Deleting

# In[12]:


def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers


# In[13]:


for i in num_cols:
    df = remove_outlier(df, i)  
    
print(df)


# In[14]:


# Let's check df after removing outliers

df[(df["Age"] > up) | (df["Age"] < low)].any(axis=None)


# In[15]:


df = pd.read_csv('../input/d/atilaysamiloglu/titanic/titanic.csv')


# ## Re-assignment with Thresholds

# In[16]:


# In this function, instead of assign variable, 'loc' has been used. Changes become permanent on their own, no need to assign

def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if low_limit > 0:
        dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit
    else:
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit


# In[17]:


for col in num_cols:
    print(col, check_outlier(df, col))


# In[18]:


for col in num_cols:
    replace_with_thresholds(df, col)


# In[19]:


# Let's check df after re=assign outliers

for col in num_cols:
    print(col, check_outlier(df, col))


# In[20]:


# RECAP

# outlier_thresholds(df, "Age")
# check_outlier(df, "Age")
# grab_outliers(df, "Age", index=True)


# # Multivariate Outlier Analysis
# 
# ## Local Outlier Factor

# In[21]:


dff = pd.read_csv('../input/diamonds/diamonds.csv')
dff.head()


# In[22]:


# Removing numerical variables 

dff.drop(['cut','color', 'clarity'], axis=1, inplace = True)
dff.head()


# In[23]:


clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(dff)
dff_scores = clf.negative_outlier_factor_


# In[24]:


# To understand LOF scores and data, adding LOF scores as a variable

dff['LOF_scores'] = dff_scores
dff.sort_values('LOF_scores', ascending=False)


# In[25]:


# Finding threshold by using cores on the chart

scores = pd.DataFrame(np.sort(dff_scores))
scores.plot(stacked=True, xlim=[0, 20], style='.-')
plt.show()


# In[26]:


# As seen, there is an elbow at around 3 point. This is threshold. 

threshold_value = np.sort(dff_scores)[3]
print(threshold_value)


# In[27]:


# Outliers

dff[dff_scores < threshold_value]


# ## Deleting outliers

# In[28]:


dff[dff_scores < threshold_value].drop(axis=0, labels=dff[dff_scores < threshold_value].index)


# # MISSING VALUES

# In[29]:


df = pd.read_csv('../input/d/atilaysamiloglu/titanic/titanic.csv')

df.head()


# In[30]:


# Is there any missing value? 

df.isnull().any()


# In[31]:


# Sum of missing values?

df.isnull().sum()


# In[32]:


# Number and ration of missing values

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)

    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)
missing_values_table(df, True)


# ## Deleting

# In[33]:


df.dropna()


# ## Fiiling with lamda and apply

# In[34]:


#  Mean works for just numerical variables

dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)

dff.isnull().sum()


# In[35]:


# Median or mod can be used for categorical variables
# If number of value more than 10, that variable has high cardinality ('Cabin')

dff = dff.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

dff.isnull().sum()


# ## Assignment with Scikit - Learn 

# In[36]:


V1 = np.array([1, 3, 6, np.NaN, 7, 1, np.NaN, 9, 15])
V2 = np.array([7, np.NaN, 5, 8, 12, np.NaN, np.NaN, 2, 3])
V3 = np.array([np.NaN, 12, 5, 6, 14, 7, np.NaN, 2, 31])

df = pd.DataFrame(
    {"V1": V1,
     "V2": V2,
     "V3": V3}
)

print(df)


# In[37]:


from sklearn.impute import SimpleImputer

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

imp_mean.fit(df)
imp_mean.transform(df)


# ## Fill in missing values according to categorical variables

# In[38]:


V1 = np.array([1, 3, 6, np.NaN, 7, 1, np.NaN, 9, 15])
V2 = np.array([7, np.NaN, 5, 8, 12, np.NaN, np.NaN, 2, 3])
V3 = np.array([np.NaN, 12, 5, 6, 14, 7, np.NaN, 2, 31])
V4 = np.array(["IT", "IT", "IK", "IK", "IK", "IK", "IT", "IT", "IT"])

df = pd.DataFrame(
    {"salary": V1,
     "V2": V2,
     "V3": V3,
     "departman": V4}
)

print(df)


# In[39]:


V1 = np.array([1, 3, 6, np.NaN, 7, 1, np.NaN, 9, 15])


# In[40]:


# Assign values according to average in their departments, more sensitive assignment

df["salary"].fillna(df.groupby("departman")["salary"].transform("mean"))


# ## Advanced Analytics

# In[41]:


df = pd.read_csv('../input/d/atilaysamiloglu/titanic/titanic.csv')


# In[42]:


# These methods are used to define non-random missing values. 


# 

# In[43]:


msno.bar(df)
plt.show()


# In[44]:


msno.matrix(df)
plt.show()


# In[45]:


msno.heatmap(df)
plt.show()


# ## Relationship between Missing Values and Dependent Variable

# In[46]:


na_cols = missing_values_table(df, True)

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
            ## ONEMLI, EKSIK GORDUGUN YERE BIR DIGERINE SIFIR YAZ
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(df, "Survived", na_cols)


# # ENCODING
# 
# ## It makes categorical variables binary as 1 and 0.
# 

# ## LABEL ENCODING
# 
# ### It is just for binary variables.

# 

# In[47]:


df = pd.read_csv('../input/d/atilaysamiloglu/titanic/titanic.csv')
df.head()


# In[48]:


from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit_transform(df['Sex'])[0:5]


# In[49]:


# Way to see which one is zero

le.inverse_transform([0, 1])


# In[50]:


def label_encoder(dataframe, binary_col):
    labelencoder = preprocessing.LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


# In[51]:


binary_cols = [col for col in df.columns if df[col].dtypes == "O"
               and len(df[col].unique()) == 2]


# In[52]:


for i in binary_cols:
    label_encoder(df,i)


# In[53]:


# In our stituation just'Sex' variable is binary.

df.head()


# # One Hot Encoding
# 
# ## It is for variables that have more than 2 unique value.

# In[54]:


df = pd.read_csv('../input/d/atilaysamiloglu/titanic/titanic.csv')
df.head()


# In[55]:


pd.get_dummies(df, columns = ['Embarked']).head()


# In[56]:


# Missing values for Embarked also added as a variable

pd.get_dummies(df, columns=["Embarked"], dummy_na=True).head()


# In[57]:


# Dummy variable trap.

pd.get_dummies(df, columns=["Embarked"], drop_first=True).head()


# In[58]:


df = pd.read_csv('../input/d/atilaysamiloglu/titanic/titanic.csv')
df.head()


# In[59]:


# functionalize

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


ohe_cols = [col for col in df.columns if 10 >= len(df[col].unique()) > 2]


# In[60]:


one_hot_encoder(df, ohe_cols).head()


# In[61]:


one_hot_encoder(df, ohe_cols, drop_first=True).head()


# #  RARE ENCODING
# 
# ## There are 3 steps.
# 
# ## 1. Analysis of scarcity and abundance of categorical variables.
# ## 2. Analyzing the relationship between rare categories and dependent variable.
# ## 3. Writing code

# ## 1. Analysis of scarcity and abundance of categorical variables

# In[62]:


df = pd.read_csv('../input/application-train/application_train.csv')
df.head()


# In[63]:


df["NAME_EDUCATION_TYPE"].value_counts()


# In[64]:


cat_cols = [col for col in df.columns if df[col].dtypes == 'O']


# In[65]:


def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print('### ### ### ### ### ### ### ')
    ax = sns.countplot(x=dataframe[col_name], data=dataframe)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()
    plt.show()


# In[66]:


for i in cat_cols:
    cat_summary(df, i)


# ## 2. Analyzing the relationship between rare categories and dependent variable.

# In[67]:


df.groupby('NAME_INCOME_TYPE').agg({'TARGET' : 'mean'})


# In[68]:


df['NAME_INCOME_TYPE'].value_counts() / len(df)


# In[69]:


# Rare variables, threshold is 10%

[i for i in df.columns if df[i].dtypes == 'O' and (df[i].value_counts() / len(df) < 0.10).any(axis = None)]


# ## Writing rare encoding

# In[70]:


df = pd.read_csv('../input/application-train/application_train.csv')
df.head()


# In[71]:


def rare_analyser(dataframe, target, rare_perc):

    rare_columns = [col for col in dataframe.columns if dataframe[col].dtypes == 'O'
                    and (dataframe[col].value_counts() / len(dataframe) < rare_perc).any(axis=None)]

    for col in rare_columns:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


rare_analyser(df, "TARGET", 0.01)


# In[72]:


def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


new_df = rare_encoder(df, 0.01)


# In[73]:


# Let's check it, after encoding 

new_df[new_df["ORGANIZATION_TYPE"].str.contains("Rare")].head()


# In[74]:


rare_analyser(new_df, "TARGET", 0.01)


# # FEATURE SCALING
# 
# ## 1.STANDARTSCALER, 
# ## 2.ROBUSTSCALER, 
# ## 3.MINMAXSCALER, 
# ## 4.LOG SCALER, 
# ## 5.NUMERIC TO CATEGORICAL SCALER

# # 1. STANDARTSCALER
# 
# ### z = (x - u) / s
# 
# ### u: mean
# ### s: standard deviation

# In[75]:


df = pd.read_csv('../input/d/atilaysamiloglu/titanic/titanic.csv')
df.head()


# In[76]:


from sklearn.preprocessing import StandardScaler


# In[77]:


scaler = StandardScaler().fit(df[["Age"]])
df["Age"] = scaler.transform(df[["Age"]])
df["Age"].head(10)


# In[78]:


df["Age"].describe().T


# ## 2. ROBUSTSCALER
# 
# ### (x - median) /IQR

# In[79]:


df = pd.read_csv('../input/d/atilaysamiloglu/titanic/titanic.csv')
df.head()


# In[80]:


from sklearn.preprocessing import RobustScaler


# In[81]:


transformer = RobustScaler().fit(df[["Age"]])
df["Age"] = transformer.transform(df[["Age"]])
df["Age"].describe().T


# # 3. MINMAXSCALER
# 
# ### It assings value between given 2 numbers.
# ### Default value is between 0 and 1.

# In[82]:


df = pd.read_csv('../input/d/atilaysamiloglu/titanic/titanic.csv')
df.head()


# In[83]:


from sklearn.preprocessing import MinMaxScaler


# In[84]:


transformer = MinMaxScaler().fit(df[["Age"]])
df["Age"] = transformer.transform(df[["Age"]])

df["Age"].describe().T


# # 4. LOG SCALER
# 
# ### Due to log, it assigns smallest to largest

# In[85]:


df = pd.read_csv('../input/d/atilaysamiloglu/titanic/titanic.csv')
df.head()


# In[86]:


df["Age"] = np.log(df["Age"])
df["Age"].describe().T


# # 5.NUMERIC TO CATEGORICAL SCALER

# In[87]:


df = pd.read_csv('../input/d/atilaysamiloglu/titanic/titanic.csv')
df.head()


# In[88]:


df.loc[df['Age'] < 18, 'NEW_AGE_CAT'] = 'young'
df.loc[(df['Age'] >= 18) & (df['Age'] < 56), 'NEW_AGE_CAT'] = 'mature'
df.loc[(df['Age'] >= 56), 'NEW_AGE_CAT'] = 'senior'

df.head(15)


# ### NA_FLAG, BOOL, BINARY

# # 7.FEATURE EXTRACTION
# 
# ## NA_FLAG, BOOL, BINARY
# ## LETTER COUNT
# ## WORD COUNT
# ## CATCHING SPECIAL FEATUES
# ## CATCHING TITLES

# In[89]:


df = pd.read_csv('../input/d/atilaysamiloglu/titanic/titanic.csv')
df.head()


# In[90]:


# A value of 1 is assigned to those whose cabin information is not empty

df['NEW_CABIN_BOOL'] = df['Cabin'].notnull().astype('int')

df.head()


# In[91]:


# if sum of 'SibSp' and 'Parch' more than 0, add a new variable called 'NEW_IS_ALONE' and assign 'NO' for these passenger.

df.loc[((df['SibSp'] + df['Parch']) > 0), "NEW_IS_ALONE"] = "NO"
df.head(10)


# ### LETTER COUNT

# In[92]:


df["NEW_NAME_COUNT"] = df['Name'].str.len()

# OR

# df["NEW_NAME_COUNT"]  = df['Name'].apply(lambda x: len(x))


# In[93]:


df.head()


# ### WORD COUNT

# In[94]:


df["NEW_NAME_WORD_COUNT"] = df["Name"].apply(lambda x: len(str(x).split(" ")))
df.head()


# ## CATCHING SPECIAL FEATURES
# 

# In[95]:


# Finding passenger who has doctor title, and assignment as new variable called 'NEW_NAME_DR'

df["NEW_NAME_DR"] = df["Name"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))


# In[96]:


df["NEW_NAME_DR"]


# In[97]:


# There were 10 doctors and half of it survived.

df.groupby("NEW_NAME_DR").agg({"Survived": ["mean", "count"]})


# ### CATCHING TITLES

# In[98]:


df = pd.read_csv('../input/d/atilaysamiloglu/titanic/titanic.csv')
df.head()


# In[99]:


df['NEW_TITLE'] = [i.split(".")[0].split(",")[-1].strip() for i in df.Name]
df['NEW_TITLE'].head()


# In[100]:


df.head()


# ## 8. INTERACTIONS

# In[101]:


df.loc[(df['Sex'] == 'male') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
df.loc[(df['Sex'] == 'male') & ((df['Age'] > 21) & (df['Age']) < 50), 'NEW_SEX_CAT'] = 'maturemale'
df.loc[(df['Sex'] == 'male') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
df.loc[(df['Sex'] == 'female') & (df['Age'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
df.loc[(df['Sex'] == 'female') & ((df['Age'] > 21) & (df['Age']) < 50), 'NEW_SEX_CAT'] = 'maturefemale'
df.loc[(df['Sex'] == 'female') & (df['Age'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'


# In[102]:


df.head(10)

