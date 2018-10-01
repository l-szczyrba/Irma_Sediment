
# coding: utf-8

# In[177]:


import os
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import sklearn as sk
import os


# In[178]:


path = '/home/catherinej/Downloads'
file = os.path.join(path, 'IrmaMudThicknessComparisons.xlsx')
mud = pd.read_excel(file)
mud
# mud.head()
# mud.info()
# mud.describe()


# In[179]:


mud.hist(bins=50, figsize=(5,5))
plt.show()


# In[180]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit()
for train_index, test_index in split.split(mud, mud['Site']):
    mud_train_set = mud.loc[train_index]
    mud_test_set = mud.loc[test_index]


# In[181]:


mud['Site'].value_counts()


# In[182]:


# mud.plot(kind='scatter', x='Lon', y="Lat", alpha=0.4,
#         cmap=plt.get_cmap('jet'), colorbar=True)
# plt.legend()


# In[183]:


corr_matrix = mud.corr()
for key in corr_matrix:
    print(key)
corr_matrix['Thickness at Date 2 (cm)'].sort_values(ascending=False)


# In[184]:


attributes = ['Lat', 'Lon', 'Thickness at Date 1 (cm)', 'Thickness at Date 2 (cm)',
             ]

pd.tools.plotting.scatter_matrix(mud[attributes], figsize=(6,4))


# In[185]:


data = mud.drop('Notes', axis=1)
data.dropna(subset=['Thickness at Date 1 (cm)'])
data.dropna(subset=['Thickness at Date 2 (cm)'])
data


# In[186]:


from sklearn.preprocessing import Imputer
imputer = Imputer(strategy='median')
mud_num = data.drop('Region', axis=1)
mud_num = mud_num.drop('Site', axis=1)
mud_num = mud_num.drop('Station', axis=1)
mud_num = mud_num.dropna(subset=['Date 2'])
mud_num = mud_num.dropna(subset=['Thickness at Date 2 (cm)'])


# In[187]:



mud_num.plot(kind='scatter',x='Lon', y='Lat', alpha=0.1)


# In[188]:


# for t in mud_num['Thickness at Date 1 (cm)']:
mud_num = mud_num.replace('<0.1', 0)
mud_num = mud_num.replace('<1', 0.5)


# In[189]:


mud_num


# In[190]:


mud_num
# mud_num = mud_num.drop('Date 1',axis=1)


# In[191]:


corr_matrix = mud_num.corr()
corr_matrix['Thickness at Date 1 (cm)'].sort_values(ascending=False)


# In[192]:


from pandas.tools.plotting import scatter_matrix
attributes = ['Lat', 'Lon', 'Thickness at Date 1 (cm)']
scatter_matrix(mud_num[attributes], figsize=(12, 9))
mud_num.plot(kind='scatter', x='Thickness at Date 1 (cm)', y='Thickness at Date 2 (cm)', alpha=0.4)


# In[194]:


imputer.statistics_
mud_num.median().values
x = imputer.transform(mud_num)


# In[196]:


mud_num_tr = pd.DataFrame(x, columns=mud_num.columns)


# In[80]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
mud_cat = mud['Date 1']
mud_cat_encoded = encoder.fit_transform(mud_cat)
mud_cat_encoded
print(encoder.classes_)


# In[82]:


mud_cat = mud_train_set['Region']


# In[89]:


def encode_text(data_frame):
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    cat = data_frame['Site'] #, 'Region']
    cat_encoded = encoder.fit_transform(cat)
    
    return cat_encoded


# In[200]:


mud_cat = encode_text(mud_train_set)
cat = ['Thickness at Date 1 (cm)', 'Thickness at Date 2 (cm)', 'Date 1', 'Date 2']
df1 = mud_test_set.dropna(subset=cat)
df1 = mud_num.drop('Station', axis=1)


# In[201]:


def clean_dataframe(dataframe1, dataframe2):
    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer()
    cat = ['Thickness at Date 1 (cm)', 'Thickness at Date 2 (cm)', 'Date 1', 'Date 2']
    df1 = dataframe1.dropna(subset=cat)
    df2 = dataframe2.dropna(subset=cat)
    df1 = df1.replace('<0.1', 0)
    df1 = df1.replace('<1', .5)
    df2 = df2.replace('<0.1', 0)
    df2 = df2.replace('<1', .5)
    
    drop_columns = ['Date 1', 'Date 2', 'Notes']
    df1 = df1.drop(drop_columns, axis=1) 
    df2 = df2.drop(drop_columns, axis=1)
    
    return df1, df2


def categorize_text(df1, df2):
    categories = df1.select_dtypes(include=[object]).columns
    df1_cat = pd.get_dummies(df1, columns=categories, drop_first=True)
    df2_cat = pd.get_dummies(df2, columns=categories, drop_first=True)
    return df1_cat, df2_cat
#     print(categories)
#     from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
    
#     enc = MultiLabelBinarizer()
#     cat_features = ['Region', 'Site', 'Station']
#     df1_cat = df1[cat_features]
#     df2_cat = df2[cat_features]
#     df1_enc = enc.fit_transform(df1_cat)
#     df2_enc = enc.fit_transform(df2_cat)
    
#     ohe = OneHotEncoder()
#     df1_hot = ohe.fit_transform(df1_enc)
#     df2_hot = ohe.fit_transform(df2_enc)
#     return df1_hot, df2_hot

# train_set = pd.concat([mud_train_set, train_code], axis=1)
train, test = clean_dataframe(mud_train_set, mud_test_set)
print(train)
train_code, test_code = categorize_text(train, test)
print(train_code)


# In[170]:


mud_median = pd.DataFrame.median(train_code['Thickness at Date 1 (cm)'])


# In[171]:


mud_median


# In[198]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
help(lin_reg.fit)
# lin_reg.fit(train_code, test_code)

