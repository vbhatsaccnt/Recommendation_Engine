#!/usr/bin/env python
# coding: utf-8

# In[1]:


#IMPORT LIBRARY

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


data =pd.read_csv('https://raw.githubusercontent.com/codeheroku/Introduction-to-Machine-Learning/master/Building%20a%20Movie%20Recommendation%20Engine/movie_dataset.csv')


# In[3]:


data.shape


# In[4]:


data.info


# In[5]:


features =['keywords','cast','genres','director']


# In[6]:


def combine_features(row):
    return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']


# In[7]:


for feature in features:
    data[feature] = data[feature].fillna('')


# In[8]:


data['combined_features'] = data.apply(combine_features,axis =1)


# In[9]:


data['combined_features'].head(5)


# In[10]:


cv = CountVectorizer()


# In[11]:


count_matrix = cv.fit_transform(data['combined_features'])


# In[12]:


print(count_matrix)


# In[13]:


cosine_sim = cosine_similarity(count_matrix)


# In[14]:


print(cosine_sim)


# In[15]:


def get_title_from_index(index):
    
    return data[data.index == index]['title'].values[0]


# In[16]:


def get_index_from_title(title):
    
    return data[data.title == title]['index'].values[0]


# In[35]:


movie_user_likes = 'Shutter Island'

movie_index = get_index_from_title(movie_user_likes)

similar_movies = list(enumerate(cosine_sim[movie_index]))


# In[36]:


sorted_similar_movies = sorted(similar_movies, key = lambda x:x[1],reverse =True)[1:]


# In[37]:


i =0
print("Top 5 similar movies to "+movie_user_likes+" are:\n")

for element in sorted_similar_movies:
    print(get_title_from_index(element[0]))
    i=i+1
    if i>=5:
        break


# In[ ]:





# In[ ]:




