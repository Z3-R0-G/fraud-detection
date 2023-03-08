#!/usr/bin/env python
# coding: utf-8

# In[1]:


### !pip install pyarrow
get_ipython().system('pip uninstall -y numpy')
get_ipython().system('pip uninstall -y setuptools')
get_ipython().system('pip install setuptools==39.1.0')
get_ipython().system('pip install numpy')
get_ipython().system('pip install tensorflow')


# # K-Means Clustering + Operation-Level Flagging
# 
# ### To Do Later: DBScan Clustering, Variational Inference
# 
# In order for there to be operation-level flagging, the user must be considered non-anomalous
# 
# In other words, the next/latest operation must be anomalous enough for it to cause the hidden state to diverge
# 
# Steps:
# 1. Time-LSTM Autoencoder yields UOBS embeddings (encoder hidden states)
# 2. UOBS embeddings are clustered to form user segmentations (or variational inference later...)
# 3. Get hidden vector
# 4. For each operation, calculate the bounds of time elapsed in which the operation keeps the user "safe"
# 
# Such bounds are the "flagging rules". Alternatively, we may perform this on the cluster center to approximate a "user profile" based on fully-safe operations and safe bounds per non-fully-safe operations for some sort of user profiling that we can understand/imagine.

# In[2]:


import pickle
import numpy as np
import sys
import pandas as pd
np.set_printoptions(threshold=sys.maxsize)


# In[3]:


with open("uobs", "rb") as fp:
    uobs = pickle.load(fp)


# In[4]:


print(len(uobs))


# In[5]:


print(uobs[0])


# # Unsupervised Clustering

# As such, "leaving" the cluster would be defined as leaving the core point

# In[10]:


# find optimal kmeans cluster (ensemble, repeated trainings)
# we want to maximize the Dunn index = min(intra cluster distance)/max (inter cluster distance)
#https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/ for KMeans ++

from sklearn.cluster import DBSCAN

clustering = DBSCAN(eps=5, min_samples=5).fit(uobs)

# the following is calculating number per labels

df = pd.DataFrame(clustering.labels_, columns = ['Labels'])

df.value_counts()

# this is to be fine tuned later


# In[11]:


# next, we want to find the nearby 'core' points of a point, and find all operations that will lead it to leave the radius


# In[12]:


clustering.components_.shape


# In[152]:


from sklearn.neighbors import NearestNeighbors

neigh = NearestNeighbors(n_neighbors=5, radius=5)
neigh.fit(uobs)


# In[155]:


neigh.radius_neighbors(uobs, return_distance=False)


# In[209]:


test = neigh.radius_neighbors(uobs[4].reshape(1,-1), 0.4, return_distance=False)[0]


# In[211]:


clustering.labels_


# In[163]:


uobs[0].reshape(1,-11)


# In[13]:


import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(uobs)

eigvec = pca.transform(uobs)


# In[14]:


sns.distplot(a=np.dot(uobs, pca.components_[0])/np.linalg.norm(pca.components_[0])**2, hist=False)
sns.distplot(a=np.dot(uobs, pca.components_[1])/np.linalg.norm(pca.components_[1])**2, hist=False)


# In[ ]:




