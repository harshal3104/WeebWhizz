#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies=pd.read_csv('animes.csv')
# reviews=pd.read_csv('reviews.csv')
profiles=pd.read_csv('profiles.csv')


# In[3]:


movies=pd.read_csv('animes.csv')


# In[4]:


#a_movies = movies.merge(reviews,on='anime_uid')


# In[5]:


# img VARIBALE FOR URLS AND ANIMA POSTER
img=movies[['img_url','link']] # ----------


# In[6]:


#movies = pd.merge(movies, img, on='anime_uid', how='left') #--------------------


# In[7]:


movies=movies[['anime_uid','title','synopsis','genre','aired','episodes']]


# In[8]:


def remove_sentence(paragraph):
    sentence_to_remove=" \r\n \r\n[Written by MAL Rewrite]"
    if isinstance(paragraph, str):
        return paragraph.replace(sentence_to_remove, '')
    else:
        return paragraph

movies['synopsis']=movies['synopsis'].apply(remove_sentence)


# In[9]:


def remove_sentence_2(paragraph):
    sentence_to_remove="  \r\n \r\n(Source: ANN)"
    if isinstance(paragraph, str):
        return paragraph.replace(sentence_to_remove, '')
    else:
        return paragraph

movies['synopsis']=movies['synopsis'].apply(remove_sentence_2)


# In[10]:


def remove_sentence_3(paragraph):
    sentence_to_remove=" \r\n"
    if isinstance(paragraph, str):
        return paragraph.replace(sentence_to_remove, '')
    else:
        return paragraph

movies['synopsis']=movies['synopsis'].apply(remove_sentence_3)


# In[11]:


movies['synopsis'] = movies['synopsis'].apply(lambda x: x.split() if isinstance(x, str) else [])


# In[12]:


movies.head()


# In[13]:


def replace_sentence(paragraph):
    sentence_to_remove="Sci-Fi"
    if isinstance(paragraph, str):
        return paragraph.replace(sentence_to_remove, 'ScienceFiction')
    else:
        return paragraph

movies['genre']=movies['genre'].apply(replace_sentence)


# In[14]:


#movies.head()


# In[15]:


def parse_genre_string(genre_string):
    genre_string = genre_string.strip('[]')
    genre_list = [genre.strip().strip("'") for genre in genre_string.split(',')]
    genre_list = [genre.replace("'", "") for genre in genre_list]
    return genre_list
movies['genre'] = movies['genre'].apply(lambda x: parse_genre_string(x))


# In[16]:


movies['tags']=movies['synopsis'] + movies['genre']


# In[17]:


#movies.head()


# In[18]:


new_df=movies[['anime_uid','title','tags']]


# In[19]:


new_df


# In[20]:


new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))


# In[21]:


#new_df.head()


# In[22]:


new_df['tags'][0]


# In[23]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[24]:


new_df.head()


# In[25]:


import nltk


# In[26]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[27]:


def stem(text):
    y = []
    
    for i in text.split():
        y.append(ps.stem(i))
        
    return " ".join(y)


# In[28]:


new_df['tags']=new_df['tags'].apply(stem)


# In[29]:


new_df['tags'][0]


# In[30]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=3000,stop_words='english')


# In[31]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[32]:


vectors[1]


# In[33]:


cv.get_feature_names_out()


# In[34]:


# stem("""following their participation at the inter high, the karasuno high school volleyball team attempts to refocus their efforts, aiming to conquer the spring tournament instead. when they receive an invitation from long standing rival nekoma high, karasuno agrees to take part in a large training camp alongside many notable volleyball teams in tokyo and even some national level players. by playing with some of the toughest teams in japan, they hope not only to sharpen their skills, but also come up with new attacks that would strengthen them. moreover, hinata and kageyama attempt to devise a more powerful weapon, one that could possibly break the sturdiest of blocks. facing what may be their last chance at victory before the senior players graduate, the members of karasuno's volleyball team must learn to settle their differences and train harder than ever if they hope to overcome formidable opponents old and new including their archrival aoba jousai and its worldclass setter tooru oikawa. comedy sports drama school shounen""")


# In[35]:


from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix, save_npz
from scipy import sparse


# In[36]:


similarity = cosine_similarity(vectors)  # similarity to similarity_matrix


# In[37]:


# Convert the NumPy array to a CSR matrix .....................................
similarity_sparse_matrix=sparse.csr_matrix(similarity)


# In[38]:


# Save the similarity matrix to an NPZ file    ........................................
save_npz('similarity_sparse_matrix.npz', similarity_sparse_matrix)
#np.savez('similarity_matrix.npz', sparse_matrix=sparse_matrix)


# In[39]:


# np.savez_compressed('similarity_matrix.npz', similarity)


# In[40]:


new_df.drop_duplicates(inplace=True)


# In[41]:


new_df.duplicated().sum()


# In[42]:


sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]


# In[43]:


def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances= similarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[44]:


recommend('Wonder')


# In[45]:


new_df['title'][144]


# In[46]:


recommend('Naruto')


# In[47]:


import pickle


# In[48]:


pickle.dump(new_df,open('movies.pkl','wb')) # throws error


# In[49]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))


# In[50]:


#pickle.dump(similarity,open('similarity.pkl','wb'))


# In[51]:


#with np.load('similarity_sparse_matrix.npz') as data: 
#    print(data.files)


# In[52]:


#row = [0, 1, 2, 3] col = [1, 2, 3, 4] data = [1, 2, 3, 4] sparse_matrix = csr_matrix((data, (row, col)))


# In[53]:


#np.savez('similarity_matrix.npz', sparse_matrix=sparse_matrix)


# In[54]:


#print(similarity)

