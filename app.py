from typing import Tuple, Any

import streamlit as st
import pickle
import pandas as pd
import numpy as np


from scipy.sparse import load_npz  # Importing load_npz function to load the sparse matrix

st.title('WeebWhizz')

# Load the sparse matrix from file @st.cache  # Cache the function output to avoid reloading the matrix on each request
def load_sparse_matrix(file_path: str) -> Any:
    sparse_matrix = load_npz(r'c:\Users\Lenovo\machine learning project\movie recommender system\similarity_sparse_matrix.npz')
    # Load the sparse matrix from the file
    return sparse_matrix

# Function to recommend movies based on selected movie

def recommend(selected_movie_name: str, similarity_matrix: Any, movies: pd.DataFrame, top_n: int = 5) -> list:
    movie_idx = movies[movies['title'] == selected_movie_name].index[0]  # Get the index of the selected movie
    # Compute similarity scores between the selected movie and all other movies
    sim_scores = list(enumerate(similarity_matrix[movie_idx].toarray().ravel()))
    # Sort the movies based on similarity scores
    sim_scores = sorted(sim_scores, reverse=True, key=lambda x: x[1])
    # Get top N similar movies (excluding the selected movie itself)
    top_similar_movies = sim_scores[1:top_n + 1]
    recommendations = [(movies.iloc[idx]['title']) for idx, score in top_similar_movies]
    return recommendations


# Load the DataFrame from the pickle file
movies_df = pd.read_pickle('movies.pkl')


# Load the movies DataFrame from the CSV file
movies = pd.read_csv('movies.csv')


selected_movie_name = st.selectbox(
    'Select an anime:',
    movies['title'].values)



# Load the sparse matrix
sparse_matrix = load_sparse_matrix('similarity_sparse_matrix.npz')


if st.button('Recommend'):
    recommendations = recommend(selected_movie_name, sparse_matrix, movies)
    for movie in recommendations:
        st.write(movie)
        


#-----------------------------------x------------------------x------------------------x-----------x
# from typing import Tuple, Any

# import streamlit as st
# import pickle
# import pandas as pd
# import numpy as np


# st.title('WeebWhizz')


# def recommend(movie):
#     movie_index=movies[movies['title']==movie].index[0]
#     distances= similarity[movie_index]
#     movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    
    # recommended_movies=[]
    # for i in movies_list:
    #     recommended_movies.append(movies.iloc[i[0]].title)
    # return recommended_movies
    # for i in similar_movies_indices:
    #     if 0 <= i[0] < len(movies):
    #         recommended_movies.append(movies.iloc[i[0]].title)

    # return recommended_movies

# Convert the similarity matrix to a numpy array
# similarity_array = np.array(similarity.pkl)

# Save the numpy array as a CSV file
# np.savetxt('similarity.csv', similarity_array, delimiter=',')



# movies_dict=pickle.load(open('movie_dict.pkl','rb'))
# movies=pd.DataFrame(movies_dict)
# movies.to_csv('similarity.csv', index=False)


#similarity=pickle.load(open('similarity.pkl','rb'))

# selected_movie_name = st.selectbox(
#     'How would you like to be contacted?',
#     movies['title'].values)


# if st.button('Recommend'):
#     recommendations=recommend(selected_movie_name)
#     for i in recommendations:
#         st.write(i)


# large_file_path = 'large_file.csv'
# df = pd.read_csv(large_file_path, low_memory=False)

#------------------------------------------------------------------------------------------------------------------------------------------------------

# def recommend(selected_movie_name: str, similarity_matrix: Any, movies: pd.DataFrame, top_n: int = 5) -> list:
#     movie_idx = movies[movies['title'] == selected_movie_name].index[0]  # Get the index of the selected movie
#     # Debug print statement
#     print("Columns:", movies.columns)
#     print("Index:", movie_idx)
#     print("Title:", movies.iloc[movie_idx]['title'])  # Ensure correct movie title is retrieved

#     if 'img_url' in movies.columns:
#         print("img_url:", movies.iloc[movie_idx]['img_url'])  # Ensure correct column name and value
#     else:
#         print("img_url column not found in movies DataFrame")

#     if 'link' in movies.columns:
#         print("link:", movies.iloc[movie_idx]['link'])  # Ensure correct column name and value
#     else:
#         print("link column not found in movies DataFrame")

#     # Compute similarity scores between the selected movie and all other movies
#     sim_scores = list(enumerate(similarity_matrix[movie_idx].toarray().ravel()))

#     # Sort the movies based on similarity scores
#     sim_scores = sorted(sim_scores, reverse=True, key=lambda x: x[1])

#     # Get top N similar movies (excluding the selected movie itself)
#     top_similar_movies = sim_scores[1:top_n + 1]

#     recommendations = [(movies.iloc[idx]['title'], movies.iloc[idx]['img_url'], movies.iloc[idx]['link'], score) for idx, score in top_similar_movies]

#     return recommendations


# Convert the DataFrame to CSV format and save it
# movies_df.to_csv('movies.csv', index=False, columns=['anime_uid', 'title', 'tags', 'img_url', 'link'])  # Add 'img_url' and 'link' columns


# Drop any duplicate rows based on the 'anime_uid' column
# movies = movies.drop_duplicates(subset='anime_uid')

# Ensure that the 'img_url' and 'link' columns are present in the movies DataFrame
# if 'img_url' not in movies.columns or 'link' not in movies.columns:
#     print("Error: 'img_url' and/or 'link' columns not found in movies DataFrame.")
# else:
#     print("Columns:", movies.columns)

#-----------------------------------------------------------
# Convert the DataFrame to CSV format and save it
# movies_df.to_csv('movies.csv', index=False, columns=['anime_uid', 'title', 'tags', 'img_url', 'link'])
# # movies_df.to_csv('movies.csv', index=False)

# # Load the movies dataframe
# movies = pd.read_csv('movies.csv')
#----------------------------------------------------


# if st.button('Recommend'):
#     recommendations = recommend(selected_movie_name, sparse_matrix, movies)
#     for movie_title, score in recommendations:
#         img_url = movies[movies['title'] == movie_title]['img_url'].values[0]
#         link = movies[movies['title'] == movie_title]['link'].values[0]
#         st.write(f"Title: {movie_title}")
#         st.image(img_url, caption='Anime Poster', use_column_width=True)
#         st.write(f"Website URL: {link}")
