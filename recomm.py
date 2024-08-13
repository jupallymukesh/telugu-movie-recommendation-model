import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(r'c:\users\jupally mukesh\appdata\local\packages\pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0\localcache\local-packages\python311\site-packages')
from sklearn.neighbors import NearestNeighbors
#from sklearn.neighbors import NearestNeighbors
import difflib


def get_closest_movie_id(movie_name, movie_meta_data):
    highest_similarity = 0.0
    closest_movie_id = None
    
    for i, title in enumerate(movie_meta_data['Movie']):
        similarity = difflib.SequenceMatcher(None, title.lower(), movie_name.lower()).ratio()
        if similarity > highest_similarity:
            highest_similarity = similarity
            closest_movie_id = movie_meta_data['movieid'][i]
    
    if closest_movie_id is None:
        raise ValueError("No matching or similar movie found")
    
    return closest_movie_id

def movie_recommender_engine(movie_name, matrix, cf_model, n_recs, movie_meta_data):
    # Fit model on matrix
    cf_model.fit(matrix)
    
    # Find closest movie ID
    closest_movie_id = get_closest_movie_id(movie_name, movie_meta_data)

    # Calculate neighbour distances
    distances, indices = cf_model.kneighbors(matrix.iloc[[closest_movie_id]], n_neighbors=n_recs)
    movie_rec_ids = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[::-1]

    # List to store recommendations
    cf_recs = []
    for idx, dist in movie_rec_ids:
        cf_recs.append({'Title': movie_meta_data.loc[idx, 'Movie'], 'Distance': dist})

    # Create DataFrame with recommendations
    df = pd.DataFrame(cf_recs, index=range(1, n_recs + 1))
    
    return df



data=pd.read_csv("TeluguMovies_dataset.csv")
ratings = pd.read_csv("user_movie_ratings.csv")


data['Certificate'].fillna("Not Rated",inplace=True)

movie_meta_data=data.iloc[:,0:2]

ratings.rename(columns={"movie_id": "movieid"}, inplace=True)

movie_data=movie_meta_data.merge(ratings,on='movieid')

item_matrix=movie_data.pivot(index=['user_id'],columns=['movieid'],values='rating').fillna(0)


cf_knn_model=NearestNeighbors(metric='cosine',algorithm='brute',n_neighbors=10,n_jobs=-1)

cf_knn_model.fit(item_matrix)

item_matrix = item_matrix.T


n_recs = 5
movie_recommendations = movie_recommender_engine('don', item_matrix, cf_knn_model, n_recs, movie_meta_data)
print(movie_recommendations[::-1])

