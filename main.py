import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from math import sqrt

ratings = pd.read_csv(r"C:\Users\Vedant\OneDrive\Desktop\recommendation project\ratings.csv")
movies = pd.read_csv(r"C:\Users\Vedant\OneDrive\Desktop\recommendation project\movies.csv")

# Ensure correct data types
ratings['movieId'] = ratings['movieId'].astype(int)
movies['movieId'] = movies['movieId'].astype(int)

print("Dataset Loaded:\n", ratings.head())

movie_counts = ratings['movieId'].value_counts()

plt.figure()
movie_counts.hist(bins=50)
plt.title("Long Tail Distribution of Movie Popularity")
plt.xlabel("Number of Ratings")
plt.ylabel("Frequency")
plt.show()

user_item = ratings.pivot(index='userId', columns='movieId', values='rating')
user_item_filled = user_item.fillna(0)

def cosine_similarity_manual(matrix):
    dot_product = np.dot(matrix, matrix.T)
    norm = np.linalg.norm(matrix, axis=1)
    similarity = dot_product / (norm[:, None] * norm[None, :])
    return similarity

user_similarity = cosine_similarity_manual(user_item_filled.values)
print("\nUser Similarity (Manual):\n", user_similarity[:5])

svd = TruncatedSVD(n_components=20)
matrix = user_item_filled.values

reduced = svd.fit_transform(matrix)
reconstructed = svd.inverse_transform(reduced)

print("\nSVD Reconstruction Sample:\n", reconstructed[:5])

actual = user_item_filled.values.flatten()
predicted = reconstructed.flatten()

rmse = sqrt(mean_squared_error(actual, predicted))
print("\nRMSE (SVD Model):", rmse)

movie_dict = dict(zip(movies['movieId'], movies['title']))

def recommend_movies(user_id, top_n=5):
    user_index = user_id - 1

    user_ratings = reconstructed[user_index]
    user_actual = user_item.iloc[user_index]

    recommendations = []

    for i, movie_id in enumerate(user_item.columns):
        predicted_rating = user_ratings[i]
        recommendations.append((int(movie_id), predicted_rating))

    recommendations.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop {top_n} Recommendations for User {user_id}:\n")

    count = 0
    for movie_id, score in recommendations:
        if user_actual[movie_id] != 0:
            continue

        title = movie_dict.get(movie_id, f"Movie ID {movie_id}")
        print(f"{title} (Predicted Rating: {round(score,2)})")

        count += 1
        if count == top_n:
            break
    recommendations.sort(key=lambda x: x[1], reverse=True)
    top_movies = recommendations[:top_n]
    
    print(f"\nTop {top_n} Recommendations for User {user_id}:\n")
    
    for movie_id, score in top_movies:
        title = movie_dict.get(int(movie_id), f"Movie ID {movie_id}")
        print(f"{title} (Predicted Rating: {round(score,2)})")
recommend_movies(1)
