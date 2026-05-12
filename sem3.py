"""
Семинар 3. Контентная фильтрация
Цель: Разработать методы контентной фильтрации по пользователям и по фильмам.
В качестве контента используем описание жанров для каждого фильма из movies.csv.
Для векторизации жанров используем CountVectorizer с разделителем "|".
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import build_user_item_matrix, id_to_movie, load_data, print_user_rated_items


class ContentRecommender:
    """
    Класс для построения рекомендаций на основе контента - описания жанров.
    Матрица эмбеддингов размером (max_movie_id+1, n_genres), где строки
    соответствуют movieId, а столбцы — one-hot кодированию жанров.
    Матрица строится при инициализации экземпляра класса.
    """

    def __init__(self):
        self.embeddings = None
        self.ui_matrix = build_user_item_matrix()
        self._build_embeddings()

    def _build_embeddings(self):
        _, movies_df = load_data()
        self.movies_df = movies_df.copy()
        self.movies_df["genres"] = self.movies_df["genres"].fillna("")
        vectorizer = CountVectorizer(
            tokenizer=lambda s: s.split("|"), 
            lowercase=False
        )
        
        genre_matrix = vectorizer.fit_transform(self.movies_df["genres"])
  
        max_movie_id = self.movies_df["movieId"].max()
        n_genres = genre_matrix.shape[1]

        self.embeddings = np.zeros((max_movie_id + 1, n_genres))

        for idx, row in self.movies_df.iterrows():
            movie_id = row["movieId"]
            if movie_id <= max_movie_id:
                self.embeddings[movie_id] = genre_matrix[idx].toarray().flatten()

        self.genre_names = vectorizer.get_feature_names_out()
        

    def predict_rating(self, user_id: int, item_id: int, k: int = 5) -> float:
        """
        Предсказывает рейтинг user_id для item_id на основе контентной фильтрации.

        Алгоритм:
        1) Берём вектор целевого фильма: target_vec.
        2) Находим все фильмы, оцененные пользователем.
        3) Считаем косинусное сходство target_vec с векторами оцененных фильмов.
        4) Отбираем топ-k похожих оцененных фильмов (k-параметр).
        5) Предсказываем рейтинг как взвешенное среднее оценок по сходствам.
        6) Если не удаётся предсказать (нет оценок или нулевые векторы), возвращаем 0.0.
        7) Клипируем результат в [0.0, 5.0].

        Args:
            user_id: индекс пользователя
            item_id: индекс фильма
            k: сколько наиболее похожих оцененных фильмов использовать

        Returns:
            float: предсказанный рейтинг
        """
        if item_id >= len(self.embeddings):
            return 0.0
        
        target_vec = self.embeddings[item_id]

        if np.sum(target_vec) == 0:
            return 0.0

        user_ratings = self.ui_matrix[user_id]
        rated_items = np.where(user_ratings > 0)[0]
        
        if len(rated_items) == 0:
            return 0.0

        rated_vectors = []
        rated_scores = []
        
        for rated_item in rated_items:
            if rated_item < len(self.embeddings):
                vec = self.embeddings[rated_item]
                if np.sum(vec) > 0:
                    rated_vectors.append(vec)
                    rated_scores.append(user_ratings[rated_item])
        
        if len(rated_vectors) == 0:
            return 0.0
        
        rated_vectors = np.array(rated_vectors)
        rated_scores = np.array(rated_scores)

        target_vec_reshaped = target_vec.reshape(1, -1)
        similarities = cosine_similarity(target_vec_reshaped, rated_vectors)[0]

        top_k_indices = np.argsort(similarities)[::-1][:k]

        top_similarities = similarities[top_k_indices]
        top_ratings = rated_scores[top_k_indices]
        
        non_zero_mask = top_similarities > 0
        if not np.any(non_zero_mask):
            return 0.0
        
        sum_similarities = np.sum(top_similarities[non_zero_mask])
        if sum_similarities == 0:
            return 0.0
        
        predicted = np.sum(top_similarities[non_zero_mask] * top_ratings[non_zero_mask]) / sum_similarities
        
        predicted = np.clip(predicted, 0.0, 5.0)
        
        return float(predicted)

    def predict_items_for_user(
        self, user_id: int, k: int = 5, n_recommendations: int = 5
    ) -> list:
        """
        Рекомендует фильмы пользователю user_id на основе контента фильма.

        Алгоритм:
        1) Берем все фильмы, которые оценил пользователь.
        3) Строим профиль пользователя как взвешенное среднее жанров оцененных фильмов.
        4) Для всех фильмов, которые пользователь не оценил, считаем сходство с профилем.
        5) Сортируем по убыванию сходства и возвращаем top-n.
        """
        user_ratings = self.ui_matrix[user_id]
        rated_items = np.where(user_ratings > 0)[0]
        
        if len(rated_items) == 0:
            return []

        user_profile = np.zeros(self.embeddings.shape[1])
        total_weight = 0
        
        for item_id in rated_items:
            if item_id < len(self.embeddings):
                vec = self.embeddings[item_id]
                if np.sum(vec) > 0:
                    weight = user_ratings[item_id]
                    user_profile += weight * vec
                    total_weight += weight
        
        if total_weight == 0 or np.sum(user_profile) == 0:
            return []

        user_profile = user_profile / total_weight

        all_items = np.arange(len(self.embeddings))
        unrated_items = [item for item in all_items if item not in rated_items]
        
        item_scores = []
        user_profile_reshaped = user_profile.reshape(1, -1)
        
        for item_id in unrated_items:
            if item_id < len(self.embeddings):
                item_vec = self.embeddings[item_id]
                if np.sum(item_vec) > 0:
                    item_vec_reshaped = item_vec.reshape(1, -1)
                    sim = cosine_similarity(user_profile_reshaped, item_vec_reshaped)[0][0]
                    if sim > 0:
                        item_scores.append((int(item_id), sim))
        
        item_scores.sort(key=lambda x: x[1], reverse=True)
        recommendations = [item_id for item_id, _ in item_scores[:n_recommendations]]
        
        return recommendations


# Пример использования для дебага:
if __name__ == "__main__":
    user_id = 10
    item_id = 2
    k = 5
    content_recommender = ContentRecommender()
    print_user_rated_items(user_id, content_recommender.ui_matrix)

    pred_rating = content_recommender.predict_rating(user_id, item_id, k)
    print(f"Predicted rating for user {user_id} and item {item_id}: {pred_rating:.2f}")

    recommendations = content_recommender.predict_items_for_user(
        user_id, k=5, n_recommendations=10
    )
    for rec in recommendations:
        print(f"Recommended movie ID: {rec}, Title: {id_to_movie(rec)}")
