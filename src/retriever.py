import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

class MusicRetriever:
    def __init__(self, knowledge_base_path=None):
        if knowledge_base_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            knowledge_base_path = os.path.join(base_dir, "data", "knowledge_base.csv")

        self.df = pd.read_csv(knowledge_base_path)
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["description"])

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Takes a natural language query and returns the top_k most
        relevant songs from the knowledge base using TF-IDF cosine similarity.
        """
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        top_indices = scores.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            song = self.df.iloc[idx].to_dict()
            song["retrieval_score"] = round(float(scores[idx]), 4)
            results.append(song)

        return results

    def retrieve_by_profile(self, genre: str, mood: str, top_k: int = 5) -> list[dict]:
        """
        Retrieves songs using a structured profile query built from
        genre and mood — feeds naturally into the existing recommender.
        """
        query = f"{mood} {genre} music"
        return self.retrieve(query, top_k)