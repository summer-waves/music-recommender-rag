from src.recommender import Recommender, UserProfile, Song
from src.retriever import MusicRetriever

def run_rag_recommender(query: str, profile: UserProfile, top_k: int = 3):
    print(f"\n{'='*60}")
    print(f"Query: '{query}'")
    print(f"Profile: {profile.favorite_genre} | {profile.favorite_mood} | energy {profile.target_energy}")
    print(f"{'='*60}")

    # Step 1: RETRIEVE — find relevant songs from knowledge base
    retriever = MusicRetriever()
    retrieved = retriever.retrieve(query, top_k=10)

    print(f"\n🔍 Retrieved {len(retrieved)} songs from knowledge base:")
    for r in retrieved[:5]:
        print(f"  - {r['title']} by {r['artist']} (retrieval score: {r['retrieval_score']})")

    # Step 2: AUGMENT — convert retrieved songs into Song objects
    songs = []
    for r in retrieved:
        song = Song(
            id=r["title"],
            title=r["title"],
            artist=r["artist"],
            genre=r["genre"],
            mood=r["mood"],
            energy=float(r["energy"]),
            acousticness=float(r["acousticness"]),
            tempo_bpm=int(float(r["tempo_bpm"])),
            valence=float(r["valence"]),
            danceability=float(r["danceability"])
        )
        songs.append(song)

    # Step 3: GENERATE — score and rank retrieved songs against user profile
    recommender = Recommender(songs)
    recommendations = recommender.recommend(profile, k=top_k)

    print(f"\n🎵 Top {top_k} Recommendations:")
    for i, song in enumerate(recommendations, 1):
        explanation = recommender.explain_recommendation(profile, song)
        print(f"  {i}. {song.title} by {song.artist}")
        print(f"     Genre: {song.genre} | Mood: {song.mood} | Energy: {song.energy}")
        print(f"     Why: {explanation}")

    return recommendations


if __name__ == "__main__":
    # Test Profile 1 — Happy Pop Fan
    profile1 = UserProfile(
        favorite_genre="pop",
        favorite_mood="happy",
        target_energy=0.8,
        likes_acoustic=False
    )
    run_rag_recommender(
        "upbeat happy pop songs for dancing and summer vibes",
        profile1
    )

    # Test Profile 2 — Chill Acoustic Listener
    profile2 = UserProfile(
        favorite_genre="acoustic",
        favorite_mood="chill",
        target_energy=0.2,
        likes_acoustic=True
    )
    run_rag_recommender(
        "peaceful acoustic chill music for studying and relaxing",
        profile2
    )

    # Test Profile 3 — High Energy Rock Fan
    profile3 = UserProfile(
        favorite_genre="rock",
        favorite_mood="intense",
        target_energy=0.95,
        likes_acoustic=False
    )
    run_rag_recommender(
        "intense high energy rock with heavy guitars and drums",
        profile3
    )