from src.recommender import Recommender, UserProfile, Song
from src.retriever import MusicRetriever


def confidence_score(retrieved: list, recommendations: list) -> float:
    """
    Calculates a confidence score (0.0 - 1.0) based on:
    - How many retrieved songs made it into recommendations
    - Average retrieval score of top results
    """
    if not retrieved or not recommendations:
        return 0.0

    rec_titles = {song.title for song in recommendations}
    retrieved_titles = [r["title"] for r in retrieved]

    overlap = sum(1 for t in retrieved_titles if t in rec_titles)
    overlap_ratio = overlap / len(recommendations)

    avg_retrieval_score = sum(r["retrieval_score"] for r in retrieved[:3]) / 3

    confidence = (overlap_ratio * 0.6) + (avg_retrieval_score * 0.4)
    return round(min(confidence, 1.0), 4)


def run_evaluation():
    """
    Runs a test harness of predefined queries and profiles.
    Prints pass/fail results and confidence scores.
    """
    test_cases = [
        {
            "name": "Happy Pop Fan",
            "query": "upbeat happy pop songs for dancing and summer vibes",
            "profile": UserProfile("pop", "happy", 0.8, False),
            "expected_genre": "pop",
            "expected_mood": "happy"
        },
        {
            "name": "Chill Acoustic Listener",
            "query": "peaceful acoustic chill music for studying and relaxing",
            "profile": UserProfile("acoustic", "chill", 0.2, True),
            "expected_genre": "acoustic",
            "expected_mood": "chill"
        },
        {
            "name": "High Energy Rock Fan",
            "query": "intense high energy rock with heavy guitars and drums",
            "profile": UserProfile("rock", "intense", 0.95, False),
            "expected_genre": "rock",
            "expected_mood": "intense"
        },
        {
            "name": "Sad Jazz Listener",
            "query": "melancholic sad jazz ballad with slow piano",
            "profile": UserProfile("jazz", "sad", 0.3, False),
            "expected_genre": "jazz",
            "expected_mood": "sad"
        },
        {
            "name": "Electronic Dance Fan",
            "query": "high energy electronic dance music with deep bass",
            "profile": UserProfile("electronic", "energetic", 0.9, False),
            "expected_genre": "electronic",
            "expected_mood": "energetic"
        },
        {
            "name": "Lofi Study Session",
            "query": "chill lofi beats for focus and concentration",
            "profile": UserProfile("lofi", "chill", 0.25, False),
            "expected_genre": "lofi",
            "expected_mood": "chill"
        },
    ]

    print("\n" + "="*60)
    print("   🎵 MUSIC RECOMMENDER RAG — EVALUATION REPORT")
    print("="*60)

    passed = 0
    total = len(test_cases)
    confidence_scores = []

    for i, tc in enumerate(test_cases, 1):
        retriever = MusicRetriever()
        retrieved = retriever.retrieve(tc["query"], top_k=10)

        songs = []
        for r in retrieved:
            songs.append(Song(
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
            ))

        recommender = Recommender(songs)
        recommendations = recommender.recommend(tc["profile"], k=3)

        # Check if top recommendation matches expected genre
        top_song = recommendations[0] if recommendations else None
        genre_match = top_song and top_song.genre.lower() == tc["expected_genre"].lower()
        mood_match = top_song and top_song.mood.lower() == tc["expected_mood"].lower()
        test_passed = genre_match and mood_match

        conf = confidence_score(retrieved, recommendations)
        confidence_scores.append(conf)

        if test_passed:
            passed += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"

        print(f"\nTest {i}: {tc['name']}")
        print(f"  Status:     {status}")
        print(f"  Top Result: {top_song.title} by {top_song.artist}" if top_song else "  Top Result: None")
        print(f"  Genre:      expected={tc['expected_genre']} | got={top_song.genre if top_song else 'N/A'}")
        print(f"  Mood:       expected={tc['expected_mood']} | got={top_song.mood if top_song else 'N/A'}")
        print(f"  Confidence: {conf}")

    avg_confidence = round(sum(confidence_scores) / len(confidence_scores), 4)

    print("\n" + "="*60)
    print(f"   RESULTS: {passed}/{total} tests passed")
    print(f"   AVG CONFIDENCE SCORE: {avg_confidence}")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_evaluation()