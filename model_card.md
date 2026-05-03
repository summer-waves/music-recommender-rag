# 🎵 Model Card — Music Recommender RAG

---

## Model Overview

**System Name:** Music Recommender RAG
**Base Project:** Module 3 — Music Recommender Simulation
**Version:** 2.0
**Author:** Marco Ortiz
**Date:** May 2026
**Type:** Retrieval-Augmented Generation (RAG) recommendation system

---

## Intended Use

This system is designed to recommend music to a user based on a natural
language query and a structured taste profile. It is intended for:

- Personal music discovery
- Educational demonstration of RAG pipelines
- Portfolio demonstration of applied AI system design

It is **not** intended for:
- Commercial music streaming integration
- Real-time recommendation at scale
- Replacing human music curation

---

## How It Works

The system follows a three-step RAG pipeline:

1. **Retrieve** — A TF-IDF vectorizer searches a 30-song knowledge base
   using cosine similarity against the user's natural language query.
2. **Augment** — The top 10 retrieved songs are converted into Song objects
   and passed to the scoring engine.
3. **Generate** — The recommender scores each retrieved song against the
   user's taste profile (genre, mood, energy, acousticness) and returns
   the top k recommendations with explanations.

---

## Training Data / Knowledge Base

The knowledge base (`data/knowledge_base.csv`) contains 30 manually
curated songs across 6 genres: pop, rock, lofi, acoustic, electronic,
and jazz. Each song includes a natural language description used for
TF-IDF retrieval, plus structured attributes used for scoring.

**Limitations of the knowledge base:**
- 30 songs is a very small catalog — real systems use millions of tracks
- All songs and descriptions were written manually — no real artist data
- Genre distribution is equal (5 songs per genre) which may not reflect
  real listening patterns

---

## Performance

| Metric | Result |
|---|---|
| Evaluator pass rate | 6/6 (100%) |
| Average confidence score | 0.7154 |
| Pytest pass rate | 8/8 (100%) |
| Genres covered | 6 (pop, rock, lofi, acoustic, electronic, jazz) |

---

## Limitations and Biases

**Genre dominance bias:**
The scoring system weights genre matches at +2.0 points, which is higher
than mood (+1.0) or energy (+1.5 max). This means the system will almost
always recommend songs from the user's preferred genre, even when a song
from a different genre might be a better emotional or energy fit. This
mirrors real-world filter bubble problems in recommendation systems.

**Keyword dependency:**
TF-IDF only matches on exact or near-exact keywords. A query like
"something to cry to" will not reliably retrieve sad songs because
"cry" does not appear in the song descriptions. A semantic embedding
model would handle this better.

**Cold start:**
The system requires the user to explicitly state their favorite genre,
mood, and energy level. It cannot infer preferences from listening
history or behavior.

**Small catalog:**
With only 30 songs, results can repeat across different profiles,
especially for less specific queries.

---

## Ethical Considerations

**Could this system be misused?**
In its current form the system poses minimal risk — it only recommends
fictional songs from a static catalog. However, if extended to real
platforms, recommendation systems can reinforce filter bubbles by
repeatedly pushing users toward the same genre or artist, limiting
exposure to diverse music and cultures.

**Mitigation strategies:**
- Add a diversity penalty so the same artist cannot appear more than
  once in the top recommendations
- Introduce a random exploration factor that occasionally recommends
  outside the user's stated preferences
- Make scoring weights transparent and user-adjustable

---

## Testing and Reliability

The system includes two layers of reliability testing:

**Automated evaluator (`src/evaluator.py`):**
Runs 6 predefined profiles and checks that the top recommendation
matches the expected genre and mood. Reports confidence scores based
on retrieval overlap and average retrieval score.

**Pytest suite (`tests/test_retriever.py`):**
8 unit tests covering result count, required fields, score range,
score ordering, profile-based retrieval, and edge cases.

**What surprised me during testing:**
Even with only 30 songs, the TF-IDF retriever was highly accurate
for descriptive queries. The biggest surprise was how well the
confidence scores clustered around 0.70 — the system was consistently
uncertain at roughly the same level across all genres, which suggests
the retrieval quality is uniform rather than strong for some genres
and weak for others.

---

## AI Collaboration Reflection

**Helpful suggestion:**
Claude suggested using TF-IDF cosine similarity as the retrieval
engine instead of a neural embedding model. This was genuinely helpful
because it meant the entire RAG pipeline could run locally with no API
key and no external dependencies — just scikit-learn. It simplified
setup significantly without sacrificing retrieval quality for this
scale of project.

**Flawed suggestion:**
Early in the build, Claude suggested using `top_k` as the argument
name when calling `recommender.recommend()`. This caused a TypeError
because the existing recommender used `k` not `top_k`. The fix was
simple once the actual source code was checked, but it was a good
reminder that AI suggestions need to be verified against the actual
codebase rather than assumed to be correct.

---

## Future Improvements

- Replace TF-IDF with sentence-transformers for semantic retrieval
- Expand knowledge base to 200+ real songs using a public music API
- Add a Streamlit UI for interactive query input
- Make scoring weights configurable by the user
- Add collaborative filtering on top of content-based scoring