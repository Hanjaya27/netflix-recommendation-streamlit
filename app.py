# app.py
import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# =====================
# KONFIGURASI
# =====================
OMDB_API_KEY = "c59861e5"

st.set_page_config(
    page_title="Netflix Movie Recommender",
    layout="wide"
)

# =====================
# CSS CUSTOM
# =====================
st.markdown("""
<style>
.movie-card {
    background-color: #111;
    padding: 10px;
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.6);
    text-align: center;
    color: white;
}
.movie-title {
    font-weight: bold;
    margin-top: 8px;
}
.movie-genre {
    font-size: 0.85rem;
    color: #bbb;
}
.movie-score {
    font-size: 0.8rem;
    color: #00ffcc;
}
.badge {
    display: inline-block;
    background-color: crimson;
    padding: 3px 8px;
    border-radius: 12px;
    font-size: 0.7rem;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# =====================
# HEADER
# =====================
st.markdown(
    "<h1 style='color:#E50914;'>🎬 Netflix Movie Recommendation System</h1>",
    unsafe_allow_html=True
)
st.write("📌 **Content-Based Filtering menggunakan TF-IDF & Cosine Similarity**")

# =====================
# SIDEBAR
# =====================
st.sidebar.header("Pengaturan Rekomendasi")

# =====================
# LOAD DATASET
# =====================
@st.cache_data
def load_data():
    movies = pd.read_csv("netflix_movies_detailed_up_to_2025.csv")
    tv = pd.read_csv("netflix_tv_shows_detailed_up_to_2025.csv")

    movies["type"] = "Movie"
    tv["type"] = "TV Show"

    df = pd.concat([movies, tv], ignore_index=True)
    df = df[["title", "genres", "description", "type"]].copy()

    df["content"] = (
        df["title"].fillna("") + " " +
        df["genres"].fillna("") + " " +
        df["description"].fillna("")
    )

    return df.reset_index(drop=True)

df = load_data()

# =====================
# TF-IDF
# =====================
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = tfidf.fit_transform(df["content"])
indices = pd.Series(df.index, index=df["title"]).drop_duplicates()

# =====================
# OMDb POSTER
# =====================
@st.cache_data(show_spinner=False)
def get_movie_poster(title):
    url = "http://www.omdbapi.com/"
    params = {"apikey": OMDB_API_KEY, "t": title}
    res = requests.get(url, params=params).json()

    if res.get("Poster") and res["Poster"] != "N/A":
        return res["Poster"]
    return "https://via.placeholder.com/300x450?text=No+Poster"

# =====================
# REKOMENDASI
# =====================
def recommend_top_n(title, top_n=5):
    idx = indices[title]
    cosine_scores = cosine_similarity(
        tfidf_matrix[idx], tfidf_matrix
    ).flatten()

    sim_scores = sorted(
        list(enumerate(cosine_scores)),
        key=lambda x: x[1],
        reverse=True
    )

    results = []
    for i, score in sim_scores:
        if i != idx:
            results.append({
                "title": df.iloc[i]["title"],
                "genres": df.iloc[i]["genres"],
                "type": df.iloc[i]["type"],
                "similarity_score": score
            })
        if len(results) == top_n:
            break

    return pd.DataFrame(results)

# =====================
# SIDEBAR INPUT
# =====================
movie_input = st.sidebar.selectbox(
    "🔍 Cari Film Favorit",
    df["title"].sort_values().tolist(),
    index=None,
    placeholder="Ketik judul film..."
)

top_n = st.sidebar.slider(
    "Masukkan Jumlah Rekomendasi",
    3, 12, 6
)

# =====================
# TAMPILKAN HASIL
# =====================
if st.sidebar.button("Tampilkan Film Rekomendasi"):
    if movie_input is None:
        st.warning("Silakan pilih film terlebih dahulu.")
    else:
        with st.spinner("🔎 Mencari film serupa..."):
            result = recommend_top_n(movie_input, top_n)

        st.subheader(f"✨ Rekomendasi Mirip **{movie_input}**")

        cols = st.columns(3)
        for i, (_, row) in enumerate(result.iterrows()):
            with cols[i % 3]:
                poster = get_movie_poster(row["title"])
                st.image(poster, use_container_width=True)

                st.markdown(f"""
                <div class="movie-card">
                    <div class="movie-title">{row['title']}</div>
                    <div class="movie-genre">{row['genres']}</div>
                    <div class="badge">{row['type']}</div>
                    <div class="movie-score">
                        Similarity: {row['similarity_score']:.3f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # =====================
        # VISUALISASI
        # =====================
        st.subheader("Visualisasi Similarity")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(result["title"], result["similarity_score"])
        ax.set_xlabel("Cosine Similarity")
        ax.invert_yaxis()
        st.pyplot(fig)

        # =====================
        # TABEL
        # =====================
        st.subheader("Detail Data")
        st.dataframe(result)
