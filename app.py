import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import plotly.express as px


st.set_page_config(page_title="News Topic Discovery Dashboard", layout="wide")


st.markdown("""
<style>
.main-card {
    background-color: #FFD8A8;
    padding: 40px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.1);
    text-align: center;
    margin-bottom: 30px;
}
.main-title {
    
    font-size: 42px;
    font-weight: 700;
}
.main-subtitle {
    font-size: 18px;
    color: #555;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="main-card">
    <div class="main-title"> News Topic Discovery Dashboard</div>
    <div class="main-subtitle">
        Hierarchical Clustering to group similar news articles based on textual similarity.
    </div>
</div>
""", unsafe_allow_html=True)


try:
    df = pd.read_csv("all_data.csv")
except:
    st.error("❌ all_data.csv not found in project folder.")
    st.stop()

df = df.iloc[:, :2]
df.columns = ["label", "text"]


st.sidebar.header("📝 Text Vectorization")

max_features = st.sidebar.slider("Maximum TF-IDF Features", 100, 2000, 1000)
use_stopwords = st.sidebar.checkbox("Use English Stopwords", True)

st.sidebar.header("Hierarchical Clustering")

linkage_method = st.sidebar.selectbox(
    "Linkage Method",
    ["ward", "complete", "average", "single"],
    index=0
)

subset_size = st.sidebar.slider("Dendrogram Sample Size", 20, 200, 50)


vectorizer = TfidfVectorizer(
    max_features=max_features,
    stop_words="english" if use_stopwords else None
)

X = vectorizer.fit_transform(df["text"].astype(str))


if st.button("🟦 Generate Dendrogram"):

    st.subheader("Dendrogram of Articles")

    subset = X[:subset_size].toarray()
    linked = linkage(subset, method=linkage_method)

    fig, ax = plt.subplots(figsize=(12, 6))

    dendrogram(
        linked,
        leaf_rotation=90,
        leaf_font_size=8
    )

    ax.set_title("Dendrogram of Articles")
    ax.set_xlabel("Articles")
    ax.set_ylabel("Euclidean Distance")

    st.pyplot(fig)


st.markdown("---")
st.header("Apply Clustering")

num_clusters = st.slider("Select Number of Clusters", 2, 10, 3)

if st.button("🟩 Apply Clustering"):

    X_dense = X.toarray()

    if linkage_method == "ward":
        model = AgglomerativeClustering(
            n_clusters=num_clusters,
            linkage="ward"
        )
    else:
        model = AgglomerativeClustering(
            n_clusters=num_clusters,
            linkage=linkage_method,
            metric="euclidean"
        )

    cluster_labels = model.fit_predict(X_dense)

    st.success("✅ Clustering completed successfully!")

 
    st.subheader("Cluster Visualization (2D PCA)")

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X_dense)

    plot_df = pd.DataFrame({
        "PCA1": reduced[:, 0],
        "PCA2": reduced[:, 1],
        "Cluster": cluster_labels.astype(str),
        "Snippet": df["text"].astype(str).str[:100]
    })

    fig = px.scatter(
        plot_df,
        x="PCA1",
        y="PCA2",
        color="Cluster",
        hover_data=["Snippet"]
    )

    st.plotly_chart(fig, use_container_width=True)

   
    st.subheader("Cluster Summary")

    feature_names = vectorizer.get_feature_names_out()
    summary_data = []

    for cluster_id in range(num_clusters):

        indices = np.where(cluster_labels == cluster_id)[0]

        if len(indices) == 0:
            continue

        cluster_vectors = X_dense[indices]
        mean_tfidf = np.mean(cluster_vectors, axis=0)

        top_indices = mean_tfidf.argsort()[-10:][::-1]
        keywords = ", ".join([feature_names[i] for i in top_indices])

        summary_data.append({
            "Cluster ID": cluster_id,
            "Number of Articles": len(indices),
            "Top Keywords": keywords
        })

    st.dataframe(pd.DataFrame(summary_data))

    st.subheader("Validation")

    if num_clusters > 1:
        score = silhouette_score(X_dense, cluster_labels)
        st.metric("Silhouette Score", round(score, 3))

        st.write("""
Close to 1 → Well separated clusters  
Close to 0 → Overlapping clusters  
Negative → Poor clustering
""")


    st.subheader("Business Interpretation")

    st.write("""
Articles grouped in the same cluster share similar vocabulary and themes.

These clusters can be used for:
• Automatic tagging  
• Content recommendation  
• News categorization  
• Discovering emerging topics
""")
