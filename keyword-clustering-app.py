import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import nltk
nltk.download('punkt_tab')
import numpy as np
from sentence_transformers import SentenceTransformer
import io
import os


def execution():
    # Set NLTK data directory
    nltk_data_dir = "/tmp/nltk_data"
    os.environ['NLTK_DATA'] = nltk_data_dir
    os.makedirs(nltk_data_dir, exist_ok=True)

    
    # Download NLTK resources
    nltk.download('punkt', quiet=True, download_dir=nltk_data_dir)
    nltk.download('stopwords', quiet=True, download_dir=nltk_data_dir)

    from nltk.corpus import stopwords

    # Function to preprocess keywords
    def preprocess_keywords(keywords):
        stop_words_english = set(stopwords.words('english'))
        stop_words_spanish = set(stopwords.words('spanish'))
        stop_words_french = set(stopwords.words('french'))
        stop_words_german = set(stopwords.words('german'))
        stop_words_italian = set(stopwords.words('italian'))
        stop_words_portuguese = set(stopwords.words('portuguese'))
        stop_words = stop_words_english.union(stop_words_spanish, stop_words_french, stop_words_german,stop_words_italian,stop_words_portuguese)
        preprocessed = []
        for keyword in keywords:
            words = nltk.word_tokenize(str(keyword).lower())
            filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
            preprocessed.append(' '.join(filtered_words))
        return preprocessed

    # Function to assign cluster names using Sentence Transformers
    def assign_cluster_names(data, model):
        cluster_names = []
        for cluster_id in sorted(data['cluster'].unique()):
            if cluster_id == -1:  # Skip noise points
                cluster_names.append("Not identified")
                continue

            cluster_keywords = data[data['cluster'] == cluster_id]['Keyword'].tolist()
            embeddings = model.encode(cluster_keywords)
            centroid = np.mean(embeddings, axis=0)
            distances = np.linalg.norm(embeddings - centroid, axis=1)
            most_central_idx = np.argmin(distances)
            cluster_names.append(cluster_keywords[most_central_idx])

        return dict(zip(sorted(data['cluster'].unique()), cluster_names))

    # Streamlit interface
    st.title("Keyword Clustering Tool")
    st.write("Upload your CSV file containing keywords to cluster them.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        # Assuming the keyword column is named 'Keyword'
        keywords = data['Keyword'].tolist()

        # Preprocess keywords
        preprocessed_keywords = preprocess_keywords(keywords)

        # Convert keywords to TF-IDF vectors and compute cosine similarity matrix
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(preprocessed_keywords)
        similarity_matrix = cosine_similarity(X)

        # Apply DBSCAN clustering algorithm
        clustering_model = DBSCAN(eps=0.2, min_samples=2, metric='precomputed')
        distance_matrix = 1 - similarity_matrix  # Convert similarity to distance
        distance_matrix[distance_matrix < 0] = 0  # Ensure non-negative distances
        clusters = clustering_model.fit_predict(distance_matrix)

        # Add clustering results to the DataFrame
        data['cluster'] = clusters

        # Load Sentence Transformer model for semantic analysis
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

        # Assign names to clusters based on their content
        cluster_name_mapping = assign_cluster_names(data, model)

        # Map cluster names to a new column in the DataFrame
        data['Cluster Name'] = data['cluster'].map(cluster_name_mapping)

        clusterized_keywords = data[["Keyword", "Cluster Name", "Volume", "CPC"]]
        
        st.write("Clustered Keywords:")
        st.dataframe(clusterized_keywords)

        # Add download button for the results
        csv = clusterized_keywords.to_csv(index=False)
        st.download_button(
            label="Download clustered keywords as CSV",
            data=csv,
            file_name="clustered_keywords.csv",
            mime="text/csv",
        )

if __name__ == "__main__":
    execution()
