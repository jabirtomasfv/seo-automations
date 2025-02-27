import streamlit as st
import pandas as pd
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Check if punkt is already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Check if words is already downloaded (if you're using it)
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    st.write("Downloading punkt tokenizer...")
    nltk.download('punkt')
    st.write("Punkt tokenizer downloaded.")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string


# --- 1. Title and Introduction ---
st.title("Keyword Clustering Tool")
st.write("Upload your CSV file containing keywords to cluster them automatically.")

# --- 2. File Upload ---
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the CSV into a Pandas DataFrame
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.stop()  # Stop execution if there's an error

    # --- 3. Keyword Clustering Stage 1: Rule-Based Classification ---

    # Define classification terms (same as in the original code)
    classification_terms = {
        "Tickets & Booking": ["fast track", "offer", "deals", "2 for ", "last minute", "ticket", "admission", "pass", "entry fee", "booking", "price", "how much", "fee", "how expensive", "cost", "fare", "availability", "charges", "discount", "ride"],
        "Visit": ["interior", "inside", "tour", "attractions", "zone", "wildlife", "children", "fish tank"],
        "Things to see (distributor)": ["see things", "things to see", "must see", "highlights", "what to see","things to see", "must see", "highlights", "what to see"],
        "Timings": ["hour", "timing", "open", "closed", "opening times", "closing", "time", "what time","best time"],
        "Photogallery ": ["photo", "pic", "image"],
        "Restaurant": ["restaurant at", "restaurants in", "restaurant in", "dinner at", "dinning at", "food", "lunch at", "cafe", "restaurant","eat near", "places eat", "eating near", "restaurant close", "food near", "eating around", "eat around", "dining near", "restaurants near", "restaurants close", "restaurant around", "places to eat near", "bars near"],
        "Exhibitions": ["exhibition", "exhibit"],
        "Special Events": ["event", "celebration", "new year", "today", "party", "halloween", "easter", "show", "light"],
        "Visit Planning": ["visit", "best time to go", "best time to see", "best time to visit", "plan your visit", "visitor info", "visitor information", "entrance", "trip to", "trips to", "best way to see", "contact"],
        "Directions": ["parking", "going to", "situated", "directions", "locat", "how to get", "address", "map", "where is", "get to", "whereabouts", "subway", "bus", "tube", "saadiyat"],
        "Store": ["shop", "store", "merchandise", "gifts", "souvenir"],
        "Accommodation": ["hotel", "hotels near", "hotels close", "accommodation near", "where to sleep", "inn"],
        "Complementary plans": ["places near", "close to", "tours around", "attractions near", "things to do near", "activities around", "what to visit near"],
        "History & Facts": ["fact", "history", "own", "built", "build", "construction", "established", "founded", "background", "info", "high", "how old", "architect"],
        "Reviews": ["evaluations", "assessments", "appraisals", "critiques", "ratings", "feedback","testimonials", "comments", "remarks", "responses", "judgments", "verdicts","analyses", "inspections", "examinations", "overviews", "reports", "write-ups","recommendations", "check-ups", "summaries", "reappraisals", "reconsiderations"],
        "Opinions": ["views", "beliefs", "perspectives", "standpoints", "outlooks", "attitudes","positions", "judgments", "conclusions", "convictions", "ideas", "impressions","perceptions", "notions", "thoughts", "sentiments", "stances", "interpretations","points of view", "estimates", "appraisals", "conjectures", "hypotheses"],
        "To do with kids": ["activities for children", "kids' activities", "family-friendly activities","fun things to do with kids", "child-friendly attractions", "entertainment for kids","things to do as a family", "places to go with children", "kid-friendly outings","adventures for kids", "family-friendly experiences", "things to do for families","recreational activities for kids", "fun family activities", "children's entertainment","places to visit with kids", "things to do with young ones", "family fun ideas","amusement for children", "leisure activities for kids", "best things to do with kids","kid-friendly events", "excursions with children", "outdoor activities for kids","things to do with toddlers", "things to do with teens", "family-friendly destinations","weekend fun with kids", "fun stuff for kids", "cool things to do with kids","places to explore with kids", "interactive activities for kids", "engaging activities for children","best places to go with kids", "exciting experiences for kids", "top attractions for families"],
        "Best Recommendatios":  [
            "the best","the top", "ranks highest","top choice", "highest rated", "best option", "number one", "leading", "top-rated","finest", "greatest", "ultimate","the superior", "most recommended", "best-rated",
            "highest rated", "top pick", "most popular", "preferred", "best available", "most effective",
            "top-rated", "highest ranked", "number one", "best option", "top pick",
            "most recommended", "preferred choice", "leading option", "finest selection",
            "ultimate choice", "superior alternative", "optimal selection", "most effective",
            "premium option", "most suitable", "highest quality", "most popular", "best-rated",
            "editor’s choice", "critically acclaimed", "top performer", "industry leader",
            "gold standard", "top-tier", "top-notch", "best in class"
        ],
        "Things to do":[
            "things to do","fun things", "stuff to do", "things to explore","things to check out","must-do activities", "things to enjoy", "things worth doing","things to discover","things not to miss", "things to keep you busy", "things to experience",
            "cool stuff to do", "places to check out","things to try", "hang out", "spend your day", "things to check out", "fun options", "things happening", "things to do near"]
                                }

    def classify_keywords_in_dataframe(df, classification_terms):
        """
        Classifies keywords in a DataFrame based on the classification_terms dictionary.
        Assumes the DataFrame has a column named 'Keyword'.
        Adds a new column 'Classification' with the classification results.
        """
        if 'Keyword' not in df.columns:
            raise ValueError("The DataFrame must contain a column named 'Keyword'.")

        df['Cluster Name'] = 'Unclassified'

        for index, row in df.iterrows():
            keyword = row['Keyword']
            keyword_lower = keyword.lower()
            for page, terms in classification_terms.items():
                if any(term in keyword_lower for term in terms) and df.at[index, 'Cluster Name'] == 'Unclassified':
                    df.at[index, 'Cluster Name'] = page
                    break
        if 'Difficulty' in df.columns:
            df.rename(columns={'Difficulty': 'KD'}, inplace=True)

        return df[["Keyword","Volume",	"KD",	"CPC",	"Cluster Name"]]

    # Perform initial classification
    try:
        classified_keywords = classify_keywords_in_dataframe(df.copy(), classification_terms)  # Use df.copy()
        st.write("Initial Classification Complete:")
        st.dataframe(classified_keywords)
    except Exception as e:
        st.error(f"Error during initial classification: {e}")
        st.stop()

    # --- 4. Keyword Clustering Stage 2: DBSCAN Clustering ---

    # Download NLTK data (only if not already downloaded)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    from nltk.corpus import stopwords

    # Keyword preprocessing
    def preprocess_keywords(keywords):
        stop_words = set(stopwords.words('english'))
        stop_words_spanish = set(stopwords.words('spanish'))
        stop_words_french = set(stopwords.words('french'))
        stop_words_german = set(stopwords.words('german'))
        stop_words_italian = set(stopwords.words('italian'))
        stop_words_portuguese = set(stopwords.words('portuguese'))
        preprocessed = []
        for keyword in keywords:
            words = nltk.word_tokenize(keyword.lower())
            filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
            preprocessed.append(' '.join(filtered_words))
        return preprocessed

    data = classified_keywords[classified_keywords['Cluster Name'] == 'Unclassified'].copy() # before it was missing .copy()

    keywords = data['Keyword'].tolist()

    preprocessed_keywords = preprocess_keywords(keywords)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(preprocessed_keywords)

    similarity_matrix = cosine_similarity(X)

    # Add a slider for epsilon
    epsilon = st.slider("Epsilon (DBSCAN)", min_value=0.1, max_value=1.0, value=0.5, step=0.05)

    # Add a slider for min_samples
    min_samples = st.slider("Min Samples (DBSCAN)", min_value=2, max_value=10, value=2, step=1)

    clustering_model = DBSCAN(eps=epsilon, min_samples=min_samples, metric='precomputed')
    distance_matrix = 1 - similarity_matrix
    distance_matrix[distance_matrix < 0] = 0
    clusters = clustering_model.fit_predict(distance_matrix)

    data['cluster'] = clusters
    classified_keywords_2 = data[["Keyword","Volume",	"KD",	"CPC",	"cluster"]].copy() #before it was missing the .copy()

    def clean_text(text):
        text = text.lower()
        text = ''.join([char for char in text if char not in string.punctuation])
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        tokens = [w for w in tokens if not w in stop_words]
        text = ' '.join(tokens)
        return text

    data['cleaned_keyword'] = data['Keyword'].apply(clean_text)

    def generate_cluster_names(df):
        cluster_names = {}
        for cluster_id, group in df.groupby('cluster'):
            cluster_keywords = group['cleaned_keyword'].tolist()
            all_words = ' '.join(cluster_keywords)
            vectorizer = TfidfVectorizer()
            vectors = vectorizer.fit_transform([all_words])
            feature_names = vectorizer.get_feature_names_out()
            dense = vectors.todense()
            word_values = dense.tolist()[0]
            word_scores = {feature_names[i]: word_values[i] for i in range(len(feature_names))}
            # Sort by score
            sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
            # Take the top N words
            top_words = [word for word, score in sorted_words[:2]]
            cluster_name = ' '.join(top_words).title()
            cluster_names[cluster_id] = cluster_name
        return cluster_names

    cluster_names = generate_cluster_names(data)

    data['Cluster Name'] = data['cluster'].map(cluster_names)
    classified_keywords_2 = data[["Keyword","Volume",	"KD",	"CPC",	"Cluster Name"]]

    # --- 5. Merging Results ---

    merged_df = pd.concat([classified_keywords, classified_keywords_2], ignore_index=True)

    selected_columns = ['Keyword', 'Volume', 'KD', 'CPC', 'Cluster Name']
    merged_df = merged_df[selected_columns]

    merged_df = merged_df.drop_duplicates(subset='Keyword', keep='last')

    merged_df = merged_df.reset_index(drop=True)

    # --- 6. Display Final Results ---
    st.write("Final Clustered Keywords:")
    st.dataframe(merged_df)

    # --- 7. Download Results ---
    csv = merged_df.to_csv(index=False)
    st.download_button(
        label="Download clustered keywords as CSV",
        data=csv,
        file_name='clustered_keywords.csv',
        mime='text/csv',
    )

