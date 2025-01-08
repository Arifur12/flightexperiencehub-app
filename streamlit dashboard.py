import re
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
import nltk
from datetime import datetime

nltk.download('wordnet')

st.set_page_config(
    page_title="Airline Reviews Analysis",
    page_icon="✈️",
    layout="wide"
)

st.title("Airline Reviews Analysis Dashboard ✈️")
st.write("""
Upload your airline reviews dataset to analyze topics, sentiment, and generate TF-IDF weighted star ratings. 
The dashboard provides insights into various service categories and offers recommendations based on overall ratings.
""")

st.sidebar.header("Configuration")

num_topics = st.sidebar.slider(
    "Number of LDA Topics",
    min_value=2,
    max_value=10,
    value=5
)

num_words = st.sidebar.slider(
    "Number of Top Words per Topic",
    min_value=5,
    max_value=20,
    value=10
)

max_features = st.sidebar.slider(
    "Max TF-IDF Features",
    min_value=500,
    max_value=5000,
    value=2000,
    step=500,
    help="Reduce vocabulary size for faster TF-IDF processing."
)

stopwords_input = st.sidebar.text_area(
    "Custom Stopwords (comma-separated)",
    "the,a,an,is,it,to,and,in,on,for,of,at,this,that,was,with,as,be,by,verified,airways,trip"
)
custom_stopwords = set(word.strip().lower() for word in stopwords_input.split(','))

st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload your CSV file (with 'review' and 'title' columns)",
    type=["csv"]
)

run_button = st.button("Run Analysis")

analyzer = SentimentIntensityAnalyzer()

category_keywords = {
    'food': ['food', 'meal', 'snack', 'dinner', 'lunch', 'breakfast'],
    'service': ['service', 'staff', 'crew', 'attendant', 'helpful'],
    'seating': ['seat', 'seating', 'legroom', 'comfort'],
    'entertainment': ['entertainment', 'movie', 'screen', 'wifi', 'music'],
    'delay': ['delay', 'late', 'time', 'departure', 'arrival']
}

def preprocess_text(text, custom_stopwords):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in custom_stopwords]
    return ' '.join(tokens)

def perform_topic_modeling(text_data, num_topics=5, num_words=10, max_features=2000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    dtm = vectorizer.fit_transform(text_data)
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42, n_jobs=-1)
    lda.fit(dtm)

    topics = []
    for index, topic in enumerate(lda.components_):
        topic_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-num_words:]]
        topics.append((index + 1, topic_words[::-1]))
    return topics

def analyze_vader_sentiment(text):
    if not isinstance(text, str) or text.strip() == "":
        return 0.0
    scores = analyzer.polarity_scores(text)
    return scores['compound']

def assign_star_rating(sentiment):
    if sentiment >= 0.4:
        return 5
    elif sentiment >= 0.2:
        return 4
    elif sentiment >= 0.0:
        return 3
    elif sentiment >= -0.25:
        return 2
    else:
        return 1

def convert_rating_to_stars(rating):
    filled_stars = "★" * rating
    empty_stars = "☆" * (5 - rating)
    return filled_stars + empty_stars

@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file, delimiter=';', on_bad_lines='skip', header=None, names=['review', 'title'])

if 'last_updated' not in st.session_state:
    st.session_state['last_updated'] = None

if run_button:
    if uploaded_file is not None:
        try:
            with st.spinner("Loading dataset..."):
                reviews_df = load_data(uploaded_file)

            if reviews_df.empty:
                st.error("Error: The dataset is empty.")
            else:
                required_columns = ['review', 'title']
                if not all(col in reviews_df.columns for col in required_columns):
                    st.error(f"CSV file must contain the following columns: {', '.join(required_columns)}")
                else:
                    st.success(f"Dataset loaded successfully with {len(reviews_df)} records.")

                    reviews_df['review'] = reviews_df['review'].fillna('')
                    reviews_df['title'] = reviews_df['title'].fillna('')

                    with st.spinner("Applying text preprocessing..."):
                        reviews_df['clean_review'] = reviews_df['review'].apply(lambda x: preprocess_text(x, custom_stopwords))
                        reviews_df['clean_title'] = reviews_df['title'].apply(lambda x: preprocess_text(x, custom_stopwords))

                    combined_text = reviews_df['clean_review'] + ' ' + reviews_df['clean_title']

                    with st.spinner("Performing topic modeling..."):
                        topics = perform_topic_modeling(combined_text, num_topics=num_topics, num_words=num_words, max_features=max_features)

                    with st.spinner("Performing sentiment analysis with VADER..."):
                        reviews_df['sentiment_review'] = reviews_df['clean_review'].apply(analyze_vader_sentiment)
                        reviews_df['sentiment_title'] = reviews_df['clean_title'].apply(analyze_vader_sentiment)
                        reviews_df['overall_sentiment'] = (reviews_df['sentiment_review'] + reviews_df['sentiment_title']) / 2

                    with st.spinner("Generating TF-IDF weighted star ratings..."):
                        vectorizer = TfidfVectorizer(max_features=max_features)
                        X = vectorizer.fit_transform(reviews_df['clean_review'])
                        feature_names = vectorizer.get_feature_names_out()
                        feature_index = {word: i for i, word in enumerate(feature_names)}

                        category_sentiments = {cat: [] for cat in category_keywords.keys()}
                        for row_idx in range(X.shape[0]):
                            row_data = X.getrow(row_idx)
                            row_dict = dict(zip(row_data.indices, row_data.data))
                            
                            for cat, keywords in category_keywords.items():
                                score = 0.0
                                total_weight = 0.0
                                for kw in keywords:
                                    if kw in feature_index:
                                        col_idx = feature_index[kw]
                                        if col_idx in row_dict:
                                            tfidf_val = row_dict[col_idx]
                                            sentiment = analyzer.polarity_scores(kw)['compound']
                                            score += sentiment * tfidf_val
                                            total_weight += tfidf_val
                                category_sentiments[cat].append(score / total_weight if total_weight > 0 else 0.0)

                        for cat in category_keywords.keys():
                            reviews_df[f'{cat}_sentiment'] = category_sentiments[cat]
                            reviews_df[f'{cat}_rating'] = reviews_df[f'{cat}_sentiment'].apply(assign_star_rating)

                    st.session_state['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    average_ratings = {}
                    for category in category_keywords.keys():
                        avg = reviews_df[f'{category}_rating'].mean()
                        average_ratings[category] = avg

                    st.markdown("---")
                    st.subheader("Average Star Ratings per Category")
                    col1, col2, col3 = st.columns(3)
                    for idx, category in enumerate(category_keywords.keys()):
                        avg_rating = reviews_df[f'{category}_rating'].mean()
                        stars = convert_rating_to_stars(round(avg_rating))
                        col = [col1, col2, col3][idx % 3]
                        with col:
                            st.markdown(f"### **{category.capitalize()}**")
                            st.markdown(f"<div style='font-size:30px;'>{stars}</div>", unsafe_allow_html=True)
                            st.markdown(f"**Average Rating:** {avg_rating:.2f} ⭐")

                    overall_avg = reviews_df['overall_sentiment'].mean()
                    overall_star = assign_star_rating(overall_avg)
                    overall_stars = convert_rating_to_stars(overall_star)

                    st.markdown("---")
                    st.subheader("Overall Average Star Rating")
                    st.markdown(f"<div style='font-size:40px;'>{overall_stars}</div>", unsafe_allow_html=True)

                    st.markdown("---")
                    st.subheader("Recommendation")
                    if overall_star >= 4:
                        recommendation = "✅ **We recommend using this airline !**"
                        recommendation_color = "green"
                    elif overall_star == 3:
                        recommendation = "⚠️ **This airline is average. You may consider other options.**"
                        recommendation_color = "orange"
                    else:
                        recommendation = "❌ **We do not recommend using this airline .**"
                        recommendation_color = "red"
                    
                    st.markdown(f"<div style='font-size:24px; color:{recommendation_color};'>{recommendation}</div>", unsafe_allow_html=True)

                    st.markdown("---")
                    st.write(f"**Last Updated:** {st.session_state['last_updated']}")

                    st.markdown("---")
                    st.subheader("Download Processed Data")
                    csv = reviews_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name='processed_reviews.csv',
                        mime='text/csv',
                    )

                    st.markdown("---")
                    if st.button("🔄 Refresh"):
                        st.experimental_rerun()

        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV file is empty. Please provide a valid dataset.")
        except pd.errors.ParserError:
            st.error("Error parsing the CSV file. Please ensure it's correctly formatted.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        st.info("Awaiting dataset upload and analysis.")
