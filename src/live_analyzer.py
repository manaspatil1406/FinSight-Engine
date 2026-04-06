import pandas as pd
import joblib
import os
import sys
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from news_fetcher import fetch_all_sectors

# Add src to path for config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import MODEL_PATH, VECTORIZER_PATH, LIVE_NEWS_RESULTS_CSV, LIVE_NEWS_CACHE_CSV

def preprocess_text_fast(text):
    """Fast preprocessing for model inference (matches app.py logic)."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 2]
    return " ".join(tokens)

def remove_duplicate_articles(df_new, df_old=None):
    """Remove articles already seen in previous fetch based on title."""
    if df_old is None or df_old.empty:
        return df_new
    
    # Deduplicate based on title similarity
    existing_titles = set(df_old['title'].str.lower().str.strip())
    
    # Check each new article
    df_new['is_new'] = df_new['title'].apply(
        lambda x: x.lower().strip() not in existing_titles
    )
    
    # Keep only NEW articles
    df_fresh = df_new[df_new['is_new']].drop(columns=['is_new'])
    
    print(f"📰 Total fetched     : {len(df_new)}")
    print(f"✅ New articles      : {len(df_fresh)}")
    print(f"🔄 Duplicates skipped: {len(df_new) - len(df_fresh)}")
    
    return df_fresh

def analyze_live_news():
    """Fetches and analyzes live news while avoiding duplicates."""
    # Load previous results if exists for deduplication
    previous_df = pd.DataFrame()
    if os.path.exists(LIVE_NEWS_RESULTS_CSV):
        try:
            previous_df = pd.read_csv(LIVE_NEWS_RESULTS_CSV)
            if 'fetched_at' not in previous_df.columns:
                previous_df['fetched_at'] = previous_df.get('analyzed_at')
            else:
                previous_df['fetched_at'] = previous_df['fetched_at'].fillna(previous_df.get('analyzed_at'))
        except Exception as e:
            print(f"Error loading previous results: {e}")

    # Step 1: Fetch all live news
    df_new = fetch_all_sectors()
    if df_new.empty:
        print("Warning: No news fetched.")
        return previous_df # Return old results if fetch fails
    
    # Step 2: Remove duplicates
    df_fresh = remove_duplicate_articles(df_new, previous_df)
    
    if df_fresh.empty:
        print("⚠️ No new articles found since last fetch.")
        return previous_df

    # Step 3: Load model and vectorizer
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    
    # Step 4: Perform analysis on FRESH articles only
    df_fresh['combined_text'] = df_fresh['title'] + " " + df_fresh['description']
    processed_titles = df_fresh['title'].apply(preprocess_text_fast)
    tfidf_matrix = vectorizer.transform(processed_titles)
    
    df_fresh['predicted_sector'] = model.predict(tfidf_matrix)
    
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(tfidf_matrix)
        df_fresh['model_confidence'] = probs.max(axis=1) * 100
    else:
        df_fresh['model_confidence'] = 100.0
    
    analyzer = SentimentIntensityAnalyzer()
    df_fresh['vader_compound'] = df_fresh['combined_text'].apply(
        lambda x: analyzer.polarity_scores(x)['compound']
    )
    
    def get_sentiment_label(score):
        if score >= 0.05: return 'Positive'
        if score <= -0.05: return 'Negative'
        return 'Neutral'
    
    df_fresh['sentiment'] = df_fresh['vader_compound'].apply(get_sentiment_label)
    df_fresh['analyzed_at'] = pd.Timestamp.now()
    
    # Step 5: Combine and Save
    if not previous_df.empty:
        # Append fresh to old
        df_combined = pd.concat([df_fresh, previous_df]).reset_index(drop=True)
    else:
        df_combined = df_fresh
        
    df_combined.to_csv(LIVE_NEWS_RESULTS_CSV, index=False)
    # Cache also updated with combined
    df_combined.to_csv(LIVE_NEWS_CACHE_CSV, index=False)
    
    return df_combined # Return combined for the UI

def smart_fetch(force_refresh=False):
    """Fetch from cache if recent, else fetch fresh. force_refresh bypasses cache."""
    # If force_refresh=True → DELETE cache and fetch fresh
    if force_refresh:
        if os.path.exists(LIVE_NEWS_CACHE_CSV):
            os.remove(LIVE_NEWS_CACHE_CSV)
            print("🗑️ Cache deleted. Fetching fresh news...")
        return analyze_live_news()

    if os.path.exists(LIVE_NEWS_CACHE_CSV):
        try:
            cached = pd.read_csv(LIVE_NEWS_CACHE_CSV)
            if not cached.empty and 'analyzed_at' in cached.columns:
                last_fetch = pd.to_datetime(cached['analyzed_at'].max())
                time_diff = (pd.Timestamp.now() - last_fetch).total_seconds() / 60
                
                # If less than 60 minutes old
                if time_diff < 60:
                    print(f"Using cache ({time_diff:.1f} mins old)")
                    return cached
        except Exception as e:
            print(f"Error reading cache: {e}")
            
    print("Cache expired or missing. Fetching fresh news...")
    return analyze_live_news()

def get_live_market_summary(df):
    """Aggregate live news by sector."""
    if df.empty:
        return pd.DataFrame(), [], []
    
    # Group by Predicted Sector
    summary = df.groupby('predicted_sector')['sentiment'].value_counts(normalize=True).unstack(fill_value=0) * 100
    
    # Ensure all columns exist
    for col in ['Positive', 'Negative', 'Neutral']:
        if col not in summary.columns:
            summary[col] = 0.0
            
    # Add Trend column
    def determine_trend(row):
        if row['Positive'] > row['Negative'] + 10: return "Bullish 📈"
        if row['Negative'] > row['Positive'] + 10: return "Bearish 📉"
        return "Neutral ⚖️"
    
    summary['Trend'] = summary.apply(determine_trend, axis=1)
    summary = summary.reset_index().rename(columns={'predicted_sector': 'Sector'})
    
    # Sort for display
    summary = summary[['Sector', 'Positive', 'Negative', 'Neutral', 'Trend']]
    
    # Top Headlines
    top_posRaw = df[df['sentiment'] == 'Positive'].sort_values('vader_compound', ascending=False).head(3)
    top_negRaw = df[df['sentiment'] == 'Negative'].sort_values('vader_compound', ascending=True).head(3)
    
    return summary, top_posRaw, top_negRaw

if __name__ == "__main__":
    df = analyze_live_news()
    if not df.empty:
        summary, pos, neg = get_live_market_summary(df)
        print("\n--- Live Market Summary ---")
        print(summary)
