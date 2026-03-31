import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
import os
import sys

# Set paths
INPUT_PATH = r"c:\Users\Rajesh Patil\Financial-News-Market-Analysis\data\processed\preprocessed_finance_news.csv"
OUTPUT_PATH = r"c:\Users\Rajesh Patil\Financial-News-Market-Analysis\financial_news_with_sentiment.csv"
PROJECT_DIR = r"c:\Users\Rajesh Patil\Financial-News-Market-Analysis"

# Force UTF-8 encoding for stdout to avoid Windows charmap errors
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def get_sentiment_label(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

def main():
    # Step 1: Install & Import VADER
    analyzer = SentimentIntensityAnalyzer()

    # Step 2: Load the Dataset
    print(f"Loading dataset from: {INPUT_PATH}")
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} not found!")
        return
    
    # Load data
    df = pd.read_csv(INPUT_PATH)
    print(f"Dataset shape: {df.shape}")
    
    # Step 3 & 4: Apply VADER on Raw Text (headline)
    print("\nApplying VADER sentiment analysis on 'headline' column...")
    # Using 'headline' as requested. VADER works best on raw text.
    # We use .astype(str) to handle any potential non-string values
    df['vader_compound'] = df['headline'].astype(str).apply(lambda x: analyzer.polarity_scores(x)['compound'])
    df['sentiment'] = df['vader_compound'].apply(get_sentiment_label)

    # Step 5: Check Sentiment Distribution
    print("\n--- Sentiment Distribution ---")
    dist = df['sentiment'].value_counts()
    print(dist)
    
    # Save pie chart
    plt.figure(figsize=(8, 8))
    # Use standard colors and ensure labels are strings
    plt.pie(dist, labels=dist.index.astype(str), autopct='%1.1f%%', colors=['#4CAF50', '#F44336', '#FFC107'], startangle=140)
    plt.title('Overall Sentiment Distribution')
    plt.savefig(os.path.join(PROJECT_DIR, 'phase6_sentiment_distribution.png'))
    print(f"Saved: phase6_sentiment_distribution.png")

    # Step 6: Sector-wise Sentiment Breakdown
    print("\n--- Sector-wise Sentiment Breakdown ---")
    # Group by sector and sentiment
    sector_sentiment = df.groupby(['sector', 'sentiment']).size().unstack(fill_value=0)
    
    # Convert to percentages
    sector_sentiment_pct = sector_sentiment.div(sector_sentiment.sum(axis=1), axis=0) * 100
    
    print("Percentage Breakdown per Sector:")
    print(sector_sentiment_pct.round(2))
    
    # Save sector-wise bar chart
    plt.figure(figsize=(12, 6))
    sector_sentiment_pct.plot(kind='bar', stacked=True, color=['#F44336', '#FFC107', '#4CAF50'], ax=plt.gca())
    plt.title('Sentiment Distribution by Sector')
    plt.ylabel('Percentage (%)')
    plt.xlabel('Sector')
    plt.legend(title='Sentiment')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, 'phase6_sector_sentiment.png'))
    print(f"Saved: phase6_sector_sentiment.png")

    # Step 7: Test with Custom Headlines
    print("\n--- Testing Custom Headlines ---")
    test_headlines = [
        "Infosys launches new AI platform with record profits",
        "RBI cuts interest rates boosting market confidence",
        "Oil prices crash causing massive losses",
        "Tesla reports disappointing quarterly earnings",
        "Sun Pharma gets breakthrough FDA approval"
    ]
    
    for h in test_headlines:
        score = analyzer.polarity_scores(h)['compound']
        label = get_sentiment_label(score)
        print(f"Headline   : {h}")
        print(f"Compound   : {score:.2f}")
        print(f"Sentiment  : {label}")
        print("-" * 30)

    # Step 8: Save the Updated Dataset
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
