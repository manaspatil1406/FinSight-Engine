import pandas as pd
from newsapi import NewsApiClient
import feedparser
import requests
import os
import sys

# Add src to path if needed for config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import PROJECT_ROOT

# --- CONFIGURATION ---
# Replace with your actual NewsAPI key or set as environment variable
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "YOUR_API_KEY_HERE")

# Sector-specific queries for NewsAPI
SECTOR_QUERIES = {
    'IT'         : 'Infosys OR TCS OR Wipro OR tech stocks OR "artificial intelligence"',
    'Banking'    : 'RBI OR HDFC OR "interest rates" OR "monetary policy"',
    'Pharma'     : 'Sun Pharma OR Dr Reddy OR "FDA approval" OR pharmaceutical',
    'Energy'     : 'oil prices OR OPEC OR Reliance OR "renewable energy"',
    'Automobile' : 'Tesla OR Tata Motors OR "EV sales" OR automobile'
}

# RSS feeds for backup
RSS_FEEDS = {
    'IT'         : 'https://feeds.finance.yahoo.com/rss/2.0/headline?s=INFY&region=US&lang=en-US',
    'Banking'    : 'https://feeds.finance.yahoo.com/rss/2.0/headline?s=HDFCBANK.NS',
    'Pharma'     : 'https://feeds.finance.yahoo.com/rss/2.0/headline?s=SUNPHARMA.NS',
    'Energy'     : 'https://feeds.finance.yahoo.com/rss/2.0/headline?s=RELIANCE.NS',
    'Automobile' : 'https://feeds.finance.yahoo.com/rss/2.0/headline?s=TATAMOTORS.NS'
}

from datetime import datetime, timedelta
import time

def fetch_newsapi(sector, query, max_articles=20):
    """Fetch latest news for a sector using NewsAPI."""
    if NEWS_API_KEY == "YOUR_API_KEY_HERE":
        print(f"Warning: NEWS_API_KEY is not set. Skipping NewsAPI for {sector}.")
        return []
    
    try:
        # Always fetch news from last 24 hours only
        from_date = (datetime.now() - timedelta(hours=24)).strftime('%Y-%m-%dT%H:%M:%S')
        to_date   = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        response = newsapi.get_everything(
            q=query,
            language='en',
            sort_by='publishedAt',
            from_param=from_date,
            to=to_date,
            page_size=max_articles
        )
        
        articles = []
        if response['status'] == 'ok':
            for art in response['articles']:
                articles.append({
                    'title': art['title'],
                    'description': art['description'] if art['description'] else "",
                    'publishedAt': art['publishedAt'],
                    'source': art['source']['name'],
                    'url': art['url'],
                    'sector': sector,
                    'fetched_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
        return articles
    except Exception as e:
        print(f"Error fetching NewsAPI for {sector}: {e}")
        return []

def fetch_rss(sector, url):
    """Fetch latest news for a sector using RSS feeds with cache-busting."""
    try:
        # Add cache-busting parameter to URL
        cache_bust_url = f"{url}&_t={int(time.time())}"
        
        feed = feedparser.parse(cache_bust_url)
        articles = []
        for entry in feed.entries:
            # Yahoo RSS usually has 'summary' or 'description'
            description = entry.summary if 'summary' in entry else (entry.description if 'description' in entry else "")
            articles.append({
                'title': entry.title,
                'description': description,
                'publishedAt': entry.published if 'published' in entry else "",
                'source': f'Yahoo Finance - {sector}',
                'url': entry.link,
                'sector': sector,
                'fetched_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        return articles
    except Exception as e:
        print(f"Error fetching RSS for {sector}: {e}")
        return []

def fetch_all_sectors():
    """Main function to fetch news for all sectors."""
    all_articles = []
    
    for sector in SECTOR_QUERIES.keys():
        print(f"Fetching news for sector: {sector}...")
        
        # Try NewsAPI first
        articles = fetch_newsapi(sector, SECTOR_QUERIES[sector])
        
        # If NewsAPI failed or returned nothing (e.g. no key), try RSS
        if not articles:
            print(f"No NewsAPI results for {sector}, falling back to RSS.")
            articles = fetch_rss(sector, RSS_FEEDS.get(sector, ""))
        
        all_articles.extend(articles)
    
    df = pd.DataFrame(all_articles)
    return df

if __name__ == "__main__":
    df = fetch_all_sectors()
    print(f"Total articles fetched: {len(df)}")
    if not df.empty:
        print(df.head())
