import os

# Project root directory (the parent of src/)
# This makes the project OS-independent and portable.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data Directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(DATA_DIR, "outputs")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

# Models Directory
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Reports Directory
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")
CHARTS_DIR = os.path.join(REPORTS_DIR, "charts")

# Model Paths
MODEL_PATH = os.path.join(MODELS_DIR, "final_sector_model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")

# Data Output Paths
MAIN_DATA_CSV = os.path.join(OUTPUTS_DIR, "financial_news_with_sentiment.csv")
SUMMARY_CSV = os.path.join(OUTPUTS_DIR, "sector_sentiment_summary.csv")
TRENDS_CSV = os.path.join(OUTPUTS_DIR, "market_trends.csv")
YEAR_TREND_CSV = os.path.join(OUTPUTS_DIR, "year_wise_sentiment.csv")
HEADLINES_CSV = os.path.join(OUTPUTS_DIR, "top_headlines_per_sector.csv")
LIVE_NEWS_RESULTS_CSV = os.path.join(OUTPUTS_DIR, "live_news_results.csv")
LIVE_NEWS_CACHE_CSV = os.path.join(OUTPUTS_DIR, "live_news_cache.csv")

# Chart Paths
CHART_FILES = {
    "chart1": os.path.join(CHARTS_DIR, "chart1_overall_sentiment_pie.png"),
    "chart2": os.path.join(CHARTS_DIR, "chart2_sector_sentiment_bar.png"),
    "chart3": os.path.join(CHARTS_DIR, "chart3_market_trends_horizontal.png"),
    "chart4": os.path.join(CHARTS_DIR, "chart4_compound_boxplot.png"),
    "chart5": os.path.join(CHARTS_DIR, "chart5_yearwise_trend_line.png"),
    "chart6": os.path.join(CHARTS_DIR, "chart6_stacked_bar_count.png"),
    "chart7": os.path.join(CHARTS_DIR, "chart7_heatmap_sector_sentiment.png"),
    "chart8": os.path.join(CHARTS_DIR, "chart8_top_headlines_table.png"),
    "dashboard": os.path.join(CHARTS_DIR, "dashboard_master.png"),
}

# Ensure directories exist
for d in [OUTPUTS_DIR, CHARTS_DIR]:
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
