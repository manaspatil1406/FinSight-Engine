import pandas as pd # type: ignore
import os
import sys

# Add current directory to path so config can be imported when running from src/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import MAIN_DATA_CSV, SUMMARY_CSV, TRENDS_CSV, YEAR_TREND_CSV, HEADLINES_CSV

# Force UTF-8 encoding for stdout
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def main():
    # Step 1: Load the Dataset
    print(f"Loading dataset from: {MAIN_DATA_CSV}")
    if not os.path.exists(MAIN_DATA_CSV):
        print(f"Error: {MAIN_DATA_CSV} not found!")
        return
    
    df = pd.read_csv(MAIN_DATA_CSV)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Step 2: Sector-wise Sentiment Count Table
    print("\n--- Generating Sector-wise Sentiment Count ---")
    # Using 'sector' as the primary sector identifier
    sector_sentiment_counts = df.groupby(['sector', 'sentiment']).size().unstack(fill_value=0)
    sector_sentiment_counts['Total'] = sector_sentiment_counts.sum(axis=1)
    
    print("Sector Sentiment Counts:")
    print(sector_sentiment_counts)

    # Step 3: Sector-wise Sentiment Percentage Table
    print("\n--- Generating Sector-wise Sentiment Percentages ---")
    sector_sentiment_pct = sector_sentiment_counts.drop(columns='Total').div(sector_sentiment_counts['Total'], axis=0) * 100
    sector_sentiment_pct = sector_sentiment_pct.round(2)
    
    # Save Step 3 as sector_sentiment_summary.csv
    sector_sentiment_pct.to_csv(SUMMARY_CSV)
    print(f"✅ Saved: {SUMMARY_CSV}")

    # Step 4: Market Trend Labeling
    print("\n--- Identifying Market Trends ---")
    def get_trend_label(row):
        max_val = row.max()
        if row['Positive'] == max_val:
            return "Bullish 📈"
        elif row['Negative'] == max_val:
            return "Bearish 📉"
        else:
            return "Stable ➡️"

    # We only care about Positive, Negative, Neutral for the trend
    trends_df = sector_sentiment_pct[['Positive', 'Negative', 'Neutral']].copy()
    trends_df['Trend'] = trends_df.apply(get_trend_label, axis=1)
    
    # Save Step 4 as market_trends.csv
    market_trends_only = trends_df[['Trend']].reset_index().rename(columns={'sector': 'Sector'})
    market_trends_only.to_csv(TRENDS_CSV, index=False)
    print(f"✅ Saved: {TRENDS_CSV}")

    # Step 5: Average Compound Score per Sector
    print("\n--- Calculating Average Compound Score per Sector ---")
    avg_compound = df.groupby('sector')['vader_compound'].mean().round(2).reset_index().rename(columns={'sector': 'Sector', 'vader_compound': 'Avg Compound Score'})
    print(avg_compound)

    # Step 6: Year-wise Sentiment Trend
    print("\n--- Year-wise Sentiment Trend ---")
    # Both 'date' and 'year' exist. Let's use 'year' if available and clean.
    if 'year' in df.columns:
        year_sentiment = df.groupby(['year', 'sentiment']).size().unstack(fill_value=0)
        year_sentiment_pct = year_sentiment.div(year_sentiment.sum(axis=1), axis=0) * 100
        year_sentiment_pct = year_sentiment_pct.round(2)
        
        # Save Step 6 as year_wise_sentiment.csv
        year_sentiment_pct.to_csv(YEAR_TREND_CSV)
        print(f"✅ Saved: {YEAR_TREND_CSV}")
        print("Year-wise Trends:")
        print(year_sentiment_pct)
    else:
        print("⚠️ No Year/Date column found. Skipping Step 6.")

    # Step 7: Top Bullish & Bearish Headlines per Sector
    print("\n--- Extracting Top 3 Headlines per Sector ---")
    top_headlines_list = []
    
    for sector in df['sector'].unique():
        sector_df = df[df['sector'] == sector]
        
        # Top 3 Positive
        top_pos = sector_df.sort_values(by='vader_compound', ascending=False).head(3)
        for _, row in top_pos.iterrows():
            top_headlines_list.append({
                'Sector': sector, 
                'Type': 'Positive', 
                'Headline': row['headline'], 
                'Score': round(row['vader_compound'], 2)
            })
            
        # Top 3 Negative
        top_neg = sector_df.sort_values(by='vader_compound', ascending=True).head(3)
        for _, row in top_neg.iterrows():
            top_headlines_list.append({
                'Sector': sector, 
                'Type': 'Negative', 
                'Headline': row['headline'], 
                'Score': round(row['vader_compound'], 2)
            })

    top_headlines_df = pd.DataFrame(top_headlines_list)
    top_headlines_df.to_csv(HEADLINES_CSV, index=False)
    print(f"✅ Saved: {HEADLINES_CSV}")

    # Step 8 & 9: Final Market Intelligence Summary
    print("\n" + "=" * 50)
    print("   📊 FINANCIAL MARKET INTELLIGENCE REPORT")
    print("=" * 50)
    
    report_lines = []
    report_lines.append("=" * 50)
    report_lines.append("   📊 FINANCIAL MARKET INTELLIGENCE REPORT")
    report_lines.append("=" * 50)
    
    total_articles = len(df)
    sectors_list = ", ".join(df['sector'].unique())
    
    line = f"Total Articles Analyzed : {total_articles}"
    print(line)
    report_lines.append(line)
    
    line = f"Sectors Covered         : {sectors_list}"
    print(line)
    report_lines.append(line)
    
    print("\nSECTOR TRENDS:")
    report_lines.append("\nSECTOR TRENDS:")
    
    # Trend Summary
    for idx, row in trends_df.iterrows():
        pos_pct = row['Positive']
        trend = row['Trend']
        line = f"  {idx:<10} → {trend:<12} ({pos_pct}% Positive)"
        print(line)
        report_lines.append(line)
        
    print("\nKEY INSIGHT:")
    report_lines.append("\nKEY INSIGHT:")
    
    # Simple insights based on Bullish count
    bullish_sectors = trends_df[trends_df['Trend'] == 'Bullish 📈'].index.tolist()
    bearish_sectors = trends_df[trends_df['Trend'] == 'Bearish 📉'].index.tolist()
    
    if bullish_sectors:
        line = f"  → {', '.join(bullish_sectors)} showing positive momentum"
        print(line)
        report_lines.append(line)
    if bearish_sectors:
        line = f"  → {', '.join(bearish_sectors)} under pressure"
        print(line)
        report_lines.append(line)
        
    print("=" * 50)
    report_lines.append("=" * 50)
    
    # Save the report as aggregation_report.txt
    import config # Import inside to avoid circular check if needed, but project root is what we want
    REPORT_PATH = os.path.join(config.REPORTS_DIR, 'aggregation_report.txt')
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    print(f"✅ Saved: {REPORT_PATH}")

if __name__ == "__main__":
    main()
