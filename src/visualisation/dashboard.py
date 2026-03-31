import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import matplotlib.patches as mpatches # type: ignore
import seaborn as sns # type: ignore
import numpy as np # type: ignore
import warnings
import os
import sys

warnings.filterwarnings('ignore')

# Set global style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 12

# Add parent directory to path so config can be imported when running from src/visualisation/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MAIN_DATA_CSV, SUMMARY_CSV, TRENDS_CSV, YEAR_TREND_CSV, HEADLINES_CSV, CHART_FILES

# Project paths
DATA_FILE = MAIN_DATA_CSV
SUMMARY_FILE = SUMMARY_CSV
TRENDS_FILE = TRENDS_CSV
YEAR_TREND_FILE = YEAR_TREND_CSV
HEADLINES_FILE = HEADLINES_CSV

# Colors
COLOR_POS = '#4CAF50' # Green
COLOR_NEG = '#F44336' # Red
COLOR_NEU = '#9E9E9E' # Gray
COLOR_STA = '#FF9800' # Orange (for trend labels)

def create_overall_sentiment_pie(df):
    print("Creating Chart 1: Overall Sentiment Pie...")
    dist = df['sentiment'].value_counts()
    
    plt.figure(figsize=(10, 8))
    # Safety: Ensure explode matches the number of segments
    explode = [0.05] + [0] * (len(dist) - 1)
    
    plt.pie(dist, labels=dist.index, autopct='%1.1f%%', 
            colors=[COLOR_POS, COLOR_NEG, COLOR_NEU][:len(dist)], 
            explode=explode, startangle=140, shadow=True)
    plt.title("Overall Sentiment Distribution in Financial News", weight='bold')
    plt.tight_layout()
    plt.savefig(CHART_FILES['chart1'])
    plt.close()

def create_sector_sentiment_bar(summary_df):
    print("Creating Chart 2: Sector-wise Sentiment Bar Chart...")
    # Plot grouped bar chart
    ax = summary_df.plot(kind='bar', figsize=(12, 7), 
                         color=[COLOR_NEG, COLOR_NEU, COLOR_POS], # Negative, Neutral, Positive order
                         width=0.8)
    
    # Add value labels
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=10, 
                    color='black', xytext=(0, 5), 
                    textcoords='offset points')
    
    plt.title("Sector-wise Sentiment Distribution", weight='bold')
    plt.xlabel("Sectors")
    plt.ylabel("Percentage (%)")
    plt.legend(title='Sentiment', loc='upper right')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(CHART_FILES['chart2'])
    plt.close()

def create_market_trends_horizontal(trends_df, summary_df):
    print("Creating Chart 3: Market Trend Horizontal Bar Chart...")
    # Ensure Case Consistency
    trends_df.columns = [c.capitalize() for c in trends_df.columns]
    summary_df.index = summary_df.index.str.capitalize() if hasattr(summary_df.index, 'str') else summary_df.index
    
    # Merge trends with positive percentage
    merged = trends_df.merge(summary_df[['Positive']], left_on='Sector', right_index=True)
    
    plt.figure(figsize=(12, 7))
    # Map colors based on Trend
    trend_color_map = {
        'Bullish 📈': COLOR_POS,
        'Bearish 📉': COLOR_NEG,
        'Stable ➡️': COLOR_STA
    }
    colors = merged['Trend'].map(trend_color_map)
    
    bars = plt.barh(merged['Sector'], merged['Positive'], color=colors)
    
    # Add labels
    for bar, trend in zip(bars, merged['Trend']):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                 trend, va='center', weight='bold')
        
    plt.title("Market Trend by Sector", weight='bold')
    plt.xlabel("Positive Sentiment (%)")
    plt.ylabel("Sectors")
    plt.xlim(0, max(merged['Positive']) + 15)
    plt.tight_layout()
    plt.savefig(CHART_FILES['chart3'])
    plt.close()

def create_compound_boxplot(df):
    print("Creating Chart 4: Compound Score Distribution per Sector (Box Plot)...")
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='sector', y='vader_compound', data=df, palette='Set2')
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.title("Sentiment Compound Score Distribution by Sector", weight='bold')
    plt.xlabel("Sectors")
    plt.ylabel("VADER Compound Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(CHART_FILES['chart4'])
    plt.close()

def create_yearwise_trend_line(year_trend_df):
    print("Creating Chart 5: Year-wise Sentiment Trend (Line Chart)...")
    plt.figure(figsize=(12, 7))
    # Check column order
    cols = year_trend_df.columns.tolist()
    plt.plot(year_trend_df.index, year_trend_df['Positive'], marker='o', color=COLOR_POS, label='Positive', linewidth=2.5)
    plt.plot(year_trend_df.index, year_trend_df['Negative'], marker='o', color=COLOR_NEG, label='Negative', linewidth=2.5)
    plt.plot(year_trend_df.index, year_trend_df['Neutral'], marker='o', color=COLOR_NEU, label='Neutral', linewidth=2.5)
    
    # Special annotation for 2020
    if 2020 in year_trend_df.index:
        y_val = year_trend_df.loc[2020, 'Negative']
        plt.annotate('2020 COVID Impact → High Negative', 
                     xy=(2020, y_val), xytext=(2019, y_val + 10),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                     fontsize=10, weight='bold', color=COLOR_NEG)
        
    plt.title("Year-wise Sentiment Trend in Financial News", weight='bold')
    plt.xlabel("Year")
    plt.ylabel("Percentage (%)")
    plt.legend(title='Sentiment')
    plt.tight_layout()
    plt.savefig(CHART_FILES['chart5'])
    plt.close()

def create_stacked_bar_count(df):
    print("Creating Chart 6: Sentiment Count per Sector (Stacked Bar Chart)...")
    count_df = df.groupby(['sector', 'sentiment']).size().unstack(fill_value=0)
    
    ax = count_df.plot(kind='bar', stacked=True, figsize=(14, 8), 
                       color=[COLOR_NEG, COLOR_NEU, COLOR_POS])
    
    # Add total count label on top
    for i, (idx, row) in enumerate(count_df.iterrows()):
        total = row.sum()
        ax.text(i, total + 10, f"{int(total)}", ha='center', weight='bold')
        
    plt.title("Article Count by Sector and Sentiment", weight='bold')
    plt.xlabel("Sectors")
    plt.ylabel("Number of Articles")
    plt.legend(title='Sentiment')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(CHART_FILES['chart6'])
    plt.close()

def create_heatmap(summary_df):
    print("Creating Chart 7: Heatmap (Sector vs Sentiment %)...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(summary_df, annot=True, fmt=".1f", cmap='RdYlGn', 
                cbar_kws={'label': 'Percentage (%)'}, 
                linewidths=.5)
    plt.title("Sector vs Sentiment Heatmap", weight='bold')
    plt.xlabel("Sentiment")
    plt.ylabel("Sector")
    plt.tight_layout()
    plt.savefig(CHART_FILES['chart7'])
    plt.close()

def create_top_headlines_table(headlines_df):
    print("Creating Chart 8: Top Positive & Negative Headlines (Table)...")
    # Take top 5 pos and top 5 neg as per instructions
    top_pos = headlines_df[headlines_df['Type'] == 'Positive'].sort_values(by='Score', ascending=False).head(5)
    top_neg = headlines_df[headlines_df['Type'] == 'Negative'].sort_values(by='Score', ascending=True).head(5)
    
    combined = pd.concat([top_pos, top_neg])
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')
    
    table_data = []
    # Add headers
    headers = ["Sector", "Type", "Score", "Headline"]
    table_data.append(headers)
    
    for _, row in combined.iterrows():
        # Truncate long headlines for better display
        h = (row['Headline'][:75] + '...') if len(row['Headline']) > 75 else row['Headline']
        table_data.append([row['Sector'], row['Type'], row['Score'], h])
        
    table = ax.table(cellText=table_data, loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Styling
    for i in range(len(table_data)):
        for j in range(len(headers)):
            cell = table[(i, j)]
            if i == 0: # Header
                cell.set_facecolor('#DDDDDD')
                cell.get_text().set_weight('bold')
            elif 1 <= i <= 5: # Positive
                cell.set_facecolor('#E8F5E9')
            elif 6 <= i <= 10: # Negative
                cell.set_facecolor('#FFEBEE')
                
    plt.title("Top Positive & Negative Financial Headlines", weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(CHART_FILES['chart8'])
    plt.close()

def create_master_dashboard(df, summary_df, trends_df):
    print("Creating Master Dashboard...")
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Subplot 1: Pie Chart
    dist = df['sentiment'].value_counts()
    axes[0, 0].pie(dist, labels=dist.index, autopct='%1.1f%%', 
                  colors=[COLOR_POS, COLOR_NEG, COLOR_NEU], 
                  startangle=140)
    axes[0, 0].set_title("Overall Sentiment Distribution", weight='bold')
    
    # Subplot 2: Grouped Bar
    summary_df.plot(kind='bar', ax=axes[0, 1], 
                    color=[COLOR_NEG, COLOR_NEU, COLOR_POS], 
                    width=0.8)
    axes[0, 1].set_title("Sector-wise Sentiment Distribution (%)", weight='bold')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Subplot 3: Horizontal Bar (Market Trends)
    merged = trends_df.merge(summary_df[['Positive']], left_on='Sector', right_index=True)
    trend_color_map = {'Bullish 📈': COLOR_POS, 'Bearish 📉': COLOR_NEG, 'Stable ➡️': COLOR_STA}
    colors = merged['Trend'].map(trend_color_map)
    bars = axes[1, 0].barh(merged['Sector'], merged['Positive'], color=colors)
    for bar, trend in zip(bars, merged['Trend']):
        axes[1, 0].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, trend, va='center', weight='bold')
    axes[1, 0].set_title("Market Trends by Sector", weight='bold')
    axes[1, 0].set_xlabel("Positive Sentiment (%)")
    
    # Subplot 4: Heatmap
    sns.heatmap(summary_df, annot=True, fmt=".1f", cmap='RdYlGn', ax=axes[1, 1])
    axes[1, 1].set_title("Sector vs Sentiment Heatmap (%)", weight='bold')
    
    plt.suptitle("Financial Market Intelligence Dashboard", fontsize=24, weight='bold', y=0.95)
    plt.savefig(CHART_FILES['dashboard'], dpi=300)
    plt.close()

def main():
    # Load data files
    try:
        df = pd.read_csv(DATA_FILE)
        
        if os.path.exists(SUMMARY_FILE):
            summary_df = pd.read_csv(SUMMARY_FILE, index_col=0)
        else:
            print(f"Warning: {SUMMARY_FILE} not found. Generating summary from data.")
            summary_df = df.groupby(['sector', 'sentiment']).size().unstack(fill_value=0)
            summary_df = summary_df.div(summary_df.sum(axis=1), axis=0) * 100
            
        trends_df = pd.read_csv(TRENDS_FILE) if os.path.exists(TRENDS_FILE) else None
        headlines_df = pd.read_csv(HEADLINES_FILE) if os.path.exists(HEADLINES_FILE) else None
        
        if trends_df is None or headlines_df is None:
            print("Error: Required trend/headline CSV files are missing. Please run the aggregation script first.")
            return
        
        # Create charts
        create_overall_sentiment_pie(df)
        create_sector_sentiment_bar(summary_df)
        create_market_trends_horizontal(trends_df, summary_df)
        create_compound_boxplot(df)
        
        if os.path.exists(YEAR_TREND_FILE):
            year_trend_df = pd.read_csv(YEAR_TREND_FILE, index_col=0)
            create_yearwise_trend_line(year_trend_df)
        
        create_stacked_bar_count(df)
        create_heatmap(summary_df)
        create_top_headlines_table(headlines_df)
        
        # Create Master Dashboard
        create_master_dashboard(df, summary_df, trends_df)
        
        print("\n" + "="*50)
        print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
        print("="*50)
        
    except Exception as e:
        print(f"❌ ERROR: {e}")

if __name__ == "__main__":
    main()
