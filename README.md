# 📊 Financial News Analysis System (FNAS)

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

An end-to-end Machine Learning and Natural Language Processing (NLP) pipeline designed to classify financial news into industry sectors and analyze market sentiment in real-time.

---

## 🌟 Project Highlights

*   **🏢 Multi-Sector Classification:** Automatically categorizes news into **Banking, IT, Pharma, Energy, Automobile, and Financial Services**.
*   **😊 Sentiment Intelligence:** Leverages VADER for deep sentiment analysis (Positive, Negative, Neutral).
*   **📈 Market Trend Detection:** Synthesizes raw data into **Bullish, Bearish, and Stable** trend labels.
*   **🗺️ Interactive Dashboard:** Professional visualizations and a consolidated "Market Intelligence Dashboard."
*   **🔍 Live News Analyzer:** Real-time interface for instant sector and sentiment prediction of custom headlines.

---

## 🚀 System Architecture & Progress

The project was developed in 9 distinct phases:

1.  **✅ Phase 1: Data Collection:** Ingested the raw financial news dataset.
2.  **✅ Phase 2: Data Cleaning:** Handled null values, duplicates, and initial formatting.
3.  **✅ Phase 3: Preprocessing:** Implemented tokenization, lemmatization, and stopword removal.
4.  **✅ Phase 4: Feature Engineering:** Transformed text into numerical vectors using **TF-IDF**.
5.  **✅ Phase 5: Sector Classification:** Trained and compared **Logistic Regression** and **Naive Bayes** models (Accuracy ~85%).
6.  **✅ Phase 6: Sentiment Analysis:** Integrated **VADER** for compound sentiment scoring.
7.  **✅ Phase 7: Aggregation:** Combined sector and sentiment data to generate final market insights.
8.  **✅ Phase 8: Visualization:** Created 8 professional charts and a **Master Dashboard**.
9.  **✅ Phase 9: Streamlit UI:** Built a multi-page web application for user interaction.

---

## 📊 Dashboard Preview

![Master Dashboard](dashboard_master.png)

---

## 🛠️ Technologies Used

*   **Core:** Python, Pandas, NumPy
*   **NLP:** NLTK, spaCy, VADER Sentiment Analyzer
*   **Machine Learning:** scikit-learn (TF-IDF, Logistic Regression, Naive Bayes)
*   **Visualization:** Matplotlib, Seaborn
*   **User Interface:** Streamlit, Pillow
*   **Deployment Tools:** joblib (Model serialization)

---

## 📂 Project Structure

```bash
Financial-News-Market-Analysis/
Financial-News-Market-Analysis/
│
├── data/
│   ├── raw/                          # Original datasets (no changes)
│   ├── processed/                    # Cleaned + transformed data
│   │   ├── finance_sector_dataset.csv
│   │   └── preprocessed_finance_news.csv
│   └── outputs/                      # Generated datasets
│       ├── financial_news_with_sentiment.csv
│       ├── market_trends.csv
│       ├── sector_sentiment_summary.csv
│       ├── top_headlines_per_sector.csv
│       └── year_wise_sentiment.csv
│
├── models/                           # Saved ML models
│   ├── final_sector_model.pkl
│   ├── sector_classifier_nb.pkl
│   └── tfidf_vectorizer.pkl
│
├── notebooks/                        # Jupyter notebooks (experiments)
│   ├── data_collection.ipynb
│   ├── sentiment_analysis.ipynb
│   ├── sector_classification_model.ipynb
│   └── final_market_dashboard.ipynb
│
├── src/                              # Core source code (production logic)
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── sentiment_analysis.py
│   ├── sector_classification.py
│   ├── aggregation.py
│   └── visualization/
│       └── dashboard.py
│
├── reports/                          # Outputs for evaluation/demo
│   ├── charts/
│   │   ├── chart1_overall_sentiment_pie.png
│   │   ├── chart2_sector_sentiment_bar.png
│   │   ├── chart3_market_trends_horizontal.png
│   │   ├── chart4_compound_boxplot.png
│   │   ├── chart5_yearwise_trend_line.png
│   │   ├── chart6_stacked_bar_count.png
│   │   ├── chart7_heatmap_sector_sentiment.png
│   │   ├── chart8_top_headlines_table.png
│   │   └── dashboard_master.png
│   │
│   ├── logs/
│   │   ├── aggregation_report.txt
│   │   ├── sentiment_analysis_output.txt
│   │   ├── sector_classification_output.txt
│   │   └── sector_classification_output_utf8.txt
│   │
│   └── results/
│       ├── sector_classification_results.png
│       ├── sector_sentiment.png
│       └── sentiment_distribution.png
│
├── app/                              # App / dashboard entry
│   └── app.py
│
├── config/                           # Config files (future scaling)
│   └── config.yaml
│
├── tests/                            # (Optional but strong for resume)
│   └── test_pipeline.py
│
├── .gitignore
├── README.md
└── requirements.txt

---

## 📦 Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/manaspatil1406/Financial-News-Market-Analysis.git
cd Financial-News-Market-Analysis
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

---

## ℹ️ About the Project

Developed as a comprehensive solution for financial market surveillance, this tool helps investors and analysts quickly grasp market sentiment and trends across various sectors using data-driven insights.

**Author:** Rajesh Patil  
**Version:** 1.0.0
