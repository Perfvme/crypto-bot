import os
import requests
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from textblob import TextBlob
from typing import Dict, List

load_dotenv()

class NewsAnalyzer:
    def __init__(self):
        self.api_key = os.getenv('NEWSAPI_KEY')
        self.cache = {}
        
    def get_coin_news(self, coin: str) -> Dict:
        if coin in self.cache:
            return self.cache[coin]
            
        url = "https://newsapi.org/v2/everything"
        params = {
            'q': f"{coin} cryptocurrency",
            'from': (datetime.now() - timedelta(days=1)).isoformat(),
            'sortBy': 'relevancy',
            'apiKey': self.api_key,
            'pageSize': 5
        }
        
        try:
            response = requests.get(url, params=params)
            articles = response.json().get('articles', [])
            analyzed = self._analyze_articles(articles)
            self.cache[coin] = analyzed
            return analyzed
        except Exception as e:
            return {
                'summary': "⚠️ News unavailable",
                'sentiment': 'neutral',
                'sentiment_score': 0.0,
                'articles': []
            }

    def _analyze_articles(self, articles: List[Dict]) -> Dict:
        sentiments = []
        summaries = []
        
        for article in articles[:3]:  # Analyze top 3 articles
            text = f"{article['title']}. {article['description']}"
            analysis = TextBlob(text)
            sentiments.append(analysis.sentiment.polarity)
            summaries.append(f"- {article['title']} ({article['source']['name']})")
            
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
        return {
            'summary': '\n'.join(summaries),
            'sentiment': self._sentiment_label(avg_sentiment),
            'sentiment_score': avg_sentiment,
            'articles': articles
        }

    def _sentiment_label(self, score: float) -> str:
        if score > 0.2:
            return 'bullish'
        elif score < -0.2:
            return 'bearish'
        return 'neutral'

    def refresh_cache(self):
        self.cache.clear()
