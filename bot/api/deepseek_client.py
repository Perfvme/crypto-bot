import requests
import os
from dotenv import load_dotenv

load_dotenv()

class DeepseekClient:
    def __init__(self):
        self.base_url = "https://api.deepseek.com/v1"
        self.headers = {
            "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",
            "Content-Type": "application/json"
        }
    
    def analyze_market(self, data):
        response = requests.post(
            f"{self.base_url}/analyze",
            headers=self.headers,
            json={"market_data": data}
        )
        return response.json().get('analysis', {})
