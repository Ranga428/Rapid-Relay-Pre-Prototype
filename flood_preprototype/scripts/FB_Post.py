import requests
from dotenv import load_dotenv
import os

# Load .env from two levels up
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"))

PAGE_ID = os.getenv("FB_PAGE_ID")
TOKEN = os.getenv("FB_PAGE_TOKEN")

res = requests.post(
    f"https://graph.facebook.com/v23.0/{PAGE_ID}/feed",
    data={
        "message": "🚨 TEST: Rapid Relay flood alert system online.",
        "access_token": TOKEN
    }
)
print(res.json())
print("This new long-lived access token will expire on May 21, 2026")