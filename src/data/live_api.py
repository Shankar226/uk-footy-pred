import os, requests, pandas as pd
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
COMP = os.getenv("COMPETITION", "PL")
BASE = "https://api.football-data.org/v4"

def get_scheduled_fixtures():
    if not API_KEY:
        print("No FOOTBALL_DATA_API_KEY; skipping live fixtures.")
        return pd.DataFrame()
    headers = {"X-Auth-Token": API_KEY}
    url = f"{BASE}/competitions/{COMP}/matches?status=SCHEDULED"
    r = requests.get(url, headers=headers, timeout=20)
    r.raise_for_status()
    j = r.json()
    rows=[]
    for m in j.get("matches", []):
        rows.append({
            "utcDate": m["utcDate"],
            "home": m["homeTeam"]["name"],
            "away": m["awayTeam"]["name"],
            "matchday": m.get("matchday"),
            "id": m["id"]
        })
    return pd.DataFrame(rows)
