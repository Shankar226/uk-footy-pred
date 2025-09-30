import os, requests
from dotenv import load_dotenv

# Load .env file
load_dotenv()

API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
COMP = os.getenv("COMPETITION", "PL")  # default Premier League

if not API_KEY:
    raise ValueError("API key not found. Check your .env file.")

url = f"https://api.football-data.org/v4/competitions/{COMP}/matches?status=SCHEDULED"

headers = {"X-Auth-Token": API_KEY}
resp = requests.get(url, headers=headers, timeout=20)

if resp.status_code == 200:
    data = resp.json()
    matches = data.get("matches", [])
    print(f"✅ API is working. Found {len(matches)} scheduled matches in {COMP}.")
    for m in matches[:5]:  # print first 5 fixtures
        print(f"{m['utcDate']}: {m['homeTeam']['name']} vs {m['awayTeam']['name']}")
else:
    print(f"❌ API call failed: {resp.status_code} {resp.text}")

