import os
import requests
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv("HENDRIK_API_KEY")  # Your HenrikDev API key
REGION = "na"  # Change if needed
NAME = "Baguette"  # Your Riot/Valorant name
TAG = "GRAlN"  # Your Riot tag
CSV_FILE = "matches.csv"

HEADERS = {
    "Authorization": API_KEY
}

def fetch_matches(region, name, tag):
    """Fetch all stored matches for a player."""
    url = f"https://api.henrikdev.xyz/valorant/v1/stored-matches/{region}/{name}/{tag}"
    matches = []
    page = 1

    while True:
        params = {"page": page, "size": 100}
        response = requests.get(url, headers=HEADERS, params=params)
        if response.status_code != 200:
            print(f"API Error: {response.status_code}, {response.json()}")
            break

        data = response.json().get("data", [])
        if not data:
            break

        matches.extend(data)
        if len(data) < params["size"]:
            break

        page += 1

    print(f"Fetched {len(matches)} matches")
    return matches


def process_match(match):
    """Process a single match and extract relevant statistics."""
    meta = match.get("meta", {})
    stats = match.get("stats", {})
    teams = match.get("teams", {})

    # Determine win/loss
    team_name = stats.get("team")
    win = None
    if team_name and teams:
        player_team_score = teams.get(team_name.lower())
        other_team_score = sum(v for k, v in teams.items() if k.lower() != team_name.lower())
        if player_team_score is not None:
            win = int(player_team_score > other_team_score)  # 1 for win, 0 for loss

    # Shots
    shots = stats.get("shots", {})
    headshots = shots.get("head", 0)
    bodyshots = shots.get("body", 0)
    legshots = shots.get("leg", 0)
    total_shots = headshots + bodyshots + legshots

    # Headshot percentage (0–100)
    hs_percent = (headshots / total_shots * 100) if total_shots > 0 else 0

    # KD ratio
    kills = stats.get("kills", 0)
    deaths = stats.get("deaths", 0)
    kd_ratio = kills / deaths if deaths not in (0, None) else kills

    processed = {
        "match_id": meta.get("id"),
        "map": meta.get("map", {}).get("name"),
        "game_start": meta.get("started_at"),
        "agent": stats.get("character", {}).get("name"),
        "kills": kills,
        "deaths": deaths,
        "assists": stats.get("assists"),
        "score": stats.get("score"),
        "shots_total": total_shots,
        "hs_percent": hs_percent,
        "kd_ratio": kd_ratio,
        "win": win
    }
    return processed


def main():
    """Main function to fetch, process, and save matches to CSV."""
    raw_matches = fetch_matches(REGION, NAME, TAG)
    if not raw_matches:
        print("No matches fetched.")
        return

    processed = [process_match(m) for m in raw_matches]
    df = pd.DataFrame(processed)
    df.to_csv(CSV_FILE, index=False)
    print(f"Saved {len(df)} matches to {CSV_FILE}")


if __name__ == "__main__":
    main()
