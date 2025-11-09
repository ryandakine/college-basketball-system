#!/usr/bin/env python3
"""
Real Betting Lines Fetcher
Fetches REAL odds from actual sportsbooks

Uses The Odds API (free tier: 500 requests/month)
https://the-odds-api.com/
"""

import requests
import logging
from datetime import datetime
from typing import Dict, List, Optional
import json
import sqlite3
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealOddsFetcher:
    """
    Fetches real betting odds from actual sportsbooks

    FREE tier: 500 requests/month
    Sportsbooks included: FanDuel, DraftKings, BetMGM, etc.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize odds fetcher

        Args:
            api_key: The Odds API key (get free at the-odds-api.com)
        """
        self.api_key = api_key or os.getenv('ODDS_API_KEY')

        if not self.api_key:
            logger.warning("No ODDS_API_KEY found. Set in .env file")
            logger.warning("Get free key at: https://the-odds-api.com/")

        self.base_url = "https://api.the-odds-api.com/v4"
        self.sport = "basketball_ncaab"  # NCAA Basketball

        self.session = requests.Session()

    def fetch_upcoming_games(self, regions: str = "us") -> List[Dict]:
        """
        Fetch upcoming games with real odds

        Args:
            regions: us, uk, eu, au (default: us)

        Returns:
            List of games with odds from multiple sportsbooks
        """
        if not self.api_key:
            logger.error("API key required!")
            return []

        url = f"{self.base_url}/sports/{self.sport}/odds/"

        params = {
            'apiKey': self.api_key,
            'regions': regions,
            'markets': 'h2h,spreads,totals',  # Moneyline, spreads, totals
            'oddsFormat': 'american',
            'dateFormat': 'iso'
        }

        try:
            logger.info("Fetching real odds from sportsbooks...")

            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()

            games = response.json()

            # Check remaining quota
            if 'x-requests-remaining' in response.headers:
                remaining = response.headers['x-requests-remaining']
                logger.info(f"API requests remaining: {remaining}")

            logger.info(f"✅ Found {len(games)} upcoming games with real odds")

            return games

        except requests.RequestException as e:
            logger.error(f"Error fetching odds: {e}")
            return []

    def parse_game_odds(self, game: Dict) -> Dict:
        """
        Parse game data into usable format

        Args:
            game: Raw game data from API

        Returns:
            Parsed game dictionary
        """
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')
        commence_time = game.get('commence_time', '')

        # Get odds from first available bookmaker (usually FanDuel)
        bookmakers = game.get('bookmakers', [])

        if not bookmakers:
            return {
                'game_id': game.get('id', ''),
                'home_team': home_team,
                'away_team': away_team,
                'commence_time': commence_time,
                'bookmaker': 'None',
                'home_odds': None,
                'away_odds': None,
                'spread_home': None,
                'spread_away': None,
                'total': None
            }

        bookmaker = bookmakers[0]  # Get first bookmaker
        bookmaker_name = bookmaker.get('title', 'Unknown')

        # Extract moneyline (h2h)
        home_odds = None
        away_odds = None
        spread_home = None
        spread_away = None
        total_over = None
        total_under = None

        for market in bookmaker.get('markets', []):
            market_key = market.get('key', '')

            if market_key == 'h2h':  # Moneyline
                outcomes = market.get('outcomes', [])
                for outcome in outcomes:
                    if outcome.get('name') == home_team:
                        home_odds = outcome.get('price')
                    elif outcome.get('name') == away_team:
                        away_odds = outcome.get('price')

            elif market_key == 'spreads':  # Point spread
                outcomes = market.get('outcomes', [])
                for outcome in outcomes:
                    if outcome.get('name') == home_team:
                        spread_home = outcome.get('point')
                    elif outcome.get('name') == away_team:
                        spread_away = outcome.get('point')

            elif market_key == 'totals':  # Over/Under
                outcomes = market.get('outcomes', [])
                for outcome in outcomes:
                    if outcome.get('name') == 'Over':
                        total_over = outcome.get('point')
                    elif outcome.get('name') == 'Under':
                        total_under = outcome.get('point')

        return {
            'game_id': game.get('id', ''),
            'home_team': home_team,
            'away_team': away_team,
            'commence_time': commence_time,
            'bookmaker': bookmaker_name,
            'home_odds': home_odds,
            'away_odds': away_odds,
            'spread_home': spread_home,
            'spread_away': spread_away,
            'total': total_over or total_under
        }

    def save_to_database(self, games: List[Dict], db_path: str = "basketball_betting.db"):
        """Save real odds to database"""
        conn = sqlite3.connect(db_path)

        # Create table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS real_odds (
                game_id TEXT PRIMARY KEY,
                home_team TEXT,
                away_team TEXT,
                commence_time TEXT,
                bookmaker TEXT,
                home_odds REAL,
                away_odds REAL,
                spread_home REAL,
                spread_away REAL,
                total REAL,
                fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Insert games
        for game in games:
            parsed = self.parse_game_odds(game)

            conn.execute("""
                INSERT OR REPLACE INTO real_odds
                (game_id, home_team, away_team, commence_time, bookmaker,
                 home_odds, away_odds, spread_home, spread_away, total)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                parsed['game_id'],
                parsed['home_team'],
                parsed['away_team'],
                parsed['commence_time'],
                parsed['bookmaker'],
                parsed['home_odds'],
                parsed['away_odds'],
                parsed['spread_home'],
                parsed['spread_away'],
                parsed['total']
            ))

        conn.commit()
        conn.close()

        logger.info(f"💾 Saved {len(games)} games with real odds to database")

    def get_game_odds(self, home_team: str, away_team: str) -> Optional[Dict]:
        """
        Get odds for a specific matchup

        Args:
            home_team: Home team name
            away_team: Away team name

        Returns:
            Odds dictionary or None
        """
        games = self.fetch_upcoming_games()

        for game in games:
            if (game.get('home_team', '').lower() == home_team.lower() and
                game.get('away_team', '').lower() == away_team.lower()):
                return self.parse_game_odds(game)

        return None


def main():
    """Test real odds fetcher"""
    print("\n💰 Real Betting Lines Fetcher\n")

    # Check for API key
    api_key = os.getenv('ODDS_API_KEY')

    if not api_key:
        print("❌ No API key found!\n")
        print("Get a FREE API key (500 requests/month):")
        print("1. Go to: https://the-odds-api.com/")
        print("2. Sign up for free account")
        print("3. Copy your API key")
        print("4. Add to .env file:")
        print("   ODDS_API_KEY=your_key_here")
        print("\nExiting...")
        return

    fetcher = RealOddsFetcher(api_key)

    print(f"✅ API key loaded\n")
    print("Fetching real odds from sportsbooks...\n")

    games = fetcher.fetch_upcoming_games()

    if games:
        print(f"✅ Found {len(games)} upcoming games\n")

        print("Sample Games with REAL Odds:")
        print("="*80)

        for i, game in enumerate(games[:5]):  # Show first 5
            parsed = fetcher.parse_game_odds(game)

            print(f"\nGame {i+1}: {parsed['away_team']} @ {parsed['home_team']}")
            print(f"  Commence: {parsed['commence_time']}")
            print(f"  Bookmaker: {parsed['bookmaker']}")

            if parsed['home_odds']:
                print(f"  Moneyline: {parsed['away_team']} ({parsed['away_odds']:+d}) / "
                      f"{parsed['home_team']} ({parsed['home_odds']:+d})")

            if parsed['spread_home']:
                print(f"  Spread: {parsed['away_team']} ({parsed['spread_away']:+.1f}) / "
                      f"{parsed['home_team']} ({parsed['spread_home']:+.1f})")

            if parsed['total']:
                print(f"  Total: O/U {parsed['total']:.1f}")

        print("\n" + "="*80)

        # Save to database
        print("\nSaving to database...")
        fetcher.save_to_database(games)

        print("\n✅ Complete! REAL sportsbook odds ready.")
        print("\n💡 Data from actual bookmakers:")
        print("  • FanDuel")
        print("  • DraftKings")
        print("  • BetMGM")
        print("  • And more...")

    else:
        print("❌ No games found")
        print("\nPossible issues:")
        print("1. API key invalid")
        print("2. No upcoming games today")
        print("3. API quota exceeded (500/month)")


if __name__ == "__main__":
    main()
