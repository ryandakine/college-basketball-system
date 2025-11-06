#!/usr/bin/env python3
"""
Automatic Game Fetcher
=====================

Fetches today's college basketball games from ESPN API
and prepares them for prediction.

Features:
- Fetches games for any date
- Filters by status (upcoming, in-progress, completed)
- Extracts team names, odds, venue info
- Identifies tournament context
- Returns game data ready for prediction
"""

import logging
import requests
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GameToPredict:
    """Game that needs a prediction."""
    game_id: str
    game_date: datetime
    home_team: str
    away_team: str
    venue: str
    neutral_site: bool
    tournament_context: str
    status: str

    # Betting lines (if available)
    spread: Optional[float] = None
    total: Optional[float] = None
    home_moneyline: Optional[int] = None
    away_moneyline: Optional[int] = None


class AutomaticGameFetcher:
    """Automatically fetch games that need predictions."""

    def __init__(self):
        self.api_base = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"

    def fetch_todays_games(self) -> List[GameToPredict]:
        """Fetch all games scheduled for today."""
        today = datetime.now().strftime("%Y%m%d")
        return self.fetch_games_for_date(today)

    def fetch_upcoming_games(self, days_ahead: int = 7) -> List[GameToPredict]:
        """Fetch games for the next N days."""
        all_games = []
        for i in range(days_ahead):
            date = (datetime.now() + timedelta(days=i)).strftime("%Y%m%d")
            games = self.fetch_games_for_date(date)
            all_games.extend(games)
        return all_games

    def fetch_games_for_date(self, date: str) -> List[GameToPredict]:
        """
        Fetch games for a specific date.

        Args:
            date: Date in YYYYMMDD format

        Returns:
            List of GameToPredict objects
        """
        try:
            url = f"{self.api_base}/scoreboard?dates={date}"
            logger.info(f"Fetching games for {date} from {url}")

            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            games = []
            for event in data.get('events', []):
                game = self._parse_game_event(event)
                if game:
                    games.append(game)

            logger.info(f"Found {len(games)} games for {date}")
            return games

        except Exception as e:
            logger.error(f"Error fetching games for {date}: {e}")
            return []

    def _parse_game_event(self, event: Dict) -> Optional[GameToPredict]:
        """Parse ESPN event data into GameToPredict."""
        try:
            game_id = event.get('id', '')
            status = event.get('status', {})
            status_type = status.get('type', {}).get('name', 'unknown')

            # Skip completed games
            if status_type == 'STATUS_FINAL':
                return None

            competition = event.get('competitions', [{}])[0]
            competitors = competition.get('competitors', [])

            if len(competitors) != 2:
                return None

            # Find home and away teams
            home = next((c for c in competitors if c.get('homeAway') == 'home'), None)
            away = next((c for c in competitors if c.get('homeAway') == 'away'), None)

            if not home or not away:
                return None

            # Extract venue info
            venue_info = competition.get('venue', {})
            venue_name = venue_info.get('fullName', 'Unknown')
            neutral_site = competition.get('neutralSite', False)

            # Determine tournament context
            tournament_context = "regular_season"
            notes = competition.get('notes', [])
            for note in notes:
                headline = note.get('headline', '').lower()
                if 'ncaa tournament' in headline or 'march madness' in headline:
                    tournament_context = "march_madness"
                elif 'tournament' in headline:
                    tournament_context = "conference_tournament"

            # Extract betting lines
            odds = competition.get('odds', [{}])[0] if competition.get('odds') else {}
            spread = None
            total = None
            home_moneyline = None
            away_moneyline = None

            if odds:
                # Spread (home team perspective)
                spread_str = odds.get('details', '')
                if spread_str:
                    try:
                        # Format: "DUKE -5.5" or "UNC +3.5"
                        parts = spread_str.split()
                        if len(parts) >= 2:
                            spread = float(parts[1])
                    except:
                        pass

                # Total
                over_under = odds.get('overUnder')
                if over_under:
                    try:
                        total = float(over_under)
                    except:
                        pass

            # Parse game date
            game_date_str = event.get('date', '')
            try:
                game_date = datetime.fromisoformat(game_date_str.replace('Z', '+00:00'))
            except:
                game_date = datetime.now()

            return GameToPredict(
                game_id=game_id,
                game_date=game_date,
                home_team=home.get('team', {}).get('displayName', ''),
                away_team=away.get('team', {}).get('displayName', ''),
                venue=venue_name,
                neutral_site=neutral_site,
                tournament_context=tournament_context,
                status=status_type,
                spread=spread,
                total=total,
                home_moneyline=home_moneyline,
                away_moneyline=away_moneyline
            )

        except Exception as e:
            logger.error(f"Error parsing game event: {e}")
            return None

    def get_games_needing_predictions(self) -> List[GameToPredict]:
        """
        Get all upcoming games that need predictions.
        Excludes games that are completed or already have predictions.
        """
        import sqlite3

        # Fetch today's and tomorrow's games
        games = []
        for i in range(2):
            date = (datetime.now() + timedelta(days=i)).strftime("%Y%m%d")
            games.extend(self.fetch_games_for_date(date))

        # Filter out games we already have predictions for
        try:
            conn = sqlite3.connect('basketball_betting.db')
            cursor = conn.cursor()

            games_needing_predictions = []
            for game in games:
                # Check if we already have a prediction
                cursor.execute("""
                    SELECT COUNT(*) FROM basketball_predictions
                    WHERE game_id = ? OR (
                        home_team = ? AND away_team = ?
                        AND DATE(prediction_date) = DATE(?)
                    )
                """, (game.game_id, game.home_team, game.away_team, game.game_date))

                count = cursor.fetchone()[0]
                if count == 0:
                    games_needing_predictions.append(game)

            conn.close()

            logger.info(f"Found {len(games_needing_predictions)} games needing predictions")
            return games_needing_predictions

        except Exception as e:
            logger.error(f"Error checking for existing predictions: {e}")
            return games


def main():
    """Test game fetching."""
    fetcher = AutomaticGameFetcher()

    print("\n" + "="*60)
    print("🏀 AUTOMATIC GAME FETCHER")
    print("="*60)

    # Fetch today's games
    games = fetcher.fetch_todays_games()

    print(f"\nGames Today: {len(games)}")
    print("-"*60)

    for game in games:
        print(f"\n{game.away_team} @ {game.home_team}")
        print(f"  Game ID: {game.game_id}")
        print(f"  Venue: {game.venue}")
        print(f"  Neutral: {game.neutral_site}")
        print(f"  Tournament: {game.tournament_context}")
        print(f"  Status: {game.status}")
        if game.spread:
            print(f"  Spread: {game.spread:+.1f}")
        if game.total:
            print(f"  Total: {game.total:.1f}")

    print("\n" + "="*60)

    # Check which need predictions
    needing_predictions = fetcher.get_games_needing_predictions()
    print(f"\nGames Needing Predictions: {len(needing_predictions)}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
