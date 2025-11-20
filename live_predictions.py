#!/usr/bin/env python3
"""
Live Predictions System
=======================

Fetches real games and odds, makes predictions, places paper bets.

Usage:
    python live_predictions.py --today        # Get today's picks
    python live_predictions.py --tomorrow     # Get tomorrow's picks
    python live_predictions.py --resolve      # Resolve yesterday's bets
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
import logging
from typing import List, Dict, Optional

from paper_trading import PaperTradingSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LivePredictionSystem:
    """
    Live prediction system with real data feeds
    """

    def __init__(self, model_path: str = "trained_model.pkl"):
        self.model_path = Path(model_path)
        self.model = None
        self.paper_trader = PaperTradingSystem(initial_bankroll=1000)

        # ESPN API
        self.espn_base = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"

        # Load model if exists
        if self.model_path.exists():
            self.load_model()

    def load_model(self):
        """Load trained prediction model"""
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        logger.info("✅ Loaded trained model")

    def get_todays_games(self) -> List[Dict]:
        """Fetch today's college basketball games from ESPN"""
        today = datetime.now().strftime("%Y%m%d")
        return self._fetch_games(today)

    def get_tomorrows_games(self) -> List[Dict]:
        """Fetch tomorrow's games"""
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y%m%d")
        return self._fetch_games(tomorrow)

    def get_yesterdays_results(self) -> List[Dict]:
        """Fetch yesterday's results to resolve bets"""
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        return self._fetch_games(yesterday, completed_only=True)

    def _fetch_games(self, date_str: str, completed_only: bool = False) -> List[Dict]:
        """Fetch games from ESPN API"""
        games = []

        try:
            url = f"{self.espn_base}/scoreboard?dates={date_str}"
            response = requests.get(url, timeout=10)

            if response.status_code != 200:
                logger.error(f"ESPN API error: {response.status_code}")
                return games

            data = response.json()
            events = data.get('events', [])

            for event in events:
                try:
                    competition = event.get('competitions', [{}])[0]
                    competitors = competition.get('competitors', [])

                    if len(competitors) != 2:
                        continue

                    # Get teams
                    home_team = None
                    away_team = None
                    home_score = None
                    away_score = None

                    for comp in competitors:
                        team_name = comp.get('team', {}).get('displayName', '')
                        score = comp.get('score', '0')
                        try:
                            score = int(score) if score else 0
                        except:
                            score = 0

                        if comp.get('homeAway') == 'home':
                            home_team = team_name
                            home_score = score
                        else:
                            away_team = team_name
                            away_score = score

                    if not home_team or not away_team:
                        continue

                    # Check if completed
                    is_completed = event.get('status', {}).get('type', {}).get('completed', False)

                    if completed_only and not is_completed:
                        continue

                    game = {
                        'game_id': event.get('id', ''),
                        'date': date_str,
                        'time': event.get('date', ''),
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_score': home_score,
                        'away_score': away_score,
                        'completed': is_completed,
                        'venue': competition.get('venue', {}).get('fullName', ''),
                        'neutral_site': competition.get('neutralSite', False),
                        'conference_game': competition.get('conferenceCompetition', False)
                    }

                    # Add winner if completed
                    if is_completed and home_score and away_score:
                        game['winner'] = 'HOME' if home_score > away_score else 'AWAY'

                    games.append(game)

                except Exception as e:
                    continue

        except Exception as e:
            logger.error(f"Error fetching games: {e}")

        return games

    def predict_game(self, game: Dict) -> Dict:
        """
        Make prediction for a single game

        Returns prediction with confidence and spread
        """
        # Simple prediction based on historical patterns
        # In production, this would use the trained model

        home_team = game['home_team']
        away_team = game['away_team']

        # Base confidence from home court advantage
        home_confidence = 0.55  # Home teams win ~55% historically

        # Adjust for neutral site
        if game.get('neutral_site'):
            home_confidence = 0.50

        # Add some variance based on team names (placeholder for real features)
        # In production, you'd look up team stats
        np.random.seed(hash(home_team + away_team) % 2**32)
        adjustment = np.random.uniform(-0.15, 0.15)
        home_confidence = np.clip(home_confidence + adjustment, 0.35, 0.75)

        # Predict spread (rough estimate)
        spread_prediction = (home_confidence - 0.5) * 30  # Scale to points

        # Determine pick
        if home_confidence >= 0.55:
            pick = 'HOME'
            confidence = home_confidence
        else:
            pick = 'AWAY'
            confidence = 1 - home_confidence

        return {
            'game_id': game['game_id'],
            'home_team': home_team,
            'away_team': away_team,
            'pick': pick,
            'confidence': confidence,
            'spread_prediction': spread_prediction,
            'home_win_prob': home_confidence
        }

    def get_predictions(self, games: List[Dict], min_confidence: float = 0.58) -> List[Dict]:
        """Get predictions for all games, filtered by confidence"""
        predictions = []

        for game in games:
            pred = self.predict_game(game)

            if pred['confidence'] >= min_confidence:
                predictions.append({
                    **game,
                    **pred
                })

        # Sort by confidence
        predictions.sort(key=lambda x: x['confidence'], reverse=True)

        return predictions

    def place_paper_bets(self, predictions: List[Dict], max_bets: int = 5):
        """Place paper bets on top predictions"""
        bets_placed = 0

        for pred in predictions[:max_bets]:
            bet = self.paper_trader.place_bet(
                game_id=pred['game_id'],
                home_team=pred['home_team'],
                away_team=pred['away_team'],
                pick=pred['pick'],
                confidence=pred['confidence'],
                spread_prediction=pred['spread_prediction']
            )

            if bet:
                bets_placed += 1

        return bets_placed

    def resolve_yesterdays_bets(self):
        """Resolve pending bets with yesterday's results"""
        results = self.get_yesterdays_results()

        resolved = 0
        for game in results:
            if game.get('completed') and game.get('winner'):
                bet = self.paper_trader.resolve_bet(
                    game_id=game['game_id'],
                    winner=game['winner'],
                    home_score=game['home_score'],
                    away_score=game['away_score']
                )
                if bet:
                    resolved += 1

        return resolved

    def print_predictions(self, predictions: List[Dict]):
        """Print formatted predictions"""
        print("\n" + "="*70)
        print("🏀 TODAY'S PREDICTIONS")
        print("="*70)

        if not predictions:
            print("\nNo high-confidence picks today.")
            print("="*70 + "\n")
            return

        for i, pred in enumerate(predictions, 1):
            conf_bar = "█" * int(pred['confidence'] * 20)
            print(f"\n{i}. {pred['pick']}: {pred['home_team'] if pred['pick'] == 'HOME' else pred['away_team']}")
            print(f"   {pred['away_team']} @ {pred['home_team']}")
            print(f"   Confidence: {pred['confidence']*100:.1f}% {conf_bar}")
            print(f"   Spread: {pred['spread_prediction']:+.1f}")

        print("\n" + "="*70 + "\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="🏀 Live Predictions System")
    parser.add_argument('--today', action='store_true', help="Get today's picks")
    parser.add_argument('--tomorrow', action='store_true', help="Get tomorrow's picks")
    parser.add_argument('--resolve', action='store_true', help="Resolve yesterday's bets")
    parser.add_argument('--dashboard', action='store_true', help="Show paper trading dashboard")
    parser.add_argument('--bet', action='store_true', help="Place paper bets on picks")
    parser.add_argument('--max-bets', type=int, default=5, help="Max bets to place")
    parser.add_argument('--min-conf', type=float, default=0.58, help="Minimum confidence")

    args = parser.parse_args()

    system = LivePredictionSystem()

    if args.resolve:
        print("\n🔄 Resolving yesterday's bets...")
        resolved = system.resolve_yesterdays_bets()
        print(f"✅ Resolved {resolved} bets")
        system.paper_trader.print_dashboard()

    elif args.dashboard:
        system.paper_trader.print_dashboard()

    elif args.tomorrow:
        print("\n📅 Fetching tomorrow's games...")
        games = system.get_tomorrows_games()
        print(f"Found {len(games)} games")

        predictions = system.get_predictions(games, min_confidence=args.min_conf)
        system.print_predictions(predictions)

        if args.bet and predictions:
            print(f"💰 Placing paper bets...")
            placed = system.place_paper_bets(predictions, max_bets=args.max_bets)
            print(f"✅ Placed {placed} bets")
            system.paper_trader.print_dashboard()

    else:  # Default: today
        print("\n📅 Fetching today's games...")
        games = system.get_todays_games()
        print(f"Found {len(games)} games")

        predictions = system.get_predictions(games, min_confidence=args.min_conf)
        system.print_predictions(predictions)

        if args.bet and predictions:
            print(f"💰 Placing paper bets...")
            placed = system.place_paper_bets(predictions, max_bets=args.max_bets)
            print(f"✅ Placed {placed} bets")
            system.paper_trader.print_dashboard()


if __name__ == "__main__":
    main()
