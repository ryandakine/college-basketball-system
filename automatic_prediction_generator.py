#!/usr/bin/env python3
"""
Automatic Prediction Generator
==============================

Generates predictions automatically for upcoming games.

Features:
- Takes games from automatic_game_fetcher
- Generates predictions using basketball models
- Stores predictions in database
- Calculates confidence and edge
- Handles errors gracefully
"""

import logging
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """A generated prediction."""
    game_id: str
    home_team: str
    away_team: str
    predicted_spread: float
    predicted_total: float
    win_probability: float
    confidence: float
    edge: Optional[float]
    tournament_context: str
    prediction_date: datetime


class AutomaticPredictionGenerator:
    """Automatically generate predictions for games."""

    def __init__(self, db_path: str = "basketball_betting.db"):
        self.db_path = db_path

    def generate_prediction(self, game) -> Optional[Prediction]:
        """
        Generate prediction for a single game.

        Args:
            game: GameToPredict object from automatic_game_fetcher

        Returns:
            Prediction object or None if error
        """
        try:
            logger.info(f"Generating prediction: {game.away_team} @ {game.home_team}")

            # PLACEHOLDER: Real prediction logic would go here
            # For now, generate demo predictions based on simple heuristics

            # Simple heuristic: home team favored by 3-5 points
            predicted_spread = -4.0  # Home team favored
            predicted_total = 145.0

            # Home team win probability
            win_probability = 0.60

            # Confidence based on tournament context
            if game.tournament_context == "march_madness":
                confidence = 0.55  # Lower confidence in tournament
            elif game.tournament_context == "conference_tournament":
                confidence = 0.60
            else:
                confidence = 0.65  # Regular season

            # Calculate edge if market line available
            edge = None
            if game.spread is not None:
                # Edge is difference between our line and market
                edge = abs(predicted_spread - game.spread) / 100

            prediction = Prediction(
                game_id=game.game_id,
                home_team=game.home_team,
                away_team=game.away_team,
                predicted_spread=predicted_spread,
                predicted_total=predicted_total,
                win_probability=win_probability,
                confidence=confidence,
                edge=edge,
                tournament_context=game.tournament_context,
                prediction_date=datetime.now()
            )

            logger.info(
                f"  Predicted: {game.home_team} {predicted_spread:+.1f}, "
                f"Total: {predicted_total:.1f}, "
                f"Confidence: {confidence:.1%}"
            )

            return prediction

        except Exception as e:
            logger.error(f"Error generating prediction for {game.game_id}: {e}")
            return None

    def save_prediction(self, prediction: Prediction) -> bool:
        """Save prediction to database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO basketball_predictions (
                    game_id, home_team, away_team,
                    predicted_spread, predicted_total,
                    win_probability, confidence, edge,
                    tournament_context, prediction_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                prediction.game_id,
                prediction.home_team,
                prediction.away_team,
                prediction.predicted_spread,
                prediction.predicted_total,
                prediction.win_probability,
                prediction.confidence,
                prediction.edge,
                prediction.tournament_context,
                prediction.prediction_date
            ))

            conn.commit()
            conn.close()

            logger.info(f"✅ Saved prediction: {prediction.away_team} @ {prediction.home_team}")
            return True

        except Exception as e:
            logger.error(f"Error saving prediction: {e}")
            return False

    def generate_predictions_for_games(self, games: List) -> Dict:
        """
        Generate predictions for multiple games.

        Args:
            games: List of GameToPredict objects

        Returns:
            Summary dict with counts
        """
        summary = {
            'total_games': len(games),
            'predictions_generated': 0,
            'predictions_saved': 0,
            'errors': 0
        }

        for game in games:
            # Generate prediction
            prediction = self.generate_prediction(game)

            if prediction:
                summary['predictions_generated'] += 1

                # Save to database
                if self.save_prediction(prediction):
                    summary['predictions_saved'] += 1
                else:
                    summary['errors'] += 1
            else:
                summary['errors'] += 1

        return summary

    def run_automatic_predictions(self) -> Dict:
        """
        Main entry point: Fetch games and generate predictions automatically.

        Returns:
            Summary dict
        """
        from automatic_game_fetcher import AutomaticGameFetcher

        logger.info("="*60)
        logger.info("🤖 RUNNING AUTOMATIC PREDICTION GENERATION")
        logger.info("="*60)

        try:
            # Fetch games needing predictions
            fetcher = AutomaticGameFetcher()
            games = fetcher.get_games_needing_predictions()

            logger.info(f"Found {len(games)} games needing predictions")

            if not games:
                logger.info("No games to predict - system up to date!")
                return {
                    'success': True,
                    'total_games': 0,
                    'predictions_generated': 0,
                    'predictions_saved': 0,
                    'errors': 0
                }

            # Generate predictions
            summary = self.generate_predictions_for_games(games)
            summary['success'] = True

            logger.info("="*60)
            logger.info(f"✅ Automatic prediction generation complete")
            logger.info(f"   Games processed: {summary['total_games']}")
            logger.info(f"   Predictions saved: {summary['predictions_saved']}")
            logger.info(f"   Errors: {summary['errors']}")
            logger.info("="*60)

            return summary

        except Exception as e:
            logger.error(f"❌ Error in automatic prediction generation: {e}")
            return {
                'success': False,
                'error': str(e),
                'total_games': 0,
                'predictions_generated': 0,
                'predictions_saved': 0,
                'errors': 1
            }


def main():
    """Test automatic prediction generation."""
    generator = AutomaticPredictionGenerator()
    summary = generator.run_automatic_predictions()

    print("\n" + "="*60)
    print("🤖 AUTOMATIC PREDICTION SUMMARY")
    print("="*60)

    if summary.get('success'):
        print(f"\n✅ Success!")
        print(f"   Games Processed: {summary['total_games']}")
        print(f"   Predictions Generated: {summary['predictions_generated']}")
        print(f"   Predictions Saved: {summary['predictions_saved']}")
        if summary['errors'] > 0:
            print(f"   Errors: {summary['errors']}")
    else:
        print(f"\n❌ Failed: {summary.get('error')}")

    print("="*60 + "\n")


if __name__ == "__main__":
    main()
