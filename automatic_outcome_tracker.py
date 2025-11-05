#!/usr/bin/env python3
"""
Automatic Outcome Tracker for College Basketball
================================================

Fetches actual game results from ESPN API and updates the database
with outcomes for continuous learning.

Features:
- Automatic game result fetching
- Score and spread tracking
- Database updates with actual outcomes
- Performance metrics calculation
- Tournament-aware tracking
"""

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GameOutcome:
    """Actual game outcome."""
    game_id: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    total_points: int
    spread_result: float
    game_date: datetime
    tournament_context: str = "regular_season"
    completed: bool = True


class AutomaticOutcomeTracker:
    """Automatically track and update game outcomes."""

    def __init__(self, db_path: str = "basketball_betting.db"):
        self.db_path = db_path
        self.api_base = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"

    def fetch_completed_games(self, date: Optional[str] = None) -> List[GameOutcome]:
        """
        Fetch completed games from ESPN API.

        Args:
            date: Date in YYYYMMDD format. Defaults to today.

        Returns:
            List of GameOutcome objects
        """
        try:
            if date is None:
                date = datetime.now().strftime("%Y%m%d")

            url = f"{self.api_base}/scoreboard?dates={date}"
            logger.info(f"Fetching games from: {url}")

            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            outcomes = []
            for event in data.get('events', []):
                status = event.get('status', {})
                if status.get('type', {}).get('completed', False):
                    outcome = self._parse_game_event(event)
                    if outcome:
                        outcomes.append(outcome)

            logger.info(f"Found {len(outcomes)} completed games")
            return outcomes

        except Exception as e:
            logger.error(f"Error fetching games: {e}")
            return []

    def _parse_game_event(self, event: Dict) -> Optional[GameOutcome]:
        """Parse ESPN event data into GameOutcome."""
        try:
            game_id = event.get('id', '')
            competition = event.get('competitions', [{}])[0]
            competitors = competition.get('competitors', [])

            if len(competitors) != 2:
                return None

            # Find home and away teams
            home = next((c for c in competitors if c.get('homeAway') == 'home'), None)
            away = next((c for c in competitors if c.get('homeAway') == 'away'), None)

            if not home or not away:
                return None

            home_score = int(home.get('score', 0))
            away_score = int(away.get('score', 0))

            # Determine tournament context
            tournament_context = "regular_season"
            notes = competition.get('notes', [])
            for note in notes:
                headline = note.get('headline', '').lower()
                if 'ncaa tournament' in headline or 'march madness' in headline:
                    tournament_context = "march_madness"
                elif 'tournament' in headline:
                    tournament_context = "conference_tournament"

            return GameOutcome(
                game_id=game_id,
                home_team=home.get('team', {}).get('displayName', ''),
                away_team=away.get('team', {}).get('displayName', ''),
                home_score=home_score,
                away_score=away_score,
                total_points=home_score + away_score,
                spread_result=home_score - away_score,
                game_date=datetime.now(),
                tournament_context=tournament_context,
                completed=True
            )

        except Exception as e:
            logger.error(f"Error parsing game event: {e}")
            return None

    def update_database_with_outcomes(self, outcomes: List[GameOutcome]) -> int:
        """
        Update database with actual outcomes.

        Returns:
            Number of predictions updated
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            updated = 0
            for outcome in outcomes:
                # Find matching predictions
                cursor.execute("""
                    SELECT id, predicted_spread, predicted_total
                    FROM basketball_predictions
                    WHERE (home_team = ? OR away_team = ?)
                    AND actual_spread IS NULL
                    AND DATE(prediction_date) = DATE(?)
                """, (outcome.home_team, outcome.away_team, outcome.game_date))

                predictions = cursor.fetchall()

                for pred_id, pred_spread, pred_total in predictions:
                    # Calculate if prediction was correct
                    spread_correct = self._check_spread_correct(
                        pred_spread, outcome.spread_result
                    )
                    total_correct = self._check_total_correct(
                        pred_total, outcome.total_points
                    )

                    # Update prediction record
                    cursor.execute("""
                        UPDATE basketball_predictions
                        SET actual_spread = ?,
                            actual_total = ?,
                            prediction_correct = ?
                        WHERE id = ?
                    """, (
                        outcome.spread_result,
                        outcome.total_points,
                        spread_correct,
                        pred_id
                    ))

                    updated += 1
                    logger.info(
                        f"Updated prediction {pred_id}: "
                        f"{outcome.home_team} vs {outcome.away_team} - "
                        f"Spread correct: {spread_correct}"
                    )

            conn.commit()
            conn.close()

            logger.info(f"Updated {updated} predictions with actual outcomes")
            return updated

        except Exception as e:
            logger.error(f"Error updating database: {e}")
            return 0

    def _check_spread_correct(self, predicted: float, actual: float) -> bool:
        """Check if spread prediction was correct (within 5 points)."""
        if predicted is None or actual is None:
            return False
        return abs(predicted - actual) <= 5.0

    def _check_total_correct(self, predicted: float, actual: float) -> bool:
        """Check if total prediction was correct (within 5 points)."""
        if predicted is None or actual is None:
            return False
        return abs(predicted - actual) <= 5.0

    def calculate_performance_metrics(self) -> Dict:
        """Calculate system performance metrics."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Overall accuracy
            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN prediction_correct = 1 THEN 1 ELSE 0 END) as correct
                FROM basketball_predictions
                WHERE actual_spread IS NOT NULL
            """)
            total, correct = cursor.fetchone()

            accuracy = (correct / total * 100) if total > 0 else 0

            # Recent performance (last 30 days)
            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN prediction_correct = 1 THEN 1 ELSE 0 END) as correct
                FROM basketball_predictions
                WHERE actual_spread IS NOT NULL
                AND prediction_date >= DATE('now', '-30 days')
            """)
            recent_total, recent_correct = cursor.fetchone()

            recent_accuracy = (recent_correct / recent_total * 100) if recent_total > 0 else 0

            # Tournament performance
            cursor.execute("""
                SELECT
                    tournament_context,
                    COUNT(*) as total,
                    SUM(CASE WHEN prediction_correct = 1 THEN 1 ELSE 0 END) as correct
                FROM basketball_predictions
                WHERE actual_spread IS NOT NULL
                GROUP BY tournament_context
            """)
            tournament_results = cursor.fetchall()

            conn.close()

            metrics = {
                'overall_accuracy': accuracy,
                'total_predictions': total,
                'correct_predictions': correct,
                'recent_accuracy': recent_accuracy,
                'recent_total': recent_total,
                'tournament_breakdown': {
                    context: {'total': t, 'correct': c, 'accuracy': (c/t*100) if t > 0 else 0}
                    for context, t, c in tournament_results
                }
            }

            logger.info(f"Performance Metrics: {accuracy:.1f}% overall, {recent_accuracy:.1f}% recent")
            return metrics

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}

    def run_daily_update(self, lookback_days: int = 7):
        """
        Run daily update of outcomes.

        Args:
            lookback_days: How many days back to check for results
        """
        logger.info(f"Running daily outcome update (lookback: {lookback_days} days)")

        total_updated = 0
        for i in range(lookback_days):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y%m%d")
            outcomes = self.fetch_completed_games(date)
            if outcomes:
                updated = self.update_database_with_outcomes(outcomes)
                total_updated += updated

        logger.info(f"Daily update complete: {total_updated} predictions updated")

        # Calculate and log performance metrics
        metrics = self.calculate_performance_metrics()
        logger.info(f"Current system accuracy: {metrics.get('overall_accuracy', 0):.1f}%")

        return metrics


def main():
    """Run outcome tracker."""
    tracker = AutomaticOutcomeTracker()

    # Run daily update
    metrics = tracker.run_daily_update(lookback_days=7)

    print("\n" + "="*60)
    print("📊 SYSTEM PERFORMANCE METRICS")
    print("="*60)
    print(f"\nOverall Accuracy: {metrics.get('overall_accuracy', 0):.1f}%")
    print(f"Total Predictions: {metrics.get('total_predictions', 0)}")
    print(f"Correct Predictions: {metrics.get('correct_predictions', 0)}")
    print(f"\nRecent (30 days) Accuracy: {metrics.get('recent_accuracy', 0):.1f}%")
    print(f"Recent Total: {metrics.get('recent_total', 0)}")
    print("\nTournament Breakdown:")
    for context, stats in metrics.get('tournament_breakdown', {}).items():
        print(f"  {context}: {stats['accuracy']:.1f}% ({stats['correct']}/{stats['total']})")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
