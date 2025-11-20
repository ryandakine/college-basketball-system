#!/usr/bin/env python3
"""
🏀 UNIFIED PREDICTION RUNNER
============================

One command to make predictions with:
- Real data sources (Barttorvik, ESPN, Odds API)
- 12-Model AI Council
- LLM Ensemble (optional)

Usage:
    python predict.py Duke "North Carolina"
    python predict.py --refresh Duke "North Carolina"
    python predict.py --all-games
"""

import argparse
import sys
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def predict_single_game(home_team: str, away_team: str, refresh: bool = False):
    """Make prediction for a single game"""
    from real_data_connector import IntegratedPredictionEngine

    engine = IntegratedPredictionEngine()

    if refresh:
        result = engine.refresh_data_and_predict(home_team, away_team)
    else:
        result = engine.predict_game(home_team, away_team)

    if result['success']:
        print_prediction(result)
    else:
        print(f"\n❌ Prediction failed: {result.get('error')}")
        print("\n💡 Make sure you have:")
        print("   1. Run: python real_data_pipeline.py")
        print("   2. Set ODDS_API_KEY in .env")

    return result


def predict_all_upcoming():
    """Predict all upcoming games with odds"""
    from real_odds_fetcher import RealOddsFetcher
    from real_data_connector import IntegratedPredictionEngine

    print("\n🏀 Fetching all upcoming games with odds...\n")

    fetcher = RealOddsFetcher()
    games = fetcher.fetch_upcoming_games()

    if not games:
        print("❌ No upcoming games found")
        print("   Check your ODDS_API_KEY in .env")
        return

    print(f"Found {len(games)} upcoming games\n")

    engine = IntegratedPredictionEngine()
    predictions = []

    for game in games[:10]:  # Limit to 10 games
        home = game.get('home_team', '')
        away = game.get('away_team', '')

        if not home or not away:
            continue

        print(f"\n{'='*60}")
        result = engine.predict_game(home, away)

        if result['success']:
            predictions.append(result)
            print_prediction(result, compact=True)

    print(f"\n\n{'='*60}")
    print(f"✅ Completed {len(predictions)} predictions")
    print(f"{'='*60}\n")


def print_prediction(result: dict, compact: bool = False):
    """Print prediction result"""
    if not result.get('success'):
        return

    home = result['home_team']
    away = result['away_team']
    pred = result.get('prediction')
    data = result.get('game_data', {})

    if compact:
        # One-line summary
        if pred:
            print(f"🎯 {away} @ {home}: {pred.recommended_pick} ({pred.confidence:.0%})")
        return

    # Full prediction
    print("\n" + "="*60)
    print(f"🏀 PREDICTION: {away} @ {home}")
    print("="*60)

    # Team efficiency
    home_eff = data.get('home_efficiency', {})
    away_eff = data.get('away_efficiency', {})

    if home_eff:
        print(f"\n{home}:")
        print(f"   Rank: #{home_eff.get('rank', 'N/A')}")
        print(f"   Record: {home_eff.get('wins', 0)}-{home_eff.get('losses', 0)}")
        print(f"   AdjOE: {home_eff.get('adj_oe', 0):.1f}")
        print(f"   AdjDE: {home_eff.get('adj_de', 0):.1f}")
        print(f"   Tempo: {home_eff.get('tempo', 0):.1f}")

    if away_eff:
        print(f"\n{away}:")
        print(f"   Rank: #{away_eff.get('rank', 'N/A')}")
        print(f"   Record: {away_eff.get('wins', 0)}-{away_eff.get('losses', 0)}")
        print(f"   AdjOE: {away_eff.get('adj_oe', 0):.1f}")
        print(f"   AdjDE: {away_eff.get('adj_de', 0):.1f}")
        print(f"   Tempo: {away_eff.get('tempo', 0):.1f}")

    # Odds
    odds = data.get('odds', {})
    if odds:
        print(f"\nOdds ({odds.get('bookmaker', 'N/A')}):")
        if odds.get('home_odds'):
            print(f"   Moneyline: {home} ({odds['home_odds']:+d}) / {away} ({odds['away_odds']:+d})")
        if odds.get('spread'):
            print(f"   Spread: {home} ({odds['spread']:+.1f})")
        if odds.get('total'):
            print(f"   Total: O/U {odds['total']:.1f}")

    # Prediction
    if pred:
        print(f"\n🎯 PREDICTION:")
        print(f"   Pick: {pred.recommended_pick}")
        print(f"   Confidence: {pred.confidence:.1%}")
        print(f"   Market: {pred.market_type}")
        if hasattr(pred, 'reasoning') and pred.reasoning:
            print(f"   Reasoning: {pred.reasoning[:200]}...")

    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description="🏀 College Basketball Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python predict.py Duke "North Carolina"
  python predict.py --refresh Duke "North Carolina"
  python predict.py --all-games
  python predict.py --setup
        """
    )

    parser.add_argument(
        'home_team',
        nargs='?',
        help='Home team name'
    )

    parser.add_argument(
        'away_team',
        nargs='?',
        help='Away team name'
    )

    parser.add_argument(
        '--refresh',
        action='store_true',
        help='Refresh all data sources before predicting'
    )

    parser.add_argument(
        '--all-games',
        action='store_true',
        help='Predict all upcoming games with odds'
    )

    parser.add_argument(
        '--setup',
        action='store_true',
        help='Run initial data setup'
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("🏀 COLLEGE BASKETBALL PREDICTION SYSTEM")
    print("   Real Data + 12-Model AI Council")
    print("="*60)

    if args.setup:
        print("\n📥 Setting up data sources...\n")
        from real_data_pipeline import RealDataPipeline
        pipeline = RealDataPipeline()
        pipeline.fetch_all_real_data()
        pipeline.print_data_summary()
        return

    if args.all_games:
        predict_all_upcoming()
        return

    if not args.home_team or not args.away_team:
        parser.print_help()
        print("\n💡 Quick start:")
        print("   python predict.py --setup")
        print("   python predict.py Duke 'North Carolina'")
        return

    predict_single_game(args.home_team, args.away_team, args.refresh)


if __name__ == "__main__":
    main()
