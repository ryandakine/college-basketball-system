#!/usr/bin/env python3
"""
College Basketball Daily Tracker
=================================

Automated daily analysis of CBB games using the 12-model AI council.
Tracks performance over time.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict

# Import system components
from line_monitor import LineMonitor
from cbb_12model_ai_council import CBB12ModelAICouncil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/cbb_daily_tracker.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_daily_cbb_analysis():
    """Run daily CBB game analysis using 13-Model AI Council."""
    print(f"\n🏀 CBB DAILY AUTO-TRACKER - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 80)
    
    try:
        # 1. Initialize components
        logger.info("Initializing components...")
        monitor = LineMonitor()
        council = CBB12ModelAICouncil()
        
        # 2. Fetch today's games
        logger.info("Fetching today's games...")
        odds_data = monitor.fetch_odds()
        games = odds_data.get('games', [])
        
        if not games:
            logger.info("No games found for today.")
            print("ℹ️  No games today")
            return

        print(f"\n📊 Analyzing {len(games)} games...")
        
        results = []
        
        # 3. Analyze each game
        for game in games:
            try:
                # Prepare game data for council
                game_data = {
                    'home_team': game.get('home_team'),
                    'away_team': game.get('away_team'),
                    'date': game.get('commence_time'),
                    'spread': _extract_spread(game),
                    'tournament_context': 'regular_season', # Default for now
                    'neutral_site': False # Default
                }
                
                # Run analysis
                recommendation = council.analyze_game(game_data)
                
                # Store result
                result = {
                    'game': f"{game_data['away_team']} @ {game_data['home_team']}",
                    'pick': recommendation.recommended_pick,
                    'confidence': recommendation.confidence,
                    'consensus': recommendation.model_consensus,
                    'reasoning': recommendation.reasoning,
                    'timestamp': datetime.now().isoformat()
                }
                results.append(result)
                
                # Print summary
                print(f"   👉 {result['game']}: {result['pick']} ({result['confidence']:.1%})")
                
            except Exception as e:
                logger.error(f"Error analyzing game {game.get('home_team')} vs {game.get('away_team')}: {e}")

        # 4. Save results
        _save_results(results)
        
    except Exception as e:
        logger.error(f"Critical error in daily tracker: {e}")
        print(f"❌ Error: {e}")

def _extract_spread(game: Dict) -> float:
    """Extract spread from odds data."""
    try:
        # Simplified extraction - would need robust parsing for real API structure
        bookmakers = game.get('bookmakers', [])
        if bookmakers:
            markets = bookmakers[0].get('markets', [])
            for market in markets:
                if market['key'] == 'spreads':
                    return market['outcomes'][0].get('point', 0)
    except:
        pass
    return 0.0

def _save_results(results: List[Dict]):
    """Save analysis results to JSON."""
    output_file = Path('data/daily_predictions.json')
    output_file.parent.mkdir(exist_ok=True)
    
    # Load existing or create new
    if output_file.exists():
        with open(output_file) as f:
            history = json.load(f)
    else:
        history = []
    
    # Append new results
    record = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'predictions': results
    }
    history.append(record)
    
    with open(output_file, 'w') as f:
        json.dump(history, f, indent=2)
        
    print(f"\n✅ Saved {len(results)} predictions to {output_file}")

if __name__ == "__main__":
    run_daily_cbb_analysis()
