
import logging
import sys
import os
import json
from datetime import datetime

# Configure logging
log_file = 'test_live_data.log'
if os.path.exists(log_file):
    os.remove(log_file)

# Force reset of logging handlers
root = logging.getLogger()
if root.handlers:
    for handler in root.handlers:
        root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("LiveIntegrationTest")
print("Test script started...")

# Add current directory to path
sys.path.append(os.getcwd())

def test_live_integration():
    logger.info("Starting Live Data Integration Test...")

    # 1. Initialize LineMonitor and Fetch Odds
    logger.info("Initializing LineMonitor...")
    try:
        from line_monitor import LineMonitor
        monitor = LineMonitor()
        logger.info("Fetching live odds...")
        odds_data = monitor.fetch_odds()
        
        if not odds_data:
            logger.error("No odds data fetched! Check API key and connectivity.")
            return
            
        logger.info(f"Successfully fetched odds for {len(odds_data)} games.")
    except Exception as e:
        logger.error(f"Error fetching odds: {e}")
        return

    # 2. Initialize IntegratedBettingSystem
    logger.info("Initializing IntegratedBettingSystem...")
    try:
        from integrated_betting_system import IntegratedBettingSystem
        system = IntegratedBettingSystem()
    except Exception as e:
        logger.error(f"Failed to initialize IntegratedBettingSystem: {e}")
        return

    # 3. Process a Sample Game
    # We'll take the first game from the fetched odds
    try:
        game_id = list(odds_data.keys())[0]
        game_info = odds_data[game_id]
        
        logger.info(f"Analyzing live game: {game_info.get('home_team')} vs {game_info.get('away_team')}")
        
        # Construct game_data object expected by IntegratedBettingSystem
        # Note: In a full production system, we would fetch stats/injuries here too.
        # For now, we use the REAL odds and placeholders for other data to verify the pipeline.
        
        game_data = {
            'game_id': game_id,
            'date': datetime.now().isoformat(),
            'home_team': game_info.get('home_team'),
            'away_team': game_info.get('away_team'),
            'betting_lines': {
                'spread': game_info.get('spread', 0.0),
                'total': game_info.get('total', 145.0),
                'home_ml': game_info.get('home_ml', -110),
                'away_ml': game_info.get('away_ml', -110)
            },
            # Placeholders for other modules (until we integrate their data sources)
            'is_tournament': False,
            'home_team_relationships': {'team_chemistry': 0.7},
            'away_team_relationships': {'team_chemistry': 0.7},
            'home_team_injuries': [],
            'away_team_injuries': [],
            'home_team_versatility': {},
            'away_team_versatility': {},
            'home_team_analytics': {'net_efficiency': 0.1},
            'away_team_analytics': {'net_efficiency': 0.05},
            'ml_predictions': {'spread_confidence': 0.6, 'total_confidence': 0.6}
        }
        
        # Run analysis
        analysis = system.analyze_game(game_data)
        
        logger.info("Analysis completed successfully!")
        logger.info(f"Recommended Bets: {len(analysis.recommended_bets)}")
        for bet in analysis.recommended_bets:
            logger.info(f"  - {bet.bet_type.upper()} {bet.side.upper()} ({bet.confidence:.2f} conf)")
            
    except Exception as e:
        logger.error(f"Error processing game: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_live_integration()
