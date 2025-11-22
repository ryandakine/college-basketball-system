
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

    # 2. Initialize DataCollector
    logger.info("Initializing DataCollector...")
    try:
        from data_collector import DataCollector
        data_collector = DataCollector()
    except Exception as e:
        logger.error(f"Failed to initialize DataCollector: {e}")
        return

    # 3. Initialize IntegratedBettingSystem
    logger.info("Initializing IntegratedBettingSystem...")
    try:
        from integrated_betting_system import IntegratedBettingSystem
        system = IntegratedBettingSystem()
    except Exception as e:
        logger.error(f"Failed to initialize IntegratedBettingSystem: {e}")
        return

    # 4. Process a Sample Game
    try:
        odds_data = monitor.fetch_odds()
        
        # The odds API returns games as a list in the 'games' key
        games_list = odds_data.get('games', [])
        
        if not games_list:
            logger.error("No games found in odds data")
            return
        
        # Take the first game
        game = games_list[0]
        game_id = game.get('id')
        home_team = game.get('home_team')
        away_team = game.get('away_team')
        
        logger.info(f"Analyzing live game: {home_team} vs {away_team}")
        
        # Extract odds from bookmakers (use first bookmaker for simplicity)
        bookmakers = game.get('bookmakers', [])
        if not bookmakers:
            logger.warning("No bookmakers found for this game")
            spread = 0.0
            total = 145.0
            home_ml = -110
            away_ml = -110
        else:
            bookmaker = bookmakers[0]
            markets = {m['key']: m for m in bookmaker.get('markets', [])}
            
            # Extract spread
            spread_market = markets.get('spreads', {})
            spread_outcomes = spread_market.get('outcomes', [])
            spread = spread_outcomes[0].get('point', 0.0) if spread_outcomes else 0.0
            
            # Extract total
            totals_market = markets.get('totals', {})
            totals_outcomes = totals_market.get('outcomes', [])
            total = totals_outcomes[0].get('point', 145.0) if totals_outcomes else 145.0
            
            # Extract moneylines
            h2h_market = markets.get('h2h', {})
            h2h_outcomes = h2h_market.get('outcomes', [])
            home_ml = -110
            away_ml = -110
            for outcome in h2h_outcomes:
                if outcome.get('name') == home_team:
                    # Convert decimal odds to American
                    decimal = outcome.get('price', 2.0)
                    home_ml = int((decimal - 1) * 100) if decimal >= 2.0 else int(-100 / (decimal - 1))
                elif outcome.get('name') == away_team:
                    decimal = outcome.get('price', 2.0)
                    away_ml = int((decimal - 1) * 100) if decimal >= 2.0 else int(-100 / (decimal - 1))
        
        logger.info(f"Odds: Spread={spread}, Total={total}, Home ML={home_ml}, Away ML={away_ml}")
        
        # Fetch real data for both teams
        logger.info(f"Fetching stats for {home_team}...")
        home_stats = data_collector.fetch_team_stats(home_team)
        
        logger.info(f"Fetching stats for {away_team}...")
        away_stats = data_collector.fetch_team_stats(away_team)
        
        logger.info(f"Fetching roster for {home_team}...")
        home_roster = data_collector.fetch_roster_data(home_team)
        
        logger.info(f"Fetching roster for {away_team}...")
        away_roster = data_collector.fetch_roster_data(away_team)
        
        # Construct game_data with REAL data
        game_data = {
            'game_id': game_id,
            'date': datetime.now().isoformat(),
            'home_team': home_team,
            'away_team': away_team,
            'betting_lines': {
                'spread': spread,
                'total': total,
                'home_ml': home_ml,
                'away_ml': away_ml
            },
            # Real data from ESPN
            'is_tournament': False,
            'home_team_relationships': {
                'team_chemistry': 0.7,  # Still placeholder - would need advanced data
            },
            'away_team_relationships': {
                'team_chemistry': 0.7,
            },
            'home_team_injuries': [],  # ESPN doesn't provide injury data
            'away_team_injuries': [],
            'home_team_versatility': {
                'position_flexibility': 0.7,
                'coaching_adaptability': 0.7,
                'depth_versatility': min(1.0, len(home_roster) / 12.0),  # Based on roster size
            },
            'away_team_versatility': {
                'position_flexibility': 0.7,
                'coaching_adaptability': 0.7,
                'depth_versatility': min(1.0, len(away_roster) / 12.0),
            },
            'home_team_analytics': {
                'net_efficiency': (home_stats['points_per_game'] - home_stats['points_allowed_per_game']) / 10.0,
                'strength_of_schedule': 0.5,  # Would need more data
                'pace': 70.0,  # Would need possessions data
                'tempo_control_rating': 0.5,
            },
            'away_team_analytics': {
                'net_efficiency': (away_stats['points_per_game'] - away_stats['points_allowed_per_game']) / 10.0,
                'strength_of_schedule': 0.5,
                'pace': 70.0,
                'tempo_control_rating': 0.5,
            },
            'ml_predictions': {
                'spread_confidence': 0.6,
                'total_confidence': 0.6
            }
        }
        
        logger.info("Game data constructed with real stats:")
        logger.info(f"  {home_team}: {home_stats['wins']}-{home_stats['losses']}, {home_stats['points_per_game']:.1f} PPG")
        logger.info(f"  {away_team}: {away_stats['wins']}-{away_stats['losses']}, {away_stats['points_per_game']:.1f} PPG")
        logger.info(f"  {home_team} roster: {len(home_roster)} players")
        logger.info(f"  {away_team} roster: {len(away_roster)} players")
        
        # Run analysis
        logger.info("Running integrated analysis...")
        analysis = system.analyze_game(game_data)
        
        logger.info("Analysis completed successfully!")
        logger.info(f"Final Confidence: {analysis.final_confidence:.3f}")
        logger.info(f"Expected ROI: {analysis.expected_roi:.2f}%")
        logger.info(f"Recommended Bets: {len(analysis.recommended_bets)}")
        for bet in analysis.recommended_bets:
            logger.info(f"  - {bet.bet_type.upper()} {bet.side.upper()} ({bet.confidence:.2f} conf, {bet.stake_percentage:.2%} stake)")
            
    except Exception as e:
        logger.error(f"Error processing game: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_live_integration()
