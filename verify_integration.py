
import logging
import sys
import os
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IntegrationTest")

# Add current directory to path
sys.path.append(os.getcwd())

def test_integration():
    logger.info("Starting Integration Test...")

    # 1. Check Config
    logger.info("Checking monitor_config.json...")
    try:
        with open('monitor_config.json', 'r') as f:
            config = json.load(f)
            api_key = config.get('api_keys', {}).get('odds_api_key')
            if api_key and api_key != "your_odds_api_key_here":
                logger.info(f"API Key found: {api_key[:4]}...{api_key[-4:]}")
            else:
                logger.error("API Key not properly configured!")
                return
    except Exception as e:
        logger.error(f"Error reading config: {e}")
        return

    # 2. Initialize System
    logger.info("Initializing IntegratedBettingSystem...")
    try:
        from integrated_betting_system import IntegratedBettingSystem
        system = IntegratedBettingSystem()
        logger.info("IntegratedBettingSystem initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize IntegratedBettingSystem: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Check Modules
    logger.info("Checking loaded modules...")
    modules = [
        ('relationship_analyzer', system.relationship_analyzer),
        ('upset_predictor', system.upset_predictor),
        ('injury_analyzer', system.injury_analyzer),
        ('versatility_analyzer', system.versatility_analyzer),
        ('analytics_engine', system.analytics_engine),
        ('ml_trainer', system.ml_trainer),
        ('backtester', system.backtester)
    ]

    for name, module in modules:
        class_name = module.__class__.__name__
        logger.info(f"Module '{name}' loaded as: {class_name}")
        if class_name == 'MockAnalyzer':
            logger.warning(f"Module '{name}' is using MockAnalyzer fallback.")
        else:
            logger.info(f"Module '{name}' is using REAL implementation.")

    # 4. Test Analysis (Basic)
    logger.info("Testing analyze_game with dummy data...")
    try:
        # Using two real team names just in case
        result = system.analyze_game("Duke", "North Carolina")
        logger.info(f"Analysis Result: {json.dumps(result, indent=2)}")
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_integration()
