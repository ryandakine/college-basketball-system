#!/usr/bin/env python3
"""
Automated Daily Betting System
==============================

Runs automatically every day to:
1. Resolve yesterday's bets
2. Get today's games and place new bets
3. Log results and send notifications

Set up with cron or Task Scheduler to run daily at 10 AM.

Cron example (Linux/Mac):
    0 10 * * * cd /path/to/college-basketball-system && python auto_daily_bets.py >> logs/daily.log 2>&1

Task Scheduler (Windows):
    Run: python auto_daily_bets.py
    Start in: C:\\path\\to\\college-basketball-system
    Schedule: Daily at 10:00 AM
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import logging
import json

# Set up logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(log_dir / f"auto_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from live_predictions import LivePredictionSystem
from paper_trading import PaperTradingSystem


def run_daily_automation():
    """Run the complete daily automation"""

    logger.info("="*60)
    logger.info("🏀 AUTOMATED DAILY BETTING SYSTEM")
    logger.info(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    logger.info("="*60)

    system = LivePredictionSystem()

    # Step 1: Resolve yesterday's bets
    logger.info("\n📊 STEP 1: Resolving yesterday's bets...")
    try:
        resolved = system.resolve_yesterdays_bets()
        logger.info(f"   Resolved {resolved} bets")
    except Exception as e:
        logger.error(f"   Error resolving bets: {e}")

    # Step 2: Get today's games
    logger.info("\n📅 STEP 2: Fetching today's games...")
    try:
        games = system.get_todays_games()
        logger.info(f"   Found {len(games)} games")
    except Exception as e:
        logger.error(f"   Error fetching games: {e}")
        games = []

    # Step 3: Make predictions
    logger.info("\n🎯 STEP 3: Making predictions...")
    predictions = []
    if games:
        try:
            predictions = system.get_predictions(games, min_confidence=0.58)
            logger.info(f"   {len(predictions)} high-confidence picks")

            for pred in predictions[:5]:
                logger.info(f"   • {pred['pick']}: {pred['home_team'] if pred['pick'] == 'HOME' else pred['away_team']} ({pred['confidence']*100:.1f}%)")
        except Exception as e:
            logger.error(f"   Error making predictions: {e}")

    # Step 4: Place paper bets
    logger.info("\n💰 STEP 4: Placing paper bets...")
    if predictions:
        try:
            placed = system.place_paper_bets(predictions, max_bets=5)
            logger.info(f"   Placed {placed} bets")
        except Exception as e:
            logger.error(f"   Error placing bets: {e}")
    else:
        logger.info("   No bets to place")

    # Step 5: Print summary
    logger.info("\n📊 STEP 5: Daily Summary")
    stats = system.paper_trader.get_stats()

    logger.info(f"   Bankroll: ${stats['bankroll']:,.2f}")
    logger.info(f"   Total Bets: {stats['total_bets']}")
    if stats['total_bets'] > 0:
        logger.info(f"   Win Rate: {stats['win_rate']*100:.1f}%")
        logger.info(f"   ROI: {stats['roi']:.1f}%")
        logger.info(f"   Profit: ${stats['profit']:+,.2f}")
    logger.info(f"   Pending Bets: {stats['pending_bets']}")

    # Save daily summary
    summary = {
        'date': datetime.now().isoformat(),
        'games_found': len(games),
        'predictions_made': len(predictions),
        'bets_placed': len(predictions[:5]) if predictions else 0,
        'stats': stats
    }

    summary_file = log_dir / f"summary_{datetime.now().strftime('%Y%m%d')}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n💾 Summary saved to {summary_file}")
    logger.info("="*60)
    logger.info("✅ Daily automation complete!")
    logger.info("="*60 + "\n")

    return summary


def setup_instructions():
    """Print setup instructions for automation"""

    print("\n" + "="*60)
    print("🔧 AUTOMATION SETUP INSTRUCTIONS")
    print("="*60)

    print("\n📌 OPTION 1: Cron Job (Linux/Mac)")
    print("-"*40)
    print("Run: crontab -e")
    print("Add this line (runs daily at 10 AM):")
    print(f"\n0 10 * * * cd {Path.cwd()} && python auto_daily_bets.py >> logs/daily.log 2>&1\n")

    print("\n📌 OPTION 2: Task Scheduler (Windows)")
    print("-"*40)
    print("1. Open Task Scheduler")
    print("2. Create Basic Task")
    print("3. Trigger: Daily at 10:00 AM")
    print("4. Action: Start a program")
    print(f"   Program: python")
    print(f"   Arguments: auto_daily_bets.py")
    print(f"   Start in: {Path.cwd()}")

    print("\n📌 OPTION 3: Keep Terminal Open (Simple)")
    print("-"*40)
    print("Run this to keep it running in background:")
    print(f"\npython auto_daily_bets.py --loop\n")

    print("="*60 + "\n")


def run_loop():
    """Run in continuous loop mode (checks every hour during game hours)"""
    import time

    logger.info("🔄 Running in loop mode - will check hourly during game hours (10 AM - 11 PM)")

    while True:
        now = datetime.now()
        hour = now.hour

        # Only run during game hours (10 AM - 11 PM)
        if 10 <= hour <= 23:
            logger.info(f"\n⏰ Hourly check at {now.strftime('%H:%M')}")
            run_daily_automation()
        else:
            logger.info(f"💤 Outside game hours ({now.strftime('%H:%M')}), sleeping...")

        # Sleep for 1 hour
        time.sleep(3600)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="🏀 Automated Daily Betting")
    parser.add_argument('--setup', action='store_true', help='Show setup instructions')
    parser.add_argument('--loop', action='store_true', help='Run in continuous loop mode')
    parser.add_argument('--test', action='store_true', help='Test run (no actual bets)')

    args = parser.parse_args()

    if args.setup:
        setup_instructions()
    elif args.loop:
        run_loop()
    else:
        run_daily_automation()
