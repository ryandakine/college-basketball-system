#!/usr/bin/env python3
"""
Main entry point for the College Basketball Betting Intelligence System.

Usage:
    python basketball_main.py --analyze-today
    python basketball_main.py --backtest --start-date 2024-01-01
    python basketball_main.py --paper-trade --bankroll 10000
    python basketball_main.py --init-db

Self-Learning:
    python basketball_main.py --update-outcomes
    python basketball_main.py --monitor-performance
    python basketball_main.py --run-learning-cycle

FULL AUTOMATION:
    python basketball_main.py --auto-predict    # Auto generate predictions
    python basketball_main.py --full-auto       # Complete daily cycle
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def init_database():
    """Initialize database schemas."""
    logger.info("Initializing database schemas...")

    try:
        import sqlite3

        # Initialize main betting database
        db_path = Path("basketball_betting.db")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS basketball_predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                predicted_spread REAL,
                predicted_total REAL,
                win_probability REAL,
                confidence REAL,
                tournament_context TEXT,
                actual_spread REAL,
                actual_total REAL,
                prediction_correct BOOLEAN
            )
        """)

        # Create bets table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS basketball_bets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                bet_type TEXT NOT NULL,
                bet_amount REAL NOT NULL,
                odds REAL NOT NULL,
                edge REAL,
                confidence REAL,
                bet_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                result TEXT,
                profit_loss REAL
            )
        """)

        # Create teams table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS basketball_teams (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_name TEXT UNIQUE NOT NULL,
                conference TEXT,
                kenpom_rating REAL,
                tempo REAL,
                offensive_efficiency REAL,
                defensive_efficiency REAL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

        logger.info(f"✅ Database initialized successfully at {db_path}")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        return False


def analyze_today():
    """Analyze today's games."""
    logger.info("🏀 Analyzing today's games...")

    try:
        # Import would use actual prediction engine
        # from core_basketball_prediction_engine import CoreBasketballPredictionEngine

        logger.info("Note: Full prediction engine integration pending")
        logger.info("This is a demonstration of the system structure")

        # Demo output
        print("\n" + "="*60)
        print("🏀 TODAY'S GAME ANALYSIS")
        print("="*60)
        print("\nGame 1: Duke vs North Carolina")
        print("  Predicted Spread: Duke -4.5")
        print("  Predicted Total: 152.5")
        print("  Win Probability: 68%")
        print("  Confidence: HIGH")
        print("  Edge: 2.3%")
        print("  Recommendation: BET Duke -4.5")
        print("\nGame 2: Kansas vs Kentucky")
        print("  Predicted Spread: Kansas -2.5")
        print("  Predicted Total: 145.0")
        print("  Win Probability: 58%")
        print("  Confidence: MEDIUM")
        print("  Edge: 1.1%")
        print("  Recommendation: SMALL BET Kansas -2.5")
        print("\n" + "="*60)

        return True

    except Exception as e:
        logger.error(f"❌ Error analyzing games: {e}")
        return False


def run_backtest(start_date: str, end_date: Optional[str] = None):
    """Run backtesting on historical data."""
    logger.info(f"📊 Running backtest from {start_date}...")

    try:
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Backtesting period: {start_date} to {end_date}")
        logger.info("Note: Full backtesting integration pending")

        # Demo output
        print("\n" + "="*60)
        print("📊 BACKTESTING RESULTS")
        print("="*60)
        print(f"\nPeriod: {start_date} to {end_date}")
        print("\nPerformance Metrics:")
        print("  Total Bets: TBD")
        print("  Win Rate: TBD")
        print("  ROI: TBD")
        print("  Sharpe Ratio: TBD")
        print("  Max Drawdown: TBD")
        print("\n" + "="*60)

        return True

    except Exception as e:
        logger.error(f"❌ Error running backtest: {e}")
        return False


def paper_trade(bankroll: float):
    """Start paper trading mode."""
    logger.info(f"💰 Starting paper trading with ${bankroll:,.2f} bankroll...")

    try:
        logger.info("Note: Paper trading system pending full integration")

        print("\n" + "="*60)
        print("💰 PAPER TRADING MODE")
        print("="*60)
        print(f"\nStarting Bankroll: ${bankroll:,.2f}")
        print("Max Bet Size: 5% of bankroll")
        print("Kelly Criterion: Enabled (Fractional: 1/4)")
        print("\nStatus: Monitoring for opportunities...")
        print("Press Ctrl+C to stop")
        print("="*60 + "\n")

        return True

    except Exception as e:
        logger.error(f"❌ Error starting paper trading: {e}")
        return False


def update_outcomes():
    """Update database with actual game outcomes."""
    logger.info("📥 Updating outcomes...")
    try:
        from automatic_outcome_tracker import AutomaticOutcomeTracker
        tracker = AutomaticOutcomeTracker()
        metrics = tracker.run_daily_update(lookback_days=7)
        print("\n" + "="*60)
        print("📥 OUTCOME UPDATE COMPLETE")
        print("="*60)
        print(f"\nOverall Accuracy: {metrics.get('overall_accuracy', 0):.1f}%")
        print(f"Total Predictions: {metrics.get('total_predictions', 0)}")
        print("="*60 + "\n")
        return True
    except Exception as e:
        logger.error(f"❌ Error updating outcomes: {e}")
        return False


def monitor_performance():
    """Monitor system performance and detect drift."""
    logger.info("🔍 Monitoring performance...")
    try:
        from performance_monitor import PerformanceMonitor
        monitor = PerformanceMonitor()
        summary = monitor.run_monitoring_cycle()

        print("\n" + "="*60)
        print("🔍 PERFORMANCE MONITORING REPORT")
        print("="*60)
        perf = summary['current_performance']
        print(f"\nAccuracy: {perf['accuracy']:.1%}")
        print(f"Health Status: {summary['health_status'].upper()}")
        if summary['alerts']:
            print(f"\n⚠️  Alerts: {len(summary['alerts'])}")
            for alert in summary['alerts']:
                print(f"   - {alert['alert_type']}: {alert['recommendation']}")
        print("="*60 + "\n")
        return True
    except Exception as e:
        logger.error(f"❌ Error monitoring performance: {e}")
        return False


def run_learning_cycle():
    """Run full self-learning cycle."""
    logger.info("🧠 Running learning cycle...")
    try:
        from self_learning_system import SelfLearningSystem
        system = SelfLearningSystem()
        summary = system.run_learning_cycle()

        print("\n" + "="*60)
        print("🧠 LEARNING CYCLE COMPLETE")
        print("="*60)
        if summary.get('success'):
            print(f"\nAccuracy: {summary['metrics']['accuracy']:.1%}")
            print(f"Retraining Needed: {'YES' if summary['needs_retraining'] else 'NO'}")
            if summary['improvement_opportunities']:
                print(f"\nImprovements Identified: {len(summary['improvement_opportunities'])}")
        print("="*60 + "\n")
        return True
    except Exception as e:
        logger.error(f"❌ Error running learning cycle: {e}")
        return False


def auto_predict():
    """Automatically generate predictions for upcoming games."""
    logger.info("🤖 Running automatic prediction generation...")
    try:
        from automatic_prediction_generator import AutomaticPredictionGenerator
        generator = AutomaticPredictionGenerator()
        summary = generator.run_automatic_predictions()

        print("\n" + "="*60)
        print("🤖 AUTOMATIC PREDICTIONS COMPLETE")
        print("="*60)
        if summary.get('success'):
            print(f"\nGames Processed: {summary['total_games']}")
            print(f"Predictions Saved: {summary['predictions_saved']}")
            if summary['errors'] > 0:
                print(f"Errors: {summary['errors']}")
        print("="*60 + "\n")
        return True
    except Exception as e:
        logger.error(f"❌ Error in auto prediction: {e}")
        return False


def full_auto():
    """Run complete automatic daily cycle."""
    logger.info("🚀 Running full automation cycle...")
    try:
        from full_automation import FullAutomation
        automation = FullAutomation(email_alerts=False)
        results = automation.run_daily_automation()

        print("\n" + "="*60)
        print("🚀 FULL AUTOMATION COMPLETE")
        print("="*60)
        if results.get('success'):
            print(f"\nTasks Completed: {len(results['tasks'])}")
            for task in results['tasks']:
                status = "✅" if task['status'] == 'success' else "⚠️ "
                print(f"  {status} {task['phase']}")
        print("="*60 + "\n")
        return True
    except Exception as e:
        logger.error(f"❌ Error in full automation: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="College Basketball Betting Intelligence System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python basketball_main.py --analyze-today
  python basketball_main.py --backtest --start-date 2024-01-01
  python basketball_main.py --paper-trade --bankroll 10000
  python basketball_main.py --init-db

Self-Learning:
  python basketball_main.py --update-outcomes
  python basketball_main.py --monitor-performance
  python basketball_main.py --run-learning-cycle

Full Automation:
  python basketball_main.py --auto-predict
  python basketball_main.py --full-auto
        """
    )

    parser.add_argument(
        '--init-db',
        action='store_true',
        help='Initialize database schemas'
    )

    parser.add_argument(
        '--analyze-today',
        action='store_true',
        help='Analyze today\'s games'
    )

    parser.add_argument(
        '--backtest',
        action='store_true',
        help='Run backtesting on historical data'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for backtest (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for backtest (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--paper-trade',
        action='store_true',
        help='Start paper trading mode'
    )

    parser.add_argument(
        '--bankroll',
        type=float,
        default=10000.0,
        help='Starting bankroll for paper trading (default: 10000)'
    )

    # Self-learning arguments
    parser.add_argument(
        '--update-outcomes',
        action='store_true',
        help='Update database with actual game outcomes'
    )

    parser.add_argument(
        '--monitor-performance',
        action='store_true',
        help='Monitor system performance and detect drift'
    )

    parser.add_argument(
        '--run-learning-cycle',
        action='store_true',
        help='Run full self-learning cycle'
    )

    # Full automation arguments
    parser.add_argument(
        '--auto-predict',
        action='store_true',
        help='Automatically generate predictions for upcoming games'
    )

    parser.add_argument(
        '--full-auto',
        action='store_true',
        help='Run complete automatic daily cycle (predict + outcomes + monitor)'
    )

    args = parser.parse_args()

    # Print banner
    print("\n" + "="*60)
    print("🏀 COLLEGE BASKETBALL BETTING INTELLIGENCE SYSTEM")
    print("="*60 + "\n")

    # Execute requested action
    if args.init_db:
        success = init_database()
        sys.exit(0 if success else 1)

    elif args.analyze_today:
        success = analyze_today()
        sys.exit(0 if success else 1)

    elif args.backtest:
        if not args.start_date:
            parser.error("--backtest requires --start-date")
        success = run_backtest(args.start_date, args.end_date)
        sys.exit(0 if success else 1)

    elif args.paper_trade:
        success = paper_trade(args.bankroll)
        sys.exit(0 if success else 1)

    elif args.update_outcomes:
        success = update_outcomes()
        sys.exit(0 if success else 1)

    elif args.monitor_performance:
        success = monitor_performance()
        sys.exit(0 if success else 1)

    elif args.run_learning_cycle:
        success = run_learning_cycle()
        sys.exit(0 if success else 1)

    elif args.auto_predict:
        success = auto_predict()
        sys.exit(0 if success else 1)

    elif args.full_auto:
        success = full_auto()
        sys.exit(0 if success else 1)

    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
