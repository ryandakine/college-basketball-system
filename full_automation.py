#!/usr/bin/env python3
"""
Full End-to-End Automation System
=================================

COMPLETELY AUTOMATIC - No manual input needed!

Daily Cycle (runs every day):
1. Fetch today's upcoming games
2. Generate predictions automatically
3. Save predictions to database
4. Fetch yesterday's outcomes
5. Update database with actual results
6. Monitor performance
7. Alert if issues detected

Weekly Cycle (runs every Sunday):
1. Run daily cycle
2. Full learning analysis
3. Identify improvements
4. Recommend retraining
5. Send weekly summary

Usage:
    python full_automation.py --daily
    python full_automation.py --weekly

Cron:
    0 10 * * * python full_automation.py --daily --email-alerts
    0 4 * * 0 python full_automation.py --weekly --email-alerts
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from automatic_game_fetcher import AutomaticGameFetcher
from automatic_prediction_generator import AutomaticPredictionGenerator
from automatic_outcome_tracker import AutomaticOutcomeTracker
from performance_monitor import PerformanceMonitor
from self_learning_system import SelfLearningSystem

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('full_automation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FullAutomation:
    """Complete end-to-end automation orchestrator."""

    def __init__(self, email_alerts: bool = False):
        self.game_fetcher = AutomaticGameFetcher()
        self.prediction_generator = AutomaticPredictionGenerator()
        self.outcome_tracker = AutomaticOutcomeTracker()
        self.performance_monitor = PerformanceMonitor()
        self.learning_system = SelfLearningSystem()
        self.email_alerts = email_alerts

    def run_daily_automation(self):
        """
        Run complete daily automation cycle.

        Morning (10 AM recommended):
        1. Fetch today's games
        2. Generate predictions
        3. Fetch yesterday's outcomes
        4. Update performance metrics
        5. Monitor for drift
        6. Alert if critical issues
        """
        logger.info("="*70)
        logger.info("🤖 STARTING DAILY AUTOMATION CYCLE")
        logger.info(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*70)

        results = {
            'timestamp': datetime.now().isoformat(),
            'success': True,
            'tasks': []
        }

        try:
            # ═══════════════════════════════════════════════════════
            # PHASE 1: GENERATE NEW PREDICTIONS
            # ═══════════════════════════════════════════════════════
            logger.info("\n" + "─"*70)
            logger.info("📊 PHASE 1: Generating Predictions for Today's Games")
            logger.info("─"*70)

            # Fetch games needing predictions
            games = self.game_fetcher.get_games_needing_predictions()
            logger.info(f"Found {len(games)} games needing predictions")

            if games:
                # Generate predictions
                pred_summary = self.prediction_generator.generate_predictions_for_games(games)
                results['tasks'].append({
                    'phase': 'prediction_generation',
                    'status': 'success',
                    'games_processed': pred_summary['total_games'],
                    'predictions_saved': pred_summary['predictions_saved']
                })
                logger.info(f"✅ Generated {pred_summary['predictions_saved']} predictions")
            else:
                logger.info("✅ No games today - system up to date")
                results['tasks'].append({
                    'phase': 'prediction_generation',
                    'status': 'no_games',
                    'games_processed': 0
                })

            # ═══════════════════════════════════════════════════════
            # PHASE 2: UPDATE OUTCOMES FROM YESTERDAY
            # ═══════════════════════════════════════════════════════
            logger.info("\n" + "─"*70)
            logger.info("📥 PHASE 2: Fetching Outcomes from Yesterday")
            logger.info("─"*70)

            outcome_metrics = self.outcome_tracker.run_daily_update(lookback_days=2)
            results['tasks'].append({
                'phase': 'outcome_tracking',
                'status': 'success',
                'accuracy': outcome_metrics.get('overall_accuracy', 0),
                'predictions_updated': outcome_metrics.get('total_predictions', 0)
            })
            logger.info(f"✅ Updated outcomes - Accuracy: {outcome_metrics.get('overall_accuracy', 0):.1f}%")

            # ═══════════════════════════════════════════════════════
            # PHASE 3: MONITOR PERFORMANCE
            # ═══════════════════════════════════════════════════════
            logger.info("\n" + "─"*70)
            logger.info("🔍 PHASE 3: Monitoring System Performance")
            logger.info("─"*70)

            monitor_summary = self.performance_monitor.run_monitoring_cycle()
            results['tasks'].append({
                'phase': 'performance_monitoring',
                'status': 'success',
                'health_status': monitor_summary['health_status'],
                'alerts': len(monitor_summary['alerts'])
            })

            # Check for critical alerts
            if monitor_summary['alerts']:
                critical_alerts = [
                    a for a in monitor_summary['alerts']
                    if a.get('severity') == 'critical'
                ]
                if critical_alerts:
                    logger.warning(f"🚨 {len(critical_alerts)} CRITICAL ALERTS detected!")
                    results['critical_alerts'] = critical_alerts
                    if self.email_alerts:
                        self._send_alert_email(critical_alerts)

            logger.info(f"✅ Performance: {monitor_summary['health_status'].upper()}")

            # ═══════════════════════════════════════════════════════
            # SUMMARY
            # ═══════════════════════════════════════════════════════
            logger.info("\n" + "="*70)
            logger.info("✅ DAILY AUTOMATION CYCLE COMPLETE")
            logger.info(f"   Predictions Generated: {pred_summary.get('predictions_saved', 0)}")
            logger.info(f"   Outcomes Updated: {outcome_metrics.get('total_predictions', 0)}")
            logger.info(f"   System Health: {monitor_summary['health_status'].upper()}")
            logger.info("="*70 + "\n")

        except Exception as e:
            logger.error(f"❌ Error in daily automation: {e}")
            results['success'] = False
            results['error'] = str(e)

        return results

    def run_weekly_automation(self):
        """
        Run complete weekly automation cycle.

        Sunday Morning (4 AM recommended):
        1. Run daily automation first
        2. Full learning cycle analysis
        3. Identify improvement opportunities
        4. Recommend retraining if needed
        5. Send weekly summary email
        """
        logger.info("="*70)
        logger.info("📅 STARTING WEEKLY AUTOMATION CYCLE")
        logger.info(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*70)

        results = {
            'timestamp': datetime.now().isoformat(),
            'success': True,
            'tasks': []
        }

        try:
            # ═══════════════════════════════════════════════════════
            # PHASE 1-3: Run Daily Automation First
            # ═══════════════════════════════════════════════════════
            logger.info("\n📊 Running Daily Automation as Prerequisites...")
            daily_results = self.run_daily_automation()
            results['tasks'].extend(daily_results.get('tasks', []))

            # ═══════════════════════════════════════════════════════
            # PHASE 4: FULL LEARNING CYCLE
            # ═══════════════════════════════════════════════════════
            logger.info("\n" + "─"*70)
            logger.info("🧠 PHASE 4: Running Full Learning Cycle")
            logger.info("─"*70)

            learning_summary = self.learning_system.run_learning_cycle()
            results['tasks'].append({
                'phase': 'learning_cycle',
                'status': 'success' if learning_summary.get('success') else 'failed',
                'accuracy': learning_summary.get('metrics', {}).get('accuracy', 0),
                'needs_retraining': learning_summary.get('needs_retraining', False)
            })

            # Extract key metrics
            metrics = learning_summary.get('metrics', {})
            opportunities = learning_summary.get('improvement_opportunities', [])
            needs_retraining = learning_summary.get('needs_retraining', False)

            logger.info(f"✅ Learning Cycle Complete")
            logger.info(f"   Accuracy: {metrics.get('accuracy', 0):.1%}")
            logger.info(f"   Retraining Needed: {'YES' if needs_retraining else 'NO'}")
            logger.info(f"   Improvement Opportunities: {len(opportunities)}")

            # ═══════════════════════════════════════════════════════
            # PHASE 5: WEEKLY SUMMARY
            # ═══════════════════════════════════════════════════════
            if self.email_alerts:
                self._send_weekly_summary(results, metrics, opportunities, needs_retraining)

            logger.info("\n" + "="*70)
            logger.info("✅ WEEKLY AUTOMATION CYCLE COMPLETE")
            logger.info("="*70 + "\n")

        except Exception as e:
            logger.error(f"❌ Error in weekly automation: {e}")
            results['success'] = False
            results['error'] = str(e)

        return results

    def _send_alert_email(self, alerts):
        """Send critical alert email."""
        try:
            import os
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            from dotenv import load_dotenv

            load_dotenv()

            sender = os.getenv('EMAIL_USER')
            password = os.getenv('EMAIL_PASS')

            if not sender or not password:
                logger.warning("Email credentials not configured")
                return

            msg = MIMEMultipart()
            msg['From'] = sender
            msg['To'] = sender
            msg['Subject'] = f"🚨 CRITICAL ALERT - Basketball System"

            body = f"""
🚨 CRITICAL ALERT - Automatic Basketball System
{'='*60}

{len(alerts)} critical issues detected!

"""
            for alert in alerts:
                body += f"""
Alert: {alert.get('alert_type', 'unknown')}
Metric: {alert.get('metric', 'unknown')}
Current: {alert.get('current_value', 0):.1%}
Expected: {alert.get('expected_value', 0):.1%}
Action: {alert.get('recommendation', 'None')}

"""

            body += f"""
{'='*60}

View logs: ./full_automation.log
"""

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
                                int(os.getenv('SMTP_PORT', '587')))
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
            server.quit()

            logger.info(f"Sent critical alert email")

        except Exception as e:
            logger.error(f"Error sending alert email: {e}")

    def _send_weekly_summary(self, results, metrics, opportunities, needs_retraining):
        """Send weekly summary email."""
        try:
            import os
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            from dotenv import load_dotenv

            load_dotenv()

            sender = os.getenv('EMAIL_USER')
            password = os.getenv('EMAIL_PASS')

            if not sender or not password:
                return

            msg = MIMEMultipart()
            msg['From'] = sender
            msg['To'] = sender
            msg['Subject'] = "📊 Weekly Automation Summary - Basketball System"

            body = f"""
📊 Weekly Automation Summary
{'='*60}

Week Ending: {datetime.now().strftime('%Y-%m-%d')}
Status: {'✅ Healthy' if results['success'] else '❌ Issues'}

🤖 FULLY AUTOMATIC OPERATION:
  System fetching games automatically ✅
  System generating predictions automatically ✅
  System tracking outcomes automatically ✅
  System learning and improving automatically ✅

📈 Performance:
  Accuracy: {metrics.get('accuracy', 0):.1%}
  Recent: {metrics.get('recent_accuracy', 0):.1%}
  Total Predictions: {metrics.get('total_predictions', 0)}

🔄 Retraining: {'⚠️  RECOMMENDED' if needs_retraining else '✅ Not Needed'}

"""

            if opportunities:
                body += "\n💡 Improvement Opportunities:\n"
                for opp in opportunities:
                    body += f"  • {opp}\n"

            body += f"""
{'='*60}

View detailed logs: ./full_automation.log

Keep learning! 🧠🏀
"""

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
                                int(os.getenv('SMTP_PORT', '587')))
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
            server.quit()

            logger.info(f"Sent weekly summary email")

        except Exception as e:
            logger.error(f"Error sending weekly summary: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Full End-to-End Automation System",
        epilog="""
Cron Examples:
  # Daily at 10 AM (after games finish, before next games start)
  0 10 * * * python full_automation.py --daily --email-alerts

  # Weekly on Sunday at 4 AM
  0 4 * * 0 python full_automation.py --weekly --email-alerts
        """
    )

    parser.add_argument(
        '--daily',
        action='store_true',
        help='Run daily automation cycle'
    )

    parser.add_argument(
        '--weekly',
        action='store_true',
        help='Run weekly automation cycle'
    )

    parser.add_argument(
        '--email-alerts',
        action='store_true',
        help='Enable email alerts'
    )

    args = parser.parse_args()

    automation = FullAutomation(email_alerts=args.email_alerts)

    if args.daily:
        results = automation.run_daily_automation()
        sys.exit(0 if results['success'] else 1)

    elif args.weekly:
        results = automation.run_weekly_automation()
        sys.exit(0 if results['success'] else 1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
