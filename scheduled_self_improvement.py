#!/usr/bin/env python3
"""
Scheduled Self-Improvement System
=================================

Automated daily/weekly self-improvement cycles:
- Daily: Fetch outcomes, update database, monitor performance
- Weekly: Run full learning cycle, retrain if needed
- Alert on critical issues

Can be run via cron:
    0 2 * * * python scheduled_self_improvement.py --daily
    0 3 * * 0 python scheduled_self_improvement.py --weekly
"""

import argparse
import logging
import smtplib
import sys
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Dict, List

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from automatic_outcome_tracker import AutomaticOutcomeTracker
from self_learning_system import SelfLearningSystem
from performance_monitor import PerformanceMonitor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('self_improvement.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ScheduledSelfImprovement:
    """Orchestrates scheduled self-improvement tasks."""

    def __init__(self, email_alerts: bool = False):
        self.outcome_tracker = AutomaticOutcomeTracker()
        self.learning_system = SelfLearningSystem()
        self.performance_monitor = PerformanceMonitor()
        self.email_alerts = email_alerts

    def run_daily_tasks(self) -> Dict:
        """
        Run daily self-improvement tasks:
        1. Fetch game outcomes
        2. Update database
        3. Monitor performance
        4. Alert if issues detected
        """
        logger.info("="*60)
        logger.info("🌅 STARTING DAILY SELF-IMPROVEMENT TASKS")
        logger.info("="*60)

        results = {
            'timestamp': datetime.now().isoformat(),
            'success': True,
            'tasks_completed': [],
            'alerts': []
        }

        try:
            # Task 1: Fetch and update outcomes
            logger.info("📥 Task 1: Fetching game outcomes...")
            metrics = self.outcome_tracker.run_daily_update(lookback_days=3)
            results['tasks_completed'].append({
                'task': 'outcome_tracking',
                'status': 'success',
                'accuracy': metrics.get('overall_accuracy', 0)
            })

            # Task 2: Monitor performance
            logger.info("🔍 Task 2: Monitoring system performance...")
            monitor_summary = self.performance_monitor.run_monitoring_cycle()
            results['tasks_completed'].append({
                'task': 'performance_monitoring',
                'status': 'success',
                'health_status': monitor_summary['health_status']
            })

            # Check for critical alerts
            if monitor_summary['alerts']:
                critical_alerts = [
                    a for a in monitor_summary['alerts']
                    if a.get('severity') == 'critical'
                ]
                if critical_alerts:
                    results['alerts'].extend(critical_alerts)
                    if self.email_alerts:
                        self._send_alert_email(critical_alerts, 'CRITICAL')

            logger.info("✅ Daily tasks completed successfully")

        except Exception as e:
            logger.error(f"❌ Error in daily tasks: {e}")
            results['success'] = False
            results['error'] = str(e)

        return results

    def run_weekly_tasks(self) -> Dict:
        """
        Run weekly self-improvement tasks:
        1. Run daily tasks first
        2. Full learning cycle
        3. Identify improvement opportunities
        4. Trigger retraining if needed
        """
        logger.info("="*60)
        logger.info("📅 STARTING WEEKLY SELF-IMPROVEMENT TASKS")
        logger.info("="*60)

        results = {
            'timestamp': datetime.now().isoformat(),
            'success': True,
            'tasks_completed': [],
            'recommendations': []
        }

        try:
            # Run daily tasks first
            logger.info("Running daily tasks as prerequisite...")
            daily_results = self.run_daily_tasks()
            results['tasks_completed'].extend(daily_results['tasks_completed'])

            # Task 3: Full learning cycle
            logger.info("🧠 Task 3: Running full learning cycle...")
            learning_summary = self.learning_system.run_learning_cycle()
            results['tasks_completed'].append({
                'task': 'learning_cycle',
                'status': 'success' if learning_summary.get('success') else 'failed',
                'accuracy': learning_summary.get('metrics', {}).get('accuracy', 0)
            })

            # Task 4: Process recommendations
            if learning_summary.get('needs_retraining'):
                results['recommendations'].append({
                    'type': 'retraining',
                    'priority': 'high',
                    'message': 'Model retraining recommended due to performance drift'
                })

            if learning_summary.get('improvement_opportunities'):
                for opp in learning_summary['improvement_opportunities']:
                    results['recommendations'].append({
                        'type': 'improvement',
                        'priority': 'medium',
                        'message': opp
                    })

            # Send weekly summary email
            if self.email_alerts:
                self._send_weekly_summary(results)

            logger.info("✅ Weekly tasks completed successfully")

        except Exception as e:
            logger.error(f"❌ Error in weekly tasks: {e}")
            results['success'] = False
            results['error'] = str(e)

        return results

    def _send_alert_email(self, alerts: List[Dict], severity: str):
        """Send email alert for critical issues."""
        try:
            # Email configuration (should be in .env)
            import os
            from dotenv import load_dotenv
            load_dotenv()

            sender = os.getenv('EMAIL_USER')
            password = os.getenv('EMAIL_PASS')
            recipient = sender  # Send to self

            if not sender or not password:
                logger.warning("Email credentials not configured")
                return

            # Create email
            msg = MIMEMultipart()
            msg['From'] = sender
            msg['To'] = recipient
            msg['Subject'] = f"🚨 {severity} ALERT - Basketball Betting System"

            body = f"""
{severity} ALERT - College Basketball Betting System
{'='*60}

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Active Alerts ({len(alerts)}):
"""
            for alert in alerts:
                body += f"""
Alert Type: {alert.get('alert_type', 'unknown')}
Metric: {alert.get('metric', 'unknown')}
Current Value: {alert.get('current_value', 0):.1%}
Expected Value: {alert.get('expected_value', 0):.1%}
Deviation: {alert.get('deviation', 0):.1%}
Recommendation: {alert.get('recommendation', 'None')}
"""

            body += f"""
{'='*60}

Action Required: Review system performance and consider immediate intervention.

Dashboard: http://localhost:8000/dashboard
Logs: ./self_improvement.log
"""

            msg.attach(MIMEText(body, 'plain'))

            # Send email
            server = smtplib.SMTP(os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
                                int(os.getenv('SMTP_PORT', '587')))
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
            server.quit()

            logger.info(f"Sent {severity} alert email to {recipient}")

        except Exception as e:
            logger.error(f"Error sending alert email: {e}")

    def _send_weekly_summary(self, results: Dict):
        """Send weekly summary email."""
        try:
            import os
            from dotenv import load_dotenv
            load_dotenv()

            sender = os.getenv('EMAIL_USER')
            password = os.getenv('EMAIL_PASS')
            recipient = sender

            if not sender or not password:
                return

            msg = MIMEMultipart()
            msg['From'] = sender
            msg['To'] = recipient
            msg['Subject'] = "📊 Weekly Self-Improvement Summary - Basketball System"

            # Find accuracy from tasks
            learning_task = next(
                (t for t in results['tasks_completed'] if t['task'] == 'learning_cycle'),
                {}
            )
            accuracy = learning_task.get('accuracy', 0)

            body = f"""
Weekly Self-Improvement Summary
{'='*60}

Week Ending: {datetime.now().strftime('%Y-%m-%d')}
System Status: {'✅ Healthy' if results['success'] else '❌ Issues Detected'}

Performance Metrics:
  Overall Accuracy: {accuracy:.1%}
  Tasks Completed: {len(results['tasks_completed'])}
  Recommendations: {len(results['recommendations'])}

"""

            if results['recommendations']:
                body += "\nRecommendations:\n"
                for rec in results['recommendations']:
                    body += f"  [{rec['priority'].upper()}] {rec['message']}\n"

            body += f"""
{'='*60}

Full logs available at: ./self_improvement.log
Dashboard: http://localhost:8000/dashboard

Keep learning! 🧠🏀
"""

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
                                int(os.getenv('SMTP_PORT', '587')))
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
            server.quit()

            logger.info(f"Sent weekly summary email to {recipient}")

        except Exception as e:
            logger.error(f"Error sending weekly summary: {e}")


def main():
    """Main entry point for scheduled tasks."""
    parser = argparse.ArgumentParser(
        description="Scheduled Self-Improvement System",
        epilog="""
Cron Examples:
  # Daily at 2 AM
  0 2 * * * cd /path/to/system && python scheduled_self_improvement.py --daily

  # Weekly on Sunday at 3 AM
  0 3 * * 0 cd /path/to/system && python scheduled_self_improvement.py --weekly
        """
    )

    parser.add_argument(
        '--daily',
        action='store_true',
        help='Run daily self-improvement tasks'
    )

    parser.add_argument(
        '--weekly',
        action='store_true',
        help='Run weekly self-improvement tasks'
    )

    parser.add_argument(
        '--email-alerts',
        action='store_true',
        help='Enable email alerts for critical issues'
    )

    args = parser.parse_args()

    # Initialize system
    system = ScheduledSelfImprovement(email_alerts=args.email_alerts)

    if args.daily:
        results = system.run_daily_tasks()
        print("\n" + "="*60)
        print("🌅 DAILY TASKS SUMMARY")
        print("="*60)
        print(f"\nStatus: {'✅ SUCCESS' if results['success'] else '❌ FAILED'}")
        print(f"Tasks Completed: {len(results['tasks_completed'])}")
        if results.get('alerts'):
            print(f"⚠️  Alerts: {len(results['alerts'])}")
        print("="*60 + "\n")

    elif args.weekly:
        results = system.run_weekly_tasks()
        print("\n" + "="*60)
        print("📅 WEEKLY TASKS SUMMARY")
        print("="*60)
        print(f"\nStatus: {'✅ SUCCESS' if results['success'] else '❌ FAILED'}")
        print(f"Tasks Completed: {len(results['tasks_completed'])}")
        if results.get('recommendations'):
            print(f"\n💡 Recommendations: {len(results['recommendations'])}")
            for rec in results['recommendations']:
                print(f"   [{rec['priority'].upper()}] {rec['message']}")
        print("="*60 + "\n")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
