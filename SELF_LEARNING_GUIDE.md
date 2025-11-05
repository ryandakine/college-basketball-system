# 🧠 Self-Learning System Guide

## Overview

The College Basketball Betting System includes a comprehensive self-learning framework that automatically improves predictions over time by:

- **Tracking actual outcomes** and comparing them to predictions
- **Detecting performance drift** when accuracy degrades
- **Identifying improvement opportunities** through pattern analysis
- **Triggering automatic retraining** when needed
- **Monitoring model calibration** to ensure confidence matches reality

## Architecture

```
Self-Learning System
├── Automatic Outcome Tracker      # Fetches game results from ESPN API
├── Performance Monitor             # Detects drift and calibration issues
├── Self-Learning Orchestrator      # Coordinates learning cycles
└── Scheduled Automation            # Daily/weekly automated runs
```

## Components

### 1. Automatic Outcome Tracker (`automatic_outcome_tracker.py`)

**Purpose**: Fetch actual game results and update the database

**Key Features**:
- Fetches completed games from ESPN API
- Parses scores, spreads, totals
- Identifies tournament context (regular season, conference tourney, March Madness)
- Updates predictions with actual outcomes
- Calculates accuracy metrics

**Usage**:
```bash
# Manual run
python automatic_outcome_tracker.py

# Via main system
python basketball_main.py --update-outcomes
```

**API Endpoint**:
```
https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard
```

### 2. Self-Learning System (`self_learning_system.py`)

**Purpose**: Analyze performance and identify improvements

**Key Features**:
- Fetches prediction history
- Calculates comprehensive learning metrics
- Detects model drift
- Identifies improvement opportunities
- Saves learning state to disk

**Metrics Tracked**:
- Overall accuracy
- Recent accuracy (7-day rolling)
- Tournament-specific performance
- Feature importance
- Improvement trends
- Confidence calibration

**Usage**:
```bash
# Manual run
python self_learning_system.py

# Via main system
python basketball_main.py --run-learning-cycle
```

**Drift Detection Thresholds**:
- Minimum samples for retraining: 50 predictions
- Accuracy threshold: 52%
- Drift threshold: 5% accuracy drop
- Retraining triggers when thresholds exceeded

### 3. Performance Monitor (`performance_monitor.py`)

**Purpose**: Real-time performance monitoring and drift detection

**Key Features**:
- Captures performance snapshots
- Calculates calibration scores
- Detects accuracy drift
- Identifies edge degradation
- Generates actionable alerts

**Alert Severities**:
- **🔴 Critical**: Immediate action required (accuracy <45%, drift >10%)
- **🟡 Warning**: Monitor closely (accuracy <50%, drift >5%)

**Monitored Metrics**:
| Metric | Description | Threshold |
|--------|-------------|-----------|
| Accuracy | Overall prediction accuracy | Warning: <50%, Critical: <45% |
| Calibration | Confidence vs actual accuracy | Warning: >15% error |
| Edge Detection | Performance on identified edges | Warning: <52% |
| Sharp Ratio | % predictions beating closing lines | Target: >52% |

**Usage**:
```bash
# Manual run
python performance_monitor.py

# Via main system
python basketball_main.py --monitor-performance
```

### 4. Scheduled Automation (`scheduled_self_improvement.py`)

**Purpose**: Automated daily/weekly self-improvement cycles

**Daily Tasks** (Recommended: 2 AM):
1. Fetch game outcomes (last 3 days)
2. Update database with actual results
3. Monitor performance for drift
4. Send email alerts if critical issues detected

**Weekly Tasks** (Recommended: Sunday 3 AM):
1. Run daily tasks
2. Full learning cycle analysis
3. Identify improvement opportunities
4. Generate recommendations
5. Send weekly summary email

**Usage**:
```bash
# Manual runs
python scheduled_self_improvement.py --daily
python scheduled_self_improvement.py --weekly --email-alerts

# Via cron (recommended)
# See cron setup below
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install requests numpy python-dotenv
```

### 2. Configure Environment

Add to `.env`:
```bash
# Email alerts (optional but recommended)
EMAIL_USER=your_email@gmail.com
EMAIL_PASS=your_app_password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
```

For Gmail, create an [App Password](https://support.google.com/accounts/answer/185833).

### 3. Initialize Database

```bash
python basketball_main.py --init-db
```

### 4. Manual Testing

Test each component:

```bash
# Test outcome tracking
python basketball_main.py --update-outcomes

# Test performance monitoring
python basketball_main.py --monitor-performance

# Test learning cycle
python basketball_main.py --run-learning-cycle
```

### 5. Setup Automated Runs (Cron)

Create cron jobs for automated execution:

```bash
# Edit crontab
crontab -e

# Add these lines (adjust paths):
# Daily at 2 AM - Update outcomes and monitor
0 2 * * * cd /path/to/college-basketball-system && /path/to/python scheduled_self_improvement.py --daily --email-alerts >> self_improvement.log 2>&1

# Weekly on Sunday at 3 AM - Full learning cycle
0 3 * * 0 cd /path/to/college-basketball-system && /path/to/python scheduled_self_improvement.py --weekly --email-alerts >> self_improvement.log 2>&1
```

**Example with virtual environment**:
```bash
0 2 * * * cd /home/user/college-basketball-system && /home/user/college-basketball-system/venv/bin/python scheduled_self_improvement.py --daily --email-alerts >> self_improvement.log 2>&1
```

## Monitoring & Alerts

### Email Alerts

**Critical Alerts** (immediate):
- Accuracy drops below 45%
- Accuracy drift exceeds 10%
- System health status is critical

**Weekly Summary** (informational):
- Overall performance metrics
- Accuracy trends
- Improvement recommendations
- Retraining recommendations

### Log Files

**Location**: `./self_improvement.log`

**What's Logged**:
- All outcome fetches
- Performance snapshots
- Drift detections
- Learning cycle results
- Errors and warnings

**Monitoring Commands**:
```bash
# View recent logs
tail -f self_improvement.log

# Check for errors
grep ERROR self_improvement.log

# View today's activity
grep "$(date +%Y-%m-%d)" self_improvement.log
```

## Performance Metrics

### Overall System Health

| Metric | Healthy | Warning | Critical |
|--------|---------|---------|----------|
| Accuracy | ≥52% | 50-52% | <50% |
| Recent Accuracy | ≥50% | 45-50% | <45% |
| Calibration Error | <10% | 10-15% | >15% |
| Edge Performance | ≥52% | 50-52% | <50% |

### Learning State File

**Location**: `models/learning_state.json`

**Contains**:
```json
{
  "metrics": {
    "total_predictions": 150,
    "correct_predictions": 82,
    "accuracy": 0.547,
    "recent_accuracy": 0.520,
    "tournament_performance": {
      "regular_season": 0.552,
      "conference_tournament": 0.538,
      "march_madness": 0.515
    },
    "feature_importance": {
      "tempo_differential": 0.85,
      "efficiency_rating": 0.80,
      "tournament_experience": 0.75
    }
  },
  "iteration_count": 12,
  "last_updated": "2025-01-15T03:00:00"
}
```

## Improvement Opportunities

The system automatically identifies:

1. **High-Confidence Failures**
   - When high-confidence predictions (>70%) are wrong
   - Action: Review confidence calibration

2. **Tournament Performance Issues**
   - When March Madness accuracy <50%
   - Action: Increase tournament-specific feature weights

3. **Spread Accuracy Problems**
   - When average spread error >8 points
   - Action: Improve tempo and efficiency modeling

4. **Total Prediction Issues**
   - When average total error >10 points
   - Action: Refine pace prediction model

## Retraining Triggers

Automatic retraining is recommended when:

1. Total predictions ≥ 50 samples
2. **AND** one of:
   - Overall accuracy < 52%
   - Recent accuracy drop > 5%
   - Declining trend for 3+ weeks
   - Edge performance degradation

## Safety Features

### Timeouts
- Maximum runtime: 5 minutes per cycle
- Prevents infinite loops
- Automatic termination via signal handling

### Token Limits
- Maximum iterations: 3 per cycle
- Maximum tokens: 24,000 per cycle
- Prevents runaway processes

### Data Validation
- Validates API responses
- Handles missing data gracefully
- Logs all errors for debugging

## Troubleshooting

### Problem: No outcomes being fetched

**Solution**:
```bash
# Check ESPN API directly
curl "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"

# Check logs
grep "Fetching games" self_improvement.log

# Verify date format
python -c "from datetime import datetime; print(datetime.now().strftime('%Y%m%d'))"
```

### Problem: Accuracy calculation is 0%

**Solution**:
- Ensure predictions exist in database
- Check that outcomes have been fetched
- Verify database contains actual_spread values

```bash
sqlite3 basketball_betting.db "SELECT COUNT(*) FROM basketball_predictions WHERE actual_spread IS NOT NULL;"
```

### Problem: Email alerts not sending

**Solution**:
```bash
# Verify environment variables
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('EMAIL_USER'))"

# Test SMTP connection
python -c "import smtplib; s = smtplib.SMTP('smtp.gmail.com', 587); s.starttls(); print('Success')"

# Check Gmail app password (not regular password)
```

### Problem: Cron job not running

**Solution**:
```bash
# Check cron is running
sudo service cron status

# View cron logs
grep CRON /var/log/syslog

# Test command manually
cd /path/to/system && python scheduled_self_improvement.py --daily

# Check file permissions
ls -la scheduled_self_improvement.py
```

## Advanced Configuration

### Custom Thresholds

Edit `self_learning_system.py`:
```python
self.min_samples_for_retraining = 50  # Minimum predictions needed
self.accuracy_threshold = 0.52        # 52% minimum accuracy
self.drift_threshold = 0.05           # 5% accuracy drop triggers retraining
```

### Custom Email Templates

Edit `scheduled_self_improvement.py`:
- `_send_alert_email()` for critical alerts
- `_send_weekly_summary()` for weekly reports

### Lookback Windows

Adjust in respective files:
- Outcome tracking: `lookback_days=7` (automatic_outcome_tracker.py)
- Performance monitoring: `'-30 days'` (performance_monitor.py)
- Learning metrics: `days=90` (self_learning_system.py)

## Best Practices

1. **Start with manual runs** - Test each component before automation
2. **Monitor logs regularly** - Check for errors and warnings
3. **Review weekly summaries** - Track improvement trends
4. **Act on critical alerts** - Don't ignore performance degradation
5. **Retrain when recommended** - Trust the drift detection
6. **Keep historical data** - Maintain at least 90 days of predictions

## Integration with Main System

The self-learning system integrates seamlessly:

```python
# In your prediction workflow
prediction = engine.predict_game(...)

# Prediction is automatically logged to database
# Self-learning system will:
#   1. Fetch actual outcome (next day)
#   2. Calculate if prediction was correct
#   3. Update learning metrics
#   4. Detect drift if accuracy drops
#   5. Recommend retraining if needed
```

## Success Metrics

Track these over time:

- **Accuracy improvement**: Target +2-5% per season
- **Calibration improvement**: Target <10% error
- **Edge detection**: Maintain >52% on identified edges
- **Sharp ratio**: Beat closing lines >52% of time

## Support

For issues:
1. Check logs: `tail -f self_improvement.log`
2. Run manual tests with verbose logging
3. Verify database schema and data
4. Check API connectivity

---

**🧠 Keep Learning! Your system improves with every prediction! 🏀**
