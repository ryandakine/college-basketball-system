#!/bin/bash
# CBB Auto-Tracker - Runs every day at 6 PM Mountain Time
#
# To install:
# crontab -e
# Add: 0 18 * * * /home/ryan/college-basketball-system/cbb_auto_tracker.sh >> /home/ryan/college-basketball-system/logs/cbb_auto_$(date +\%Y\%m\%d).log 2>&1

cd /home/ryan/college-basketball-system || exit

echo "================================================================"
echo "CBB Auto-Tracker running at $(date)"
echo "================================================================"

# Run the tracker
python3 cbb_daily_tracker.py

echo "================================================================"
echo "Completed at $(date)"
echo ""
