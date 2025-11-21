#!/bin/bash
# Simple wrapper to start the betting automation in loop mode
cd /home/ryan/college_basketball_system
echo "🏀 Starting College Basketball Betting Automation..."
echo "Logs will be saved to logs/automation.log"
echo "Press Ctrl+C to stop."
venv/bin/python auto_daily_bets.py --loop
