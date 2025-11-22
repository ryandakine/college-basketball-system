#!/bin/bash
# Wrapper to run the women's basketball scraper in background

# Ensure logs directory exists
mkdir -p logs

echo "🏀 Starting Women's Basketball Data Scrape..."
echo "Logs will be saved to logs/womens_scrape.log"
echo "Running in background..."

nohup venv/bin/python scrape_womens_basketball.py --all > logs/womens_scrape.log 2>&1 &

echo "Process started with PID $!"
echo "You can check progress with: tail -f logs/womens_scrape.log"
