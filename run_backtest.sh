#!/bin/bash
# Quick Backtest Runner - Tests your system on historical data

set -e

echo "🏀 Basketball System Backtest"
echo "=============================="
echo ""

# Check if data exists
if [ ! -f "scraped_games.csv" ] && [ ! -f "basketball_betting.db" ]; then
    echo "❌ No historical data found!"
    echo ""
    echo "First, scrape real games:"
    echo "  python real_historical_data_scraper.py"
    echo ""
    exit 1
fi

echo "✅ Historical data found"
echo ""

# Run backtest
echo "Running backtest..."
echo "- Initial Bankroll: \$10,000"
echo "- Bet Sizing: Quarter Kelly"
echo "- Min Edge: 2%"
echo "- Min Confidence: 55%"
echo ""

python basketball_main.py --backtest

echo ""
echo "📊 Results saved to: backtest_results.json"
echo ""
