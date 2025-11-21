#!/bin/bash
cd /home/ryan/college_basketball_system
echo "🏀 Loading College Basketball Dashboard..."
echo "=========================================="
venv/bin/python paper_trading.py --dashboard
echo ""
echo "=========================================="
echo "Press Enter to close..."
read
