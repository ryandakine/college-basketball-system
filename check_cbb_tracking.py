#!/usr/bin/env python3
"""
Simple CBB Tracking Status Checker
"""

import json
from pathlib import Path


tracking_file = Path('data/cbb_auto_tracking.json')

print("\n📊 CBB TRACKING STATUS")
print("=" * 60)

if tracking_file.exists():
    with open(tracking_file) as f:
        data = json.load(f)
    
    runs = data.get('runs', [])
    print(f"✅ Tracking Active")
    print(f"   Started: {data.get('start_date')}")
    print(f"   Total Runs: {len(runs)}")
    
    if runs:
        print(f"\n   Latest run: {runs[-1]['date']} at {runs[-1]['time']}")
else:
    print("⚠️  No tracking data yet")
    print("\nTo start tracking:")
    print("   python3 cbb_daily_tracker.py")

print("\n🤖 AUTOMATION SETUP:")
print("=" * 60)
print("To run automatically every day at 6 PM:")
print("\n1. Run: crontab -e")
print("2. Add: 0 18 * * * /home/ryan/college-basketball-system/cbb_auto_tracker.sh")
print("3. Save")

print("\n📋 MANUAL COMMANDS:")
print("=" * 60)
print("Run tracker now:  python3 cbb_daily_tracker.py")
print("Check status:     python3 check_cbb_tracking.py")
print("View data:        cat data/cbb_auto_tracking.json")
