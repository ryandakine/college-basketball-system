#!/usr/bin/env python3
"""
CBB Revenue Script Detector
============================

Detects revenue maximization patterns in college basketball:
- Primetime games with packed arenas
- March Madness manufactured drama
- Overtime engineering in big games
- Conference tournament manipulation
"""

import json
import logging
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_cbb_revenue_patterns(games_data: List[Dict]) -> Dict:
    """
    Analyze CBB games for revenue maximization patterns.
    
    Patterns to detect:
    1. ESPN GameDay primetime games with drama
    2. March Madness overtime games (ads)
    3. Conference tournament championship finishes
    4. Blue blood rivalry overtime rate
    5. Sweet 16/Elite 8 close games (max engagement)
    """
    print("🏀 CBB REVENUE MAXIMIZATION DETECTOR")
    print("=" * 100)
    print(f"Analyzing {len(games_data)} games...")
    print()
    
    revenue_games = []
    overtime_games = []
    primetime_drama = []
    
    for game in games_data:
        home = game.get('home_team', '')
        away = game.get('away_team', '')
        home_score = game.get('home_score', 0)
        away_score = game.get('away_score', 0)
        margin = abs(home_score - away_score)
        attendance = game.get('attendance', 0)
        is_primetime = game.get('is_primetime', False)
        tournament = game.get('tournament_context') == 'march_madness'
        went_ot = game.get('overtime', False)
        tv_coverage = game.get('tv_coverage', '')
        
        # Blue bloods
        blue_bloods = ['Duke', 'UNC', 'Kansas', 'Kentucky', 'UCLA']
        is_blue_blood_game = home in blue_bloods or away in blue_bloods
        
        # Revenue maximization criteria:
        # 1. Primetime + packed arena + close game
        # 2. March Madness + overtime (extra ads)
        # 3. ESPN GameDay + tight finish
        
        if tournament and went_ot and margin <= 5:
            revenue_games.append({
                'game': f"{away} @ {home}",
                'score': f"{away_score}-{home_score}",
                'margin': margin,
                'overtime': went_ot,
                'pattern': 'March Madness OT Drama'
            })
        
        if is_primetime and is_blue_blood_game and margin <= 3:
            primetime_drama.append({
                'game': f"{away} @ {home}",
                'margin': margin,
                'pattern': 'Primetime Blue Blood Drama'
            })
        
        if went_ot:
            overtime_games.append(game)
    
    # Analysis
    print(f"🚨 REVENUE PATTERNS DETECTED:")
    print(f"   Total potential revenue games: {len(revenue_games)}")
    print(f"   March Madness OT games: {len([g for g in revenue_games if 'March Madness OT' in g['pattern']])}")
    print(f"   Primetime drama games: {len(primetime_drama)}")
    print(f"   All overtime games: {len(overtime_games)}")
    print()
    
    if revenue_games:
        print("Sample 'Perfect Storm' Revenue Games:")
        print("-" * 100)
        for game in revenue_games[:5]:
            print(f"   {game['game']}: {game['score']} (margin: {game['margin']})")
            print(f"   Pattern: {game['pattern']}")
            print()
    
    # Calculate OT rate
    total_games = len(games_data)
    ot_rate = len(overtime_games) / total_games if total_games > 0 else 0
    
    print(f"📊 OVERTIME ANALYSIS:")
    print(f"   Overtime rate: {ot_rate:.1%}")
    print(f"   Expected OT rate: ~6-8%")
    if ot_rate > 0.10:
        print(f"   ⚠️  ELEVATED OT RATE - Possible manipulation")
    
    print("\n" + "=" * 100)
    print("💡 KEY INSIGHTS:")
    print("   - March Madness OT games = max ad revenue")
    print("   - Primetime blue bloods = max engagement")
    print("   - Conference tournament finals often go OT")
    print("   - Sweet 16/Elite 8 consistently close")
    
    return {
        'revenue_games': revenue_games,
        'overtime_games': len(overtime_games),
        'ot_rate': ot_rate,
        'primetime_drama': len(primetime_drama)
    }


if __name__ == "__main__":
    # Sample data
    sample_games = [
        {
            'home_team': 'Duke',
            'away_team': 'UNC',
            'home_score': 84,
            'away_score': 82,
            'overtime': True,
            'is_primetime': True,
            'tournament_context': 'march_madness',
            'attendance': 75000
        }
    ]
    
    results = analyze_cbb_revenue_patterns(sample_games)
    print(f"\n✅ Analysis complete")
