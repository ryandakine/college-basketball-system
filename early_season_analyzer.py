#!/usr/bin/env python3
"""
Early Season Game Analyzer
Identifies soft lines on early season games using advanced analytics
"""

import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import math

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EarlySeasonAnalyzer:
    def __init__(self):
        """Initialize early season game analyzer"""
        self.team_analytics = self.load_team_analytics()
        self.injury_impact = {
            'major': 0.15,    # 15% performance drop for star player
            'moderate': 0.08, # 8% drop for key rotation player
            'minor': 0.03     # 3% drop for bench player
        }
        
    def load_team_analytics(self) -> Dict:
        """Load team analytics from preseason analysis"""
        try:
            with open('preseason_analysis_results.json', 'r') as f:
                data = json.load(f)
            return data['teams']
        except FileNotFoundError:
            logging.error("Preseason analysis results not found")
            return {}
    
    def calculate_team_strength(self, team: str, adjustments: Dict = None) -> float:
        """Calculate adjusted team strength"""
        team_key = team.lower().replace(' ', '_').replace("'", "")
        
        if team_key not in self.team_analytics:
            # Return neutral strength for unknown teams
            return 0.50
        
        base_strength = self.team_analytics[team_key]['analytics_strength']
        
        # Apply adjustments
        if adjustments:
            for adj_type, adj_value in adjustments.items():
                if adj_type == 'injury':
                    base_strength *= (1 - adj_value)
                elif adj_type == 'momentum':
                    base_strength *= (1 + adj_value)
                elif adj_type == 'rest':
                    base_strength *= (1 + adj_value)
        
        return min(max(base_strength, 0.0), 1.0)
    
    def analyze_game(self, home_team: str, away_team: str, 
                    market_line: float, total: float,
                    game_info: Dict = None) -> Dict:
        """Analyze a single game for betting opportunities"""
        
        # Get team adjustments
        home_adjustments = self.get_team_adjustments(home_team, game_info)
        away_adjustments = self.get_team_adjustments(away_team, game_info)
        
        # Calculate adjusted strengths
        home_strength = self.calculate_team_strength(home_team, home_adjustments)
        away_strength = self.calculate_team_strength(away_team, away_adjustments)
        
        # Home court advantage (typically 2-4 points)
        home_advantage = 3.0
        
        # Calculate our predicted line
        strength_diff = home_strength - away_strength
        predicted_line = (strength_diff * 25) + home_advantage  # Scale to points
        
        # Calculate edge
        line_edge = abs(predicted_line - market_line)
        line_direction = "HOME" if predicted_line > market_line else "AWAY"
        
        # Tempo and total analysis
        home_tempo = self.get_team_tempo(home_team)
        away_tempo = self.get_team_tempo(away_team)
        avg_tempo = (home_tempo + away_tempo) / 2
        
        # Predicted total based on team strengths and tempo
        base_total = 140 + (avg_tempo - 70) * 0.8  # Adjust for tempo
        predicted_total = base_total * (0.9 + (home_strength + away_strength) * 0.2)
        
        total_edge = abs(predicted_total - total)
        total_direction = "OVER" if predicted_total > total else "UNDER"
        
        return {
            'game': f"{away_team} @ {home_team}",
            'market_line': market_line,
            'predicted_line': round(predicted_line, 1),
            'line_edge': round(line_edge, 1),
            'line_recommendation': line_direction if line_edge > 2.5 else None,
            'market_total': total,
            'predicted_total': round(predicted_total, 1),
            'total_edge': round(total_edge, 1),
            'total_recommendation': total_direction if total_edge > 4.0 else None,
            'home_strength': round(home_strength, 3),
            'away_strength': round(away_strength, 3),
            'confidence': self.calculate_confidence(line_edge, total_edge, home_strength, away_strength),
            'game_info': game_info or {}
        }
    
    def get_team_adjustments(self, team: str, game_info: Dict = None) -> Dict:
        """Get team-specific adjustments for injuries, rest, etc."""
        adjustments = {}
        
        if game_info:
            # Injury adjustments
            injuries = game_info.get('injuries', {}).get(team, [])
            injury_impact = 0
            for injury in injuries:
                if injury['severity'] in self.injury_impact:
                    injury_impact += self.injury_impact[injury['severity']]
            if injury_impact > 0:
                adjustments['injury'] = min(injury_impact, 0.30)  # Cap at 30%
            
            # Rest adjustments (days since last game)
            rest_days = game_info.get('rest_days', {}).get(team, 2)
            if rest_days == 0:  # Back-to-back
                adjustments['rest'] = -0.05
            elif rest_days >= 7:  # Long layoff
                adjustments['rest'] = -0.03
            elif rest_days >= 4:  # Good rest
                adjustments['rest'] = 0.02
        
        return adjustments
    
    def get_team_tempo(self, team: str) -> float:
        """Get team tempo (possessions per game)"""
        # Default tempo if not found
        base_tempo = 70.0
        
        team_key = team.lower().replace(' ', '_').replace("'", "")
        if team_key in self.team_analytics:
            # Use analytics strength as proxy for tempo adjustment
            strength = self.team_analytics[team_key]['analytics_strength']
            # Higher strength teams often play faster
            return base_tempo + (strength - 0.5) * 10
        
        return base_tempo
    
    def calculate_confidence(self, line_edge: float, total_edge: float, 
                           home_strength: float, away_strength: float) -> str:
        """Calculate betting confidence level"""
        
        # Higher edges and more extreme strength differences = higher confidence
        max_edge = max(line_edge, total_edge)
        strength_gap = abs(home_strength - away_strength)
        
        confidence_score = max_edge * 2 + strength_gap * 10
        
        if confidence_score >= 8:
            return "HIGH"
        elif confidence_score >= 5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def generate_sample_games(self) -> List[Dict]:
        """Generate sample early season games for analysis"""
        # Early season matchups that often have soft lines
        sample_games = [
            {
                'home_team': 'Duke',
                'away_team': 'Kansas',
                'market_line': -2.5,
                'total': 155.5,
                'date': '2024-11-12',
                'game_info': {
                    'tournament': 'Champions Classic',
                    'neutral_site': True,
                    'injuries': {
                        'Duke': [{'player': 'Key Player', 'severity': 'minor'}]
                    },
                    'rest_days': {'Duke': 3, 'Kansas': 4}
                }
            },
            {
                'home_team': 'UNC',
                'away_team': 'Dayton',
                'market_line': -8.5,
                'total': 142.0,
                'date': '2024-11-15',
                'game_info': {
                    'tournament': 'Maui Invitational',
                    'injuries': {},
                    'rest_days': {'UNC': 2, 'Dayton': 1}
                }
            },
            {
                'home_team': 'Cincinnati',
                'away_team': 'Xavier',
                'market_line': 3.5,
                'total': 138.5,
                'date': '2024-11-18',
                'game_info': {
                    'tournament': 'Crosstown Shootout',
                    'rivalry': True,
                    'injuries': {
                        'Xavier': [{'player': 'Starter', 'severity': 'moderate'}]
                    },
                    'rest_days': {'Cincinnati': 3, 'Xavier': 2}
                }
            },
            {
                'home_team': 'Auburn',
                'away_team': 'Houston',
                'market_line': -1.0,
                'total': 148.5,
                'date': '2024-11-20',
                'game_info': {
                    'tournament': 'Battle 4 Atlantis',
                    'neutral_site': True,
                    'injuries': {},
                    'rest_days': {'Auburn': 2, 'Houston': 3}
                }
            },
            {
                'home_team': 'Providence',
                'away_team': 'Marquette',
                'market_line': 6.5,
                'total': 144.0,
                'date': '2024-11-22',
                'game_info': {
                    'conference': 'Big East Preview',
                    'injuries': {
                        'Providence': [{'player': 'Guard', 'severity': 'minor'}]
                    },
                    'rest_days': {'Providence': 4, 'Marquette': 3}
                }
            }
        ]
        
        return sample_games
    
    def analyze_early_season_slate(self) -> Dict:
        """Analyze a slate of early season games"""
        games = self.generate_sample_games()
        analyzed_games = []
        opportunities = []
        
        for game_data in games:
            analysis = self.analyze_game(
                game_data['home_team'],
                game_data['away_team'],
                game_data['market_line'],
                game_data['total'],
                game_data['game_info']
            )
            
            # Add date and tournament info
            analysis['date'] = game_data['date']
            analysis['tournament'] = game_data['game_info'].get('tournament', 'Regular Season')
            
            analyzed_games.append(analysis)
            
            # Check for betting opportunities
            if analysis['line_recommendation'] and analysis['confidence'] in ['HIGH', 'MEDIUM']:
                opportunities.append({
                    'type': 'Spread',
                    'game': analysis['game'],
                    'recommendation': f"{analysis['line_recommendation']} {abs(analysis['market_line'])}",
                    'edge': analysis['line_edge'],
                    'confidence': analysis['confidence'],
                    'reasoning': f"Model predicts {analysis['predicted_line']}, market {analysis['market_line']}",
                    'date': analysis['date']
                })
            
            if analysis['total_recommendation'] and analysis['confidence'] in ['HIGH', 'MEDIUM']:
                opportunities.append({
                    'type': 'Total',
                    'game': analysis['game'],
                    'recommendation': f"{analysis['total_recommendation']} {analysis['market_total']}",
                    'edge': analysis['total_edge'],
                    'confidence': analysis['confidence'],
                    'reasoning': f"Model predicts {analysis['predicted_total']}, market {analysis['market_total']}",
                    'date': analysis['date']
                })
        
        return {
            'analyzed_games': analyzed_games,
            'opportunities': opportunities,
            'analysis_date': datetime.now().isoformat(),
            'games_analyzed': len(analyzed_games),
            'opportunities_found': len(opportunities)
        }
    
    def save_analysis(self, results: Dict, filename: str = 'early_season_analysis.json') -> None:
        """Save analysis results to file"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Analysis saved to {filename}")
    
    def print_opportunities_report(self, results: Dict) -> None:
        """Print formatted opportunities report"""
        print("🏀 EARLY SEASON GAME ANALYSIS REPORT")
        print("=" * 50)
        print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🎯 Games Analyzed: {results['games_analyzed']}")
        print(f"💎 Opportunities Found: {results['opportunities_found']}")
        
        if results['opportunities']:
            print("\n🔥 BETTING OPPORTUNITIES:")
            print("-" * 40)
            
            high_confidence = [opp for opp in results['opportunities'] if opp['confidence'] == 'HIGH']
            medium_confidence = [opp for opp in results['opportunities'] if opp['confidence'] == 'MEDIUM']
            
            if high_confidence:
                print("\n⭐ HIGH CONFIDENCE:")
                for opp in high_confidence:
                    print(f"  🎯 {opp['game']} ({opp['date']})")
                    print(f"     {opp['type']}: {opp['recommendation']}")
                    print(f"     Edge: {opp['edge']:.1f} points")
                    print(f"     Reasoning: {opp['reasoning']}")
                    print()
            
            if medium_confidence:
                print("📊 MEDIUM CONFIDENCE:")
                for opp in medium_confidence:
                    print(f"  • {opp['game']} - {opp['type']}: {opp['recommendation']} (Edge: {opp['edge']:.1f})")
        
        print("\n📋 DETAILED GAME ANALYSIS:")
        print("-" * 40)
        for game in results['analyzed_games']:
            print(f"\n🏀 {game['game']} ({game['date']})")
            print(f"   Tournament: {game['tournament']}")
            print(f"   Market Line: {game['market_line']:+.1f} | Predicted: {game['predicted_line']:+.1f} | Edge: {game['line_edge']:.1f}")
            print(f"   Market Total: {game['market_total']:.1f} | Predicted: {game['predicted_total']:.1f} | Edge: {game['total_edge']:.1f}")
            print(f"   Team Strengths: Home {game['home_strength']:.3f} | Away {game['away_strength']:.3f}")
            print(f"   Confidence: {game['confidence']}")
            
            if game['line_recommendation']:
                print(f"   🎯 Line Bet: {game['line_recommendation']} {abs(game['market_line'])}")
            if game['total_recommendation']:
                print(f"   🎯 Total Bet: {game['total_recommendation']} {game['market_total']}")
        
        print("\n⚠️  EARLY SEASON EDGE FACTORS:")
        print("• Books haven't adjusted to real team performance")
        print("• Small sample sizes create uncertainty")
        print("• Injury news may not be fully priced in")
        print("• Tournament games have unique dynamics")
        print("• Lines often softer for mid-major matchups")
        
        print(f"\n🚀 READY TO CAPITALIZE ON SOFT EARLY SEASON LINES!")

def main():
    """Main function"""
    print("🏀 EARLY SEASON GAME ANALYZER")
    print("=" * 40)
    
    analyzer = EarlySeasonAnalyzer()
    
    if not analyzer.team_analytics:
        print("❌ No team analytics found. Run preseason Monte Carlo analyzer first.")
        return
    
    print("🔍 Analyzing early season games for soft lines...")
    results = analyzer.analyze_early_season_slate()
    
    # Save results
    analyzer.save_analysis(results)
    
    # Print report
    analyzer.print_opportunities_report(results)

if __name__ == "__main__":
    main()