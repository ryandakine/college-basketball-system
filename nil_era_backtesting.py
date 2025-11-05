#!/usr/bin/env python3
"""
NIL Era Backtesting System for College Basketball
Comprehensive backtesting of agent-player-coach relationships impact on betting outcomes
from July 1, 2021 through present, including line movement analysis and ROI calculations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NIL_START_DATE = datetime(2021, 7, 1)
SEASONS_ANALYZED = ['2021-22', '2022-23', '2023-24', '2024-25']

@dataclass
class BettingLine:
    """Individual betting line data point."""
    game_id: str
    date: datetime
    home_team: str
    away_team: str
    spread_open: float
    spread_close: float
    total_open: float
    total_close: float
    home_ml_open: int
    home_ml_close: int
    away_ml_open: int
    away_ml_close: int
    
@dataclass
class GameResult:
    """Game result with betting outcomes."""
    game_id: str
    home_score: int
    away_score: int
    total_points: int
    home_covered: bool
    over_hit: bool
    favorite_won: bool
    margin_of_victory: int
    
@dataclass
class RelationshipFactors:
    """Relationship factors that influenced the game."""
    game_id: str
    home_team_chemistry: float
    away_team_chemistry: float
    home_agent_influence: float
    away_agent_influence: float
    home_nil_tension: float
    away_nil_tension: float
    home_transfer_risk: float
    away_transfer_risk: float
    
@dataclass
class BacktestResult:
    """Results from backtesting analysis."""
    season: str
    total_games: int
    traditional_model_accuracy: float
    relationship_enhanced_accuracy: float
    improvement_pct: float
    roi_traditional: float
    roi_enhanced: float
    key_insights: List[str]
    edge_opportunities: Dict[str, float]

class NILEraBacktester:
    """Main backtesting engine for NIL era analysis."""
    
    def __init__(self):
        self.nil_start = NIL_START_DATE
        self.betting_thresholds = {
            'chemistry_edge': 0.15,  # Minimum team chemistry difference to bet
            'agent_risk': 0.6,       # Agent influence threshold for fade
            'nil_tension': 0.5,      # NIL tension threshold for volatility
            'transfer_risk': 0.7     # Transfer risk threshold
        }
        
        # Historical market inefficiencies discovered
        self.market_edges = {
            'high_agent_influence_fade': {'roi': 0.0, 'games': 0, 'wins': 0},
            'team_chemistry_differential': {'roi': 0.0, 'games': 0, 'wins': 0},
            'nil_tension_over_betting': {'roi': 0.0, 'games': 0, 'wins': 0},
            'transfer_portal_disruption': {'roi': 0.0, 'games': 0, 'wins': 0},
            'coaching_stability_edge': {'roi': 0.0, 'games': 0, 'wins': 0}
        }
    
    def load_historical_data(self, data_source: str = "sample") -> Dict[str, List[Dict]]:
        """Load historical betting and relationship data."""
        logger.info(f"Loading historical data from {data_source}")
        
        if data_source == "sample":
            # Generate sample data for demonstration
            return self._generate_sample_historical_data()
        else:
            # In production, would load from actual data sources
            return self._load_production_data(data_source)
    
    def _generate_sample_historical_data(self) -> Dict[str, List[Dict]]:
        """Generate comprehensive sample data for backtesting demo."""
        historical_data = {}
        
        for season in SEASONS_ANALYZED:
            season_data = []
            
            # Generate 500 sample games per season
            for game_num in range(500):
                game_date = self._generate_season_date(season, game_num)
                
                # Sample teams with varying relationship dynamics
                teams = [
                    ('Duke', 'UNC'), ('Gonzaga', 'UCLA'), ('Kansas', 'Kentucky'),
                    ('Villanova', 'Baylor'), ('Michigan', 'Ohio State')
                ]
                home_team, away_team = teams[game_num % len(teams)]
                
                # Generate betting lines with NIL-era volatility
                spread_open = np.random.normal(0, 8)
                spread_close = spread_open + np.random.normal(0, 2)  # Line movement
                total_open = np.random.normal(145, 15)
                total_close = total_open + np.random.normal(0, 3)
                
                # Generate relationship factors (post-NIL era patterns)
                home_chemistry = max(0, min(1, np.random.normal(0.6, 0.2)))
                away_chemistry = max(0, min(1, np.random.normal(0.6, 0.2)))
                
                # Higher agent influence in NIL era
                home_agent_influence = max(0, min(1, np.random.normal(0.4, 0.25)))
                away_agent_influence = max(0, min(1, np.random.normal(0.4, 0.25)))
                
                # NIL tension varies significantly
                home_nil_tension = max(0, min(1, np.random.normal(0.3, 0.2)))
                away_nil_tension = max(0, min(1, np.random.normal(0.3, 0.2)))
                
                # Transfer risk elevated post-NIL
                home_transfer_risk = max(0, min(1, np.random.normal(0.25, 0.15)))
                away_transfer_risk = max(0, min(1, np.random.normal(0.25, 0.15)))
                
                # Generate realistic game results influenced by relationship factors
                chemistry_diff = home_chemistry - away_chemistry
                expected_margin = spread_close + (chemistry_diff * 5)  # Chemistry impact
                
                # Add volatility from agent influence and NIL tension
                volatility = (home_agent_influence + away_agent_influence + 
                            home_nil_tension + away_nil_tension) / 4
                actual_margin = expected_margin + np.random.normal(0, 5 + volatility * 10)
                
                home_score = int(np.random.normal(75, 10))
                away_score = max(40, int(home_score - actual_margin))
                
                game_data = {
                    'game_id': f"{season}_{game_num:03d}",
                    'date': game_date.isoformat(),
                    'home_team': home_team,
                    'away_team': away_team,
                    'betting_lines': {
                        'spread_open': spread_open,
                        'spread_close': spread_close,
                        'total_open': total_open,
                        'total_close': total_close,
                        'home_ml_open': self._spread_to_moneyline(spread_open),
                        'home_ml_close': self._spread_to_moneyline(spread_close),
                        'away_ml_open': self._spread_to_moneyline(-spread_open),
                        'away_ml_close': self._spread_to_moneyline(-spread_close)
                    },
                    'results': {
                        'home_score': home_score,
                        'away_score': away_score,
                        'total_points': home_score + away_score,
                        'margin': home_score - away_score,
                        'home_covered': (home_score - away_score) > spread_close,
                        'over_hit': (home_score + away_score) > total_close,
                        'favorite_won': (home_score > away_score) if spread_close < 0 else (away_score > home_score)
                    },
                    'relationship_factors': {
                        'home_team_chemistry': home_chemistry,
                        'away_team_chemistry': away_chemistry,
                        'home_agent_influence': home_agent_influence,
                        'away_agent_influence': away_agent_influence,
                        'home_nil_tension': home_nil_tension,
                        'away_nil_tension': away_nil_tension,
                        'home_transfer_risk': home_transfer_risk,
                        'away_transfer_risk': away_transfer_risk
                    }
                }
                
                season_data.append(game_data)
            
            historical_data[season] = season_data
        
        return historical_data
    
    def _generate_season_date(self, season: str, game_num: int) -> datetime:
        """Generate realistic dates within a season."""
        season_start_year = int(season.split('-')[0])
        
        # College basketball season roughly Nov-March
        if game_num < 100:  # Non-conference games
            start_date = datetime(season_start_year, 11, 15)
            return start_date + timedelta(days=np.random.randint(0, 45))
        elif game_num < 400:  # Conference games
            start_date = datetime(season_start_year + 1, 1, 1)
            return start_date + timedelta(days=np.random.randint(0, 60))
        else:  # Tournament games
            start_date = datetime(season_start_year + 1, 3, 1)
            return start_date + timedelta(days=np.random.randint(0, 30))
    
    def _spread_to_moneyline(self, spread: float) -> int:
        """Convert point spread to moneyline odds."""
        if spread > 0:
            return int(-110 - spread * 5)
        else:
            return int(110 - spread * 5)
    
    def _load_production_data(self, data_source: str) -> Dict[str, List[Dict]]:
        """Load actual production data (placeholder for real implementation)."""
        # In production, would connect to actual data sources:
        # - Betting line databases
        # - Team chemistry tracking systems
        # - Transfer portal databases
        # - NIL deal databases
        logger.warning("Production data loading not implemented - using sample data")
        return self._generate_sample_historical_data()
    
    def run_backtest_analysis(self, historical_data: Dict[str, List[Dict]]) -> Dict[str, BacktestResult]:
        """Run comprehensive backtest analysis across all seasons."""
        logger.info("Starting comprehensive NIL era backtest analysis")
        
        backtest_results = {}
        
        for season, games in historical_data.items():
            logger.info(f"Analyzing season: {season}")
            
            # Traditional model predictions (spreads/totals only)
            traditional_predictions = self._generate_traditional_predictions(games)
            traditional_accuracy = self._calculate_prediction_accuracy(games, traditional_predictions)
            traditional_roi = self._calculate_roi(games, traditional_predictions, 'traditional')
            
            # Enhanced model with relationship factors
            enhanced_predictions = self._generate_enhanced_predictions(games)
            enhanced_accuracy = self._calculate_prediction_accuracy(games, enhanced_predictions)
            enhanced_roi = self._calculate_roi(games, enhanced_predictions, 'enhanced')
            
            # Calculate improvement
            improvement_pct = ((enhanced_accuracy - traditional_accuracy) / traditional_accuracy) * 100
            
            # Identify key insights and edge opportunities
            insights = self._generate_season_insights(games, enhanced_predictions)
            edge_opportunities = self._calculate_edge_opportunities(games)
            
            backtest_results[season] = BacktestResult(
                season=season,
                total_games=len(games),
                traditional_model_accuracy=traditional_accuracy,
                relationship_enhanced_accuracy=enhanced_accuracy,
                improvement_pct=improvement_pct,
                roi_traditional=traditional_roi,
                roi_enhanced=enhanced_roi,
                key_insights=insights,
                edge_opportunities=edge_opportunities
            )
        
        return backtest_results
    
    def _generate_traditional_predictions(self, games: List[Dict]) -> List[Dict]:
        """Generate traditional model predictions (spreads/totals only)."""
        predictions = []
        
        for game in games:
            # Simple traditional prediction based only on betting lines
            spread_prediction = {
                'game_id': game['game_id'],
                'spread_pick': 'home' if game['betting_lines']['spread_close'] > -3 else 'away',
                'total_pick': 'over' if game['betting_lines']['total_close'] < 150 else 'under',
                'confidence': 0.52,  # Traditional models ~52% accuracy
                'bet_size': 1.0
            }
            predictions.append(spread_prediction)
        
        return predictions
    
    def _generate_enhanced_predictions(self, games: List[Dict]) -> List[Dict]:
        """Generate enhanced predictions using relationship factors."""
        predictions = []
        
        for game in games:
            rel_factors = game['relationship_factors']
            
            # Calculate relationship edges
            chemistry_edge = rel_factors['home_team_chemistry'] - rel_factors['away_team_chemistry']
            agent_risk_home = rel_factors['home_agent_influence']
            agent_risk_away = rel_factors['away_agent_influence']
            nil_tension_total = rel_factors['home_nil_tension'] + rel_factors['away_nil_tension']
            
            # Enhanced spread prediction
            spread_adjustment = chemistry_edge * 3  # Chemistry worth ~3 points
            if agent_risk_home > self.betting_thresholds['agent_risk']:
                spread_adjustment -= 2  # Fade high agent influence teams
            if agent_risk_away > self.betting_thresholds['agent_risk']:
                spread_adjustment += 2
            
            adjusted_spread = game['betting_lines']['spread_close'] + spread_adjustment
            spread_pick = 'home' if adjusted_spread < 0 else 'away'
            
            # Enhanced total prediction
            total_adjustment = nil_tension_total * -5  # High NIL tension = lower scoring
            adjusted_total = game['betting_lines']['total_close'] + total_adjustment
            total_pick = 'over' if adjusted_total < game['betting_lines']['total_close'] else 'under'
            
            # Calculate confidence based on edge strength
            edge_strength = abs(chemistry_edge) + (nil_tension_total * 0.5)
            confidence = min(0.65, 0.52 + edge_strength * 0.2)
            
            # Bet sizing based on confidence
            if confidence > 0.58:
                bet_size = 2.0  # Double bet on strong edges
            elif confidence < 0.54:
                bet_size = 0.5  # Half bet on weak edges
            else:
                bet_size = 1.0
            
            enhanced_prediction = {
                'game_id': game['game_id'],
                'spread_pick': spread_pick,
                'total_pick': total_pick,
                'confidence': confidence,
                'bet_size': bet_size,
                'chemistry_edge': chemistry_edge,
                'agent_risk_factor': max(agent_risk_home, agent_risk_away),
                'nil_tension_factor': nil_tension_total
            }
            
            predictions.append(enhanced_prediction)
        
        return predictions
    
    def _calculate_prediction_accuracy(self, games: List[Dict], predictions: List[Dict]) -> float:
        """Calculate prediction accuracy."""
        correct_predictions = 0
        total_predictions = 0
        
        for game, pred in zip(games, predictions):
            # Check spread accuracy
            actual_home_covered = game['results']['home_covered']
            predicted_home = pred['spread_pick'] == 'home'
            
            if actual_home_covered == predicted_home:
                correct_predictions += 1
            total_predictions += 1
            
            # Check total accuracy
            actual_over = game['results']['over_hit']
            predicted_over = pred['total_pick'] == 'over'
            
            if actual_over == predicted_over:
                correct_predictions += 1
            total_predictions += 1
        
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    def _calculate_roi(self, games: List[Dict], predictions: List[Dict], model_type: str) -> float:
        """Calculate Return on Investment (ROI) for betting strategy."""
        total_wagered = 0.0
        total_won = 0.0
        
        for game, pred in zip(games, predictions):
            bet_size = pred.get('bet_size', 1.0)
            
            # Spread bet
            spread_wager = bet_size * 110  # Standard -110 odds
            total_wagered += spread_wager
            
            actual_home_covered = game['results']['home_covered']
            predicted_home = pred['spread_pick'] == 'home'
            
            if actual_home_covered == predicted_home:
                total_won += bet_size * 100  # Win pays 100 on 110 bet
            
            # Total bet
            total_wager = bet_size * 110
            total_wagered += total_wager
            
            actual_over = game['results']['over_hit']
            predicted_over = pred['total_pick'] == 'over'
            
            if actual_over == predicted_over:
                total_won += bet_size * 100
        
        roi = ((total_won - total_wagered) / total_wagered) * 100 if total_wagered > 0 else 0.0
        return roi
    
    def _generate_season_insights(self, games: List[Dict], predictions: List[Dict]) -> List[str]:
        """Generate key insights from season analysis."""
        insights = []
        
        # Analyze chemistry edge performance
        chemistry_games = [
            (g, p) for g, p in zip(games, predictions) 
            if abs(p.get('chemistry_edge', 0)) > self.betting_thresholds['chemistry_edge']
        ]
        
        if chemistry_games:
            chemistry_accuracy = sum(
                1 for g, p in chemistry_games 
                if g['results']['home_covered'] == (p['spread_pick'] == 'home')
            ) / len(chemistry_games)
            
            if chemistry_accuracy > 0.55:
                insights.append(f"Team chemistry edge games hit at {chemistry_accuracy:.1%} rate")
        
        # Analyze high agent influence fades
        high_agent_games = [
            (g, p) for g, p in zip(games, predictions) 
            if p.get('agent_risk_factor', 0) > self.betting_thresholds['agent_risk']
        ]
        
        if high_agent_games:
            agent_fade_accuracy = sum(
                1 for g, p in high_agent_games 
                if g['results']['home_covered'] != (p['spread_pick'] == 'home')  # Fade logic
            ) / len(high_agent_games)
            
            if agent_fade_accuracy > 0.54:
                insights.append(f"Fading high agent influence teams successful {agent_fade_accuracy:.1%} of time")
        
        # Analyze NIL tension impact on totals
        nil_tension_games = [
            (g, p) for g, p in zip(games, predictions) 
            if p.get('nil_tension_factor', 0) > self.betting_thresholds['nil_tension']
        ]
        
        if nil_tension_games:
            under_rate = sum(
                1 for g, p in nil_tension_games 
                if not g['results']['over_hit']
            ) / len(nil_tension_games)
            
            if under_rate > 0.55:
                insights.append(f"High NIL tension games go UNDER {under_rate:.1%} of time")
        
        return insights
    
    def _calculate_edge_opportunities(self, games: List[Dict]) -> Dict[str, float]:
        """Calculate specific edge opportunities and their ROI."""
        edges = {}
        
        # Team chemistry differential edge
        chemistry_edge_games = [
            g for g in games 
            if abs(g['relationship_factors']['home_team_chemistry'] - 
                   g['relationship_factors']['away_team_chemistry']) > 0.2
        ]
        
        if chemistry_edge_games:
            chemistry_wins = sum(
                1 for g in chemistry_edge_games 
                if (g['relationship_factors']['home_team_chemistry'] > 
                    g['relationship_factors']['away_team_chemistry']) == g['results']['home_covered']
            )
            edges['chemistry_differential'] = (chemistry_wins / len(chemistry_edge_games) - 0.5) * 200
        
        # High agent influence fade
        high_agent_games = [
            g for g in games 
            if max(g['relationship_factors']['home_agent_influence'],
                   g['relationship_factors']['away_agent_influence']) > 0.6
        ]
        
        if high_agent_games:
            # Fade the team with higher agent influence
            fade_wins = 0
            for g in high_agent_games:
                if g['relationship_factors']['home_agent_influence'] > g['relationship_factors']['away_agent_influence']:
                    # Fade home team
                    if not g['results']['home_covered']:
                        fade_wins += 1
                else:
                    # Fade away team
                    if g['results']['home_covered']:
                        fade_wins += 1
            
            edges['agent_influence_fade'] = (fade_wins / len(high_agent_games) - 0.5) * 200
        
        # NIL tension under betting
        nil_tension_games = [
            g for g in games 
            if (g['relationship_factors']['home_nil_tension'] + 
                g['relationship_factors']['away_nil_tension']) > 0.6
        ]
        
        if nil_tension_games:
            under_wins = sum(1 for g in nil_tension_games if not g['results']['over_hit'])
            edges['nil_tension_under'] = (under_wins / len(nil_tension_games) - 0.5) * 200
        
        return edges
    
    def generate_comprehensive_report(self, backtest_results: Dict[str, BacktestResult]) -> Dict[str, any]:
        """Generate comprehensive backtest report."""
        logger.info("Generating comprehensive NIL era backtest report")
        
        # Overall statistics
        total_games = sum(result.total_games for result in backtest_results.values())
        avg_traditional_accuracy = np.mean([result.traditional_model_accuracy for result in backtest_results.values()])
        avg_enhanced_accuracy = np.mean([result.relationship_enhanced_accuracy for result in backtest_results.values()])
        avg_improvement = np.mean([result.improvement_pct for result in backtest_results.values()])
        avg_traditional_roi = np.mean([result.roi_traditional for result in backtest_results.values()])
        avg_enhanced_roi = np.mean([result.roi_enhanced for result in backtest_results.values()])
        
        # Best performing strategies
        all_edges = {}
        for result in backtest_results.values():
            for edge_name, edge_value in result.edge_opportunities.items():
                if edge_name not in all_edges:
                    all_edges[edge_name] = []
                all_edges[edge_name].append(edge_value)
        
        best_edges = {
            edge: np.mean(values) for edge, values in all_edges.items() 
            if np.mean(values) > 2.0  # Only profitable edges
        }
        
        # Season-by-season trends
        season_trends = {}
        for season, result in backtest_results.items():
            season_trends[season] = {
                'accuracy_improvement': result.improvement_pct,
                'roi_improvement': result.roi_enhanced - result.roi_traditional,
                'games_analyzed': result.total_games
            }
        
        report = {
            'summary': {
                'total_games_analyzed': total_games,
                'average_traditional_accuracy': avg_traditional_accuracy,
                'average_enhanced_accuracy': avg_enhanced_accuracy,
                'average_improvement_percent': avg_improvement,
                'traditional_roi': avg_traditional_roi,
                'enhanced_roi': avg_enhanced_roi,
                'roi_improvement': avg_enhanced_roi - avg_traditional_roi
            },
            'best_performing_edges': best_edges,
            'season_trends': season_trends,
            'detailed_results': {season: vars(result) for season, result in backtest_results.items()},
            'recommendations': self._generate_betting_recommendations(best_edges, season_trends)
        }
        
        return report
    
    def _generate_betting_recommendations(self, best_edges: Dict[str, float], 
                                        season_trends: Dict[str, Dict]) -> List[str]:
        """Generate actionable betting recommendations."""
        recommendations = []
        
        if 'chemistry_differential' in best_edges and best_edges['chemistry_differential'] > 5:
            recommendations.append(
                f"Target games with significant team chemistry differentials (>{self.betting_thresholds['chemistry_edge']}). "
                f"Historical ROI: {best_edges['chemistry_differential']:.1f}%"
            )
        
        if 'agent_influence_fade' in best_edges and best_edges['agent_influence_fade'] > 3:
            recommendations.append(
                f"Fade teams with high agent influence (>{self.betting_thresholds['agent_risk']}). "
                f"Historical ROI: {best_edges['agent_influence_fade']:.1f}%"
            )
        
        if 'nil_tension_under' in best_edges and best_edges['nil_tension_under'] > 4:
            recommendations.append(
                f"Bet UNDER in games with high NIL tension (>{self.betting_thresholds['nil_tension']}). "
                f"Historical ROI: {best_edges['nil_tension_under']:.1f}%"
            )
        
        # Trend-based recommendations
        recent_seasons = list(season_trends.keys())[-2:]  # Last 2 seasons
        if recent_seasons:
            avg_recent_improvement = np.mean([
                season_trends[season]['accuracy_improvement'] 
                for season in recent_seasons
            ])
            
            if avg_recent_improvement > 5:
                recommendations.append(
                    f"Relationship factors are becoming increasingly valuable. "
                    f"Recent seasons show {avg_recent_improvement:.1f}% improvement over traditional models."
                )
        
        return recommendations

def main():
    """Demo the NIL era backtesting system."""
    print("NIL Era College Basketball Backtesting System")
    print("=" * 50)
    print(f"Analyzing from: {NIL_START_DATE.strftime('%B %d, %Y')} to present")
    print(f"Seasons: {', '.join(SEASONS_ANALYZED)}")
    
    # Initialize backtester
    backtester = NILEraBacktester()
    
    try:
        # Load historical data
        print("\nLoading historical data...")
        historical_data = backtester.load_historical_data("sample")
        
        total_games = sum(len(games) for games in historical_data.values())
        print(f"Loaded {total_games:,} games across {len(historical_data)} seasons")
        
        # Run backtest analysis
        print("\nRunning comprehensive backtest analysis...")
        backtest_results = backtester.run_backtest_analysis(historical_data)
        
        # Generate comprehensive report
        print("\nGenerating comprehensive report...")
        report = backtester.generate_comprehensive_report(backtest_results)
        
        # Display results
        print("\n" + "="*50)
        print("BACKTEST RESULTS SUMMARY")
        print("="*50)
        
        summary = report['summary']
        print(f"Total Games Analyzed: {summary['total_games_analyzed']:,}")
        print(f"Traditional Model Accuracy: {summary['average_traditional_accuracy']:.2%}")
        print(f"Enhanced Model Accuracy: {summary['average_enhanced_accuracy']:.2%}")
        print(f"Accuracy Improvement: {summary['average_improvement_percent']:.1f}%")
        print(f"Traditional ROI: {summary['traditional_roi']:.2f}%")
        print(f"Enhanced ROI: {summary['enhanced_roi']:.2f}%")
        print(f"ROI Improvement: {summary['roi_improvement']:.2f}%")
        
        print(f"\n{'='*30}")
        print("BEST PERFORMING EDGES:")
        for edge_name, roi in report['best_performing_edges'].items():
            print(f"  {edge_name.replace('_', ' ').title()}: {roi:.2f}% ROI")
        
        print(f"\n{'='*30}")
        print("SEASON-BY-SEASON TRENDS:")
        for season, trends in report['season_trends'].items():
            print(f"  {season}: {trends['accuracy_improvement']:.1f}% improvement, "
                  f"{trends['roi_improvement']:.2f}% ROI boost ({trends['games_analyzed']} games)")
        
        print(f"\n{'='*30}")
        print("BETTING RECOMMENDATIONS:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print(f"\n{'='*30}")
        print("KEY INSIGHTS FROM LATEST SEASON:")
        latest_season = max(backtest_results.keys())
        latest_result = backtest_results[latest_season]
        for insight in latest_result.key_insights:
            print(f"  • {insight}")
            
    except Exception as e:
        logger.error(f"Backtesting failed: {e}")
        print(f"Backtesting failed: {e}")

if __name__ == "__main__":
    main()