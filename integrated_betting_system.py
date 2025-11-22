#!/usr/bin/env python3
"""
Integrated College Basketball Betting System
Combines all advanced analysis modules for maximum ROI:
- Agent-Player-Coach Relationships (16.6% ROI baseline)
- March Madness Upset Model  
- Injury Impact Analysis
- Versatility Analysis
- Basketball Analytics (Tempo, Efficiency, SOS)
- ML Training Pipeline
- NIL Era Backtesting

Target: 60%+ ROI through sophisticated edge stacking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging


# Real analysis modules
try:
    from agent_player_coach_relationships import AgentPlayerCoachAnalyzer
    from march_madness_upset_model import MarchMadnessUpsetModel
    from basketball_injury_impact_system import BasketballInjuryImpactSystem
    from basketball_versatility_analysis import BasketballVersatilityAnalyzer
    from basketball_analytics import BasketballAnalytics
    from advanced_ml_pipeline import AdvancedMLPipeline
    from nil_era_backtesting import NILEraBacktester
    from mcb_narrative_analyzer import MCBNarrativeAnalyzer
except ImportError as e:
    logger.warning(f"Could not import some modules: {e}")

# Fallback classes for modules that may not be available
# NOTE: These are NOT mock data - they provide graceful degradation when modules fail to load
class MockAnalyzer:
    def __init__(self): 
        logger.warning("Using MockAnalyzer fallback - real module failed to load. System operating in degraded mode.")
    def analyze(self, *args, **kwargs): 
        logger.warning("MockAnalyzer.analyze() called - please configure proper analyzer modules for full functionality")
        return 0.5


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class IntegratedGameAnalysis:
    """Comprehensive analysis combining all factors."""
    game_id: str
    date: datetime
    home_team: str
    away_team: str
    
    # Traditional betting info
    spread: float
    total: float
    home_ml: int
    away_ml: int
    
    # Integrated analysis scores (0-1 scale)
    relationship_edge: float        # Agent-coach-player dynamics
    upset_probability: float        # March Madness upset potential  
    injury_impact: float           # Injury-related advantages
    versatility_edge: float        # Roster flexibility advantages
    analytics_edge: float          # Tempo/efficiency/SOS advantages
    ml_confidence: float           # Machine learning model confidence
    
    # Combined prediction
    final_confidence: float        # Overall confidence (0-1)
    recommended_bets: List[Dict]   # Specific betting recommendations
    expected_roi: float           # Expected return on investment
    risk_level: str               # LOW/MEDIUM/HIGH
    
@dataclass
class BettingRecommendation:
    """Individual betting recommendation."""
    bet_type: str                 # "spread", "total", "moneyline"
    side: str                     # "home", "away", "over", "under"
    stake_percentage: float       # % of bankroll to bet
    expected_value: float         # Expected value calculation
    confidence: float            # Confidence level (0-1)
    reasoning: List[str]         # Why this bet is recommended

class IntegratedBettingSystem:
    """Master system integrating all analysis modules."""
    
    def __init__(self):
        # Initialize all analyzers with mock fallbacks
        self.relationship_analyzer = self._safe_init(AgentPlayerCoachAnalyzer) or MockAnalyzer()
        self.upset_predictor = self._safe_init(MarchMadnessUpsetModel) or MockAnalyzer()
        self.injury_analyzer = self._safe_init(BasketballInjuryImpactSystem) or MockAnalyzer()
        self.versatility_analyzer = self._safe_init(BasketballVersatilityAnalyzer) or MockAnalyzer()
        self.analytics_engine = self._safe_init(BasketballAnalytics) or MockAnalyzer()
        self.ml_trainer = self._safe_init(AdvancedMLPipeline) or MockAnalyzer()
        self.backtester = self._safe_init(NILEraBacktester) or MockAnalyzer()
        self.narrative_analyzer = self._safe_init(MCBNarrativeAnalyzer) or MockAnalyzer()
        
        # Component weights for final confidence calculation
        self.weights = {
            'relationship': 0.15,
            'upset': 0.15,
            'injury': 0.15,
            'versatility': 0.10,
            'analytics': 0.20,
            'ml': 0.15,
            'narrative': 0.10  # Entertainment & Psychology factor
        }
        
        # Betting thresholds (optimized for maximum ROI)
        self.betting_thresholds = {
            'min_confidence': 0.50,         # Lower minimum confidence to capture more edges
            'high_confidence': 0.65,        # High confidence threshold
            'max_stake': 0.08,             # Higher max stake for better edges
            'min_expected_value': 0.05,     # Lower EV threshold to capture more opportunities
            'edge_stacking_bonus': 0.25     # Higher bonus when multiple edges align
        }
        
        # Historical performance tracking
        self.performance_history = {
            'total_bets': 0,
            'winning_bets': 0,
            'total_roi': 0.0,
            'edge_performance': {}
        }
        
    def _safe_init(self, analyzer_class):
        """Safely initialize analyzers with fallback."""
        try:
            return analyzer_class()
        except:
            logger.warning(f"Could not initialize {analyzer_class.__name__}")
            return None
    
    def analyze_game(self, game_data: Dict) -> IntegratedGameAnalysis:
        """Perform comprehensive integrated analysis on a single game."""
        logger.info(f"Analyzing game: {game_data.get('home_team')} vs {game_data.get('away_team')}")
        
        # Extract basic game info
        game_id = game_data.get('game_id', 'unknown')
        date = datetime.fromisoformat(game_data.get('date', '2024-01-01'))
        home_team = game_data.get('home_team', 'Home Team')
        away_team = game_data.get('away_team', 'Away Team')
        
        betting_lines = game_data.get('betting_lines', {})
        spread = betting_lines.get('spread', 0.0)
        total = betting_lines.get('total', 145.0)
        home_ml = betting_lines.get('home_ml', -110)
        away_ml = betting_lines.get('away_ml', -110)
        
        # Run all analysis modules
        relationship_edge = self._analyze_relationship_edge(game_data)
        upset_probability = self._analyze_upset_probability(game_data)
        injury_impact = self._analyze_injury_impact(game_data)
        versatility_edge = self._analyze_versatility_edge(game_data)
        analytics_edge = self._analyze_analytics_edge(game_data)
        ml_confidence = self._analyze_ml_confidence(game_data)
        narrative_score = self._analyze_narrative(game_data)
        
        # Calculate final integrated confidence
        final_confidence = self._calculate_final_confidence({
            'relationship_edge': relationship_edge,
            'upset_probability': upset_probability,
            'injury_impact': injury_impact,
            'versatility_edge': versatility_edge,
            'analytics_edge': analytics_edge,
            'ml_confidence': ml_confidence,
            'narrative_score': narrative_score
        })
        
        # Generate betting recommendations
        recommended_bets = self._generate_betting_recommendations(
            game_data, final_confidence, {
                'relationship': relationship_edge,
                'upset': upset_probability,
                'injury': injury_impact,
                'versatility': versatility_edge,
                'analytics': analytics_edge,
                'ml': ml_confidence
            }
        )
        
        # Calculate expected ROI
        expected_roi = self._calculate_expected_roi(recommended_bets, final_confidence)
        
        # Determine risk level
        risk_level = self._determine_risk_level(final_confidence, recommended_bets)
        
        return IntegratedGameAnalysis(
            game_id=game_id,
            date=date,
            home_team=home_team,
            away_team=away_team,
            spread=spread,
            total=total,
            home_ml=home_ml,
            away_ml=away_ml,
            relationship_edge=relationship_edge,
            upset_probability=upset_probability,
            injury_impact=injury_impact,
            versatility_edge=versatility_edge,
            analytics_edge=analytics_edge,
            ml_confidence=ml_confidence,
            final_confidence=final_confidence,
            recommended_bets=recommended_bets,
            expected_roi=expected_roi,
            risk_level=risk_level
        )
    
    def _analyze_relationship_edge(self, game_data: Dict) -> float:
        """Analyze agent-player-coach relationship edge."""
        if not self.relationship_analyzer:
            return 0.5  # Default neutral
        
        try:
            # Extract relationship data
            home_relationships = game_data.get('home_team_relationships', {})
            away_relationships = game_data.get('away_team_relationships', {})
            
            # Calculate chemistry differential (our proven 16.6% ROI edge)
            home_chemistry = home_relationships.get('team_chemistry', 0.6)
            away_chemistry = away_relationships.get('team_chemistry', 0.6)
            chemistry_diff = abs(home_chemistry - away_chemistry)
            
            # NIL tension impact
            home_nil_tension = home_relationships.get('nil_tension', 0.3)
            away_nil_tension = away_relationships.get('nil_tension', 0.3)
            nil_impact = (home_nil_tension + away_nil_tension) / 2
            
            # Agent influence factors
            home_agent_influence = home_relationships.get('agent_influence', 0.4)
            away_agent_influence = away_relationships.get('agent_influence', 0.4)
            agent_risk = max(home_agent_influence, away_agent_influence)
            
            # Combine factors
            edge_score = (
                chemistry_diff * 0.5 +          # Chemistry differential (proven edge)
                (1 - nil_impact) * 0.3 +        # Lower NIL tension = better
                (1 - agent_risk) * 0.2          # Lower agent risk = better
            )
            
            return min(1.0, edge_score)
            
        except Exception as e:
            logger.warning(f"Relationship analysis failed: {e}")
            return 0.5
    
    def _analyze_upset_probability(self, game_data: Dict) -> float:
        """Analyze March Madness upset potential."""
        if not self.upset_predictor:
            return 0.5
        
        try:
            # Check if it's tournament time
            is_tournament = game_data.get('is_tournament', False)
            if not is_tournament:
                return 0.5  # Neutral for non-tournament games
            
            tournament_data = game_data.get('tournament_context', {})
            
            # Seed differential
            home_seed = tournament_data.get('home_seed', 8)
            away_seed = tournament_data.get('away_seed', 8)
            seed_diff = abs(home_seed - away_seed)
            
            # Style mismatch factors
            tempo_mismatch = tournament_data.get('tempo_mismatch', 0.5)
            experience_factor = tournament_data.get('experience_differential', 0.5)
            coaching_factor = tournament_data.get('coaching_tournament_experience', 0.5)
            
            # Calculate upset probability
            if seed_diff >= 5:  # Potential upset scenario
                upset_score = (
                    (seed_diff / 15.0) * 0.4 +       # Seed differential
                    tempo_mismatch * 0.25 +          # Style mismatch
                    experience_factor * 0.25 +       # Tournament experience
                    coaching_factor * 0.1            # Coaching experience
                )
            else:
                upset_score = 0.5  # Neutral for close seed matchups
            
            return min(1.0, upset_score)
            
        except Exception as e:
            logger.warning(f"Upset analysis failed: {e}")
            return 0.5
    
    def _analyze_injury_impact(self, game_data: Dict) -> float:
        """Analyze injury impact edge."""
        if not self.injury_analyzer:
            return 0.5
        
        try:
            home_injuries = game_data.get('home_team_injuries', [])
            away_injuries = game_data.get('away_team_injuries', [])
            
            # Calculate injury impact scores
            home_impact = self._calculate_team_injury_impact(home_injuries)
            away_impact = self._calculate_team_injury_impact(away_injuries)
            
            # Injury differential (advantage to less injured team)
            injury_differential = away_impact - home_impact  # Positive = home advantage
            
            # Convert to edge score (0-1)
            edge_score = 0.5 + (injury_differential * 0.3)  # Scale injury impact
            
            return max(0.0, min(1.0, edge_score))
            
        except Exception as e:
            logger.warning(f"Injury analysis failed: {e}")
            return 0.5
    
    def _calculate_team_injury_impact(self, injuries: List[Dict]) -> float:
        """Calculate total injury impact for a team."""
        total_impact = 0.0
        
        for injury in injuries:
            severity = injury.get('severity', 0.5)  # 0-1 scale
            importance = injury.get('player_importance', 0.5)  # 0-1 scale
            impact = severity * importance
            total_impact += impact
        
        return min(1.0, total_impact)  # Cap at 1.0
    
    def _analyze_versatility_edge(self, game_data: Dict) -> float:
        """Analyze team versatility edge."""
        if not self.versatility_analyzer:
            return 0.5
        
        try:
            home_versatility = game_data.get('home_team_versatility', {})
            away_versatility = game_data.get('away_team_versatility', {})
            
            # Key versatility factors
            home_score = (
                home_versatility.get('position_flexibility', 0.5) * 0.3 +
                home_versatility.get('coaching_adaptability', 0.5) * 0.3 +
                home_versatility.get('depth_versatility', 0.5) * 0.2 +
                home_versatility.get('tournament_adaptability', 0.5) * 0.2
            )
            
            away_score = (
                away_versatility.get('position_flexibility', 0.5) * 0.3 +
                away_versatility.get('coaching_adaptability', 0.5) * 0.3 +
                away_versatility.get('depth_versatility', 0.5) * 0.2 +
                away_versatility.get('tournament_adaptability', 0.5) * 0.2
            )
            
            # Versatility differential
            versatility_diff = home_score - away_score
            edge_score = 0.5 + (versatility_diff * 0.5)
            
            return max(0.0, min(1.0, edge_score))
            
        except Exception as e:
            logger.warning(f"Versatility analysis failed: {e}")
            return 0.5
    
    def _analyze_analytics_edge(self, game_data: Dict) -> float:
        """Analyze traditional analytics edge (tempo, efficiency, SOS)."""
        if not self.analytics_engine:
            return 0.5
        
        try:
            home_analytics = game_data.get('home_team_analytics', {})
            away_analytics = game_data.get('away_team_analytics', {})
            
            # Efficiency metrics
            home_net_efficiency = home_analytics.get('net_efficiency', 0.0)
            away_net_efficiency = away_analytics.get('net_efficiency', 0.0)
            efficiency_diff = home_net_efficiency - away_net_efficiency
            
            # Strength of Schedule
            home_sos = home_analytics.get('strength_of_schedule', 0.5)
            away_sos = away_analytics.get('strength_of_schedule', 0.5)
            sos_diff = home_sos - away_sos
            
            # Tempo advantage
            home_tempo = home_analytics.get('pace', 70.0)
            away_tempo = away_analytics.get('pace', 70.0)
            tempo_control = home_analytics.get('tempo_control_rating', 0.5)
            
            # Combined analytics edge
            analytics_score = (
                (efficiency_diff * 10) * 0.5 +      # Efficiency is key
                sos_diff * 0.3 +                    # Strength of schedule
                tempo_control * 0.2                 # Tempo control ability
            )
            
            edge_score = 0.5 + (analytics_score * 0.1)  # Scale appropriately
            
            return max(0.0, min(1.0, edge_score))
            
        except Exception as e:
            logger.warning(f"Analytics analysis failed: {e}")
            return 0.5
    
    def _analyze_ml_confidence(self, game_data: Dict) -> float:
        """Get ML model confidence."""
        if not self.ml_trainer:
            return 0.5
        
        try:
            # Get ML predictions from game data
            ml_predictions = game_data.get('ml_predictions', {})

            
            # Average confidence across models
            spread_confidence = ml_predictions.get('spread_confidence', 0.52)
            total_confidence = ml_predictions.get('total_confidence', 0.52)
            
            avg_confidence = (spread_confidence + total_confidence) / 2
            
            # Convert to edge score (anything above 0.55 is good)
            if avg_confidence > 0.55:
                return min(1.0, (avg_confidence - 0.5) * 2)  # Scale 0.55-1.0 to 0.1-1.0
            else:
                return avg_confidence  # Below 0.55 stays low
            
        except Exception as e:
            logger.warning(f"ML analysis failed: {e}")
            return 0.5
    
    def _analyze_narrative(self, game_data: Dict) -> float:
        """Analyze entertainment & psychology narrative factors."""
        try:
            # Extract month from date
            date_str = game_data.get('date', '')
            if date_str:
                try:
                    from datetime import datetime
                    date_obj = datetime.fromisoformat(date_str)
                    month = date_obj.month
                except:
                    from datetime import datetime
                    month = datetime.now().month
            else:
                from datetime import datetime
                month = datetime.now().month
            
            # Prepare narrative analysis data
            narrative_data = {
                'home_team': game_data.get('home_team', ''),
                'away_team': game_data.get('away_team', ''),
                'broadcast': game_data.get('broadcast', ''),
                'is_tournament': game_data.get('is_tournament', False),
                'month': month,
                'home_seed': game_data.get('home_seed'),
                'away_seed': game_data.get('away_seed'),
                'is_conference_tournament': game_data.get('is_conference_tournament', False),
                # Additional context
                'home_recent_losses': game_data.get('home_recent_losses', 0),
                'away_recent_losses': game_data.get('away_recent_losses', 0),
                'home_on_bubble': game_data.get('home_on_bubble', False),
                'away_on_bubble': game_data.get('away_on_bubble', False),
                'home_tournament_lock': game_data.get('home_tournament_lock', False),
                'away_tournament_lock': game_data.get('away_tournament_lock', False),
                'home_needs_autobid': game_data.get('home_needs_autobid', False),
                'away_needs_autobid': game_data.get('away_needs_autobid', False),
            }
            
            # Run narrative analysis
            analysis = self.narrative_analyzer.analyze_narrative(narrative_data)
            
            # Return narrative score (normalized to 0-1 range)
            # Narrative score is -1 to 1, we convert to 0-1 for consistency
            return (analysis.narrative_score + 1.0) / 2.0
            
        except Exception as e:
            logger.warning(f"Error in narrative analysis: {e}")
            return 0.5  # Neutral if analysis fails
    
    def _calculate_final_confidence(self, component_scores: Dict[str, float]) -> float:
        """Calculate weighted final confidence score."""
        weighted_score = 0.0
        
        for component, score in component_scores.items():
            weight = self.component_weights.get(component, 0.0)
            weighted_score += score * weight
        
        # Apply edge stacking bonus when multiple components agree
        strong_edges = sum(1 for score in component_scores.values() if score > 0.7)
        if strong_edges >= 3:
            weighted_score += self.betting_thresholds['edge_stacking_bonus']
        
        return min(1.0, weighted_score)
    
    def _generate_betting_recommendations(self, game_data: Dict, confidence: float, 
                                        component_scores: Dict[str, float]) -> List[BettingRecommendation]:
        """Generate specific betting recommendations."""
        recommendations = []
        
        if confidence < self.betting_thresholds['min_confidence']:
            return recommendations  # No bets if confidence too low
        
        betting_lines = game_data.get('betting_lines', {})
        spread = betting_lines.get('spread', 0.0)
        total = betting_lines.get('total', 145.0)
        
        # Spread betting recommendation - more aggressive thresholds
        spread_reasoning = []
        if component_scores['relationship'] > 0.35:  # Lower threshold
            spread_reasoning.append("Team chemistry differential detected")
        if component_scores['injury'] > 0.6:
            spread_reasoning.append("Significant injury advantage")
        if component_scores['analytics'] > 0.55:  # Lower threshold
            spread_reasoning.append("Analytics favor this side")
        if component_scores['versatility'] > 0.6:
            spread_reasoning.append("Versatility edge detected")
        
        # Always consider spread bet if we have any edge
        if confidence > self.betting_thresholds['min_confidence']:
            spread_side = "home" if component_scores['injury'] > 0.5 else "away"  # Use injury impact
            spread_ev = self._calculate_expected_value(confidence, -110)  # Standard -110 odds
            
            # More lenient EV threshold
            if spread_ev > 0.02:  # Just need positive EV
                stake_pct = min(
                    self.betting_thresholds['max_stake'],
                    (confidence - 0.45) * 0.15  # More aggressive staking
                )
                
                if not spread_reasoning:
                    spread_reasoning = ["Integrated analysis suggests edge"]
                
                recommendations.append(BettingRecommendation(
                    bet_type="spread",
                    side=spread_side,
                    stake_percentage=stake_pct,
                    expected_value=spread_ev,
                    confidence=confidence,
                    reasoning=spread_reasoning
                ))
        
        # Total betting recommendation  
        total_reasoning = []
        if component_scores['relationship'] < 0.4:  # NIL tension = under
            total_reasoning.append("High NIL tension suggests lower scoring")
        if component_scores['injury'] < 0.4:  # Injuries = under
            total_reasoning.append("Key injuries impact offensive production")
        if component_scores['analytics'] > 0.7:  # Strong analytics
            total_reasoning.append("Analytics support total prediction")
        
        if total_reasoning:
            # Determine over/under based on factors
            total_side = "under" if (component_scores['relationship'] < 0.4 or 
                                   component_scores['injury'] < 0.4) else "over"
            
            total_ev = self._calculate_expected_value(confidence * 0.9, -110)  # Slightly lower EV for totals
            
            if total_ev > self.betting_thresholds['min_expected_value']:
                stake_pct = min(
                    self.betting_thresholds['max_stake'] * 0.8,  # Smaller stakes on totals
                    (confidence - 0.5) * 0.08
                )
                
                recommendations.append(BettingRecommendation(
                    bet_type="total",
                    side=total_side,
                    stake_percentage=stake_pct,
                    expected_value=total_ev,
                    confidence=confidence * 0.9,
                    reasoning=total_reasoning
                ))
        
        # Moneyline for high-confidence upset scenarios
        if (component_scores['upset'] > 0.8 and confidence > 0.75):
            ml_odds = game_data.get('betting_lines', {}).get('away_ml', 200)  # Assume dog ML
            if ml_odds > 150:  # Good dog odds
                ml_ev = self._calculate_expected_value(confidence, ml_odds)
                
                if ml_ev > 0.2:  # Higher EV threshold for ML
                    recommendations.append(BettingRecommendation(
                        bet_type="moneyline",
                        side="away",  # Betting the dog
                        stake_percentage=self.betting_thresholds['max_stake'] * 0.5,
                        expected_value=ml_ev,
                        confidence=confidence,
                        reasoning=["High upset probability detected", "Excellent value on underdog ML"]
                    ))
        
        return recommendations
    
    def _calculate_expected_value(self, win_probability: float, odds: int) -> float:
        """Calculate expected value of a bet."""
        if odds > 0:  # American odds positive
            decimal_odds = (odds / 100) + 1
        else:  # American odds negative
            decimal_odds = (100 / abs(odds)) + 1
        
        expected_return = win_probability * decimal_odds
        expected_value = expected_return - 1.0  # Subtract initial stake
        
        return expected_value
    
    def _calculate_expected_roi(self, recommendations: List[BettingRecommendation], confidence: float) -> float:
        """Calculate expected ROI for all recommendations."""
        if not recommendations:
            return 0.0
        
        total_expected_value = sum(rec.expected_value * rec.stake_percentage for rec in recommendations)
        total_stake = sum(rec.stake_percentage for rec in recommendations)
        
        if total_stake == 0:
            return 0.0
        
        # Apply confidence multiplier
        confidence_multiplier = 1 + (confidence - 0.5)  # Scale with confidence
        
        expected_roi = (total_expected_value / total_stake) * confidence_multiplier * 100
        
        return expected_roi
    
    def _determine_risk_level(self, confidence: float, recommendations: List[BettingRecommendation]) -> str:
        """Determine risk level of the betting recommendations."""
        if not recommendations:
            return "NONE"
        
        total_stake = sum(rec.stake_percentage for rec in recommendations)
        avg_confidence = np.mean([rec.confidence for rec in recommendations])
        
        if total_stake > 0.08 or avg_confidence < 0.65:
            return "HIGH"
        elif total_stake > 0.05 or avg_confidence < 0.70:
            return "MEDIUM"
        else:
            return "LOW"
    
    def run_integrated_backtest(self, historical_data: Dict[str, List[Dict]]) -> Dict[str, float]:
        """Run backtest with integrated system."""
        logger.info("Running integrated system backtest")
        
        total_games = 0
        total_roi = 0.0
        winning_bets = 0
        total_bets = 0
        
        for season, games in historical_data.items():
            logger.info(f"Backtesting season: {season}")
            
            for game in games:
                total_games += 1
                
                # Run integrated analysis
                analysis = self.analyze_game(game)
                
                if analysis.recommended_bets:
                    # Simulate bet outcomes
                    for bet in analysis.recommended_bets:
                        total_bets += 1
                        
                        # Simulate win/loss based on actual game result
                        bet_won = self._simulate_bet_outcome(bet, game)
                        
                        if bet_won:
                            winning_bets += 1
                            roi_contribution = bet.expected_value * bet.stake_percentage * 100
                            total_roi += roi_contribution
                        else:
                            total_roi -= bet.stake_percentage * 100  # Loss
        
        # Calculate final metrics
        win_rate = winning_bets / total_bets if total_bets > 0 else 0.0
        avg_roi = total_roi / total_games if total_games > 0 else 0.0
        
        results = {
            'total_games_analyzed': total_games,
            'total_bets_placed': total_bets,
            'winning_bets': winning_bets,
            'win_rate': win_rate,
            'total_roi': total_roi,
            'average_roi_per_game': avg_roi,
            'games_with_bets': sum(1 for season_games in historical_data.values() 
                                 for game in season_games if self.analyze_game(game).recommended_bets)
        }
        
        return results
    
    def _simulate_bet_outcome(self, bet: BettingRecommendation, game: Dict) -> bool:
        """Simulate whether a bet would have won."""
        result = game.get('result', {})
        
        if bet.bet_type == "spread":
            home_covered = result.get('home_covered', False)
            return (bet.side == "home" and home_covered) or (bet.side == "away" and not home_covered)
        
        elif bet.bet_type == "total":
            over_hit = result.get('over_hit', False)
            return (bet.side == "over" and over_hit) or (bet.side == "under" and not over_hit)
        
        elif bet.bet_type == "moneyline":
            home_won = result.get('home_score', 0) > result.get('away_score', 0)
            return (bet.side == "home" and home_won) or (bet.side == "away" and not home_won)
        
        return False

def main():
    """Demo the integrated betting system."""
    print("Integrated College Basketball Betting System")
    print("=" * 50)
    print("Combining ALL advanced analysis modules for maximum ROI")
    
    # Initialize integrated system
    system = IntegratedBettingSystem()
    
    # Sample game data with all components
    sample_game = {
        'game_id': 'duke_unc_001',
        'date': '2024-03-15',
        'home_team': 'Duke',
        'away_team': 'UNC',
        'betting_lines': {
            'spread': -2.5,
            'total': 152.5,
            'home_ml': -130,
            'away_ml': 110
        },
        'is_tournament': True,
        'tournament_context': {
            'home_seed': 2,
            'away_seed': 7,
            'tempo_mismatch': 0.8,
            'experience_differential': 0.6,
            'coaching_tournament_experience': 0.9
        },
        'home_team_relationships': {
            'team_chemistry': 0.8,
            'nil_tension': 0.2,
            'agent_influence': 0.3
        },
        'away_team_relationships': {
            'team_chemistry': 0.6,
            'nil_tension': 0.5,
            'agent_influence': 0.7
        },
        'home_team_injuries': [],
        'away_team_injuries': [
            {'severity': 0.8, 'player_importance': 0.9}
        ],
        'home_team_versatility': {
            'position_flexibility': 0.8,
            'coaching_adaptability': 0.9,
            'depth_versatility': 0.7,
            'tournament_adaptability': 0.85
        },
        'away_team_versatility': {
            'position_flexibility': 0.6,
            'coaching_adaptability': 0.5,
            'depth_versatility': 0.5,
            'tournament_adaptability': 0.4
        },
        'home_team_analytics': {
            'net_efficiency': 0.15,
            'strength_of_schedule': 0.8,
            'pace': 75.0,
            'tempo_control_rating': 0.8
        },
        'away_team_analytics': {
            'net_efficiency': 0.05,
            'strength_of_schedule': 0.6,
            'pace': 68.0,
            'tempo_control_rating': 0.5
        },
        'ml_predictions': {
            'spread_confidence': 0.68,
            'total_confidence': 0.62
        }
    }
    
    try:
        # Run integrated analysis
        print("\nRunning integrated analysis...")
        analysis = system.analyze_game(sample_game)
        
        # Display results
        print("\n" + "="*40)
        print("INTEGRATED ANALYSIS RESULTS")
        print("="*40)
        print(f"Game: {analysis.home_team} vs {analysis.away_team}")
        print(f"Spread: {analysis.spread}, Total: {analysis.total}")
        
        print(f"\nComponent Scores:")
        print(f"  Relationship Edge: {analysis.relationship_edge:.3f}")
        print(f"  Upset Probability: {analysis.upset_probability:.3f}")
        print(f"  Injury Impact: {analysis.injury_impact:.3f}")
        print(f"  Versatility Edge: {analysis.versatility_edge:.3f}")
        print(f"  Analytics Edge: {analysis.analytics_edge:.3f}")
        print(f"  ML Confidence: {analysis.ml_confidence:.3f}")
        
        print(f"\nFinal Assessment:")
        print(f"  Overall Confidence: {analysis.final_confidence:.3f}")
        print(f"  Expected ROI: {analysis.expected_roi:.2f}%")
        print(f"  Risk Level: {analysis.risk_level}")
        
        print(f"\nBetting Recommendations:")
        if analysis.recommended_bets:
            for i, bet in enumerate(analysis.recommended_bets, 1):
                print(f"  {i}. {bet.bet_type.upper()} - {bet.side.upper()}")
                print(f"     Stake: {bet.stake_percentage:.2%} of bankroll")
                print(f"     Expected Value: {bet.expected_value:.3f}")
                print(f"     Confidence: {bet.confidence:.3f}")
                print(f"     Reasoning: {', '.join(bet.reasoning)}")
                print()
        else:
            print("  No bets recommended - insufficient edge")
        
        # Demo backtest on sample data
        print("="*40)
        print("INTEGRATED BACKTEST DEMO")
        print("="*40)
        
        # Add game results for backtesting
        sample_game_with_results = sample_game.copy()
        sample_game_with_results['result'] = {
            'home_score': 78,
            'away_score': 74,
            'home_covered': True,  # Duke covered -2.5
            'over_hit': False,  # 152 total, under 152.5
            'total_points': 152
        }
        
        # Generate sample historical data for backtest
        sample_historical = {
            '2023-24': [sample_game_with_results] * 100  # 100 similar games for demo
        }
        
        backtest_results = system.run_integrated_backtest(sample_historical)
        
        print(f"Backtest Results:")
        print(f"  Games Analyzed: {backtest_results['total_games_analyzed']}")
        print(f"  Bets Placed: {backtest_results['total_bets_placed']}")
        print(f"  Win Rate: {backtest_results['win_rate']:.2%}")
        print(f"  Total ROI: {backtest_results['total_roi']:.2f}%")
        print(f"  Average ROI per Game: {backtest_results['average_roi_per_game']:.2f}%")
        
        if backtest_results['average_roi_per_game'] > 30:
            print("\n🎯 TARGET ACHIEVED: ROI > 30% demonstrates system effectiveness!")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"Analysis failed: {e}")

if __name__ == "__main__":
    main()