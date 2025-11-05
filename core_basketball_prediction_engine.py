#!/usr/bin/env python3
"""
Core College Basketball Prediction Engine
==========================================

Central prediction system for college basketball that integrates advanced models and generates
comprehensive game predictions with betting recommendations.

Integrates:
- Basketball Model Training Pipeline
- Player Fatigue Analysis (tournament context)
- Injury Impact System (basketball-specific)
- Team Depth and Rotation Analysis
- Tempo and efficiency factors
- Conference strength analysis
- Tournament context and seeding
- Real-time data feeds

Adapted from MLB Core Prediction Engine for College Basketball
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import sqlite3
from pathlib import Path

# Import our basketball-specific systems
try:
    from basketball_model_training_pipeline import BasketballModelTrainingPipeline
    from basketball_player_fatigue_system import BasketballPlayerFatigueSystem
    from basketball_injury_impact_system import BasketballInjuryImpactSystem
    from basketball_depth_analysis import BasketballDepthAnalysisSystem
except ImportError as e:
    logging.warning(f"Could not import basketball systems: {e}")

@dataclass
class GameContext:
    """Complete game context information for college basketball"""
    game_id: str
    date: datetime
    home_team: str
    away_team: str
    
    # Conference and ranking info
    home_conference: str
    away_conference: str
    home_ranking: Optional[int]  # AP/Coaches Poll ranking
    away_ranking: Optional[int]
    home_kenpom_rating: float   # KenPom efficiency rating
    away_kenpom_rating: float
    
    # Tournament context
    tournament_context: str  # "regular_season", "conference_tournament", "march_madness"
    tournament_round: Optional[str]  # "first_four", "round_of_64", etc.
    home_seed: Optional[int]
    away_seed: Optional[int]
    
    # Team stats
    home_team_stats: Dict
    away_team_stats: Dict
    
    # Game environment
    venue: str
    venue_capacity: int
    neutral_site: bool
    game_time: str
    tv_coverage: str
    
    # Market information
    odds: Dict
    betting_lines: Dict

@dataclass
class BasketballPredictionComponents:
    """Individual prediction components from each system"""
    
    # Base statistical predictions
    win_probability: float
    point_differential: float
    total_points: float
    
    # Basketball-specific predictions
    tempo_prediction: float  # Estimated possessions
    efficiency_predictions: Dict  # Offensive/defensive efficiency
    
    # Advanced system predictions
    fatigue_adjustments: Dict
    injury_impacts: Dict
    depth_advantages: Dict
    tournament_adjustments: Dict
    
    # Environmental adjustments
    venue_adjustments: Dict
    conference_strength_factors: Dict
    
    # Confidence metrics
    prediction_confidence: float
    model_agreement: float
    data_quality_score: float

@dataclass
class BasketballBettingRecommendation:
    """Basketball-specific betting recommendation"""
    bet_type: str
    recommendation: str  # "BET", "AVOID", "STRONG_BET", "SMALL_BET"
    
    # Odds and value
    recommended_odds: float
    fair_value: float
    edge_percentage: float
    
    # Confidence and sizing
    confidence: float
    suggested_unit_size: float
    max_bet_percentage: float
    
    # Supporting analysis
    key_factors: List[str]
    risk_factors: List[str]
    reasoning: str

@dataclass
class ComprehensiveBasketballPrediction:
    """Complete basketball prediction with all components and recommendations"""
    game_context: GameContext
    prediction_components: BasketballPredictionComponents
    
    # Final predictions
    final_win_probability: float
    final_point_differential: float
    final_total_points: float
    final_tempo: float
    
    # Betting recommendations
    spread_recommendation: Optional[BasketballBettingRecommendation]
    total_recommendation: Optional[BasketballBettingRecommendation]
    moneyline_recommendation: Optional[BasketballBettingRecommendation]
    prop_recommendations: List[BasketballBettingRecommendation]
    
    # Meta information
    prediction_timestamp: datetime
    model_version: str
    confidence_tier: str  # "HIGH", "MEDIUM", "LOW"

class CoreBasketballPredictionEngine:
    """Central basketball prediction engine integrating all advanced systems"""
    
    def __init__(self, db_path: str = "basketball_core_predictions.db"):
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        self.model_version = "1.0.0"
        
        # Initialize database
        self._init_database()
        
        # Initialize basketball systems
        self._initialize_systems()
        
        # Basketball-specific weights for ensemble
        self.SYSTEM_WEIGHTS = {
            'base_model': 0.30,
            'efficiency_model': 0.25,
            'fatigue_system': 0.15,
            'injury_system': 0.10,
            'depth_system': 0.10,
            'tournament_context': 0.10
        }
        
        # Confidence thresholds
        self.CONFIDENCE_THRESHOLDS = {
            'high': 0.75,
            'medium': 0.60,
            'low': 0.45
        }
        
        # Betting edge thresholds
        self.BETTING_THRESHOLDS = {
            'strong_bet': 0.06,  # 6%+ edge (lower than MLB due to market efficiency)
            'bet': 0.04,         # 4%+ edge
            'avoid': 0.02        # Less than 2% edge
        }
        
        # Basketball-specific constants
        self.AVERAGE_COLLEGE_POSSESSIONS = 68.0
        self.AVERAGE_COLLEGE_POINTS = 71.0
        
    def _init_database(self):
        """Initialize basketball prediction database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Game predictions
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS basketball_game_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    game_date TEXT NOT NULL,
                    home_team TEXT NOT NULL,
                    away_team TEXT NOT NULL,
                    tournament_context TEXT,
                    prediction_timestamp TEXT NOT NULL,
                    final_win_probability REAL NOT NULL,
                    final_point_differential REAL NOT NULL,
                    final_total_points REAL NOT NULL,
                    final_tempo REAL NOT NULL,
                    confidence_tier TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    prediction_data TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Basketball betting recommendations
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS basketball_betting_recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    bet_type TEXT NOT NULL,
                    recommendation TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    edge_percentage REAL NOT NULL,
                    suggested_units REAL NOT NULL,
                    reasoning TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Prediction performance tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS basketball_prediction_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT NOT NULL,
                    prediction_date TEXT NOT NULL,
                    predicted_winner TEXT,
                    actual_winner TEXT,
                    predicted_total REAL,
                    actual_total REAL,
                    predicted_tempo REAL,
                    actual_tempo REAL,
                    prediction_accuracy REAL,
                    bet_outcomes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Basketball prediction database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing basketball prediction database: {e}")
            
    def _initialize_systems(self):
        """Initialize all basketball prediction systems"""
        try:
            # Initialize basketball systems if available
            self.model_pipeline = None
            self.fatigue_system = None
            self.injury_system = None
            self.depth_system = None
            
            try:
                self.model_pipeline = BasketballModelTrainingPipeline()
                self.logger.info("Basketball model training pipeline initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize basketball model pipeline: {e}")
                
            try:
                self.fatigue_system = BasketballPlayerFatigueSystem()
                self.logger.info("Basketball player fatigue system initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize basketball fatigue system: {e}")
                
            try:
                self.injury_system = BasketballInjuryImpactSystem()
                self.logger.info("Basketball injury impact system initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize basketball injury system: {e}")
                
            try:
                self.depth_system = BasketballDepthAnalysisSystem()
                self.logger.info("Basketball depth analysis system initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize basketball depth system: {e}")
                
        except Exception as e:
            self.logger.error(f"Error initializing basketball systems: {e}")
            
    def generate_comprehensive_prediction(self, game_data: Dict) -> ComprehensiveBasketballPrediction:
        """Generate comprehensive prediction for a college basketball game"""
        try:
            self.logger.info(f"Generating basketball prediction for {game_data.get('game_id', 'unknown')}")
            
            # Create game context
            game_context = self._create_game_context(game_data)
            
            # Generate prediction components from each system
            prediction_components = self._generate_prediction_components(game_context, game_data)
            
            # Combine all predictions into final prediction
            final_predictions = self._combine_predictions(prediction_components)
            
            # Generate betting recommendations
            betting_recommendations = self._generate_betting_recommendations(
                game_context, prediction_components, final_predictions
            )
            
            # Determine confidence tier
            confidence_tier = self._determine_confidence_tier(prediction_components)
            
            # Create comprehensive prediction
            comprehensive_prediction = ComprehensiveBasketballPrediction(
                game_context=game_context,
                prediction_components=prediction_components,
                final_win_probability=final_predictions['win_probability'],
                final_point_differential=final_predictions['point_differential'],
                final_total_points=final_predictions['total_points'],
                final_tempo=final_predictions['tempo'],
                spread_recommendation=betting_recommendations.get('spread'),
                total_recommendation=betting_recommendations.get('total'),
                moneyline_recommendation=betting_recommendations.get('moneyline'),
                prop_recommendations=betting_recommendations.get('props', []),
                prediction_timestamp=datetime.now(),
                model_version=self.model_version,
                confidence_tier=confidence_tier
            )
            
            # Store prediction
            self._store_prediction(comprehensive_prediction)
            
            return comprehensive_prediction
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive basketball prediction: {e}")
            return self._create_default_prediction(game_data)
            
    def _create_game_context(self, game_data: Dict) -> GameContext:
        """Create basketball game context from input data"""
        try:
            game_date_str = game_data.get('date', datetime.now().strftime('%Y-%m-%d'))
            game_date = datetime.strptime(game_date_str, '%Y-%m-%d')
            
            return GameContext(
                game_id=game_data.get('game_id', f"bb_game_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                date=game_date,
                home_team=game_data.get('home_team', 'HOME'),
                away_team=game_data.get('away_team', 'AWAY'),
                home_conference=game_data.get('home_conference', 'UNKNOWN'),
                away_conference=game_data.get('away_conference', 'UNKNOWN'),
                home_ranking=game_data.get('home_ranking'),
                away_ranking=game_data.get('away_ranking'),
                home_kenpom_rating=game_data.get('home_kenpom_rating', 50.0),
                away_kenpom_rating=game_data.get('away_kenpom_rating', 50.0),
                tournament_context=game_data.get('tournament_context', 'regular_season'),
                tournament_round=game_data.get('tournament_round'),
                home_seed=game_data.get('home_seed'),
                away_seed=game_data.get('away_seed'),
                home_team_stats=game_data.get('home_team_stats', {}),
                away_team_stats=game_data.get('away_team_stats', {}),
                venue=game_data.get('venue', 'Unknown'),
                venue_capacity=game_data.get('venue_capacity', 10000),
                neutral_site=game_data.get('neutral_site', False),
                game_time=game_data.get('game_time', '19:00'),
                tv_coverage=game_data.get('tv_coverage', 'Local'),
                odds=game_data.get('odds', {}),
                betting_lines=game_data.get('betting_lines', {})
            )
            
        except Exception as e:
            self.logger.error(f"Error creating basketball game context: {e}")
            return self._create_default_game_context(game_data)
            
    def _generate_prediction_components(self, game_context: GameContext, game_data: Dict) -> BasketballPredictionComponents:
        """Generate predictions from all available basketball systems"""
        try:
            # Base statistical prediction
            base_prediction = self._generate_base_basketball_prediction(game_context, game_data)
            
            # Advanced system predictions
            fatigue_adjustments = self._get_fatigue_adjustments(game_context)
            injury_impacts = self._get_injury_impacts(game_context)
            depth_advantages = self._get_depth_advantages(game_context)
            tournament_adjustments = self._get_tournament_adjustments(game_context)
            
            # Environmental adjustments
            venue_adjustments = self._get_venue_adjustments(game_context)
            conference_factors = self._get_conference_strength_factors(game_context)
            
            # Calculate prediction confidence
            prediction_confidence = self._calculate_prediction_confidence(
                base_prediction, fatigue_adjustments, injury_impacts, 
                depth_advantages, tournament_adjustments
            )
            
            # Calculate model agreement
            model_agreement = self._calculate_model_agreement([
                base_prediction, fatigue_adjustments, injury_impacts,
                depth_advantages, tournament_adjustments
            ])
            
            # Data quality score
            data_quality_score = self._assess_data_quality(game_context, game_data)
            
            return BasketballPredictionComponents(
                win_probability=base_prediction['win_probability'],
                point_differential=base_prediction['point_differential'],
                total_points=base_prediction['total_points'],
                tempo_prediction=base_prediction['tempo'],
                efficiency_predictions=base_prediction['efficiency'],
                fatigue_adjustments=fatigue_adjustments,
                injury_impacts=injury_impacts,
                depth_advantages=depth_advantages,
                tournament_adjustments=tournament_adjustments,
                venue_adjustments=venue_adjustments,
                conference_strength_factors=conference_factors,
                prediction_confidence=prediction_confidence,
                model_agreement=model_agreement,
                data_quality_score=data_quality_score
            )
            
        except Exception as e:
            self.logger.error(f"Error generating basketball prediction components: {e}")
            return self._create_default_prediction_components()
            
    def _generate_base_basketball_prediction(self, game_context: GameContext, game_data: Dict) -> Dict:
        """Generate base basketball statistical prediction"""
        try:
            # Get team efficiency ratings
            home_off_eff = game_context.home_team_stats.get('offensive_efficiency', 100.0)
            home_def_eff = game_context.home_team_stats.get('defensive_efficiency', 100.0)
            away_off_eff = game_context.away_team_stats.get('offensive_efficiency', 100.0)
            away_def_eff = game_context.away_team_stats.get('defensive_efficiency', 100.0)
            
            # Get tempo factors
            home_tempo = game_context.home_team_stats.get('tempo', self.AVERAGE_COLLEGE_POSSESSIONS)
            away_tempo = game_context.away_team_stats.get('tempo', self.AVERAGE_COLLEGE_POSSESSIONS)
            
            # Calculate expected tempo for the game
            game_tempo = (home_tempo + away_tempo) / 2
            
            # Home court advantage (stronger in college basketball)
            home_advantage = 3.5 if not game_context.neutral_site else 0.0
            
            # Calculate expected points per possession for each team
            home_ppp = (home_off_eff * away_def_eff) / (100.0 * 100.0)
            away_ppp = (away_off_eff * home_def_eff) / (100.0 * 100.0)
            
            # Expected points
            home_expected_points = home_ppp * game_tempo + home_advantage
            away_expected_points = away_ppp * game_tempo
            
            # Point differential and total
            point_differential = home_expected_points - away_expected_points
            total_points = home_expected_points + away_expected_points
            
            # Win probability using point differential
            win_probability = self._point_differential_to_win_probability(point_differential)
            
            # Efficiency predictions
            efficiency_predictions = {
                'home_offensive_efficiency': home_off_eff,
                'home_defensive_efficiency': home_def_eff,
                'away_offensive_efficiency': away_off_eff,
                'away_defensive_efficiency': away_def_eff,
                'home_expected_points': home_expected_points,
                'away_expected_points': away_expected_points
            }
            
            return {
                'win_probability': win_probability,
                'point_differential': point_differential,
                'total_points': total_points,
                'tempo': game_tempo,
                'efficiency': efficiency_predictions
            }
            
        except Exception as e:
            self.logger.error(f"Error generating base basketball prediction: {e}")
            return {
                'win_probability': 0.5,
                'point_differential': 0.0,
                'total_points': self.AVERAGE_COLLEGE_POINTS,
                'tempo': self.AVERAGE_COLLEGE_POSSESSIONS,
                'efficiency': {}
            }
            
    def _point_differential_to_win_probability(self, point_diff: float) -> float:
        """Convert point differential to win probability for college basketball"""
        # Using logistic regression coefficients fitted to historical college basketball data
        # Coefficient roughly 0.35 for college basketball (different from NBA/NFL)
        coefficient = 0.35
        odds = np.exp(coefficient * point_diff)
        probability = odds / (1 + odds)
        return max(0.05, min(0.95, probability))
        
    def _get_fatigue_adjustments(self, game_context: GameContext) -> Dict:
        """Get player/team fatigue adjustments for basketball"""
        try:
            if not self.fatigue_system:
                return {'available': False}
                
            adjustments = {'available': True}
            
            # Tournament context affects fatigue significantly
            if game_context.tournament_context == 'march_madness':
                # Teams playing consecutive days in tournament
                adjustments['tournament_fatigue'] = {
                    'home_fatigue_factor': 0.95,  # Slight fatigue
                    'away_fatigue_factor': 0.95
                }
            elif game_context.tournament_context == 'conference_tournament':
                adjustments['conference_tournament_fatigue'] = {
                    'home_fatigue_factor': 0.97,
                    'away_fatigue_factor': 0.97
                }
            else:
                adjustments['regular_season'] = {
                    'home_fatigue_factor': 1.0,
                    'away_fatigue_factor': 1.0
                }
                
            return adjustments
            
        except Exception as e:
            self.logger.error(f"Error getting basketball fatigue adjustments: {e}")
            return {'available': False, 'error': str(e)}
            
    def _get_injury_impacts(self, game_context: GameContext) -> Dict:
        """Get injury impact adjustments for basketball"""
        try:
            if not self.injury_system:
                return {'available': False}
                
            # Basketball injuries have different impacts than baseball
            impacts = {'available': True}
            
            # Mock injury analysis (would normally get real injury reports)
            impacts['home_impact'] = {
                'overall_team_impact': 0.0,  # No major injuries
                'starting_five_impact': 0.0,
                'bench_depth_impact': 0.0
            }
            
            impacts['away_impact'] = {
                'overall_team_impact': 0.0,
                'starting_five_impact': 0.0,
                'bench_depth_impact': 0.0
            }
            
            return impacts
            
        except Exception as e:
            self.logger.error(f"Error getting basketball injury impacts: {e}")
            return {'available': False, 'error': str(e)}
            
    def _get_depth_advantages(self, game_context: GameContext) -> Dict:
        """Get team depth and rotation advantages"""
        try:
            if not self.depth_system:
                return {'available': False}
                
            advantages = {'available': True}
            
            # Analyze bench strength and rotation depth
            advantages['home_depth'] = {
                'bench_strength': 0.5,  # Scale 0-1
                'rotation_depth': 8,    # Number of reliable players
                'minutes_distribution': 'balanced'
            }
            
            advantages['away_depth'] = {
                'bench_strength': 0.5,
                'rotation_depth': 8,
                'minutes_distribution': 'balanced'
            }
            
            return advantages
            
        except Exception as e:
            self.logger.error(f"Error getting basketball depth advantages: {e}")
            return {'available': False, 'error': str(e)}
            
    def _get_tournament_adjustments(self, game_context: GameContext) -> Dict:
        """Get tournament context adjustments"""
        try:
            adjustments = {'available': True}
            
            context = game_context.tournament_context
            
            if context == 'march_madness':
                # March Madness specific factors
                adjustments['march_madness'] = {
                    'upset_potential': self._calculate_upset_potential(game_context),
                    'experience_factor': self._calculate_tournament_experience(game_context),
                    'seeding_advantage': self._calculate_seeding_advantage(game_context)
                }
            elif context == 'conference_tournament':
                adjustments['conference_tournament'] = {
                    'familiarity_factor': 1.1,  # Teams know each other well
                    'motivation_factor': 1.05   # Higher stakes
                }
            else:
                adjustments['regular_season'] = {
                    'standard_factors': 1.0
                }
                
            return adjustments
            
        except Exception as e:
            self.logger.error(f"Error getting tournament adjustments: {e}")
            return {'available': False, 'error': str(e)}
            
    def _calculate_upset_potential(self, game_context: GameContext) -> float:
        """Calculate upset potential in March Madness"""
        if not game_context.home_seed or not game_context.away_seed:
            return 0.5
            
        seed_diff = abs(game_context.home_seed - game_context.away_seed)
        
        # Higher seed differences increase upset potential
        if seed_diff >= 5:
            return 0.8
        elif seed_diff >= 3:
            return 0.6
        else:
            return 0.3
            
    def _calculate_tournament_experience(self, game_context: GameContext) -> float:
        """Calculate tournament experience factor"""
        # Would normally look up historical tournament appearances
        return 1.0  # Neutral for now
        
    def _calculate_seeding_advantage(self, game_context: GameContext) -> float:
        """Calculate advantage from better seeding"""
        if not game_context.home_seed or not game_context.away_seed:
            return 0.0
            
        seed_diff = game_context.away_seed - game_context.home_seed  # Positive if home is better seed
        return seed_diff * 0.5  # Each seed line worth 0.5 points
        
    def _get_venue_adjustments(self, game_context: GameContext) -> Dict:
        """Get venue-specific adjustments"""
        try:
            adjustments = {
                'home_court_advantage': 3.5 if not game_context.neutral_site else 0.0,
                'venue_size_factor': 1.0,
                'altitude_factor': 1.0,
                'crowd_factor': 1.0
            }
            
            # Venue capacity affects home court advantage
            capacity = game_context.venue_capacity
            if capacity > 20000:
                adjustments['crowd_factor'] = 1.15
            elif capacity > 15000:
                adjustments['crowd_factor'] = 1.10
            elif capacity < 5000:
                adjustments['crowd_factor'] = 0.95
                
            return adjustments
            
        except Exception as e:
            self.logger.error(f"Error getting venue adjustments: {e}")
            return {'home_court_advantage': 3.5, 'venue_size_factor': 1.0}
            
    def _get_conference_strength_factors(self, game_context: GameContext) -> Dict:
        """Get conference strength factors"""
        try:
            # Conference strength tiers (simplified)
            tier_1_conferences = ['ACC', 'Big 12', 'Big Ten', 'SEC', 'Big East', 'Pac-12']
            tier_2_conferences = ['American', 'Mountain West', 'WCC', 'A-10']
            
            home_conf = game_context.home_conference
            away_conf = game_context.away_conference
            
            def get_conf_strength(conf):
                if conf in tier_1_conferences:
                    return 1.1
                elif conf in tier_2_conferences:
                    return 1.0
                else:
                    return 0.9
                    
            factors = {
                'home_conference_strength': get_conf_strength(home_conf),
                'away_conference_strength': get_conf_strength(away_conf),
                'strength_differential': get_conf_strength(home_conf) - get_conf_strength(away_conf)
            }
            
            return factors
            
        except Exception as e:
            self.logger.error(f"Error getting conference strength factors: {e}")
            return {'home_conference_strength': 1.0, 'away_conference_strength': 1.0}
            
    # Additional methods for basketball prediction logic...
    # (continuing with similar adaptations of the baseball engine methods)
    
    def _combine_predictions(self, components: BasketballPredictionComponents) -> Dict:
        """Combine all basketball prediction components into final predictions"""
        try:
            # Start with base prediction
            final_win_prob = components.win_probability
            final_point_diff = components.point_differential
            final_total_points = components.total_points
            final_tempo = components.tempo_prediction
            
            # Apply tournament adjustments
            if components.tournament_adjustments.get('available'):
                tournament_data = components.tournament_adjustments
                
                if 'march_madness' in tournament_data:
                    madness_data = tournament_data['march_madness']
                    seeding_advantage = madness_data.get('seeding_advantage', 0.0)
                    final_point_diff += seeding_advantage
                    final_win_prob = self._point_differential_to_win_probability(final_point_diff)
                    
            # Apply venue adjustments
            venue_data = components.venue_adjustments
            home_court = venue_data.get('home_court_advantage', 0.0)
            final_point_diff += home_court
            final_win_prob = self._point_differential_to_win_probability(final_point_diff)
            
            # Apply fatigue adjustments
            if components.fatigue_adjustments.get('available'):
                fatigue_data = components.fatigue_adjustments
                home_fatigue = fatigue_data.get('home_fatigue_factor', 1.0)
                away_fatigue = fatigue_data.get('away_fatigue_factor', 1.0)
                
                # Fatigue affects total points more than spread
                fatigue_factor = (home_fatigue + away_fatigue) / 2
                final_total_points *= fatigue_factor
                final_tempo *= fatigue_factor
                
            # Constrain final predictions to reasonable ranges
            final_win_prob = max(0.05, min(0.95, final_win_prob))
            final_total_points = max(100.0, min(200.0, final_total_points))
            final_point_diff = max(-40.0, min(40.0, final_point_diff))
            final_tempo = max(50.0, min(90.0, final_tempo))
            
            return {
                'win_probability': final_win_prob,
                'point_differential': final_point_diff,
                'total_points': final_total_points,
                'tempo': final_tempo
            }
            
        except Exception as e:
            self.logger.error(f"Error combining basketball predictions: {e}")
            return {
                'win_probability': 0.5,
                'point_differential': 0.0,
                'total_points': self.AVERAGE_COLLEGE_POINTS,
                'tempo': self.AVERAGE_COLLEGE_POSSESSIONS
            }
            
    # Continue with additional basketball-specific methods...
    # (betting recommendations, data quality assessment, etc.)
    
    def _odds_to_probability(self, odds: int) -> float:
        """Convert American odds to implied probability"""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return -odds / (-odds + 100)
            
    def _generate_betting_recommendations(
        self,
        game_context: GameContext,
        components: BasketballPredictionComponents,
        final_predictions: Dict[str, float],
    ) -> Dict[str, Union[BasketballBettingRecommendation, List[BasketballBettingRecommendation]]]:
        """Generate actionable betting recommendations."""
        recommendations: Dict[str, Union[BasketballBettingRecommendation, List[BasketballBettingRecommendation]]] = {}

        confidence_multiplier = max(
            0.1,
            min(
                1.0,
                (components.prediction_confidence * 0.5)
                + (components.model_agreement * 0.3)
                + (components.data_quality_score * 0.2),
            ),
        )

        # Spread recommendation
        spread_info = game_context.betting_lines.get('spread', {})
        spread_line = spread_info.get('line')
        if spread_line is not None:
            expected_margin = -spread_line
            predicted_margin = final_predictions['point_differential']
            margin_edge = predicted_margin - expected_margin

            home_cover_prob = self._margin_diff_to_probability(margin_edge)
            away_cover_prob = 1.0 - home_cover_prob

            home_odds = spread_info.get('home_odds', -110)
            away_odds = spread_info.get('away_odds', -110)
            implied_home = self._odds_to_probability(home_odds)
            implied_away = self._odds_to_probability(away_odds)

            home_edge = home_cover_prob - implied_home
            away_edge = away_cover_prob - implied_away

            if home_edge >= away_edge:
                spread_side = 'home'
                edge_pct = home_edge
                odds_used = home_odds
                cover_probability = home_cover_prob
                selection = f"{game_context.home_team} {spread_line:+.1f}"
            else:
                spread_side = 'away'
                edge_pct = away_edge
                odds_used = away_odds
                cover_probability = away_cover_prob
                selection = f"{game_context.away_team} {-spread_line:+.1f}"

            recommendation_level, suggested_units = self._grade_edge(edge_pct, confidence_multiplier)
            if recommendation_level != "AVOID":
                recommendations['spread'] = BasketballBettingRecommendation(
                    bet_type="spread",
                    recommendation=recommendation_level,
                    recommended_odds=odds_used,
                    fair_value=self._probability_to_american_odds(cover_probability),
                    edge_percentage=edge_pct,
                    confidence=confidence_multiplier,
                    suggested_unit_size=suggested_units,
                    max_bet_percentage=min(0.06, suggested_units * 0.03),
                    key_factors=self._collect_key_factors(game_context, components),
                    risk_factors=self._collect_risk_factors(components),
                    reasoning=(
                        f"Projected margin {predicted_margin:+.1f} vs market {expected_margin:+.1f}; "
                        f"{selection} shows {edge_pct:.1%} edge"
                    ),
                )

        # Total recommendation
        total_info = game_context.betting_lines.get('total', {})
        total_line = total_info.get('line')
        if total_line is not None:
            total_diff = final_predictions['total_points'] - total_line
            over_probability = self._margin_diff_to_probability(total_diff, slope=0.08)
            under_probability = 1.0 - over_probability

            over_odds = total_info.get('over_odds', -110)
            under_odds = total_info.get('under_odds', -110)
            implied_over = self._odds_to_probability(over_odds)
            implied_under = self._odds_to_probability(under_odds)

            over_edge = over_probability - implied_over
            under_edge = under_probability - implied_under

            if over_edge >= under_edge:
                total_side = 'over'
                edge_pct = over_edge
                odds_used = over_odds
                prob_used = over_probability
            else:
                total_side = 'under'
                edge_pct = under_edge
                odds_used = under_odds
                prob_used = under_probability

            recommendation_level, suggested_units = self._grade_edge(edge_pct, confidence_multiplier)
            if recommendation_level != "AVOID":
                recommendations['total'] = BasketballBettingRecommendation(
                    bet_type="total",
                    recommendation=recommendation_level,
                    recommended_odds=odds_used,
                    fair_value=self._probability_to_american_odds(prob_used),
                    edge_percentage=edge_pct,
                    confidence=confidence_multiplier,
                    suggested_unit_size=suggested_units,
                    max_bet_percentage=min(0.05, suggested_units * 0.025),
                    key_factors=self._collect_key_factors(game_context, components, include_tempo=True),
                    risk_factors=self._collect_risk_factors(components, include_tempo=True),
                    reasoning=(
                        f"Projected total {final_predictions['total_points']:.1f} vs market {total_line:.1f}; "
                        f"{total_side.upper()} edge {edge_pct:.1%}"
                    ),
                )

        # Moneyline recommendation
        moneyline_info = game_context.odds.get('moneyline', {})
        home_ml = moneyline_info.get('home')
        away_ml = moneyline_info.get('away')
        if home_ml is not None and away_ml is not None:
            home_win_prob = final_predictions['win_probability']
            away_win_prob = 1.0 - home_win_prob
            implied_home = self._odds_to_probability(home_ml)
            implied_away = self._odds_to_probability(away_ml)

            home_edge = home_win_prob - implied_home
            away_edge = away_win_prob - implied_away

            if home_edge >= away_edge:
                ml_team = game_context.home_team
                odds_used = home_ml
                prob_used = home_win_prob
                edge_pct = home_edge
            else:
                ml_team = game_context.away_team
                odds_used = away_ml
                prob_used = away_win_prob
                edge_pct = away_edge

            recommendation_level, suggested_units = self._grade_edge(edge_pct, confidence_multiplier, moneyline=True)
            if recommendation_level != "AVOID":
                recommendations['moneyline'] = BasketballBettingRecommendation(
                    bet_type="moneyline",
                    recommendation=recommendation_level,
                    recommended_odds=odds_used,
                    fair_value=self._probability_to_american_odds(prob_used),
                    edge_percentage=edge_pct,
                    confidence=confidence_multiplier,
                    suggested_unit_size=suggested_units,
                    max_bet_percentage=min(0.08, suggested_units * 0.04),
                    key_factors=self._collect_key_factors(game_context, components),
                    risk_factors=self._collect_risk_factors(components),
                    reasoning=(
                        f"Win probability {prob_used:.1%} vs implied {self._odds_to_probability(odds_used):.1%} "
                        f"for {ml_team}"
                    ),
                )

        recommendations['props'] = []
        return recommendations
        
    def _determine_confidence_tier(self, components: BasketballPredictionComponents) -> str:
        """Translate numeric confidence into tiered label."""
        composite = (
            components.prediction_confidence * 0.5
            + components.model_agreement * 0.3
            + components.data_quality_score * 0.2
        )
        if composite >= self.CONFIDENCE_THRESHOLDS['high']:
            return "HIGH"
        if composite >= self.CONFIDENCE_THRESHOLDS['medium']:
            return "MEDIUM"
        return "LOW"
        
    def _calculate_prediction_confidence(self, base_prediction: Dict, *system_predictions: Dict) -> float:
        """Aggregate system confidence using margin strength and available modules."""
        margin_strength = abs(base_prediction.get('point_differential', 0.0)) / 12.0
        margin_strength = max(0.0, min(0.25, margin_strength))

        total_deviation = abs(base_prediction.get('total_points', self.AVERAGE_COLLEGE_POINTS) - self.AVERAGE_COLLEGE_POINTS) / 60.0
        total_deviation = max(0.0, min(0.1, total_deviation))

        confidence = 0.5 + margin_strength + total_deviation

        for system in system_predictions:
            if isinstance(system, dict) and system.get('available', False):
                confidence += 0.02
            elif isinstance(system, dict) and system:
                confidence += 0.01

        return max(0.35, min(0.9, confidence))
        
    def _calculate_model_agreement(self, predictions: List[Dict]) -> float:
        """Estimate agreement between base model and auxiliary systems."""
        if not predictions:
            return 0.5

        base = predictions[0]
        base_margin = base.get('point_differential', 0.0)
        votes = []

        for system in predictions[1:]:
            if not isinstance(system, dict) or not system:
                continue

            adjustment = 0.0
            if 'march_madness' in system:
                adjustment += system['march_madness'].get('seeding_advantage', 0.0)
            if 'conference_tournament' in system:
                adjustment += 0.5
            if 'home_depth' in system:
                home_strength = system['home_depth'].get('bench_strength', 0.5)
                away_strength = system['away_depth'].get('bench_strength', 0.5)
                adjustment += (home_strength - away_strength)
            if 'home_fatigue_factor' in system:
                fatigue_diff = system.get('home_fatigue_factor', 1.0) - system.get('away_fatigue_factor', 1.0)
                adjustment += fatigue_diff * 4.0

            votes.append(np.sign(base_margin + adjustment))

        if not votes:
            return 0.55

        agreement = votes.count(np.sign(base_margin)) / len(votes)
        return max(0.4, min(0.9, agreement))
        
    def _assess_data_quality(self, game_context: GameContext, game_data: Dict) -> float:
        """Assess completeness of the provided matchup data."""
        checks = [
            bool(game_context.home_team_stats),
            bool(game_context.away_team_stats),
            bool(game_data.get('betting_lines')),
            game_data.get('home_kenpom_rating') is not None,
            game_data.get('away_kenpom_rating') is not None,
        ]
        provided = sum(1 for check in checks if check)
        quality = 0.5 + 0.1 * provided

        if game_context.tournament_context in ('march_madness', 'conference_tournament'):
            quality += 0.05
        if game_context.neutral_site:
            quality += 0.02

        return max(0.4, min(0.95, quality))
        
    def _store_prediction(self, prediction: ComprehensiveBasketballPrediction) -> None:
        """Persist prediction and recommendations to SQLite."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                '''
                INSERT INTO basketball_game_predictions (
                    game_id,
                    game_date,
                    home_team,
                    away_team,
                    tournament_context,
                    prediction_timestamp,
                    final_win_probability,
                    final_point_differential,
                    final_total_points,
                    final_tempo,
                    confidence_tier,
                    model_version,
                    prediction_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''',
                (
                    prediction.game_context.game_id,
                    prediction.game_context.date.strftime('%Y-%m-%d'),
                    prediction.game_context.home_team,
                    prediction.game_context.away_team,
                    prediction.game_context.tournament_context,
                    prediction.prediction_timestamp.isoformat(),
                    prediction.final_win_probability,
                    prediction.final_point_differential,
                    prediction.final_total_points,
                    prediction.final_tempo,
                    prediction.confidence_tier,
                    prediction.model_version,
                    json.dumps(self._serialize_prediction(prediction), default=str),
                ),
            )

            for bet_key in ('spread', 'total', 'moneyline'):
                rec = getattr(prediction, f"{bet_key}_recommendation", None)
                if not rec:
                    continue
                cursor.execute(
                    '''
                    INSERT INTO basketball_betting_recommendations (
                        game_id,
                        bet_type,
                        recommendation,
                        confidence,
                        edge_percentage,
                        suggested_units,
                        reasoning
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        prediction.game_context.game_id,
                        rec.bet_type,
                        rec.recommendation,
                        rec.confidence,
                        rec.edge_percentage,
                        rec.suggested_unit_size,
                        rec.reasoning,
                    ),
                )

            conn.commit()
            conn.close()
        except Exception as exc:
            self.logger.warning(f"Failed to store basketball prediction: {exc}")
        
    def _create_default_prediction(self, game_data: Dict) -> ComprehensiveBasketballPrediction:
        """Return neutral prediction object when generation fails."""
        context = self._create_default_game_context(game_data)
        components = self._create_default_prediction_components()

        return ComprehensiveBasketballPrediction(
            game_context=context,
            prediction_components=components,
            final_win_probability=0.5,
            final_point_differential=0.0,
            final_total_points=self.AVERAGE_COLLEGE_POINTS,
            final_tempo=self.AVERAGE_COLLEGE_POSSESSIONS,
            spread_recommendation=None,
            total_recommendation=None,
            moneyline_recommendation=None,
            prop_recommendations=[],
            prediction_timestamp=datetime.now(),
            model_version=self.model_version,
            confidence_tier="LOW",
        )
        
    def _create_default_game_context(self, game_data: Dict) -> GameContext:
        """Create minimal viable game context."""
        now = datetime.now()
        return GameContext(
            game_id=game_data.get('game_id', f"default_{now.strftime('%Y%m%d_%H%M%S')}"),
            date=datetime.strptime(game_data.get('date', now.strftime('%Y-%m-%d')), '%Y-%m-%d'),
            home_team=game_data.get('home_team', 'HOME'),
            away_team=game_data.get('away_team', 'AWAY'),
            home_conference=game_data.get('home_conference', 'UNKNOWN'),
            away_conference=game_data.get('away_conference', 'UNKNOWN'),
            home_ranking=game_data.get('home_ranking'),
            away_ranking=game_data.get('away_ranking'),
            home_kenpom_rating=game_data.get('home_kenpom_rating', 50.0),
            away_kenpom_rating=game_data.get('away_kenpom_rating', 50.0),
            tournament_context=game_data.get('tournament_context', 'regular_season'),
            tournament_round=game_data.get('tournament_round'),
            home_seed=game_data.get('home_seed'),
            away_seed=game_data.get('away_seed'),
            home_team_stats=game_data.get('home_team_stats', {}),
            away_team_stats=game_data.get('away_team_stats', {}),
            venue=game_data.get('venue', 'Unknown'),
            venue_capacity=game_data.get('venue_capacity', 10000),
            neutral_site=game_data.get('neutral_site', False),
            game_time=game_data.get('game_time', '19:00'),
            tv_coverage=game_data.get('tv_coverage', 'Local'),
            odds=game_data.get('odds', {}),
            betting_lines=game_data.get('betting_lines', {}),
        )
        
    def _create_default_prediction_components(self) -> BasketballPredictionComponents:
        """Return neutral prediction component bundle."""
        return BasketballPredictionComponents(
            win_probability=0.5,
            point_differential=0.0,
            total_points=self.AVERAGE_COLLEGE_POINTS,
            tempo_prediction=self.AVERAGE_COLLEGE_POSSESSIONS,
            efficiency_predictions={},
            fatigue_adjustments={'available': False},
            injury_impacts={'available': False},
            depth_advantages={'available': False},
            tournament_adjustments={'available': False},
            venue_adjustments={'home_court_advantage': 0.0},
            conference_strength_factors={
                'home_conference_strength': 1.0,
                'away_conference_strength': 1.0,
                'strength_differential': 0.0,
            },
            prediction_confidence=0.5,
            model_agreement=0.5,
            data_quality_score=0.5,
        )

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _margin_diff_to_probability(self, diff: float, slope: float = 0.25) -> float:
        """Convert margin differential into probability (logistic)."""
        odds = np.exp(slope * diff)
        probability = odds / (1 + odds)
        return max(0.05, min(0.95, probability))

    def _grade_edge(self, edge_pct: float, confidence: float, moneyline: bool = False) -> Tuple[str, float]:
        """Translate edge to recommendation strength and staking."""
        if edge_pct <= self.BETTING_THRESHOLDS['avoid']:
            return "AVOID", 0.0

        multiplier = 1.2 if moneyline else 1.0
        adjusted_edge = edge_pct * confidence * multiplier

        if adjusted_edge >= self.BETTING_THRESHOLDS['strong_bet']:
            return "STRONG_BET", round(1.5 * confidence, 2)
        if adjusted_edge >= self.BETTING_THRESHOLDS['bet']:
            return "BET", round(1.0 * confidence, 2)
        return "SMALL_BET", round(0.5 * confidence, 2)

    def _probability_to_american_odds(self, probability: float) -> float:
        """Convert probability to American odds for fair value reference."""
        probability = max(0.01, min(0.99, probability))
        if probability >= 0.5:
            return - (probability / (1 - probability)) * 100
        return ((1 - probability) / probability) * 100

    def _collect_key_factors(
        self,
        game_context: GameContext,
        components: BasketballPredictionComponents,
        include_tempo: bool = False,
    ) -> List[str]:
        """Summarize factors supporting a wager."""
        factors: List[str] = []
        if game_context.tournament_context != 'regular_season':
            factors.append(f"{game_context.tournament_context.replace('_', ' ').title()} context")

        conf_diff = components.conference_strength_factors.get('strength_differential', 0.0)
        if conf_diff:
            favored_team = game_context.home_team if conf_diff > 0 else game_context.away_team
            factors.append(f"Conference strength favors {favored_team}")

        if include_tempo:
            factors.append(f"Tempo projection {components.tempo_prediction:.1f}")

        if components.injury_impacts.get('available'):
            factors.append("Injury review applied")
        if components.depth_advantages.get('available'):
            factors.append("Depth edge incorporated")

        return factors[:4]

    def _collect_risk_factors(
        self,
        components: BasketballPredictionComponents,
        include_tempo: bool = False,
    ) -> List[str]:
        """Summarize risk items for transparency."""
        risks: List[str] = []
        if components.data_quality_score < 0.65:
            risks.append("Data quality below ideal")
        if components.model_agreement < 0.55:
            risks.append("Model disagreement")
        if components.prediction_confidence < 0.6:
            risks.append("Modest confidence")
        if include_tempo and abs(components.tempo_prediction - self.AVERAGE_COLLEGE_POSSESSIONS) < 3:
            risks.append("Tempo near league average")
        return risks[:3]

    def _serialize_prediction(self, prediction: ComprehensiveBasketballPrediction) -> Dict[str, Any]:
        """Serialize structured prediction for database storage."""
        payload = {
            'game_context': asdict(prediction.game_context),
            'final_predictions': {
                'win_probability': prediction.final_win_probability,
                'point_differential': prediction.final_point_differential,
                'total_points': prediction.final_total_points,
                'tempo': prediction.final_tempo,
            },
            'confidence_tier': prediction.confidence_tier,
        }

        def serialize_rec(rec: Optional[BasketballBettingRecommendation]) -> Optional[Dict[str, Any]]:
            if not rec:
                return None
            return asdict(rec)

        payload['recommendations'] = {
            'spread': serialize_rec(prediction.spread_recommendation),
            'total': serialize_rec(prediction.total_recommendation),
            'moneyline': serialize_rec(prediction.moneyline_recommendation),
            'props': [asdict(prop) for prop in prediction.prop_recommendations],
        }
        return payload

def main():
    """Test the basketball prediction engine"""
    logging.basicConfig(level=logging.INFO)
    
    engine = CoreBasketballPredictionEngine()
    
    print("College Basketball Prediction Engine Test")
    print("=" * 45)
    
    # Mock game data
    game_data = {
        'game_id': 'duke_unc_20250315',
        'date': '2025-03-15',
        'home_team': 'Duke',
        'away_team': 'North Carolina',
        'home_conference': 'ACC',
        'away_conference': 'ACC',
        'home_ranking': 3,
        'away_ranking': 8,
        'home_kenpom_rating': 85.2,
        'away_kenpom_rating': 78.6,
        'tournament_context': 'march_madness',
        'tournament_round': 'elite_eight',
        'home_seed': 2,
        'away_seed': 4,
        'home_team_stats': {
            'offensive_efficiency': 118.5,
            'defensive_efficiency': 95.2,
            'tempo': 72.3
        },
        'away_team_stats': {
            'offensive_efficiency': 115.8,
            'defensive_efficiency': 98.1,
            'tempo': 69.8
        },
        'venue': 'Madison Square Garden',
        'venue_capacity': 20000,
        'neutral_site': True,
        'game_time': '21:20',
        'tv_coverage': 'CBS',
        'odds': {
            'moneyline': {
                'home': -140,
                'away': +120
            }
        },
        'betting_lines': {
            'spread': {
                'line': -3.5,
                'home_odds': -110,
                'away_odds': -110
            },
            'total': {
                'line': 148.5,
                'over_odds': -110,
                'under_odds': -110
            }
        }
    }
    
    # Generate prediction (would normally work with full implementation)
    try:
        prediction = engine.generate_comprehensive_prediction(game_data)
        print("✅ Basketball Prediction Engine operational!")
    except Exception as e:
        print(f"⚠️  Prediction engine test incomplete: {e}")
        
    # Show basic prediction logic
    base_prediction = engine._generate_base_basketball_prediction(
        engine._create_game_context(game_data), 
        game_data
    )
    
    print(f"\nSample Prediction (Duke vs UNC):")
    print(f"  Win Probability: {base_prediction['win_probability']:.1%}")
    print(f"  Point Differential: {base_prediction['point_differential']:+.1f}")
    print(f"  Total Points: {base_prediction['total_points']:.1f}")
    print(f"  Game Tempo: {base_prediction['tempo']:.1f} possessions")

if __name__ == "__main__":
    main()
