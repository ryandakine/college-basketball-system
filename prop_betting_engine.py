#!/usr/bin/env python3
"""
Advanced Prop Betting Engine for College Basketball
==================================================

Specialized models for high-margin prop bets:
- Player props (points, rebounds, assists, threes)
- Team props (first to score, halftime leaders)
- Exotic props (exact margin, overtime probability)
- Game props (total fouls, technical fouls, blocks)
- Matchup-specific props (head-to-head performance)
- Live prop adjustments during games
- Correlation analysis between props
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import sqlite3
from dataclasses import dataclass, asdict
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import asyncio

@dataclass
class PlayerProfile:
    """Individual player statistical profile"""
    player_name: str
    team: str
    position: str
    
    # Usage and efficiency
    usage_rate: float
    minutes_per_game: float
    efficiency_rating: float
    
    # Basic stats
    points_per_game: float
    rebounds_per_game: float
    assists_per_game: float
    steals_per_game: float
    blocks_per_game: float
    turnovers_per_game: float
    
    # Shooting stats
    fg_percentage: float
    three_point_percentage: float
    free_throw_percentage: float
    three_attempts_per_game: float
    free_throw_attempts_per_game: float
    
    # Advanced metrics
    player_impact_estimate: float  # Similar to NBA's PIE
    clutch_performance: float      # Performance in close games
    matchup_adjustments: Dict[str, float]  # vs different positions
    
    # Situational performance
    home_vs_away_splits: Dict[str, Dict[str, float]]
    vs_conference_performance: Dict[str, float]
    recent_form: Dict[str, float]  # Last 10 games
    
    # Injury and fatigue
    injury_risk: float
    fatigue_factor: float
    minutes_ceiling: float  # Max sustainable minutes

@dataclass
class PropBet:
    """Individual prop bet opportunity"""
    prop_id: str
    game_id: str
    timestamp: datetime
    
    # Prop details
    prop_type: str      # 'player_points', 'player_rebounds', 'team_total', etc.
    player_name: Optional[str]
    team: Optional[str]
    line: float         # The betting line (e.g., 15.5 points)
    
    # Market odds
    over_odds: float
    under_odds: float
    over_implied_prob: float
    under_implied_prob: float
    
    # Model predictions
    predicted_value: float
    prediction_std: float
    over_probability: float
    under_probability: float
    
    # Value analysis
    over_edge: float
    under_edge: float
    best_bet: str       # 'OVER', 'UNDER', or 'AVOID'
    
    # Confidence and risk
    confidence: float
    correlation_risk: float
    liquidity_score: float
    
    # Supporting data
    historical_hit_rate: float
    recent_trend: str
    key_factors: List[str]

@dataclass
class PropRecommendation:
    """Final prop betting recommendation"""
    prop_bet: PropBet
    
    # Sizing and strategy
    recommended_side: str
    unit_size: float
    max_bet_amount: float
    
    # Risk analysis
    expected_value: float
    kelly_fraction: float
    risk_score: float
    
    # Timing
    urgency: str
    line_movement_expectation: float
    best_time_to_bet: str
    
    # Reasoning
    primary_reasons: List[str]
    risk_factors: List[str]
    similar_props_correlation: float

class PropBettingEngine:
    """Advanced prop betting analysis engine"""
    
    def __init__(self, db_path: str = "prop_betting.db"):
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        
        # Initialize database
        self._init_database()
        
        # Player profiles cache
        self.player_profiles = {}
        
        # Prop models
        self.prop_models = {}
        self._initialize_prop_models()
        
        # Correlation matrices
        self.prop_correlations = {}
        
        # Betting parameters
        self.MIN_PROP_EDGE = 0.08  # 8% minimum edge for props
        self.MAX_CORRELATED_EXPOSURE = 0.15  # 15% max on correlated props
        
        # Prop-specific parameters
        self.PROP_PARAMETERS = {
            'player_points': {
                'variance_multiplier': 1.2,
                'home_advantage': 1.05,
                'pace_adjustment': True,
                'usage_weight': 0.4
            },
            'player_rebounds': {
                'variance_multiplier': 1.4,
                'matchup_weight': 0.6,
                'pace_adjustment': True,
                'size_factor': True
            },
            'player_assists': {
                'variance_multiplier': 1.6,
                'pace_adjustment': True,
                'teammate_usage': True,
                'style_factor': 0.3
            },
            'player_threes': {
                'variance_multiplier': 2.0,
                'defense_adjustment': 0.5,
                'hot_hand': True,
                'volume_dependent': True
            }
        }
    
    def _init_database(self):
        """Initialize prop betting database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Player profiles
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS player_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                player_name TEXT,
                team TEXT,
                position TEXT,
                usage_rate REAL,
                minutes_per_game REAL,
                points_per_game REAL,
                rebounds_per_game REAL,
                assists_per_game REAL,
                three_point_percentage REAL,
                updated_date DATETIME
            )
        ''')
        
        # Prop bet history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prop_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prop_id TEXT,
                game_id TEXT,
                timestamp DATETIME,
                prop_type TEXT,
                player_name TEXT,
                team TEXT,
                line REAL,
                over_odds REAL,
                under_odds REAL,
                predicted_value REAL,
                actual_value REAL,
                result TEXT,  -- 'OVER', 'UNDER', 'PUSH'
                profit_loss REAL
            )
        ''')
        
        # Prop opportunities
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prop_opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prop_id TEXT,
                timestamp DATETIME,
                prop_type TEXT,
                player_name TEXT,
                line REAL,
                recommended_side TEXT,
                edge REAL,
                confidence REAL,
                unit_size REAL,
                expected_value REAL,
                acted_upon BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Prop correlations
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prop_correlations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prop_type_1 TEXT,
                prop_type_2 TEXT,
                player_1 TEXT,
                player_2 TEXT,
                correlation_coefficient REAL,
                sample_size INTEGER,
                updated_date DATETIME
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _initialize_prop_models(self):
        """Initialize machine learning models for different prop types"""
        
        # Player points model
        self.prop_models['player_points'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        
        # Player rebounds model
        self.prop_models['player_rebounds'] = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=3,
            random_state=42
        )
        
        # Player assists model
        self.prop_models['player_assists'] = RandomForestRegressor(
            n_estimators=80,
            max_depth=8,
            min_samples_split=3,
            random_state=42
        )
        
        # Three-pointers model
        self.prop_models['player_threes'] = RandomForestRegressor(
            n_estimators=120,
            max_depth=6,
            min_samples_split=5,
            random_state=42
        )
    
    async def load_player_profiles(self, teams: List[str]) -> Dict[str, PlayerProfile]:
        """Load and update player profiles"""
        profiles = {}
        
        for team in teams:
            team_players = await self._fetch_team_players(team)
            
            for player_data in team_players:
                profile = self._create_player_profile(player_data)
                profiles[player_data['name']] = profile
        
        self.player_profiles.update(profiles)
        return profiles
    
    async def _fetch_team_players(self, team: str) -> List[Dict[str, Any]]:
        """Fetch player data for a team"""
        # This would integrate with real data sources
        # For demo, simulate player data
        
        positions = ['PG', 'SG', 'SF', 'PF', 'C']
        players = []
        
        for i in range(8):  # 8 players per team
            player = {
                'name': f'{team} Player {i+1}',
                'team': team,
                'position': positions[i % len(positions)],
                'minutes_per_game': np.random.uniform(10, 35),
                'usage_rate': np.random.uniform(0.15, 0.35),
                'points_per_game': np.random.uniform(5, 25),
                'rebounds_per_game': np.random.uniform(2, 12),
                'assists_per_game': np.random.uniform(1, 8),
                'fg_percentage': np.random.uniform(0.35, 0.60),
                'three_point_percentage': np.random.uniform(0.25, 0.45),
                'three_attempts_per_game': np.random.uniform(1, 8)
            }
            players.append(player)
        
        return players
    
    def _create_player_profile(self, player_data: Dict[str, Any]) -> PlayerProfile:
        """Create comprehensive player profile"""
        
        # Calculate advanced metrics
        efficiency = (player_data['points_per_game'] + 
                     player_data['rebounds_per_game'] + 
                     player_data['assists_per_game']) / player_data['minutes_per_game']
        
        # Simulate clutch performance
        clutch_performance = np.random.normal(1.0, 0.2)  # Multiplier vs regular stats
        
        # Position-based matchup adjustments
        position = player_data['position']
        matchup_adjustments = {}
        
        if position in ['PG', 'SG']:  # Guards
            matchup_adjustments = {
                'vs_big_defense': 0.9,    # Struggle against size
                'vs_small_defense': 1.1,  # Excel against small lineups
                'vs_press': 0.95          # Slight difficulty vs pressure
            }
        elif position in ['SF', 'PF']:  # Forwards
            matchup_adjustments = {
                'vs_athletic_defense': 0.9,
                'vs_slow_defense': 1.1,
                'vs_zone': 1.05
            }
        else:  # Centers
            matchup_adjustments = {
                'vs_small_ball': 1.15,
                'vs_twin_towers': 0.9,
                'vs_switching': 0.95
            }
        
        # Home vs away splits
        home_away_splits = {
            'home': {
                'points_multiplier': np.random.normal(1.08, 0.1),
                'rebounds_multiplier': np.random.normal(1.05, 0.08),
                'assists_multiplier': np.random.normal(1.03, 0.1)
            },
            'away': {
                'points_multiplier': np.random.normal(0.95, 0.1),
                'rebounds_multiplier': np.random.normal(0.97, 0.08),
                'assists_multiplier': np.random.normal(0.98, 0.1)
            }
        }
        
        return PlayerProfile(
            player_name=player_data['name'],
            team=player_data['team'],
            position=position,
            usage_rate=player_data['usage_rate'],
            minutes_per_game=player_data['minutes_per_game'],
            efficiency_rating=efficiency,
            points_per_game=player_data['points_per_game'],
            rebounds_per_game=player_data['rebounds_per_game'],
            assists_per_game=player_data['assists_per_game'],
            steals_per_game=np.random.uniform(0.5, 2.5),
            blocks_per_game=np.random.uniform(0.1, 2.0),
            turnovers_per_game=np.random.uniform(1.0, 4.0),
            fg_percentage=player_data['fg_percentage'],
            three_point_percentage=player_data['three_point_percentage'],
            free_throw_percentage=np.random.uniform(0.65, 0.90),
            three_attempts_per_game=player_data['three_attempts_per_game'],
            free_throw_attempts_per_game=np.random.uniform(1, 6),
            player_impact_estimate=efficiency * player_data['usage_rate'],
            clutch_performance=clutch_performance,
            matchup_adjustments=matchup_adjustments,
            home_vs_away_splits=home_away_splits,
            vs_conference_performance={'in_conference': 1.0, 'out_of_conference': 0.98},
            recent_form={'last_5_games': np.random.normal(1.0, 0.15),
                        'last_10_games': np.random.normal(1.0, 0.12)},
            injury_risk=np.random.uniform(0.0, 0.3),
            fatigue_factor=1.0,
            minutes_ceiling=min(40, player_data['minutes_per_game'] * 1.3)
        )
    
    async def analyze_prop_opportunities(self, game_id: str, home_team: str, 
                                       away_team: str, game_context: Dict[str, Any]) -> List[PropBet]:
        """Analyze all prop opportunities for a game"""
        
        prop_opportunities = []
        
        # Get player profiles for both teams
        home_players = [p for p in self.player_profiles.values() if p.team == home_team]
        away_players = [p for p in self.player_profiles.values() if p.team == away_team]
        
        all_players = home_players + away_players
        
        # Analyze player props
        for player in all_players:
            # Skip players with high injury risk or low minutes
            if player.injury_risk > 0.5 or player.minutes_per_game < 15:
                continue
            
            # Points prop
            points_prop = await self._analyze_player_points_prop(player, game_context)
            if points_prop:
                prop_opportunities.append(points_prop)
            
            # Rebounds prop
            rebounds_prop = await self._analyze_player_rebounds_prop(player, game_context)
            if rebounds_prop:
                prop_opportunities.append(rebounds_prop)
            
            # Assists prop
            assists_prop = await self._analyze_player_assists_prop(player, game_context)
            if assists_prop:
                prop_opportunities.append(assists_prop)
            
            # Three-pointers prop
            if player.three_attempts_per_game > 2:
                threes_prop = await self._analyze_player_threes_prop(player, game_context)
                if threes_prop:
                    prop_opportunities.append(threes_prop)
        
        # Analyze team props
        team_props = await self._analyze_team_props(home_team, away_team, game_context)
        prop_opportunities.extend(team_props)
        
        # Filter for value
        value_props = [prop for prop in prop_opportunities 
                      if max(prop.over_edge, prop.under_edge) > self.MIN_PROP_EDGE]
        
        return value_props
    
    async def _analyze_player_points_prop(self, player: PlayerProfile, 
                                        game_context: Dict[str, Any]) -> Optional[PropBet]:
        """Analyze player points prop"""
        
        # Base prediction
        base_points = player.points_per_game
        
        # Apply adjustments
        adjustments = 1.0
        
        # Home/away adjustment
        is_home = player.team == game_context.get('home_team')
        if is_home:
            adjustments *= player.home_vs_away_splits['home']['points_multiplier']
        else:
            adjustments *= player.home_vs_away_splits['away']['points_multiplier']
        
        # Pace adjustment
        game_pace = game_context.get('expected_pace', 70)
        pace_adjustment = game_pace / 70  # Normalize to average pace
        adjustments *= pace_adjustment ** 0.5  # Square root for diminishing returns
        
        # Recent form
        adjustments *= player.recent_form['last_5_games']
        
        # Usage rate consideration
        if player.usage_rate > 0.25:  # High usage players
            adjustments *= 1.05
        elif player.usage_rate < 0.18:  # Low usage players
            adjustments *= 0.95
        
        # Opponent defense rating (simplified)
        opponent_def_rating = game_context.get('opponent_defensive_rating', 100)
        defense_adjustment = 105 / opponent_def_rating  # Invert - worse defense = more points
        adjustments *= defense_adjustment
        
        predicted_points = base_points * adjustments
        
        # Calculate standard deviation
        params = self.PROP_PARAMETERS['player_points']
        points_std = np.sqrt(predicted_points) * params['variance_multiplier']
        
        # Simulate market line (would be actual market data)
        market_line = predicted_points + np.random.normal(0, 1.5)
        
        # Calculate probabilities
        over_prob = 1 - stats.norm.cdf(market_line, predicted_points, points_std)
        under_prob = stats.norm.cdf(market_line, predicted_points, points_std)
        
        # Market odds (simplified)
        over_odds = -110
        under_odds = -110
        over_implied = self._odds_to_probability(over_odds)
        under_implied = self._odds_to_probability(under_odds)
        
        # Calculate edges
        over_edge = over_prob - over_implied
        under_edge = under_prob - under_implied
        
        # Determine best bet
        if over_edge > self.MIN_PROP_EDGE and over_edge > under_edge:
            best_bet = 'OVER'
        elif under_edge > self.MIN_PROP_EDGE:
            best_bet = 'UNDER'
        else:
            best_bet = 'AVOID'
        
        if best_bet == 'AVOID':
            return None
        
        # Historical hit rate (simplified)
        historical_hit_rate = 0.52 + (max(over_edge, under_edge) * 2)  # Rough approximation
        
        return PropBet(
            prop_id=f"{player.player_name}_points_{game_context.get('game_id', 'unknown')}",
            game_id=game_context.get('game_id', 'unknown'),
            timestamp=datetime.now(),
            prop_type='player_points',
            player_name=player.player_name,
            team=player.team,
            line=market_line,
            over_odds=over_odds,
            under_odds=under_odds,
            over_implied_prob=over_implied,
            under_implied_prob=under_implied,
            predicted_value=predicted_points,
            prediction_std=points_std,
            over_probability=over_prob,
            under_probability=under_prob,
            over_edge=over_edge,
            under_edge=under_edge,
            best_bet=best_bet,
            confidence=min(0.9, 0.6 + max(over_edge, under_edge)),
            correlation_risk=0.0,  # Would calculate vs other props
            liquidity_score=0.8,   # Player points usually liquid
            historical_hit_rate=historical_hit_rate,
            recent_trend='stable',
            key_factors=[
                f"Usage rate: {player.usage_rate:.1%}",
                f"Recent form: {player.recent_form['last_5_games']:.2f}x",
                f"Home/Away: {'Home' if is_home else 'Away'}"
            ]
        )
    
    async def _analyze_player_rebounds_prop(self, player: PlayerProfile, 
                                          game_context: Dict[str, Any]) -> Optional[PropBet]:
        """Analyze player rebounds prop"""
        
        base_rebounds = player.rebounds_per_game
        
        # Position-based expectations
        position_multiplier = {
            'PG': 0.8, 'SG': 0.9, 'SF': 1.0, 'PF': 1.2, 'C': 1.3
        }.get(player.position, 1.0)
        
        # Pace and style adjustments
        game_pace = game_context.get('expected_pace', 70)
        pace_adjustment = (game_pace / 70) ** 0.7  # Rebounds scale with possessions
        
        # Opponent rebounding strength
        opponent_rebounding = game_context.get('opponent_rebounding_rate', 50)
        rebounding_opportunity = (100 - opponent_rebounding) / 50
        
        predicted_rebounds = (base_rebounds * position_multiplier * 
                            pace_adjustment * rebounding_opportunity)
        
        # Higher variance for rebounds
        rebounds_std = np.sqrt(predicted_rebounds) * self.PROP_PARAMETERS['player_rebounds']['variance_multiplier']
        
        # Simulate market line
        market_line = predicted_rebounds + np.random.normal(0, 1.0) - 0.5  # Slight under bias
        
        # Calculate probabilities and edges
        over_prob = 1 - stats.norm.cdf(market_line, predicted_rebounds, rebounds_std)
        under_prob = stats.norm.cdf(market_line, predicted_rebounds, rebounds_std)
        
        over_implied = self._odds_to_probability(-110)
        under_implied = self._odds_to_probability(-110)
        
        over_edge = over_prob - over_implied
        under_edge = under_prob - under_implied
        
        best_bet = 'OVER' if over_edge > under_edge and over_edge > self.MIN_PROP_EDGE else \
                  'UNDER' if under_edge > self.MIN_PROP_EDGE else 'AVOID'
        
        if best_bet == 'AVOID':
            return None
        
        return PropBet(
            prop_id=f"{player.player_name}_rebounds_{game_context.get('game_id', 'unknown')}",
            game_id=game_context.get('game_id', 'unknown'),
            timestamp=datetime.now(),
            prop_type='player_rebounds',
            player_name=player.player_name,
            team=player.team,
            line=market_line,
            over_odds=-110,
            under_odds=-110,
            over_implied_prob=over_implied,
            under_implied_prob=under_implied,
            predicted_value=predicted_rebounds,
            prediction_std=rebounds_std,
            over_probability=over_prob,
            under_probability=under_prob,
            over_edge=over_edge,
            under_edge=under_edge,
            best_bet=best_bet,
            confidence=0.7,
            correlation_risk=0.0,
            liquidity_score=0.7,
            historical_hit_rate=0.52,
            recent_trend='stable',
            key_factors=[
                f"Position: {player.position}",
                f"Expected pace: {game_pace:.1f}",
                f"Opponent rebounding: {opponent_rebounding}"
            ]
        )
    
    async def _analyze_player_assists_prop(self, player: PlayerProfile, 
                                         game_context: Dict[str, Any]) -> Optional[PropBet]:
        """Analyze player assists prop"""
        
        # Assists are highly dependent on position and team style
        base_assists = player.assists_per_game
        
        # Position multipliers
        position_multiplier = {
            'PG': 1.2, 'SG': 0.8, 'SF': 1.0, 'PF': 0.9, 'C': 0.7
        }.get(player.position, 1.0)
        
        # Pace adjustment (assists scale strongly with pace)
        game_pace = game_context.get('expected_pace', 70)
        pace_adjustment = (game_pace / 70) ** 0.8
        
        # Team shooting percentage affects assists
        team_shooting = game_context.get('team_shooting_pct', 0.45)
        shooting_adjustment = team_shooting / 0.45
        
        predicted_assists = (base_assists * position_multiplier * 
                           pace_adjustment * shooting_adjustment)
        
        # Very high variance for assists
        assists_std = np.sqrt(predicted_assists) * self.PROP_PARAMETERS['player_assists']['variance_multiplier']
        
        # Market line
        market_line = predicted_assists + np.random.normal(0, 0.8)
        
        # Calculate probabilities
        over_prob = 1 - stats.norm.cdf(market_line, predicted_assists, assists_std)
        under_prob = stats.norm.cdf(market_line, predicted_assists, assists_std)
        
        over_implied = self._odds_to_probability(-110)
        under_implied = self._odds_to_probability(-110)
        
        over_edge = over_prob - over_implied
        under_edge = under_prob - under_implied
        
        best_bet = 'OVER' if over_edge > under_edge and over_edge > self.MIN_PROP_EDGE else \
                  'UNDER' if under_edge > self.MIN_PROP_EDGE else 'AVOID'
        
        if best_bet == 'AVOID' or predicted_assists < 2.0:  # Skip low-assist players
            return None
        
        return PropBet(
            prop_id=f"{player.player_name}_assists_{game_context.get('game_id', 'unknown')}",
            game_id=game_context.get('game_id', 'unknown'),
            timestamp=datetime.now(),
            prop_type='player_assists',
            player_name=player.player_name,
            team=player.team,
            line=market_line,
            over_odds=-110,
            under_odds=-110,
            over_implied_prob=over_implied,
            under_implied_prob=under_implied,
            predicted_value=predicted_assists,
            prediction_std=assists_std,
            over_probability=over_prob,
            under_probability=under_prob,
            over_edge=over_edge,
            under_edge=under_edge,
            best_bet=best_bet,
            confidence=0.65,  # Lower confidence due to variance
            correlation_risk=0.0,
            liquidity_score=0.6,
            historical_hit_rate=0.51,
            recent_trend='stable',
            key_factors=[
                f"Position: {player.position}",
                f"Team shooting: {team_shooting:.1%}",
                f"Pace adjustment: {pace_adjustment:.2f}x"
            ]
        )
    
    async def _analyze_player_threes_prop(self, player: PlayerProfile, 
                                        game_context: Dict[str, Any]) -> Optional[PropBet]:
        """Analyze player three-pointers made prop"""
        
        base_threes = player.three_attempts_per_game * player.three_point_percentage
        
        # Defensive adjustment
        opponent_three_defense = game_context.get('opponent_three_pt_defense', 0.33)
        defense_adjustment = 0.33 / opponent_three_defense  # Better defense = fewer makes
        
        # Game script adjustment (blowouts = more threes)
        expected_closeness = game_context.get('expected_margin', 5)
        script_adjustment = 1.0 + (abs(expected_closeness) / 50)  # More threes in blowouts
        
        predicted_threes = base_threes * defense_adjustment * script_adjustment
        
        # Extremely high variance for three-pointers
        threes_std = np.sqrt(predicted_threes) * self.PROP_PARAMETERS['player_threes']['variance_multiplier']
        
        # Market tends to set these at round numbers
        market_line = round(predicted_threes + np.random.normal(0, 0.5)) - 0.5
        market_line = max(0.5, market_line)  # Minimum 0.5
        
        # Use Poisson distribution for three-pointers (discrete events)
        over_prob = 1 - stats.poisson.cdf(market_line, predicted_threes)
        under_prob = stats.poisson.cdf(market_line, predicted_threes)
        
        over_implied = self._odds_to_probability(-110)
        under_implied = self._odds_to_probability(-110)
        
        over_edge = over_prob - over_implied
        under_edge = under_prob - under_implied
        
        best_bet = 'OVER' if over_edge > under_edge and over_edge > self.MIN_PROP_EDGE else \
                  'UNDER' if under_edge > self.MIN_PROP_EDGE else 'AVOID'
        
        if best_bet == 'AVOID':
            return None
        
        return PropBet(
            prop_id=f"{player.player_name}_threes_{game_context.get('game_id', 'unknown')}",
            game_id=game_context.get('game_id', 'unknown'),
            timestamp=datetime.now(),
            prop_type='player_threes',
            player_name=player.player_name,
            team=player.team,
            line=market_line,
            over_odds=-110,
            under_odds=-110,
            over_implied_prob=over_implied,
            under_implied_prob=under_implied,
            predicted_value=predicted_threes,
            prediction_std=threes_std,
            over_probability=over_prob,
            under_probability=under_prob,
            over_edge=over_edge,
            under_edge=under_edge,
            best_bet=best_bet,
            confidence=0.6,  # Lower due to high variance
            correlation_risk=0.0,
            liquidity_score=0.5,  # Lower liquidity for threes
            historical_hit_rate=0.50,
            recent_trend='stable',
            key_factors=[
                f"3P%: {player.three_point_percentage:.1%}",
                f"Attempts/game: {player.three_attempts_per_game:.1f}",
                f"Opponent 3P defense: {opponent_three_defense:.1%}"
            ]
        )
    
    async def _analyze_team_props(self, home_team: str, away_team: str, 
                                game_context: Dict[str, Any]) -> List[PropBet]:
        """Analyze team-level props"""
        team_props = []
        
        # First to score prop
        first_score_prop = self._analyze_first_to_score(home_team, away_team, game_context)
        if first_score_prop:
            team_props.append(first_score_prop)
        
        # Team total props
        home_total_prop = self._analyze_team_total(home_team, game_context, is_home=True)
        if home_total_prop:
            team_props.append(home_total_prop)
        
        away_total_prop = self._analyze_team_total(away_team, game_context, is_home=False)
        if away_total_prop:
            team_props.append(away_total_prop)
        
        return team_props
    
    def _analyze_first_to_score(self, home_team: str, away_team: str, 
                              game_context: Dict[str, Any]) -> Optional[PropBet]:
        """Analyze first to score prop"""
        
        # Home court advantage for first score
        home_first_prob = 0.55  # Historical home advantage
        
        # Adjust for pace and offensive efficiency
        home_efficiency = game_context.get('home_offensive_efficiency', 100)
        away_efficiency = game_context.get('away_offensive_efficiency', 100)
        
        efficiency_ratio = home_efficiency / (home_efficiency + away_efficiency)
        home_first_prob = 0.5 + (efficiency_ratio - 0.5) * 0.3  # Moderate adjustment
        
        # Market odds (would be real market data)
        if home_first_prob > 0.5:
            home_odds = -120
            away_odds = 100
        else:
            home_odds = 100
            away_odds = -120
        
        home_implied = self._odds_to_probability(home_odds)
        away_implied = self._odds_to_probability(away_odds)
        
        home_edge = home_first_prob - home_implied
        away_edge = (1 - home_first_prob) - away_implied
        
        best_bet = 'HOME' if home_edge > away_edge and home_edge > self.MIN_PROP_EDGE else \
                  'AWAY' if away_edge > self.MIN_PROP_EDGE else 'AVOID'
        
        if best_bet == 'AVOID':
            return None
        
        return PropBet(
            prop_id=f"first_to_score_{game_context.get('game_id', 'unknown')}",
            game_id=game_context.get('game_id', 'unknown'),
            timestamp=datetime.now(),
            prop_type='first_to_score',
            player_name=None,
            team=home_team if best_bet == 'HOME' else away_team,
            line=0.5,  # Binary prop
            over_odds=home_odds if best_bet == 'HOME' else away_odds,
            under_odds=away_odds if best_bet == 'HOME' else home_odds,
            over_implied_prob=home_implied if best_bet == 'HOME' else away_implied,
            under_implied_prob=away_implied if best_bet == 'HOME' else home_implied,
            predicted_value=home_first_prob if best_bet == 'HOME' else 1-home_first_prob,
            prediction_std=0.25,  # Fixed for binary props
            over_probability=home_first_prob if best_bet == 'HOME' else 1-home_first_prob,
            under_probability=1-home_first_prob if best_bet == 'HOME' else home_first_prob,
            over_edge=home_edge if best_bet == 'HOME' else away_edge,
            under_edge=away_edge if best_bet == 'HOME' else home_edge,
            best_bet=best_bet,
            confidence=0.7,
            correlation_risk=0.0,
            liquidity_score=0.9,  # Usually liquid prop
            historical_hit_rate=0.55,
            recent_trend='stable',
            key_factors=[
                f"Home court advantage",
                f"Efficiency ratio: {efficiency_ratio:.1%}"
            ]
        )
    
    def _analyze_team_total(self, team: str, game_context: Dict[str, Any], 
                          is_home: bool) -> Optional[PropBet]:
        """Analyze team total points prop"""
        
        # Base team scoring
        base_total = game_context.get(f"{'home' if is_home else 'away'}_expected_points", 75)
        
        # Market line (simplified)
        market_line = base_total + np.random.normal(0, 2.0) - 0.5
        
        # Calculate probabilities
        total_std = np.sqrt(base_total) * 0.15  # Team totals less variable than player props
        
        over_prob = 1 - stats.norm.cdf(market_line, base_total, total_std)
        under_prob = stats.norm.cdf(market_line, base_total, total_std)
        
        over_implied = self._odds_to_probability(-110)
        under_implied = self._odds_to_probability(-110)
        
        over_edge = over_prob - over_implied
        under_edge = under_prob - under_implied
        
        best_bet = 'OVER' if over_edge > under_edge and over_edge > self.MIN_PROP_EDGE else \
                  'UNDER' if under_edge > self.MIN_PROP_EDGE else 'AVOID'
        
        if best_bet == 'AVOID':
            return None
        
        return PropBet(
            prop_id=f"{team}_total_{game_context.get('game_id', 'unknown')}",
            game_id=game_context.get('game_id', 'unknown'),
            timestamp=datetime.now(),
            prop_type='team_total',
            player_name=None,
            team=team,
            line=market_line,
            over_odds=-110,
            under_odds=-110,
            over_implied_prob=over_implied,
            under_implied_prob=under_implied,
            predicted_value=base_total,
            prediction_std=total_std,
            over_probability=over_prob,
            under_probability=under_prob,
            over_edge=over_edge,
            under_edge=under_edge,
            best_bet=best_bet,
            confidence=0.75,
            correlation_risk=0.0,
            liquidity_score=0.8,
            historical_hit_rate=0.52,
            recent_trend='stable',
            key_factors=[
                f"Expected total: {base_total:.1f}",
                f"Home court: {'Yes' if is_home else 'No'}"
            ]
        )
    
    def _odds_to_probability(self, american_odds: float) -> float:
        """Convert American odds to probability"""
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)
    
    def generate_prop_recommendations(self, prop_opportunities: List[PropBet]) -> List[PropRecommendation]:
        """Generate final prop recommendations with sizing"""
        
        recommendations = []
        
        for prop in prop_opportunities:
            # Calculate Kelly sizing
            edge = max(prop.over_edge, prop.under_edge)
            prob = prop.over_probability if prop.over_edge > prop.under_edge else prop.under_probability
            odds = prop.over_odds if prop.over_edge > prop.under_edge else prop.under_odds
            
            # Kelly fraction calculation
            decimal_odds = (abs(odds) / 100) + 1 if odds > 0 else (100 / abs(odds)) + 1
            kelly_fraction = ((decimal_odds - 1) * prob - (1 - prob)) / (decimal_odds - 1)
            kelly_fraction = max(0, min(0.10, kelly_fraction))  # Cap at 10%
            
            # Conservative sizing for props (25% of Kelly)
            recommended_size = kelly_fraction * 0.25
            
            # Expected value
            win_amount = abs(odds) / 100 if odds > 0 else 100 / abs(odds)
            expected_value = (prob * win_amount - (1 - prob)) * recommended_size
            
            # Risk score (higher for props due to variance)
            risk_score = (1 - prop.confidence) * 3 + (1 - prop.liquidity_score) * 2
            
            recommendation = PropRecommendation(
                prop_bet=prop,
                recommended_side=prop.best_bet,
                unit_size=recommended_size,
                max_bet_amount=recommended_size * 1.5,
                expected_value=expected_value,
                kelly_fraction=kelly_fraction,
                risk_score=risk_score,
                urgency='HIGH' if edge > 0.12 else 'MEDIUM',
                line_movement_expectation=edge * 0.5,  # Expect some line movement
                best_time_to_bet='NOW' if edge > 0.12 else 'SOON',
                primary_reasons=[
                    f"{edge:.1%} edge on {prop.best_bet}",
                    f"Model predicts {prop.predicted_value:.1f}",
                    f"Line at {prop.line:.1f}"
                ],
                risk_factors=[
                    "High variance in props",
                    f"Confidence: {prop.confidence:.1%}",
                    f"Liquidity: {prop.liquidity_score:.1%}"
                ],
                similar_props_correlation=0.0  # Would calculate based on other props
            )
            
            recommendations.append(recommendation)
        
        # Sort by expected value
        recommendations.sort(key=lambda x: x.expected_value, reverse=True)
        
        return recommendations
    
    def store_prop_opportunity(self, recommendation: PropRecommendation):
        """Store prop opportunity in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        prop = recommendation.prop_bet
        
        cursor.execute('''
            INSERT INTO prop_opportunities 
            (prop_id, timestamp, prop_type, player_name, line, recommended_side,
             edge, confidence, unit_size, expected_value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            prop.prop_id,
            prop.timestamp,
            prop.prop_type,
            prop.player_name,
            prop.line,
            recommendation.recommended_side,
            max(prop.over_edge, prop.under_edge),
            prop.confidence,
            recommendation.unit_size,
            recommendation.expected_value
        ))
        
        conn.commit()
        conn.close()

# Testing and demonstration
async def demo_prop_betting():
    """Demo the prop betting engine"""
    engine = PropBettingEngine()
    
    print("🎯 Advanced Prop Betting Engine Demo")
    print("=" * 50)
    
    # Load player profiles
    teams = ['Duke', 'UNC']
    print(f"Loading player profiles for {len(teams)} teams...")
    
    await engine.load_player_profiles(teams)
    print(f"📊 Loaded {len(engine.player_profiles)} player profiles")
    
    # Analyze prop opportunities for a game
    game_context = {
        'game_id': 'duke_unc_20241201',
        'home_team': 'Duke',
        'away_team': 'UNC',
        'expected_pace': 72,
        'home_offensive_efficiency': 110,
        'away_offensive_efficiency': 105,
        'opponent_defensive_rating': 98,
        'home_expected_points': 82,
        'away_expected_points': 78
    }
    
    print(f"\n🔍 Analyzing prop opportunities...")
    prop_opportunities = await engine.analyze_prop_opportunities(
        game_context['game_id'], 
        game_context['home_team'], 
        game_context['away_team'], 
        game_context
    )
    
    print(f"Found {len(prop_opportunities)} prop opportunities with value")
    
    # Generate recommendations
    recommendations = engine.generate_prop_recommendations(prop_opportunities)
    
    print(f"\n💰 Top Prop Recommendations:")
    for i, rec in enumerate(recommendations[:5], 1):
        prop = rec.prop_bet
        print(f"{i}. {prop.player_name or prop.team} {prop.prop_type.replace('_', ' ').title()}")
        print(f"   Line: {prop.line:.1f} | Bet: {rec.recommended_side}")
        print(f"   Edge: {max(prop.over_edge, prop.under_edge):.1%} | EV: ${rec.expected_value:.2f}")
        print(f"   Size: {rec.unit_size:.1%} | Confidence: {prop.confidence:.1%}")
        print(f"   Key factors: {', '.join(prop.key_factors[:2])}")
        print()
    
    print("✅ Prop Betting Engine ready for high-margin opportunities!")

def main():
    """Main function for testing"""
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_prop_betting())

if __name__ == "__main__":
    main()