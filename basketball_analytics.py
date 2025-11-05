#!/usr/bin/env python3
"""
Basketball-Specific Analytics Modules
=====================================

Comprehensive analytics for college basketball including:
- Tempo Analysis
- Offensive and Defensive Efficiency Ratings
- Strength of Schedule Analysis
- Conference Power Rankings
- Tournament Performance Metrics
- Player Fatigue Analysis (tournament context)
- Matchup Analysis and Trends

Adapted for College Basketball specifics from baseball analytics
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import sqlite3
from collections import defaultdict
import statistics

@dataclass
class TempoAnalysis:
    """Team tempo analysis and predictions"""
    team: str
    season_tempo: float  # Average possessions per game
    recent_tempo: float  # Last 10 games tempo
    home_tempo: float
    away_tempo: float
    vs_fast_teams: float  # Tempo vs teams with >72 possessions
    vs_slow_teams: float  # Tempo vs teams with <65 possessions
    conference_tempo: float
    tournament_tempo: float
    tempo_consistency: float  # Standard deviation
    tempo_trend: str  # "increasing", "decreasing", "stable"

@dataclass
class EfficiencyMetrics:
    """Team offensive and defensive efficiency metrics"""
    team: str
    
    # Offensive efficiency (points per 100 possessions)
    offensive_efficiency: float
    offensive_efficiency_rank: int
    home_offensive_eff: float
    away_offensive_eff: float
    recent_offensive_eff: float  # Last 10 games
    
    # Defensive efficiency (points allowed per 100 possessions)
    defensive_efficiency: float
    defensive_efficiency_rank: int
    home_defensive_eff: float
    away_defensive_eff: float
    recent_defensive_eff: float
    
    # Net efficiency
    net_efficiency: float
    net_efficiency_rank: int
    
    # Advanced metrics
    effective_fg_pct: float
    turnover_pct: float
    offensive_rebounding_pct: float
    free_throw_rate: float

@dataclass
class StrengthOfSchedule:
    """Strength of schedule analysis"""
    team: str
    overall_sos: float  # 0-100 scale
    conference_sos: float
    non_conference_sos: float
    recent_sos: float  # Last 10 games
    
    # Quality of wins/losses
    quad_1_record: str  # Record vs Quad 1 teams
    quad_2_record: str
    quad_3_record: str
    quad_4_record: str
    
    # Opponent metrics
    avg_opponent_kenpom: float
    avg_opponent_net_rating: float
    road_sos: float
    neutral_sos: float
    
    # Tournament implications
    tournament_resume_strength: float
    sos_rank: int

@dataclass
class ConferencePowerRanking:
    """Conference strength and power ranking"""
    conference: str
    power_rating: float  # Overall conference strength
    depth_rating: float  # How many good teams
    top_end_rating: float  # Quality of best teams
    
    # Tournament metrics
    projected_bids: int
    historical_tournament_success: float
    kenpom_average: float
    
    # Head-to-head records
    vs_power_conferences: Dict[str, str]  # Win-loss records
    non_conference_record: str
    
    ranking_among_conferences: int

@dataclass
class TournamentMetrics:
    """Tournament-specific performance metrics"""
    team: str
    
    # Historical tournament performance
    tournament_appearances: int
    tournament_wins: int
    tournament_win_pct: float
    avg_tournament_seed: float
    best_tournament_finish: str
    
    # Current season tournament profile
    projected_seed: Optional[int]
    seed_range: Tuple[int, int]  # Min, max likely seed
    tournament_resume_score: float
    
    # Tournament-relevant metrics
    late_season_performance: float
    march_record: str  # Record in March games
    close_game_record: str  # Games decided by <5 points
    experience_factor: float  # Based on upperclassmen

@dataclass
class PlayerFatigueAnalysis:
    """Player and team fatigue analysis"""
    team: str
    analysis_date: datetime
    
    # Team fatigue metrics
    avg_minutes_per_game: float
    rotation_depth: int  # Number of players averaging >15 min
    bench_contribution: float  # Percentage of points from bench
    
    # Recent workload
    games_in_last_7_days: int
    games_in_last_14_days: int
    travel_miles_last_week: float
    
    # Tournament context
    tournament_games_played: int
    days_since_last_game: int
    consecutive_tournament_days: int
    
    # Fatigue indicators
    recent_performance_decline: bool
    fourth_quarter_performance: float  # Relative to season average
    overtime_record: str
    
    fatigue_risk_score: float  # 0-1 scale
    recommended_rotation_adjustments: List[str]

class BasketballAnalytics:
    """Comprehensive basketball analytics system"""
    
    def __init__(self, db_path: str = "basketball_analytics.db"):
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        
        # Initialize database
        self._init_database()
        
        # Basketball constants
        self.AVERAGE_COLLEGE_POSSESSIONS = 68.0
        self.AVERAGE_COLLEGE_EFFICIENCY = 100.0
        self.POWER_CONFERENCE_THRESHOLD = 15.0  # KenPom rating threshold
        
        # Conference tiers
        self.POWER_CONFERENCES = {
            'ACC', 'Big 12', 'Big Ten', 'SEC', 'Big East', 'Pac-12'
        }
        
        self.MID_MAJOR_CONFERENCES = {
            'American', 'Mountain West', 'WCC', 'A-10', 'Colonial', 'Horizon',
            'Missouri Valley', 'Sun Belt', 'Conference USA'
        }
        
    def _init_database(self):
        """Initialize basketball analytics database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Tempo analysis table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tempo_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team TEXT NOT NULL,
                    season INTEGER NOT NULL,
                    analysis_date TEXT NOT NULL,
                    season_tempo REAL,
                    recent_tempo REAL,
                    home_tempo REAL,
                    away_tempo REAL,
                    tempo_consistency REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Efficiency metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS efficiency_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team TEXT NOT NULL,
                    season INTEGER NOT NULL,
                    analysis_date TEXT NOT NULL,
                    offensive_efficiency REAL,
                    defensive_efficiency REAL,
                    net_efficiency REAL,
                    efficiency_data TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Strength of schedule table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strength_of_schedule (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team TEXT NOT NULL,
                    season INTEGER NOT NULL,
                    overall_sos REAL,
                    conference_sos REAL,
                    quad_records TEXT,
                    sos_data TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Tournament metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tournament_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team TEXT NOT NULL,
                    season INTEGER NOT NULL,
                    projected_seed INTEGER,
                    tournament_resume_score REAL,
                    historical_data TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Basketball analytics database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing basketball analytics database: {e}")
            
    def analyze_tempo(self, team: str, game_log: List[Dict], season_stats: Dict) -> TempoAnalysis:
        """Analyze team tempo patterns and trends"""
        try:
            self.logger.info(f"Analyzing tempo for {team}")
            
            # Calculate season tempo
            total_possessions = sum(game.get('possessions', 68) for game in game_log)
            season_tempo = total_possessions / len(game_log) if game_log else self.AVERAGE_COLLEGE_POSSESSIONS
            
            # Recent tempo (last 10 games)
            recent_games = game_log[-10:] if len(game_log) >= 10 else game_log
            recent_possessions = [game.get('possessions', 68) for game in recent_games]
            recent_tempo = statistics.mean(recent_possessions)
            
            # Home vs Away tempo
            home_games = [g for g in game_log if g.get('location') == 'home']
            away_games = [g for g in game_log if g.get('location') == 'away']
            
            home_tempo = statistics.mean([g.get('possessions', 68) for g in home_games]) if home_games else season_tempo
            away_tempo = statistics.mean([g.get('possessions', 68) for g in away_games]) if away_games else season_tempo
            
            # Tempo vs different pace teams
            all_possessions = [game.get('possessions', 68) for game in game_log]
            fast_games = [p for p in all_possessions if p > 72]
            slow_games = [p for p in all_possessions if p < 65]
            
            vs_fast_teams = statistics.mean(fast_games) if fast_games else season_tempo
            vs_slow_teams = statistics.mean(slow_games) if slow_games else season_tempo
            
            # Tempo consistency (standard deviation)
            tempo_consistency = statistics.stdev(all_possessions) if len(all_possessions) > 1 else 0.0
            
            # Conference and tournament tempo
            conf_games = [g for g in game_log if g.get('game_type') == 'conference']
            tournament_games = [g for g in game_log if g.get('game_type') == 'tournament']
            
            conference_tempo = statistics.mean([g.get('possessions', 68) for g in conf_games]) if conf_games else season_tempo
            tournament_tempo = statistics.mean([g.get('possessions', 68) for g in tournament_games]) if tournament_games else season_tempo
            
            # Tempo trend analysis
            if len(game_log) >= 6:
                first_half = game_log[:len(game_log)//2]
                second_half = game_log[len(game_log)//2:]
                
                first_half_tempo = statistics.mean([g.get('possessions', 68) for g in first_half])
                second_half_tempo = statistics.mean([g.get('possessions', 68) for g in second_half])
                
                if second_half_tempo - first_half_tempo > 2.0:
                    tempo_trend = "increasing"
                elif first_half_tempo - second_half_tempo > 2.0:
                    tempo_trend = "decreasing"
                else:
                    tempo_trend = "stable"
            else:
                tempo_trend = "insufficient_data"
                
            return TempoAnalysis(
                team=team,
                season_tempo=season_tempo,
                recent_tempo=recent_tempo,
                home_tempo=home_tempo,
                away_tempo=away_tempo,
                vs_fast_teams=vs_fast_teams,
                vs_slow_teams=vs_slow_teams,
                conference_tempo=conference_tempo,
                tournament_tempo=tournament_tempo,
                tempo_consistency=tempo_consistency,
                tempo_trend=tempo_trend
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing tempo for {team}: {e}")
            return self._create_default_tempo_analysis(team)
            
    def calculate_efficiency_metrics(self, team: str, game_log: List[Dict], season_stats: Dict) -> EfficiencyMetrics:
        """Calculate comprehensive efficiency metrics"""
        try:
            self.logger.info(f"Calculating efficiency metrics for {team}")
            
            # Overall efficiency calculations
            total_points = sum(game.get('points_scored', 70) for game in game_log)
            total_possessions = sum(game.get('possessions', 68) for game in game_log)
            total_allowed = sum(game.get('points_allowed', 70) for game in game_log)
            
            offensive_efficiency = (total_points / total_possessions) * 100 if total_possessions > 0 else 100.0
            defensive_efficiency = (total_allowed / total_possessions) * 100 if total_possessions > 0 else 100.0
            net_efficiency = offensive_efficiency - defensive_efficiency
            
            # Home/Away splits
            home_games = [g for g in game_log if g.get('location') == 'home']
            away_games = [g for g in game_log if g.get('location') == 'away']
            
            if home_games:
                home_points = sum(g.get('points_scored', 70) for g in home_games)
                home_poss = sum(g.get('possessions', 68) for g in home_games)
                home_allowed = sum(g.get('points_allowed', 70) for g in home_games)
                home_offensive_eff = (home_points / home_poss) * 100 if home_poss > 0 else offensive_efficiency
                home_defensive_eff = (home_allowed / home_poss) * 100 if home_poss > 0 else defensive_efficiency
            else:
                home_offensive_eff = offensive_efficiency
                home_defensive_eff = defensive_efficiency
                
            if away_games:
                away_points = sum(g.get('points_scored', 70) for g in away_games)
                away_poss = sum(g.get('possessions', 68) for g in away_games)
                away_allowed = sum(g.get('points_allowed', 70) for g in away_games)
                away_offensive_eff = (away_points / away_poss) * 100 if away_poss > 0 else offensive_efficiency
                away_defensive_eff = (away_allowed / away_poss) * 100 if away_poss > 0 else defensive_efficiency
            else:
                away_offensive_eff = offensive_efficiency
                away_defensive_eff = defensive_efficiency
                
            # Recent efficiency (last 10 games)
            recent_games = game_log[-10:] if len(game_log) >= 10 else game_log
            if recent_games:
                recent_points = sum(g.get('points_scored', 70) for g in recent_games)
                recent_poss = sum(g.get('possessions', 68) for g in recent_games)
                recent_allowed = sum(g.get('points_allowed', 70) for g in recent_games)
                recent_offensive_eff = (recent_points / recent_poss) * 100 if recent_poss > 0 else offensive_efficiency
                recent_defensive_eff = (recent_allowed / recent_poss) * 100 if recent_poss > 0 else defensive_efficiency
            else:
                recent_offensive_eff = offensive_efficiency
                recent_defensive_eff = defensive_efficiency
                
            # Advanced metrics (simplified calculations)
            effective_fg_pct = season_stats.get('effective_fg_pct', 0.52)
            turnover_pct = season_stats.get('turnover_pct', 0.18)
            offensive_rebounding_pct = season_stats.get('offensive_rebounding_pct', 0.30)
            free_throw_rate = season_stats.get('free_throw_rate', 0.25)
            
            # Mock rankings (would be calculated against all teams)
            offensive_efficiency_rank = max(1, int(150 - (offensive_efficiency - 100) * 3))
            defensive_efficiency_rank = max(1, int(150 - (100 - defensive_efficiency) * 3))
            net_efficiency_rank = max(1, int(150 - net_efficiency * 2))
            
            return EfficiencyMetrics(
                team=team,
                offensive_efficiency=offensive_efficiency,
                offensive_efficiency_rank=offensive_efficiency_rank,
                home_offensive_eff=home_offensive_eff,
                away_offensive_eff=away_offensive_eff,
                recent_offensive_eff=recent_offensive_eff,
                defensive_efficiency=defensive_efficiency,
                defensive_efficiency_rank=defensive_efficiency_rank,
                home_defensive_eff=home_defensive_eff,
                away_defensive_eff=away_defensive_eff,
                recent_defensive_eff=recent_defensive_eff,
                net_efficiency=net_efficiency,
                net_efficiency_rank=net_efficiency_rank,
                effective_fg_pct=effective_fg_pct,
                turnover_pct=turnover_pct,
                offensive_rebounding_pct=offensive_rebounding_pct,
                free_throw_rate=free_throw_rate
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating efficiency metrics for {team}: {e}")
            return self._create_default_efficiency_metrics(team)
            
    def analyze_strength_of_schedule(self, team: str, game_log: List[Dict], 
                                   opponent_rankings: Dict[str, float]) -> StrengthOfSchedule:
        """Analyze strength of schedule and quality of wins/losses"""
        try:
            self.logger.info(f"Analyzing strength of schedule for {team}")
            
            # Calculate overall SOS
            opponent_ratings = []
            quad_1_games = []
            quad_2_games = []
            quad_3_games = []
            quad_4_games = []
            
            conference_opponents = []
            non_conference_opponents = []
            recent_opponents = []
            
            for game in game_log:
                opponent = game.get('opponent', '')
                opponent_rating = opponent_rankings.get(opponent, 50.0)  # Default to average
                opponent_ratings.append(opponent_rating)
                
                # Categorize by quadrant
                location = game.get('location', 'neutral')
                if self._is_quad_1(opponent_rating, location):
                    quad_1_games.append(game)
                elif self._is_quad_2(opponent_rating, location):
                    quad_2_games.append(game)
                elif self._is_quad_3(opponent_rating, location):
                    quad_3_games.append(game)
                else:
                    quad_4_games.append(game)
                    
                # Conference vs non-conference
                if game.get('game_type') == 'conference':
                    conference_opponents.append(opponent_rating)
                else:
                    non_conference_opponents.append(opponent_rating)
                    
            # Recent SOS (last 10 games)
            recent_games = game_log[-10:] if len(game_log) >= 10 else game_log
            recent_ratings = [opponent_rankings.get(g.get('opponent', ''), 50.0) for g in recent_games]
            
            # Calculate SOS metrics
            overall_sos = statistics.mean(opponent_ratings) if opponent_ratings else 50.0
            conference_sos = statistics.mean(conference_opponents) if conference_opponents else 50.0
            non_conference_sos = statistics.mean(non_conference_opponents) if non_conference_opponents else 50.0
            recent_sos = statistics.mean(recent_ratings) if recent_ratings else 50.0
            
            # Quality records
            quad_1_record = self._calculate_record(quad_1_games)
            quad_2_record = self._calculate_record(quad_2_games)
            quad_3_record = self._calculate_record(quad_3_games)
            quad_4_record = self._calculate_record(quad_4_games)
            
            # Additional metrics
            avg_opponent_kenpom = overall_sos
            avg_opponent_net_rating = overall_sos - 50.0  # Centered at 0
            
            # Location-specific SOS
            road_games = [g for g in game_log if g.get('location') == 'away']
            neutral_games = [g for g in game_log if g.get('location') == 'neutral']
            
            road_sos = statistics.mean([opponent_rankings.get(g.get('opponent', ''), 50.0) for g in road_games]) if road_games else overall_sos
            neutral_sos = statistics.mean([opponent_rankings.get(g.get('opponent', ''), 50.0) for g in neutral_games]) if neutral_games else overall_sos
            
            # Tournament resume strength
            tournament_resume_strength = self._calculate_tournament_resume_strength(
                quad_1_record, quad_2_record, quad_3_record, quad_4_record, overall_sos
            )
            
            # Mock SOS rank
            sos_rank = max(1, min(350, int((60.0 - overall_sos) * 6 + 175)))
            
            return StrengthOfSchedule(
                team=team,
                overall_sos=overall_sos,
                conference_sos=conference_sos,
                non_conference_sos=non_conference_sos,
                recent_sos=recent_sos,
                quad_1_record=quad_1_record,
                quad_2_record=quad_2_record,
                quad_3_record=quad_3_record,
                quad_4_record=quad_4_record,
                avg_opponent_kenpom=avg_opponent_kenpom,
                avg_opponent_net_rating=avg_opponent_net_rating,
                road_sos=road_sos,
                neutral_sos=neutral_sos,
                tournament_resume_strength=tournament_resume_strength,
                sos_rank=sos_rank
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing strength of schedule for {team}: {e}")
            return self._create_default_sos(team)
            
    def analyze_tournament_profile(self, team: str, season_stats: Dict, 
                                 historical_data: Dict) -> TournamentMetrics:
        """Analyze tournament profile and March Madness readiness"""
        try:
            self.logger.info(f"Analyzing tournament profile for {team}")
            
            # Historical tournament data
            tournament_appearances = historical_data.get('tournament_appearances', 0)
            tournament_wins = historical_data.get('tournament_wins', 0)
            tournament_win_pct = tournament_wins / max(tournament_appearances, 1)
            
            avg_tournament_seed = historical_data.get('avg_seed', 8.0)
            best_tournament_finish = historical_data.get('best_finish', 'First Round')
            
            # Current season tournament profile
            net_rating = season_stats.get('net_rating', 0.0)
            kenpom_rating = season_stats.get('kenpom_rating', 50.0)
            
            # Project seed based on metrics
            if kenpom_rating >= 95:
                projected_seed = 1
                seed_range = (1, 2)
            elif kenpom_rating >= 85:
                projected_seed = 2
                seed_range = (1, 3)
            elif kenpom_rating >= 75:
                projected_seed = 4
                seed_range = (3, 6)
            elif kenpom_rating >= 65:
                projected_seed = 7
                seed_range = (6, 9)
            elif kenpom_rating >= 55:
                projected_seed = 10
                seed_range = (9, 12)
            else:
                projected_seed = None  # Not projected to make tournament
                seed_range = (12, 16)
                
            # Tournament resume score
            tournament_resume_score = self._calculate_tournament_resume_score(season_stats)
            
            # Tournament-relevant metrics
            march_games = season_stats.get('march_games', [])
            march_record = self._calculate_record(march_games) if march_games else "0-0"
            
            close_games = season_stats.get('close_games', [])
            close_game_record = self._calculate_record(close_games) if close_games else "0-0"
            
            late_season_games = season_stats.get('late_season_games', [])
            late_season_performance = self._calculate_late_season_performance(late_season_games)
            
            # Experience factor
            upperclassmen_pct = season_stats.get('upperclassmen_percentage', 0.5)
            tournament_experience = season_stats.get('players_with_tournament_exp', 0)
            experience_factor = (upperclassmen_pct * 0.7) + (min(tournament_experience, 5) * 0.06)
            
            return TournamentMetrics(
                team=team,
                tournament_appearances=tournament_appearances,
                tournament_wins=tournament_wins,
                tournament_win_pct=tournament_win_pct,
                avg_tournament_seed=avg_tournament_seed,
                best_tournament_finish=best_tournament_finish,
                projected_seed=projected_seed,
                seed_range=seed_range,
                tournament_resume_score=tournament_resume_score,
                late_season_performance=late_season_performance,
                march_record=march_record,
                close_game_record=close_game_record,
                experience_factor=experience_factor
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing tournament profile for {team}: {e}")
            return self._create_default_tournament_metrics(team)
            
    def analyze_player_fatigue(self, team: str, roster_data: List[Dict], 
                             schedule_data: Dict) -> PlayerFatigueAnalysis:
        """Analyze player and team fatigue factors"""
        try:
            self.logger.info(f"Analyzing player fatigue for {team}")
            
            # Calculate team fatigue metrics
            minutes_data = [player.get('minutes_per_game', 25.0) for player in roster_data]
            avg_minutes_per_game = statistics.mean(minutes_data)
            
            rotation_depth = len([p for p in roster_data if p.get('minutes_per_game', 0) >= 15])
            
            # Bench contribution
            starter_points = sum(p.get('points_per_game', 0) for p in roster_data[:5])  # Top 5 scorers
            total_points = sum(p.get('points_per_game', 0) for p in roster_data)
            bench_contribution = (total_points - starter_points) / total_points if total_points > 0 else 0.2
            
            # Recent workload
            games_last_7 = schedule_data.get('games_last_7_days', 0)
            games_last_14 = schedule_data.get('games_last_14_days', 0)
            travel_miles = schedule_data.get('travel_miles_last_week', 500.0)
            
            # Tournament context
            tournament_games = schedule_data.get('tournament_games_played', 0)
            days_since_last = schedule_data.get('days_since_last_game', 2)
            consecutive_tournament_days = schedule_data.get('consecutive_tournament_days', 0)
            
            # Fatigue indicators
            recent_performance = schedule_data.get('recent_performance_vs_season', 0.0)
            recent_performance_decline = recent_performance < -0.05  # 5% decline
            
            fourth_quarter_perf = schedule_data.get('fourth_quarter_relative_performance', 1.0)
            overtime_record = schedule_data.get('overtime_record', "0-0")
            
            # Calculate fatigue risk score
            fatigue_risk_score = self._calculate_fatigue_risk_score(
                games_last_7, rotation_depth, avg_minutes_per_game, 
                consecutive_tournament_days, recent_performance_decline
            )
            
            # Rotation recommendations
            recommendations = self._generate_rotation_recommendations(
                fatigue_risk_score, rotation_depth, bench_contribution
            )
            
            return PlayerFatigueAnalysis(
                team=team,
                analysis_date=datetime.now(),
                avg_minutes_per_game=avg_minutes_per_game,
                rotation_depth=rotation_depth,
                bench_contribution=bench_contribution,
                games_in_last_7_days=games_last_7,
                games_in_last_14_days=games_last_14,
                travel_miles_last_week=travel_miles,
                tournament_games_played=tournament_games,
                days_since_last_game=days_since_last,
                consecutive_tournament_days=consecutive_tournament_days,
                recent_performance_decline=recent_performance_decline,
                fourth_quarter_performance=fourth_quarter_perf,
                overtime_record=overtime_record,
                fatigue_risk_score=fatigue_risk_score,
                recommended_rotation_adjustments=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing player fatigue for {team}: {e}")
            return self._create_default_fatigue_analysis(team)
            
    # Helper methods
    
    def _is_quad_1(self, opponent_rating: float, location: str) -> bool:
        """Determine if game is Quad 1 based on opponent rating and location"""
        if location == 'home':
            return opponent_rating >= 75  # Top 30 equivalent
        elif location == 'neutral':
            return opponent_rating >= 70  # Top 50 equivalent
        else:  # away
            return opponent_rating >= 60  # Top 75 equivalent
            
    def _is_quad_2(self, opponent_rating: float, location: str) -> bool:
        """Determine if game is Quad 2"""
        if location == 'home':
            return 60 <= opponent_rating < 75  # 31-75
        elif location == 'neutral':
            return 55 <= opponent_rating < 70  # 51-100
        else:  # away
            return 50 <= opponent_rating < 60  # 76-135
            
    def _is_quad_3(self, opponent_rating: float, location: str) -> bool:
        """Determine if game is Quad 3"""
        if location == 'home':
            return 45 <= opponent_rating < 60  # 76-160
        elif location == 'neutral':
            return 40 <= opponent_rating < 55  # 101-200
        else:  # away
            return 35 <= opponent_rating < 50  # 136-240
            
    def _calculate_record(self, games: List[Dict]) -> str:
        """Calculate win-loss record from game list"""
        wins = sum(1 for game in games if game.get('result') == 'W')
        losses = len(games) - wins
        return f"{wins}-{losses}"
        
    def _calculate_tournament_resume_strength(self, quad_1: str, quad_2: str, 
                                            quad_3: str, quad_4: str, sos: float) -> float:
        """Calculate tournament resume strength score"""
        try:
            q1_wins = int(quad_1.split('-')[0])
            q2_wins = int(quad_2.split('-')[0])
            
            # Resume score based on quality wins and SOS
            resume_score = (q1_wins * 3.0) + (q2_wins * 1.5) + ((sos - 50.0) / 10.0)
            return max(0.0, min(10.0, resume_score))
            
        except:
            return 5.0
            
    def _calculate_tournament_resume_score(self, season_stats: Dict) -> float:
        """Calculate overall tournament resume score"""
        try:
            net_rating = season_stats.get('net_rating', 0.0)
            kenpom = season_stats.get('kenpom_rating', 50.0)
            sos = season_stats.get('strength_of_schedule', 50.0)
            
            # Weighted combination
            resume_score = (kenpom - 50.0) * 0.4 + net_rating * 0.3 + (sos - 50.0) * 0.3
            return max(0.0, min(100.0, resume_score + 50.0))
            
        except:
            return 50.0
            
    def _calculate_late_season_performance(self, late_games: List[Dict]) -> float:
        """Calculate late season performance metric"""
        if not late_games:
            return 1.0
            
        wins = sum(1 for game in late_games if game.get('result') == 'W')
        return wins / len(late_games)
        
    def _calculate_fatigue_risk_score(self, games_7_days: int, rotation_depth: int,
                                    avg_minutes: float, consecutive_days: int, 
                                    performance_decline: bool) -> float:
        """Calculate fatigue risk score"""
        risk_score = 0.0
        
        # Games in last 7 days
        if games_7_days >= 4:
            risk_score += 0.3
        elif games_7_days >= 3:
            risk_score += 0.2
            
        # Rotation depth
        if rotation_depth <= 6:
            risk_score += 0.2
        elif rotation_depth <= 7:
            risk_score += 0.1
            
        # Average minutes
        if avg_minutes >= 35:
            risk_score += 0.2
        elif avg_minutes >= 32:
            risk_score += 0.1
            
        # Consecutive tournament days
        risk_score += min(0.3, consecutive_days * 0.1)
        
        # Performance decline
        if performance_decline:
            risk_score += 0.1
            
        return min(1.0, risk_score)
        
    def _generate_rotation_recommendations(self, fatigue_risk: float, 
                                         rotation_depth: int, bench_contrib: float) -> List[str]:
        """Generate rotation adjustment recommendations"""
        recommendations = []
        
        if fatigue_risk > 0.6:
            recommendations.append("Consider expanding rotation to 8-9 players")
            recommendations.append("Reduce starter minutes by 2-3 minutes per game")
            
        if rotation_depth <= 6:
            recommendations.append("Develop deeper bench options")
            
        if bench_contrib < 0.25:
            recommendations.append("Increase bench scoring contributions")
            
        if fatigue_risk > 0.8:
            recommendations.append("Consider rest for key players if possible")
            
        return recommendations
        
    # Default creation methods
    
    def _create_default_tempo_analysis(self, team: str) -> TempoAnalysis:
        """Create default tempo analysis"""
        return TempoAnalysis(
            team=team,
            season_tempo=self.AVERAGE_COLLEGE_POSSESSIONS,
            recent_tempo=self.AVERAGE_COLLEGE_POSSESSIONS,
            home_tempo=self.AVERAGE_COLLEGE_POSSESSIONS,
            away_tempo=self.AVERAGE_COLLEGE_POSSESSIONS,
            vs_fast_teams=self.AVERAGE_COLLEGE_POSSESSIONS,
            vs_slow_teams=self.AVERAGE_COLLEGE_POSSESSIONS,
            conference_tempo=self.AVERAGE_COLLEGE_POSSESSIONS,
            tournament_tempo=self.AVERAGE_COLLEGE_POSSESSIONS,
            tempo_consistency=5.0,
            tempo_trend="stable"
        )
        
    def _create_default_efficiency_metrics(self, team: str) -> EfficiencyMetrics:
        """Create default efficiency metrics"""
        return EfficiencyMetrics(
            team=team,
            offensive_efficiency=100.0,
            offensive_efficiency_rank=175,
            home_offensive_eff=100.0,
            away_offensive_eff=100.0,
            recent_offensive_eff=100.0,
            defensive_efficiency=100.0,
            defensive_efficiency_rank=175,
            home_defensive_eff=100.0,
            away_defensive_eff=100.0,
            recent_defensive_eff=100.0,
            net_efficiency=0.0,
            net_efficiency_rank=175,
            effective_fg_pct=0.50,
            turnover_pct=0.18,
            offensive_rebounding_pct=0.30,
            free_throw_rate=0.25
        )
        
    def _create_default_sos(self, team: str) -> StrengthOfSchedule:
        """Create default strength of schedule"""
        return StrengthOfSchedule(
            team=team,
            overall_sos=50.0,
            conference_sos=50.0,
            non_conference_sos=50.0,
            recent_sos=50.0,
            quad_1_record="0-0",
            quad_2_record="0-0",
            quad_3_record="0-0",
            quad_4_record="0-0",
            avg_opponent_kenpom=50.0,
            avg_opponent_net_rating=0.0,
            road_sos=50.0,
            neutral_sos=50.0,
            tournament_resume_strength=5.0,
            sos_rank=175
        )
        
    def _create_default_tournament_metrics(self, team: str) -> TournamentMetrics:
        """Create default tournament metrics"""
        return TournamentMetrics(
            team=team,
            tournament_appearances=0,
            tournament_wins=0,
            tournament_win_pct=0.0,
            avg_tournament_seed=10.0,
            best_tournament_finish="Never made tournament",
            projected_seed=None,
            seed_range=(12, 16),
            tournament_resume_score=40.0,
            late_season_performance=0.5,
            march_record="0-0",
            close_game_record="0-0",
            experience_factor=0.5
        )
        
    def _create_default_fatigue_analysis(self, team: str) -> PlayerFatigueAnalysis:
        """Create default fatigue analysis"""
        return PlayerFatigueAnalysis(
            team=team,
            analysis_date=datetime.now(),
            avg_minutes_per_game=28.0,
            rotation_depth=8,
            bench_contribution=0.30,
            games_in_last_7_days=2,
            games_in_last_14_days=4,
            travel_miles_last_week=500.0,
            tournament_games_played=0,
            days_since_last_game=3,
            consecutive_tournament_days=0,
            recent_performance_decline=False,
            fourth_quarter_performance=1.0,
            overtime_record="0-0",
            fatigue_risk_score=0.3,
            recommended_rotation_adjustments=[]
        )

def main():
    """Test the basketball analytics system"""
    logging.basicConfig(level=logging.INFO)
    
    analytics = BasketballAnalytics()
    
    print("Basketball Analytics System Test")
    print("=" * 35)
    
    # Mock game log
    game_log = [
        {'possessions': 70, 'points_scored': 75, 'points_allowed': 68, 'location': 'home', 
         'opponent': 'Team A', 'result': 'W', 'game_type': 'conference'},
        {'possessions': 68, 'points_scored': 72, 'points_allowed': 74, 'location': 'away', 
         'opponent': 'Team B', 'result': 'L', 'game_type': 'conference'},
        {'possessions': 72, 'points_scored': 80, 'points_allowed': 76, 'location': 'neutral', 
         'opponent': 'Team C', 'result': 'W', 'game_type': 'tournament'},
    ]
    
    # Mock opponent rankings
    opponent_rankings = {'Team A': 65.0, 'Team B': 75.0, 'Team C': 80.0}
    
    # Mock season stats
    season_stats = {
        'net_rating': 8.5,
        'kenpom_rating': 72.0,
        'effective_fg_pct': 0.54,
        'turnover_pct': 0.16,
        'offensive_rebounding_pct': 0.32,
        'free_throw_rate': 0.28
    }
    
    # Test tempo analysis
    tempo_analysis = analytics.analyze_tempo('Duke', game_log, season_stats)
    print(f"Tempo Analysis for {tempo_analysis.team}:")
    print(f"  Season Tempo: {tempo_analysis.season_tempo:.1f} possessions")
    print(f"  Home Tempo: {tempo_analysis.home_tempo:.1f}")
    print(f"  Away Tempo: {tempo_analysis.away_tempo:.1f}")
    print(f"  Trend: {tempo_analysis.tempo_trend}")
    print()
    
    # Test efficiency metrics
    efficiency = analytics.calculate_efficiency_metrics('Duke', game_log, season_stats)
    print(f"Efficiency Metrics for {efficiency.team}:")
    print(f"  Offensive Efficiency: {efficiency.offensive_efficiency:.1f}")
    print(f"  Defensive Efficiency: {efficiency.defensive_efficiency:.1f}")
    print(f"  Net Efficiency: {efficiency.net_efficiency:.1f}")
    print(f"  Net Efficiency Rank: #{efficiency.net_efficiency_rank}")
    print()
    
    # Test strength of schedule
    sos = analytics.analyze_strength_of_schedule('Duke', game_log, opponent_rankings)
    print(f"Strength of Schedule for {sos.team}:")
    print(f"  Overall SOS: {sos.overall_sos:.1f}")
    print(f"  Quad 1 Record: {sos.quad_1_record}")
    print(f"  Quad 2 Record: {sos.quad_2_record}")
    print(f"  Tournament Resume Strength: {sos.tournament_resume_strength:.1f}/10")
    print()
    
    print("✅ Basketball Analytics System operational!")

if __name__ == "__main__":
    main()