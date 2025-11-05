#!/usr/bin/env python3
"""
Basketball Injury Impact System
===============================

College basketball specific injury tracking and impact analysis including:
- Basketball-specific injury types and recovery times
- Roster depth analysis (critical in college basketball)
- Starting five vs bench impact assessment
- Position-specific replacement quality
- Tournament context injury implications
- Transfer portal and eligibility impacts

Adapted from baseball injury system for basketball specifics
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import sqlite3
from collections import defaultdict

@dataclass
class BasketballInjuryReport:
    """Basketball-specific injury report"""
    player_id: str
    player_name: str
    position: str  # PG, SG, SF, PF, C
    team: str
    
    # Basketball injury details
    injury_type: str  # ankle, knee, back, wrist, concussion, etc.
    injury_severity: str  # "Minor", "Moderate", "Major", "Season-ending"
    injury_date: datetime
    
    # Basketball status
    current_status: str  # "Active", "Day-to-day", "Out", "Doubtful", "Questionable"
    expected_return: Optional[datetime]
    games_missed: int
    
    # Basketball performance impact
    minutes_per_game: float
    points_per_game: float
    assists_per_game: float
    rebounds_per_game: float
    shooting_percentage: float
    player_efficiency_rating: float  # PER
    
    # Replacement analysis
    replacement_player_id: Optional[str]
    replacement_quality: float  # 0-1 scale vs injured player
    depth_chart_position: int  # 1=starter, 2=6th man, etc.

@dataclass
class BasketballTeamInjuryStatus:
    """Team-wide injury status for basketball"""
    team: str
    analysis_date: datetime
    
    # Current injuries
    active_injuries: List[BasketballInjuryReport]
    starting_five_injuries: int
    key_bench_injuries: int
    total_minutes_lost: float  # Minutes per game lost to injury
    
    # Basketball-specific depth analysis
    point_guard_depth: float  # Quality 0-1
    shooting_guard_depth: float
    small_forward_depth: float
    power_forward_depth: float
    center_depth: float
    
    # Impact assessments
    offensive_impact: float  # Points per game lost
    rebounding_impact: float  # Rebounds per game lost
    assist_impact: float  # Assists per game lost
    defensive_impact: float  # Defensive rating impact
    overall_team_impact: float  # Overall win probability impact
    
    # Basketball rotation analysis
    rotation_flexibility: float  # Ability to adjust rotations
    freshman_reliance: float  # How much team relies on freshmen
    bench_scoring_impact: float  # Bench points impact

@dataclass
class BasketballInjuryRiskAssessment:
    """Basketball injury risk prediction"""
    player_id: str
    player_name: str
    position: str
    
    # Basketball-specific risk factors
    minutes_load_risk: float  # High minutes = higher risk
    contact_risk: float  # Based on position and play style
    injury_history_risk: float
    body_type_risk: float  # Height/weight considerations
    playing_surface_risk: float  # Home vs away court factors
    
    # College basketball factors
    age_risk: float  # Freshmen vs seniors
    conditioning_risk: float  # Based on season progression
    tournament_fatigue_risk: float  # March Madness intensity
    
    overall_risk_score: float
    risk_category: str
    likely_injury_types: List[str]
    expected_games_at_risk: int

class BasketballInjuryImpactSystem:
    """Basketball-specific injury impact analysis"""
    
    def __init__(self, db_path: str = "basketball_injury_tracking.db"):
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        
        # Initialize database
        self._init_database()
        
        # Basketball position importance weights
        self.POSITION_IMPORTANCE = {
            'PG': 1.0,   # Point Guard - most important
            'SG': 0.8,   # Shooting Guard
            'SF': 0.8,   # Small Forward
            'PF': 0.7,   # Power Forward  
            'C': 0.9     # Center - very important
        }
        
        # Basketball injury severity multipliers
        self.SEVERITY_MULTIPLIERS = {
            'Minor': 0.1,      # Day-to-day, minimal impact
            'Moderate': 0.4,   # 1-2 weeks out
            'Major': 0.8,      # Month+ recovery
            'Season-ending': 1.0
        }
        
        # Basketball-specific injury types and typical recovery
        self.BASKETBALL_INJURIES = {
            'ankle_sprain': {'severity': 'Moderate', 'typical_games': 3, 'recurrence_risk': 0.3},
            'knee_injury': {'severity': 'Major', 'typical_games': 8, 'recurrence_risk': 0.4},
            'back_injury': {'severity': 'Moderate', 'typical_games': 4, 'recurrence_risk': 0.5},
            'wrist_injury': {'severity': 'Minor', 'typical_games': 2, 'recurrence_risk': 0.2},
            'concussion': {'severity': 'Moderate', 'typical_games': 2, 'recurrence_risk': 0.6},
            'hamstring': {'severity': 'Minor', 'typical_games': 3, 'recurrence_risk': 0.4},
            'shoulder': {'severity': 'Moderate', 'typical_games': 5, 'recurrence_risk': 0.3}
        }
        
        # College basketball specific factors
        self.TOURNAMENT_IMPACT_MULTIPLIER = 1.5  # Injuries hurt more in tournaments
        self.FRESHMAN_REPLACEMENT_PENALTY = 0.3  # Freshmen replacements are riskier
        
    def _init_database(self):
        """Initialize basketball injury tracking database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Basketball injury history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS basketball_injury_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id TEXT NOT NULL,
                    player_name TEXT NOT NULL,
                    team TEXT NOT NULL,
                    position TEXT NOT NULL,
                    injury_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    injury_date TEXT NOT NULL,
                    return_date TEXT,
                    games_missed INTEGER DEFAULT 0,
                    minutes_per_game REAL DEFAULT 0,
                    replacement_quality REAL DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Team injury summaries
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS basketball_team_injury_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    team TEXT NOT NULL,
                    analysis_date TEXT NOT NULL,
                    starting_five_injuries INTEGER DEFAULT 0,
                    total_minutes_lost REAL DEFAULT 0,
                    overall_impact REAL DEFAULT 0,
                    injury_data TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Risk assessments
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS basketball_injury_risk (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id TEXT NOT NULL,
                    player_name TEXT NOT NULL,
                    team TEXT NOT NULL,
                    assessment_date TEXT NOT NULL,
                    overall_risk_score REAL NOT NULL,
                    risk_category TEXT NOT NULL,
                    risk_data TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Basketball injury database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing basketball injury database: {e}")
            
    def analyze_team_injury_status(self, team: str, current_injuries: List[Dict],
                                 roster_data: List[Dict], tournament_context: str = "regular_season") -> BasketballTeamInjuryStatus:
        """Analyze basketball team injury impact"""
        try:
            self.logger.info(f"Analyzing basketball injury status for {team}")
            
            # Process current injuries
            injury_reports = []
            starting_five_injuries = 0
            key_bench_injuries = 0
            total_minutes_lost = 0.0
            
            for injury_data in current_injuries:
                injury_report = self._create_basketball_injury_report(injury_data)
                injury_reports.append(injury_report)
                
                # Check if starter (top 5 in minutes)
                if injury_report.depth_chart_position <= 5:
                    starting_five_injuries += 1
                elif injury_report.depth_chart_position <= 8:  # Key bench players
                    key_bench_injuries += 1
                    
                # Add to total minutes lost
                total_minutes_lost += injury_report.minutes_per_game
                
            # Analyze position depth
            position_depth = self._analyze_basketball_position_depth(roster_data, injury_reports)
            
            # Calculate basketball-specific impacts
            offensive_impact = self._calculate_basketball_offensive_impact(injury_reports)
            rebounding_impact = self._calculate_rebounding_impact(injury_reports)
            assist_impact = self._calculate_assist_impact(injury_reports)
            defensive_impact = self._calculate_basketball_defensive_impact(injury_reports)
            
            # Overall team impact
            base_impact = (offensive_impact + defensive_impact) / 2
            
            # Tournament context multiplier
            if tournament_context in ['march_madness', 'conference_tournament']:
                tournament_multiplier = self.TOURNAMENT_IMPACT_MULTIPLIER
            else:
                tournament_multiplier = 1.0
                
            overall_impact = base_impact * tournament_multiplier
            
            # Rotation analysis
            rotation_flexibility = self._calculate_rotation_flexibility(roster_data, injury_reports)
            freshman_reliance = self._calculate_freshman_reliance(roster_data, injury_reports)
            bench_scoring_impact = self._calculate_bench_scoring_impact(injury_reports)
            
            return BasketballTeamInjuryStatus(
                team=team,
                analysis_date=datetime.now(),
                active_injuries=injury_reports,
                starting_five_injuries=starting_five_injuries,
                key_bench_injuries=key_bench_injuries,
                total_minutes_lost=total_minutes_lost,
                point_guard_depth=position_depth['PG'],
                shooting_guard_depth=position_depth['SG'],
                small_forward_depth=position_depth['SF'],
                power_forward_depth=position_depth['PF'],
                center_depth=position_depth['C'],
                offensive_impact=offensive_impact,
                rebounding_impact=rebounding_impact,
                assist_impact=assist_impact,
                defensive_impact=defensive_impact,
                overall_team_impact=overall_impact,
                rotation_flexibility=rotation_flexibility,
                freshman_reliance=freshman_reliance,
                bench_scoring_impact=bench_scoring_impact
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing basketball injury status: {e}")
            return self._create_default_basketball_team_status(team)
            
    def _create_basketball_injury_report(self, injury_data: Dict) -> BasketballInjuryReport:
        """Create basketball injury report from raw data"""
        try:
            injury_date_str = injury_data.get('injury_date', datetime.now().strftime('%Y-%m-%d'))
            injury_date = datetime.strptime(injury_date_str, '%Y-%m-%d')
            
            expected_return = None
            if injury_data.get('expected_return'):
                expected_return = datetime.strptime(injury_data['expected_return'], '%Y-%m-%d')
                
            # Calculate replacement quality
            player_per = injury_data.get('player_efficiency_rating', 15.0)
            replacement_per = injury_data.get('replacement_per', 10.0)
            replacement_quality = replacement_per / max(10.0, player_per)
            
            return BasketballInjuryReport(
                player_id=injury_data.get('player_id', 'unknown'),
                player_name=injury_data.get('player_name', 'Unknown Player'),
                position=injury_data.get('position', 'SF'),
                team=injury_data.get('team', 'UNK'),
                injury_type=injury_data.get('injury_type', 'Unknown'),
                injury_severity=injury_data.get('severity', 'Moderate'),
                injury_date=injury_date,
                current_status=injury_data.get('status', 'Day-to-day'),
                expected_return=expected_return,
                games_missed=injury_data.get('games_missed', 0),
                minutes_per_game=injury_data.get('minutes_per_game', 20.0),
                points_per_game=injury_data.get('points_per_game', 8.0),
                assists_per_game=injury_data.get('assists_per_game', 2.0),
                rebounds_per_game=injury_data.get('rebounds_per_game', 4.0),
                shooting_percentage=injury_data.get('shooting_percentage', 0.45),
                player_efficiency_rating=player_per,
                replacement_player_id=injury_data.get('replacement_player_id'),
                replacement_quality=replacement_quality,
                depth_chart_position=injury_data.get('depth_chart_position', 6)
            )
            
        except Exception as e:
            self.logger.error(f"Error creating basketball injury report: {e}")
            return self._create_default_basketball_injury_report()
            
    def _analyze_basketball_position_depth(self, roster_data: List[Dict], 
                                         injuries: List[BasketballInjuryReport]) -> Dict[str, float]:
        """Analyze depth at each basketball position"""
        try:
            position_depth = {}
            
            # Get injured positions
            injured_positions = {injury.position for injury in injuries}
            
            positions = ['PG', 'SG', 'SF', 'PF', 'C']
            
            for position in positions:
                # Count healthy players at this position
                healthy_players = [p for p in roster_data 
                                 if p.get('position') == position 
                                 and p.get('status') == 'Healthy'
                                 and p.get('minutes_per_game', 0) > 5]  # Meaningful contributors
                
                # Count injured players at this position
                injured_at_position = [inj for inj in injuries if inj.position == position]
                
                total_players = len(healthy_players) + len(injured_at_position)
                
                if total_players == 0:
                    position_depth[position] = 0.1  # No depth
                elif len(healthy_players) >= 3:
                    position_depth[position] = 0.9  # Great depth
                elif len(healthy_players) == 2:
                    position_depth[position] = 0.6  # Good depth
                elif len(healthy_players) == 1:
                    position_depth[position] = 0.3  # Poor depth
                else:
                    position_depth[position] = 0.1  # No healthy players
                    
                # Adjust for quality of injured players
                if position in injured_positions:
                    injured_impact = sum(inj.minutes_per_game for inj in injured_at_position) / 40.0
                    position_depth[position] = max(0.1, position_depth[position] - injured_impact)
                    
            return position_depth
            
        except Exception as e:
            self.logger.error(f"Error analyzing basketball position depth: {e}")
            return {pos: 0.5 for pos in ['PG', 'SG', 'SF', 'PF', 'C']}
            
    def _calculate_basketball_offensive_impact(self, injuries: List[BasketballInjuryReport]) -> float:
        """Calculate basketball offensive impact from injuries"""
        try:
            total_points_lost = 0.0
            total_assists_lost = 0.0
            
            for injury in injuries:
                severity_multiplier = self.SEVERITY_MULTIPLIERS.get(injury.injury_severity, 0.5)
                position_weight = self.POSITION_IMPORTANCE.get(injury.position, 0.5)
                
                # Points impact
                points_impact = (injury.points_per_game * severity_multiplier * position_weight)
                total_points_lost += points_impact
                
                # Assists impact (especially important for PGs)
                assists_weight = 2.0 if injury.position == 'PG' else 1.0
                assists_impact = (injury.assists_per_game * severity_multiplier * assists_weight)
                total_assists_lost += assists_impact
                
            # Convert to overall impact (-1 to 0 scale)
            offensive_impact = -(total_points_lost / 80.0 + total_assists_lost / 20.0)  # Normalize
            return max(-1.0, offensive_impact)
            
        except Exception as e:
            self.logger.error(f"Error calculating basketball offensive impact: {e}")
            return 0.0
            
    def _calculate_rebounding_impact(self, injuries: List[BasketballInjuryReport]) -> float:
        """Calculate rebounding impact"""
        try:
            total_rebounds_lost = 0.0
            
            for injury in injuries:
                severity_multiplier = self.SEVERITY_MULTIPLIERS.get(injury.injury_severity, 0.5)
                
                # Big men have higher rebounding weight
                rebounding_weight = 1.5 if injury.position in ['PF', 'C'] else 1.0
                
                rebounds_impact = (injury.rebounds_per_game * severity_multiplier * rebounding_weight)
                total_rebounds_lost += rebounds_impact
                
            return -min(1.0, total_rebounds_lost / 40.0)  # Normalize to team rebounding
            
        except Exception as e:
            self.logger.error(f"Error calculating rebounding impact: {e}")
            return 0.0
            
    def _calculate_assist_impact(self, injuries: List[BasketballInjuryReport]) -> float:
        """Calculate assist/playmaking impact"""
        try:
            total_assists_lost = 0.0
            
            for injury in injuries:
                severity_multiplier = self.SEVERITY_MULTIPLIERS.get(injury.injury_severity, 0.5)
                
                # Guards have higher assist weight
                assist_weight = 2.0 if injury.position in ['PG', 'SG'] else 1.0
                
                assists_impact = (injury.assists_per_game * severity_multiplier * assist_weight)
                total_assists_lost += assists_impact
                
            return -min(1.0, total_assists_lost / 20.0)  # Normalize to team assists
            
        except Exception as e:
            self.logger.error(f"Error calculating assist impact: {e}")
            return 0.0
            
    def _calculate_basketball_defensive_impact(self, injuries: List[BasketballInjuryReport]) -> float:
        """Calculate basketball defensive impact"""
        try:
            total_defensive_impact = 0.0
            
            for injury in injuries:
                severity_multiplier = self.SEVERITY_MULTIPLIERS.get(injury.injury_severity, 0.5)
                position_weight = self.POSITION_IMPORTANCE.get(injury.position, 0.5)
                
                # Minutes-based defensive impact
                minutes_factor = injury.minutes_per_game / 40.0
                defensive_impact = minutes_factor * severity_multiplier * position_weight
                total_defensive_impact += defensive_impact
                
            return -min(1.0, total_defensive_impact)
            
        except Exception as e:
            self.logger.error(f"Error calculating basketball defensive impact: {e}")
            return 0.0
            
    def _calculate_rotation_flexibility(self, roster_data: List[Dict], 
                                      injuries: List[BasketballInjuryReport]) -> float:
        """Calculate team's rotation flexibility"""
        try:
            # Count versatile players (can play multiple positions)
            versatile_players = [p for p in roster_data 
                               if len(p.get('positions', [])) >= 2 
                               and p.get('status') == 'Healthy']
            
            # Count total healthy rotation players
            rotation_players = [p for p in roster_data 
                              if p.get('minutes_per_game', 0) >= 10 
                              and p.get('status') == 'Healthy']
            
            base_flexibility = len(rotation_players) / 10.0  # Ideal 10-man rotation
            versatility_bonus = len(versatile_players) / 8.0  # Bonus for versatility
            
            flexibility = min(1.0, base_flexibility + versatility_bonus)
            
            # Penalty for key injuries
            key_injury_penalty = len([inj for inj in injuries 
                                    if inj.depth_chart_position <= 5]) * 0.2
            
            return max(0.1, flexibility - key_injury_penalty)
            
        except Exception as e:
            self.logger.error(f"Error calculating rotation flexibility: {e}")
            return 0.5
            
    def _calculate_freshman_reliance(self, roster_data: List[Dict], 
                                   injuries: List[BasketballInjuryReport]) -> float:
        """Calculate reliance on freshmen due to injuries"""
        try:
            # Count freshmen in current rotation
            freshmen_in_rotation = [p for p in roster_data 
                                  if p.get('class') == 'FR' 
                                  and p.get('minutes_per_game', 0) >= 15]
            
            total_rotation = [p for p in roster_data 
                            if p.get('minutes_per_game', 0) >= 15]
            
            base_reliance = len(freshmen_in_rotation) / max(1, len(total_rotation))
            
            # Increase reliance if injuries force more freshmen into action
            injury_forced_freshmen = 0
            for injury in injuries:
                if injury.replacement_player_id:
                    replacement = next((p for p in roster_data 
                                     if p.get('player_id') == injury.replacement_player_id), None)
                    if replacement and replacement.get('class') == 'FR':
                        injury_forced_freshmen += 1
                        
            injury_reliance_increase = injury_forced_freshmen * 0.2
            
            return min(1.0, base_reliance + injury_reliance_increase)
            
        except Exception as e:
            self.logger.error(f"Error calculating freshman reliance: {e}")
            return 0.3
            
    def _calculate_bench_scoring_impact(self, injuries: List[BasketballInjuryReport]) -> float:
        """Calculate impact on bench scoring"""
        try:
            bench_points_lost = 0.0
            
            for injury in injuries:
                if injury.depth_chart_position > 5:  # Bench player
                    severity_multiplier = self.SEVERITY_MULTIPLIERS.get(injury.injury_severity, 0.5)
                    points_lost = injury.points_per_game * severity_multiplier
                    bench_points_lost += points_lost
                    
            # Normalize to typical bench scoring (20-25 PPG)
            return -min(1.0, bench_points_lost / 25.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating bench scoring impact: {e}")
            return 0.0
            
    def assess_basketball_injury_risk(self, player_data: Dict, 
                                    injury_history: List[Dict],
                                    schedule_context: Dict) -> BasketballInjuryRiskAssessment:
        """Assess basketball injury risk for individual player"""
        try:
            player_id = player_data.get('player_id', 'unknown')
            player_name = player_data.get('player_name', 'Unknown')
            position = player_data.get('position', 'SF')
            
            # Calculate risk factors
            minutes_risk = self._calculate_minutes_load_risk(player_data)
            contact_risk = self._calculate_contact_risk(player_data)
            history_risk = self._calculate_basketball_injury_history_risk(injury_history)
            body_risk = self._calculate_body_type_risk(player_data)
            surface_risk = self._calculate_playing_surface_risk(schedule_context)
            age_risk = self._calculate_age_risk_basketball(player_data)
            conditioning_risk = self._calculate_conditioning_risk(player_data, schedule_context)
            tournament_risk = self._calculate_tournament_fatigue_risk(schedule_context)
            
            # Weighted overall risk
            weights = [0.20, 0.15, 0.25, 0.10, 0.05, 0.10, 0.10, 0.05]
            risks = [minutes_risk, contact_risk, history_risk, body_risk, 
                    surface_risk, age_risk, conditioning_risk, tournament_risk]
            
            overall_risk = sum(w * r for w, r in zip(weights, risks))
            
            # Risk category
            if overall_risk < 0.3:
                risk_category = "Low"
            elif overall_risk < 0.5:
                risk_category = "Moderate" 
            elif overall_risk < 0.7:
                risk_category = "High"
            else:
                risk_category = "Very High"
                
            # Predict likely injury types
            likely_injuries = self._predict_basketball_injury_types(player_data, injury_history)
            
            # Expected games at risk
            expected_games = int(overall_risk * 15)  # Up to 15 games
            
            return BasketballInjuryRiskAssessment(
                player_id=player_id,
                player_name=player_name,
                position=position,
                minutes_load_risk=minutes_risk,
                contact_risk=contact_risk,
                injury_history_risk=history_risk,
                body_type_risk=body_risk,
                playing_surface_risk=surface_risk,
                age_risk=age_risk,
                conditioning_risk=conditioning_risk,
                tournament_fatigue_risk=tournament_risk,
                overall_risk_score=overall_risk,
                risk_category=risk_category,
                likely_injury_types=likely_injuries,
                expected_games_at_risk=expected_games
            )
            
        except Exception as e:
            self.logger.error(f"Error assessing basketball injury risk: {e}")
            return self._create_default_basketball_risk_assessment(player_id)
            
    def _calculate_minutes_load_risk(self, player_data: Dict) -> float:
        """Calculate risk based on minutes load"""
        minutes_per_game = player_data.get('minutes_per_game', 20.0)
        
        if minutes_per_game >= 35:
            return 0.8
        elif minutes_per_game >= 30:
            return 0.5
        elif minutes_per_game >= 25:
            return 0.3
        else:
            return 0.1
            
    def _calculate_contact_risk(self, player_data: Dict) -> float:
        """Calculate risk based on position and play style"""
        position = player_data.get('position', 'SF')
        
        # Contact risk by position
        contact_risks = {
            'C': 0.8,   # Centers get beat up in paint
            'PF': 0.7,  # Power forwards also in paint
            'SF': 0.5,  # Small forwards moderate contact
            'SG': 0.4,  # Shooting guards less contact
            'PG': 0.3   # Point guards least contact (usually)
        }
        
        base_risk = contact_risks.get(position, 0.5)
        
        # Adjust for play style
        fouls_per_game = player_data.get('fouls_per_game', 2.0)
        if fouls_per_game >= 3.5:
            base_risk += 0.2
        
        return min(1.0, base_risk)
        
    def _calculate_basketball_injury_history_risk(self, injury_history: List[Dict]) -> float:
        """Calculate risk based on basketball injury history"""
        if not injury_history:
            return 0.2
            
        # Recent injuries (last 2 years)
        current_year = datetime.now().year
        recent_injuries = [i for i in injury_history 
                         if datetime.strptime(i.get('injury_date', '2020-01-01'), '%Y-%m-%d').year >= current_year - 2]
        
        # Chronic basketball injuries
        chronic_types = ['knee', 'ankle', 'back', 'foot']
        chronic_injuries = [i for i in injury_history 
                          if any(chronic in i.get('injury_type', '').lower() for chronic in chronic_types)]
        
        base_risk = min(0.6, len(recent_injuries) * 0.15)
        chronic_risk = min(0.4, len(chronic_injuries) * 0.2)
        
        return min(0.9, base_risk + chronic_risk)
        
    def _calculate_body_type_risk(self, player_data: Dict) -> float:
        """Calculate risk based on body type"""
        height = player_data.get('height_inches', 72)  # 6'0" default
        weight = player_data.get('weight_pounds', 180)
        
        # Very tall players have higher injury risk
        height_risk = 0.0
        if height >= 84:  # 7'0"+
            height_risk = 0.8
        elif height >= 81:  # 6'9"+
            height_risk = 0.5
        elif height >= 78:  # 6'6"+
            height_risk = 0.3
            
        # BMI considerations
        bmi = (weight / (height ** 2)) * 703
        bmi_risk = 0.0
        if bmi < 20:  # Underweight
            bmi_risk = 0.3
        elif bmi > 28:  # Overweight for athlete
            bmi_risk = 0.4
            
        return min(1.0, height_risk + bmi_risk)
        
    def _calculate_playing_surface_risk(self, schedule_context: Dict) -> float:
        """Calculate risk based on playing surfaces and travel"""
        road_games_recent = schedule_context.get('road_games_last_10', 5)
        
        # More road games = higher risk
        if road_games_recent >= 7:
            return 0.6
        elif road_games_recent >= 5:
            return 0.3
        else:
            return 0.1
            
    def _calculate_age_risk_basketball(self, player_data: Dict) -> float:
        """Calculate age-related risk for college basketball"""
        class_year = player_data.get('class', 'SO')
        
        # College basketball age risk
        class_risks = {
            'FR': 0.4,  # Freshmen - higher risk due to adjustment
            'SO': 0.2,  # Sophomores - lowest risk
            'JR': 0.3,  # Juniors - moderate risk
            'SR': 0.5,  # Seniors - higher risk due to wear
            'GR': 0.6   # Grad students - highest risk
        }
        
        return class_risks.get(class_year, 0.3)
        
    def _calculate_conditioning_risk(self, player_data: Dict, schedule_context: Dict) -> float:
        """Calculate conditioning-based risk"""
        games_played = schedule_context.get('games_in_last_14_days', 4)
        
        # More games = higher fatigue risk
        if games_played >= 6:
            return 0.8
        elif games_played >= 4:
            return 0.4
        else:
            return 0.1
            
    def _calculate_tournament_fatigue_risk(self, schedule_context: Dict) -> float:
        """Calculate tournament fatigue risk"""
        tournament_context = schedule_context.get('tournament_context', 'regular_season')
        consecutive_tournament_games = schedule_context.get('consecutive_tournament_games', 0)
        
        if tournament_context == 'march_madness':
            base_risk = 0.4
            consecutive_risk = consecutive_tournament_games * 0.15
            return min(1.0, base_risk + consecutive_risk)
        elif tournament_context == 'conference_tournament':
            return 0.2
        else:
            return 0.0
            
    def _predict_basketball_injury_types(self, player_data: Dict, injury_history: List[Dict]) -> List[str]:
        """Predict most likely basketball injury types"""
        position = player_data.get('position', 'SF')
        
        # Common injuries by basketball position
        position_injuries = {
            'PG': ['ankle', 'knee', 'wrist'],
            'SG': ['ankle', 'hamstring', 'shoulder'],
            'SF': ['ankle', 'knee', 'back'],
            'PF': ['knee', 'back', 'ankle'],
            'C': ['knee', 'back', 'foot']
        }
        
        likely_injuries = position_injuries.get(position, ['ankle', 'knee']).copy()
        
        # Add from injury history
        for injury in injury_history:
            injury_type = injury.get('injury_type', '').lower()
            if injury_type and injury_type not in likely_injuries:
                likely_injuries.append(injury_type)
                
        return likely_injuries[:4]  # Top 4 most likely
        
    # Default creation methods
    
    def _create_default_basketball_team_status(self, team: str) -> BasketballTeamInjuryStatus:
        """Create default basketball team injury status"""
        return BasketballTeamInjuryStatus(
            team=team,
            analysis_date=datetime.now(),
            active_injuries=[],
            starting_five_injuries=0,
            key_bench_injuries=0,
            total_minutes_lost=0.0,
            point_guard_depth=0.7,
            shooting_guard_depth=0.7,
            small_forward_depth=0.7,
            power_forward_depth=0.7,
            center_depth=0.7,
            offensive_impact=0.0,
            rebounding_impact=0.0,
            assist_impact=0.0,
            defensive_impact=0.0,
            overall_team_impact=0.0,
            rotation_flexibility=0.6,
            freshman_reliance=0.3,
            bench_scoring_impact=0.0
        )
        
    def _create_default_basketball_injury_report(self) -> BasketballInjuryReport:
        """Create default basketball injury report"""
        return BasketballInjuryReport(
            player_id='unknown',
            player_name='Unknown Player',
            position='SF',
            team='UNK',
            injury_type='Unknown',
            injury_severity='Minor',
            injury_date=datetime.now(),
            current_status='Day-to-day',
            expected_return=None,
            games_missed=0,
            minutes_per_game=20.0,
            points_per_game=8.0,
            assists_per_game=2.0,
            rebounds_per_game=4.0,
            shooting_percentage=0.45,
            player_efficiency_rating=15.0,
            replacement_player_id=None,
            replacement_quality=0.7,
            depth_chart_position=6
        )
        
    def _create_default_basketball_risk_assessment(self, player_id: str) -> BasketballInjuryRiskAssessment:
        """Create default basketball risk assessment"""
        return BasketballInjuryRiskAssessment(
            player_id=player_id,
            player_name='Unknown Player',
            position='SF',
            minutes_load_risk=0.3,
            contact_risk=0.3,
            injury_history_risk=0.3,
            body_type_risk=0.3,
            playing_surface_risk=0.3,
            age_risk=0.3,
            conditioning_risk=0.3,
            tournament_fatigue_risk=0.3,
            overall_risk_score=0.3,
            risk_category="Moderate",
            likely_injury_types=['ankle', 'knee'],
            expected_games_at_risk=5
        )

def main():
    """Test basketball injury impact system"""
    logging.basicConfig(level=logging.INFO)
    
    injury_system = BasketballInjuryImpactSystem()
    
    print("Basketball Injury Impact System Test")
    print("=" * 40)
    
    # Mock injury data
    current_injuries = [
        {
            'player_id': 'player_001',
            'player_name': 'Star Point Guard',
            'position': 'PG',
            'team': 'Duke',
            'injury_type': 'ankle_sprain',
            'severity': 'Moderate',
            'status': 'Doubtful',
            'injury_date': '2024-10-10',
            'games_missed': 2,
            'minutes_per_game': 32.0,
            'points_per_game': 15.0,
            'assists_per_game': 8.0,
            'rebounds_per_game': 4.0,
            'player_efficiency_rating': 22.0,
            'depth_chart_position': 1
        }
    ]
    
    # Mock roster data
    roster_data = [
        {'player_id': 'player_002', 'position': 'PG', 'status': 'Healthy', 'minutes_per_game': 15, 'class': 'SO'},
        {'player_id': 'player_003', 'position': 'SG', 'status': 'Healthy', 'minutes_per_game': 25, 'class': 'JR'},
        {'player_id': 'player_004', 'position': 'SF', 'status': 'Healthy', 'minutes_per_game': 20, 'class': 'SR'},
    ]
    
    # Analyze team injury status
    team_status = injury_system.analyze_team_injury_status('Duke', current_injuries, roster_data, 'march_madness')
    
    print(f"Team: {team_status.team}")
    print(f"Active Injuries: {len(team_status.active_injuries)}")
    print(f"Starting Five Injuries: {team_status.starting_five_injuries}")
    print(f"Total Minutes Lost: {team_status.total_minutes_lost:.1f}")
    print(f"Overall Team Impact: {team_status.overall_team_impact:.3f}")
    print()
    
    print("Basketball Impact Breakdown:")
    print(f"  Offensive Impact: {team_status.offensive_impact:.3f}")
    print(f"  Rebounding Impact: {team_status.rebounding_impact:.3f}")
    print(f"  Assist Impact: {team_status.assist_impact:.3f}")
    print(f"  Defensive Impact: {team_status.defensive_impact:.3f}")
    print()
    
    print("Position Depth:")
    depths = [
        ('Point Guard', team_status.point_guard_depth),
        ('Shooting Guard', team_status.shooting_guard_depth),
        ('Small Forward', team_status.small_forward_depth),
        ('Power Forward', team_status.power_forward_depth),
        ('Center', team_status.center_depth)
    ]
    
    for pos_name, depth in depths:
        if depth < 0.8:  # Only show positions with concerns
            print(f"  {pos_name}: {depth:.2f}")
    print()
    
    print("Team Metrics:")
    print(f"  Rotation Flexibility: {team_status.rotation_flexibility:.2f}")
    print(f"  Freshman Reliance: {team_status.freshman_reliance:.2f}")
    print(f"  Bench Scoring Impact: {team_status.bench_scoring_impact:.3f}")
    
    print()
    print("✅ Basketball Injury Impact System operational!")

if __name__ == "__main__":
    main()