#!/usr/bin/env python3
"""
Basketball Versatility Analysis System
Analyzes player and team versatility in college basketball for strategic and betting insights.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PlayerVersatilityProfile:
    """Comprehensive player versatility metrics."""
    player_id: str
    name: str
    primary_position: str
    secondary_positions: List[str]
    
    # Position versatility scores (0-1)
    position_flexibility: float
    defensive_versatility: float
    offensive_versatility: float
    
    # Situational versatility
    clutch_versatility: float
    matchup_adaptability: float
    pace_adaptability: float
    
    # Playing time versatility
    minutes_consistency: float
    role_stability: float
    substitution_patterns: Dict[str, float]
    
    # Basketball IQ and adaptability
    court_vision_score: float
    decision_making_score: float
    leadership_score: float
    
    # Physical versatility
    size_versatility: float
    athleticism_score: float
    
    # Overall versatility rating
    overall_versatility: float

@dataclass
class TeamVersatilityProfile:
    """Team-wide versatility analysis."""
    team_id: str
    team_name: str
    
    # Roster versatility
    depth_versatility: float
    position_flexibility: float
    style_adaptability: float
    
    # Coaching versatility
    coaching_adaptability: float
    timeout_strategy_flexibility: float
    substitution_creativity: float
    
    # Tactical versatility
    offensive_scheme_variety: float
    defensive_scheme_variety: float
    pace_control_ability: float
    
    # Situational versatility
    clutch_lineup_options: float
    matchup_response_ability: float
    tournament_adaptability: float
    
    # Overall team versatility rating
    overall_team_versatility: float

class BasketballVersatilityAnalyzer:
    """Main class for basketball versatility analysis."""
    
    def __init__(self):
        self.position_weights = {
            'PG': {'ball_handling': 1.0, 'court_vision': 0.9, 'three_point': 0.7, 'defense': 0.6},
            'SG': {'scoring': 1.0, 'three_point': 0.9, 'defense': 0.8, 'ball_handling': 0.6},
            'SF': {'versatility': 1.0, 'three_point': 0.8, 'rebounding': 0.7, 'defense': 0.8},
            'PF': {'rebounding': 1.0, 'post_play': 0.9, 'defense': 0.9, 'mid_range': 0.6},
            'C': {'rebounding': 1.0, 'post_play': 1.0, 'defense': 1.0, 'rim_protection': 1.0}
        }
        
        self.versatility_factors = {
            'position_flexibility': 0.25,
            'skill_versatility': 0.30,
            'situational_adaptability': 0.25,
            'basketball_iq': 0.20
        }
    
    def analyze_player_versatility(self, player_data: Dict) -> PlayerVersatilityProfile:
        """Analyze individual player versatility."""
        logger.info(f"Analyzing versatility for {player_data.get('name', 'Unknown Player')}")
        
        # Basic info
        player_id = player_data.get('player_id', 'unknown')
        name = player_data.get('name', 'Unknown Player')
        primary_pos = player_data.get('primary_position', 'SF')
        secondary_pos = player_data.get('secondary_positions', [])
        
        # Calculate position flexibility
        position_flex = self._calculate_position_flexibility(player_data)
        
        # Calculate skill versatility
        defensive_versatility = self._calculate_defensive_versatility(player_data)
        offensive_versatility = self._calculate_offensive_versatility(player_data)
        
        # Calculate situational versatility
        clutch_versatility = self._calculate_clutch_versatility(player_data)
        matchup_adaptability = self._calculate_matchup_adaptability(player_data)
        pace_adaptability = self._calculate_pace_adaptability(player_data)
        
        # Calculate playing time patterns
        minutes_consistency = self._calculate_minutes_consistency(player_data)
        role_stability = self._calculate_role_stability(player_data)
        substitution_patterns = self._analyze_substitution_patterns(player_data)
        
        # Calculate basketball IQ factors
        court_vision = self._calculate_court_vision(player_data)
        decision_making = self._calculate_decision_making(player_data)
        leadership = self._calculate_leadership(player_data)
        
        # Calculate physical versatility
        size_versatility = self._calculate_size_versatility(player_data)
        athleticism = self._calculate_athleticism(player_data)
        
        # Calculate overall versatility
        overall_versatility = self._calculate_overall_player_versatility({
            'position_flexibility': position_flex,
            'offensive_versatility': offensive_versatility,
            'defensive_versatility': defensive_versatility,
            'situational_adaptability': (clutch_versatility + matchup_adaptability + pace_adaptability) / 3,
            'basketball_iq': (court_vision + decision_making + leadership) / 3,
            'physical_versatility': (size_versatility + athleticism) / 2
        })
        
        return PlayerVersatilityProfile(
            player_id=player_id,
            name=name,
            primary_position=primary_pos,
            secondary_positions=secondary_pos,
            position_flexibility=position_flex,
            defensive_versatility=defensive_versatility,
            offensive_versatility=offensive_versatility,
            clutch_versatility=clutch_versatility,
            matchup_adaptability=matchup_adaptability,
            pace_adaptability=pace_adaptability,
            minutes_consistency=minutes_consistency,
            role_stability=role_stability,
            substitution_patterns=substitution_patterns,
            court_vision_score=court_vision,
            decision_making_score=decision_making,
            leadership_score=leadership,
            size_versatility=size_versatility,
            athleticism_score=athleticism,
            overall_versatility=overall_versatility
        )
    
    def _calculate_position_flexibility(self, player_data: Dict) -> float:
        """Calculate how flexibly a player can play multiple positions."""
        positions_played = player_data.get('positions_played', {})
        minutes_by_position = player_data.get('minutes_by_position', {})
        
        if not positions_played:
            return 0.0
        
        # Number of positions effectively played
        position_count = len([pos for pos, games in positions_played.items() if games > 5])
        
        # Minutes distribution across positions
        total_minutes = sum(minutes_by_position.values())
        if total_minutes == 0:
            return 0.0
        
        # Calculate entropy of position distribution (higher = more flexible)
        position_entropy = 0.0
        for pos, minutes in minutes_by_position.items():
            if minutes > 0:
                prob = minutes / total_minutes
                position_entropy -= prob * np.log2(prob)
        
        # Normalize entropy (max is log2(5) for 5 positions)
        normalized_entropy = position_entropy / np.log2(5)
        
        # Combine position count and distribution
        flexibility = (position_count / 5.0 * 0.6) + (normalized_entropy * 0.4)
        
        return min(1.0, flexibility)
    
    def _calculate_offensive_versatility(self, player_data: Dict) -> float:
        """Calculate offensive skill versatility."""
        skills = {
            'three_point_shooting': player_data.get('three_point_pct', 0.3),
            'mid_range_shooting': player_data.get('mid_range_pct', 0.4),
            'post_scoring': player_data.get('post_scoring_efficiency', 0.5),
            'driving_ability': player_data.get('drives_per_game', 3) / 10.0,
            'passing_ability': player_data.get('assists_per_game', 2) / 8.0,
            'ball_handling': player_data.get('turnovers_per_game', 3) / 5.0  # Lower is better
        }
        
        # Normalize ball handling (lower turnovers = better)
        skills['ball_handling'] = 1.0 - min(1.0, skills['ball_handling'])
        
        # Calculate versatility as spread of skills
        skill_values = list(skills.values())
        mean_skill = np.mean(skill_values)
        skill_variance = np.var(skill_values)
        
        # High mean with low variance indicates versatile offense
        versatility = mean_skill * (1 - skill_variance)
        
        return max(0.0, min(1.0, versatility))
    
    def _calculate_defensive_versatility(self, player_data: Dict) -> float:
        """Calculate defensive versatility across different matchups."""
        defensive_metrics = {
            'perimeter_defense': player_data.get('steals_per_game', 1) / 3.0,
            'post_defense': player_data.get('post_defense_efficiency', 0.5),
            'help_defense': player_data.get('help_defense_rating', 0.5),
            'rebounding': player_data.get('rebounds_per_game', 5) / 12.0,
            'rim_protection': player_data.get('blocks_per_game', 0.5) / 3.0,
            'defensive_switches': player_data.get('defensive_switches_success', 0.6)
        }
        
        # Weight by position requirements
        primary_pos = player_data.get('primary_position', 'SF')
        position_weights = self.position_weights.get(primary_pos, {})
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for skill, value in defensive_metrics.items():
            weight = position_weights.get(skill, 0.5)
            weighted_score += value * weight
            total_weight += weight
        
        if total_weight > 0:
            return min(1.0, weighted_score / total_weight)
        else:
            return np.mean(list(defensive_metrics.values()))
    
    def _calculate_clutch_versatility(self, player_data: Dict) -> float:
        """Calculate performance versatility in clutch situations."""
        clutch_stats = player_data.get('clutch_stats', {})
        
        if not clutch_stats:
            return 0.5
        
        regular_stats = player_data.get('season_stats', {})
        
        # Compare clutch vs regular performance
        metrics = ['fg_pct', 'three_point_pct', 'ft_pct', 'assists', 'turnovers']
        consistency_scores = []
        
        for metric in metrics:
            regular_val = regular_stats.get(metric, 0.5)
            clutch_val = clutch_stats.get(metric, 0.5)
            
            if regular_val > 0:
                # How well does clutch performance match regular performance
                consistency = 1 - abs(clutch_val - regular_val) / regular_val
                consistency_scores.append(max(0, consistency))
        
        if consistency_scores:
            return np.mean(consistency_scores)
        else:
            return 0.5
    
    def _calculate_matchup_adaptability(self, player_data: Dict) -> float:
        """Calculate ability to adapt to different opponent types."""
        matchup_stats = player_data.get('matchup_performance', {})
        
        if not matchup_stats:
            return 0.5
        
        # Performance against different opponent types
        opponent_types = ['small', 'big', 'athletic', 'shooter', 'physical']
        performance_variance = []
        
        for opp_type in opponent_types:
            performance = matchup_stats.get(f'vs_{opp_type}', {})
            if performance:
                # Calculate composite performance score
                score = (
                    performance.get('fg_pct', 0.45) * 0.3 +
                    performance.get('efficiency', 1.0) * 0.4 +
                    performance.get('plus_minus', 0) / 10.0 * 0.3
                )
                performance_variance.append(score)
        
        if performance_variance:
            # Lower variance = more consistent across matchups
            variance = np.var(performance_variance)
            mean_performance = np.mean(performance_variance)
            adaptability = mean_performance * (1 - variance)
            return max(0.0, min(1.0, adaptability))
        
        return 0.5
    
    def _calculate_pace_adaptability(self, player_data: Dict) -> float:
        """Calculate ability to perform in different game paces."""
        pace_stats = player_data.get('pace_performance', {})
        
        if not pace_stats:
            return 0.5
        
        # Performance in different pace scenarios
        slow_pace = pace_stats.get('slow_pace_efficiency', 1.0)
        medium_pace = pace_stats.get('medium_pace_efficiency', 1.0)
        fast_pace = pace_stats.get('fast_pace_efficiency', 1.0)
        
        pace_scores = [slow_pace, medium_pace, fast_pace]
        
        # High mean with low variance indicates good pace adaptability
        mean_pace = np.mean(pace_scores)
        pace_variance = np.var(pace_scores)
        
        adaptability = mean_pace * (1 - pace_variance)
        return max(0.0, min(1.0, adaptability))
    
    def _calculate_minutes_consistency(self, player_data: Dict) -> float:
        """Calculate consistency in playing time."""
        minutes_log = player_data.get('minutes_by_game', [])
        
        if len(minutes_log) < 5:
            return 0.5
        
        mean_minutes = np.mean(minutes_log)
        std_minutes = np.std(minutes_log)
        
        if mean_minutes == 0:
            return 0.0
        
        # Coefficient of variation (lower = more consistent)
        cv = std_minutes / mean_minutes
        consistency = 1 - min(1.0, cv)
        
        return consistency
    
    def _calculate_role_stability(self, player_data: Dict) -> float:
        """Calculate stability in player's role within team."""
        role_changes = player_data.get('role_changes', 0)
        games_played = player_data.get('games_played', 20)
        
        if games_played == 0:
            return 0.0
        
        # Fewer role changes = more stability
        stability = 1 - min(1.0, role_changes / games_played * 5)
        return max(0.0, stability)
    
    def _analyze_substitution_patterns(self, player_data: Dict) -> Dict[str, float]:
        """Analyze substitution and rotation patterns."""
        sub_data = player_data.get('substitution_data', {})
        
        patterns = {
            'starter_frequency': sub_data.get('games_started', 0) / max(1, player_data.get('games_played', 1)),
            'sixth_man_usage': sub_data.get('sixth_man_games', 0) / max(1, player_data.get('games_played', 1)),
            'emergency_minutes': sub_data.get('emergency_minutes', 0) / max(1, player_data.get('total_minutes', 1)),
            'position_sub_flexibility': len(sub_data.get('substitution_positions', [])) / 5.0,
            'game_situation_usage': sub_data.get('situational_substitutions', 0) / max(1, player_data.get('games_played', 1))
        }
        
        return patterns
    
    def _calculate_court_vision(self, player_data: Dict) -> float:
        """Calculate court vision and playmaking ability."""
        assists = player_data.get('assists_per_game', 2)
        turnovers = player_data.get('turnovers_per_game', 2)
        potential_assists = player_data.get('potential_assists_per_game', assists * 1.5)
        
        # Assist to turnover ratio
        ast_to_ratio = assists / max(1, turnovers)
        
        # Assist creation rate
        assist_rate = assists / max(1, potential_assists)
        
        # Combine metrics
        vision_score = (ast_to_ratio / 5.0 * 0.6) + (assist_rate * 0.4)
        
        return min(1.0, vision_score)
    
    def _calculate_decision_making(self, player_data: Dict) -> float:
        """Calculate decision-making quality."""
        bad_decisions = player_data.get('bad_decisions_per_game', 2)
        shot_selection = player_data.get('shot_selection_rating', 0.5)
        turnover_rate = player_data.get('turnover_rate', 0.15)
        
        # Lower bad decisions and turnovers = better decision making
        decision_quality = (1 - min(1.0, bad_decisions / 5)) * 0.4
        decision_quality += shot_selection * 0.3
        decision_quality += (1 - min(1.0, turnover_rate)) * 0.3
        
        return max(0.0, min(1.0, decision_quality))
    
    def _calculate_leadership(self, player_data: Dict) -> float:
        """Calculate leadership and intangible qualities."""
        leadership_metrics = {
            'team_record_with': player_data.get('team_record_with_player', 0.5),
            'team_record_without': player_data.get('team_record_without_player', 0.5),
            'vocal_leadership': player_data.get('vocal_leadership_rating', 0.5),
            'example_setting': player_data.get('example_setting_rating', 0.5),
            'clutch_moments': player_data.get('clutch_moments_stepped_up', 0) / max(1, player_data.get('clutch_opportunities', 1))
        }
        
        # Impact differential
        record_impact = leadership_metrics['team_record_with'] - leadership_metrics['team_record_without']
        record_impact = max(-0.5, min(0.5, record_impact)) + 0.5  # Normalize to 0-1
        
        # Combine all leadership factors
        leadership_score = (
            record_impact * 0.3 +
            leadership_metrics['vocal_leadership'] * 0.2 +
            leadership_metrics['example_setting'] * 0.2 +
            leadership_metrics['clutch_moments'] * 0.3
        )
        
        return max(0.0, min(1.0, leadership_score))
    
    def _calculate_size_versatility(self, player_data: Dict) -> float:
        """Calculate versatility based on physical measurements."""
        height = player_data.get('height_inches', 78)  # Default 6'6"
        weight = player_data.get('weight_lbs', 200)
        wingspan = player_data.get('wingspan_inches', height + 2)
        
        # Calculate how well size translates across positions
        position_fits = {
            'PG': max(0, (75 - height) / 10) if height <= 75 else 0,  # Under 6'3"
            'SG': max(0, 1 - abs(height - 77) / 5),  # Around 6'5"
            'SF': max(0, 1 - abs(height - 79) / 6),  # Around 6'7"
            'PF': max(0, 1 - abs(height - 81) / 6),  # Around 6'9"
            'C': max(0, (height - 80) / 8) if height >= 80 else 0  # Over 6'8"
        }
        
        # Number of positions player's size fits
        size_flexibility = sum(1 for fit in position_fits.values() if fit > 0.5) / 5.0
        
        # Wingspan advantage
        wingspan_advantage = min(1.0, (wingspan - height) / 8.0)
        
        versatility = size_flexibility * 0.7 + wingspan_advantage * 0.3
        return min(1.0, versatility)
    
    def _calculate_athleticism(self, player_data: Dict) -> float:
        """Calculate athleticism versatility."""
        athletic_metrics = {
            'speed': player_data.get('speed_rating', 0.5),
            'agility': player_data.get('agility_rating', 0.5),
            'vertical_leap': player_data.get('vertical_inches', 30) / 40.0,
            'endurance': player_data.get('endurance_rating', 0.5),
            'lateral_movement': player_data.get('lateral_movement_rating', 0.5)
        }
        
        # High athleticism in multiple areas = versatile athlete
        athleticism_score = np.mean(list(athletic_metrics.values()))
        
        # Bonus for being well-rounded (low variance)
        variance_penalty = np.var(list(athletic_metrics.values()))
        athleticism_score *= (1 - variance_penalty * 0.5)
        
        return max(0.0, min(1.0, athleticism_score))
    
    def _calculate_overall_player_versatility(self, component_scores: Dict[str, float]) -> float:
        """Calculate overall player versatility score."""
        weighted_score = 0.0
        
        for factor, weight in self.versatility_factors.items():
            if factor in component_scores:
                weighted_score += component_scores[factor] * weight
        
        return min(1.0, weighted_score)
    
    def analyze_team_versatility(self, team_data: Dict, player_profiles: List[PlayerVersatilityProfile]) -> TeamVersatilityProfile:
        """Analyze team-wide versatility capabilities."""
        logger.info(f"Analyzing team versatility for {team_data.get('team_name', 'Unknown Team')}")
        
        team_id = team_data.get('team_id', 'unknown')
        team_name = team_data.get('team_name', 'Unknown Team')
        
        # Roster versatility analysis
        depth_versatility = self._calculate_team_depth_versatility(player_profiles)
        position_flexibility = self._calculate_team_position_flexibility(player_profiles)
        style_adaptability = self._calculate_team_style_adaptability(team_data, player_profiles)
        
        # Coaching versatility
        coaching_adaptability = self._calculate_coaching_adaptability(team_data)
        timeout_flexibility = self._calculate_timeout_strategy_flexibility(team_data)
        substitution_creativity = self._calculate_substitution_creativity(team_data, player_profiles)
        
        # Tactical versatility
        offensive_variety = self._calculate_offensive_scheme_variety(team_data)
        defensive_variety = self._calculate_defensive_scheme_variety(team_data)
        pace_control = self._calculate_pace_control_ability(team_data)
        
        # Situational versatility
        clutch_options = self._calculate_clutch_lineup_options(player_profiles)
        matchup_response = self._calculate_matchup_response_ability(team_data, player_profiles)
        tournament_adaptability = self._calculate_tournament_adaptability(team_data, player_profiles)
        
        # Overall team versatility
        overall_versatility = self._calculate_overall_team_versatility({
            'roster_versatility': (depth_versatility + position_flexibility + style_adaptability) / 3,
            'coaching_versatility': (coaching_adaptability + timeout_flexibility + substitution_creativity) / 3,
            'tactical_versatility': (offensive_variety + defensive_variety + pace_control) / 3,
            'situational_versatility': (clutch_options + matchup_response + tournament_adaptability) / 3
        })
        
        return TeamVersatilityProfile(
            team_id=team_id,
            team_name=team_name,
            depth_versatility=depth_versatility,
            position_flexibility=position_flexibility,
            style_adaptability=style_adaptability,
            coaching_adaptability=coaching_adaptability,
            timeout_strategy_flexibility=timeout_flexibility,
            substitution_creativity=substitution_creativity,
            offensive_scheme_variety=offensive_variety,
            defensive_scheme_variety=defensive_variety,
            pace_control_ability=pace_control,
            clutch_lineup_options=clutch_options,
            matchup_response_ability=matchup_response,
            tournament_adaptability=tournament_adaptability,
            overall_team_versatility=overall_versatility
        )
    
    def _calculate_team_depth_versatility(self, player_profiles: List[PlayerVersatilityProfile]) -> float:
        """Calculate team's depth and bench versatility."""
        if len(player_profiles) < 8:
            return 0.3  # Insufficient depth
        
        # Sort players by overall versatility
        sorted_players = sorted(player_profiles, key=lambda p: p.overall_versatility, reverse=True)
        
        # Analyze top 8-10 rotation players
        rotation_players = sorted_players[:10]
        bench_players = sorted_players[5:10]  # Typical bench players
        
        # Bench versatility average
        bench_versatility = np.mean([p.overall_versatility for p in bench_players]) if bench_players else 0.0
        
        # Starter-to-bench versatility difference (smaller is better)
        starter_versatility = np.mean([p.overall_versatility for p in sorted_players[:5]])
        versatility_dropoff = starter_versatility - bench_versatility
        
        # Calculate depth score
        depth_score = bench_versatility * 0.7 + (1 - min(1.0, versatility_dropoff)) * 0.3
        
        return min(1.0, depth_score)
    
    def _calculate_team_position_flexibility(self, player_profiles: List[PlayerVersatilityProfile]) -> float:
        """Calculate team's overall position flexibility."""
        position_coverage = {'PG': 0, 'SG': 0, 'SF': 0, 'PF': 0, 'C': 0}
        
        for player in player_profiles:
            # Count primary position
            if player.primary_position in position_coverage:
                position_coverage[player.primary_position] += 1
            
            # Count secondary positions with weight
            for sec_pos in player.secondary_positions:
                if sec_pos in position_coverage:
                    position_coverage[sec_pos] += 0.5 * player.position_flexibility
        
        # Check position coverage balance
        min_coverage = min(position_coverage.values())
        max_coverage = max(position_coverage.values())
        balance_score = 1 - (max_coverage - min_coverage) / max(1, max_coverage)
        
        # Calculate average position flexibility
        avg_flexibility = np.mean([p.position_flexibility for p in player_profiles])
        
        # Combine balance and flexibility
        overall_flexibility = balance_score * 0.4 + avg_flexibility * 0.6
        
        return min(1.0, overall_flexibility)
    
    def _calculate_team_style_adaptability(self, team_data: Dict, player_profiles: List[PlayerVersatilityProfile]) -> float:
        """Calculate team's ability to adapt playing style."""
        style_options = {
            'fast_break_ability': np.mean([p.pace_adaptability for p in player_profiles]),
            'half_court_offense': np.mean([p.offensive_versatility for p in player_profiles]),
            'defensive_switching': np.mean([p.defensive_versatility for p in player_profiles]),
            'small_ball_capability': len([p for p in player_profiles if p.position_flexibility > 0.6]) / len(player_profiles),
            'post_up_game': len([p for p in player_profiles if 'PF' in [p.primary_position] + p.secondary_positions or 'C' in [p.primary_position] + p.secondary_positions]) / 5.0
        }
        
        # Check if team can effectively run different styles
        style_versatility = np.mean(list(style_options.values()))
        
        # Bonus for being able to play multiple contrasting styles
        style_variance = np.var(list(style_options.values()))
        versatility_bonus = 1 - style_variance * 0.5
        
        adaptability = style_versatility * versatility_bonus
        return min(1.0, adaptability)
    
    def _calculate_coaching_adaptability(self, team_data: Dict) -> float:
        """Calculate coaching staff's adaptability."""
        coaching_metrics = team_data.get('coaching_metrics', {})
        
        adaptability_factors = {
            'in_game_adjustments': coaching_metrics.get('in_game_adjustment_rating', 0.5),
            'scheme_variety': coaching_metrics.get('scheme_variety_rating', 0.5),
            'player_development': coaching_metrics.get('player_development_rating', 0.5),
            'game_plan_flexibility': coaching_metrics.get('game_plan_flexibility', 0.5),
            'timeout_effectiveness': coaching_metrics.get('timeout_effectiveness', 0.5)
        }
        
        return np.mean(list(adaptability_factors.values()))
    
    def _calculate_timeout_strategy_flexibility(self, team_data: Dict) -> float:
        """Calculate timeout usage and strategy flexibility."""
        timeout_data = team_data.get('timeout_usage', {})
        
        if not timeout_data:
            return 0.5
        
        # Analyze timeout usage patterns
        strategic_timeouts = timeout_data.get('strategic_timeouts', 0)
        total_timeouts = timeout_data.get('total_timeouts_used', 1)
        timeout_success_rate = timeout_data.get('timeout_success_rate', 0.5)
        
        flexibility_score = (strategic_timeouts / max(1, total_timeouts)) * timeout_success_rate
        
        return min(1.0, flexibility_score)
    
    def _calculate_substitution_creativity(self, team_data: Dict, player_profiles: List[PlayerVersatilityProfile]) -> float:
        """Calculate creativity and effectiveness of substitution patterns."""
        sub_data = team_data.get('substitution_patterns', {})
        
        # Unique lineup combinations used
        unique_lineups = sub_data.get('unique_lineups_used', 10)
        total_games = team_data.get('games_played', 20)
        lineup_creativity = min(1.0, unique_lineups / (total_games * 2))
        
        # Substitution timing variety
        sub_timing_variety = sub_data.get('substitution_timing_variety', 0.5)
        
        # Player versatility enabling creative substitutions
        avg_player_versatility = np.mean([p.overall_versatility for p in player_profiles])
        
        creativity = (lineup_creativity * 0.4 + sub_timing_variety * 0.3 + avg_player_versatility * 0.3)
        
        return min(1.0, creativity)
    
    def _calculate_offensive_scheme_variety(self, team_data: Dict) -> float:
        """Calculate variety in offensive schemes and plays."""
        offensive_data = team_data.get('offensive_schemes', {})
        
        scheme_variety = {
            'motion_offense': offensive_data.get('motion_offense_usage', 0.3),
            'pick_and_roll': offensive_data.get('pick_and_roll_usage', 0.4),
            'isolation_plays': offensive_data.get('isolation_usage', 0.2),
            'post_ups': offensive_data.get('post_up_usage', 0.2),
            'fast_breaks': offensive_data.get('fast_break_usage', 0.3),
            'set_plays': offensive_data.get('set_play_usage', 0.3)
        }
        
        # Higher entropy = more variety
        scheme_entropy = 0.0
        total_usage = sum(scheme_variety.values())
        
        if total_usage > 0:
            for usage in scheme_variety.values():
                if usage > 0:
                    prob = usage / total_usage
                    scheme_entropy -= prob * np.log2(prob)
        
        # Normalize entropy
        max_entropy = np.log2(len(scheme_variety))
        normalized_entropy = scheme_entropy / max_entropy if max_entropy > 0 else 0
        
        return normalized_entropy
    
    def _calculate_defensive_scheme_variety(self, team_data: Dict) -> float:
        """Calculate variety in defensive schemes."""
        defensive_data = team_data.get('defensive_schemes', {})
        
        scheme_usage = {
            'man_to_man': defensive_data.get('man_to_man_usage', 0.6),
            'zone_defense': defensive_data.get('zone_usage', 0.2),
            'press_defense': defensive_data.get('press_usage', 0.1),
            'switching_defense': defensive_data.get('switching_usage', 0.3),
            'help_and_recover': defensive_data.get('help_and_recover_usage', 0.4)
        }
        
        # Calculate scheme entropy
        scheme_entropy = 0.0
        total_usage = sum(scheme_usage.values())
        
        if total_usage > 0:
            for usage in scheme_usage.values():
                if usage > 0:
                    prob = usage / total_usage
                    scheme_entropy -= prob * np.log2(prob)
        
        max_entropy = np.log2(len(scheme_usage))
        normalized_entropy = scheme_entropy / max_entropy if max_entropy > 0 else 0
        
        return normalized_entropy
    
    def _calculate_pace_control_ability(self, team_data: Dict) -> float:
        """Calculate team's ability to control game pace."""
        pace_data = team_data.get('pace_control', {})
        
        pace_metrics = {
            'slow_down_ability': pace_data.get('slow_down_games_won', 0) / max(1, pace_data.get('slow_down_attempts', 1)),
            'speed_up_ability': pace_data.get('speed_up_games_won', 0) / max(1, pace_data.get('speed_up_attempts', 1)),
            'pace_variance': 1 - min(1.0, pace_data.get('pace_std_dev', 10) / 15),  # Lower variance = better control
            'situational_pace': pace_data.get('situational_pace_adjustments', 0.5)
        }
        
        return np.mean(list(pace_metrics.values()))
    
    def _calculate_clutch_lineup_options(self, player_profiles: List[PlayerVersatilityProfile]) -> float:
        """Calculate number and quality of clutch lineup options."""
        clutch_players = [p for p in player_profiles if p.clutch_versatility > 0.6]
        
        if len(clutch_players) < 5:
            return 0.3
        
        # Number of viable clutch lineups
        clutch_lineup_options = len(clutch_players) - 4  # Subtract minimum 5 for starting lineup
        
        # Quality of clutch players
        avg_clutch_versatility = np.mean([p.clutch_versatility for p in clutch_players])
        
        # Position flexibility in clutch situations
        clutch_position_flex = np.mean([p.position_flexibility for p in clutch_players])
        
        options_score = min(1.0, clutch_lineup_options / 5.0) * 0.4
        quality_score = avg_clutch_versatility * 0.4
        flexibility_score = clutch_position_flex * 0.2
        
        return options_score + quality_score + flexibility_score
    
    def _calculate_matchup_response_ability(self, team_data: Dict, player_profiles: List[PlayerVersatilityProfile]) -> float:
        """Calculate ability to respond to different opponent matchups."""
        matchup_data = team_data.get('matchup_responses', {})
        
        # Historical success against different opponent types
        matchup_success = {
            'vs_big_teams': matchup_data.get('vs_big_teams_record', 0.5),
            'vs_small_teams': matchup_data.get('vs_small_teams_record', 0.5),
            'vs_athletic_teams': matchup_data.get('vs_athletic_teams_record', 0.5),
            'vs_shooting_teams': matchup_data.get('vs_shooting_teams_record', 0.5),
            'vs_defensive_teams': matchup_data.get('vs_defensive_teams_record', 0.5)
        }
        
        # Player versatility enabling matchup responses
        avg_matchup_adaptability = np.mean([p.matchup_adaptability for p in player_profiles])
        
        # Combine historical success with player capabilities
        historical_success = np.mean(list(matchup_success.values()))
        
        response_ability = historical_success * 0.6 + avg_matchup_adaptability * 0.4
        
        return response_ability
    
    def _calculate_tournament_adaptability(self, team_data: Dict, player_profiles: List[PlayerVersatilityProfile]) -> float:
        """Calculate adaptability for tournament play."""
        tournament_factors = {
            'tournament_experience': team_data.get('tournament_games_played', 0) / 20.0,
            'coaching_tournament_experience': team_data.get('coaching_tournament_experience', 0) / 10.0,
            'player_tournament_versatility': np.mean([p.overall_versatility for p in player_profiles]),
            'depth_for_tournament': len([p for p in player_profiles if p.overall_versatility > 0.5]) / 8.0,
            'clutch_performance': np.mean([p.clutch_versatility for p in player_profiles])
        }
        
        # Weight factors for tournament importance
        tournament_adaptability = (
            min(1.0, tournament_factors['tournament_experience']) * 0.2 +
            min(1.0, tournament_factors['coaching_tournament_experience']) * 0.2 +
            tournament_factors['player_tournament_versatility'] * 0.3 +
            min(1.0, tournament_factors['depth_for_tournament']) * 0.15 +
            tournament_factors['clutch_performance'] * 0.15
        )
        
        return min(1.0, tournament_adaptability)
    
    def _calculate_overall_team_versatility(self, component_scores: Dict[str, float]) -> float:
        """Calculate overall team versatility score."""
        weights = {
            'roster_versatility': 0.35,
            'coaching_versatility': 0.25,
            'tactical_versatility': 0.25,
            'situational_versatility': 0.15
        }
        
        weighted_score = 0.0
        for component, weight in weights.items():
            if component in component_scores:
                weighted_score += component_scores[component] * weight
        
        return min(1.0, weighted_score)
    
    def generate_versatility_insights(self, team_profile: TeamVersatilityProfile, player_profiles: List[PlayerVersatilityProfile]) -> Dict[str, str]:
        """Generate actionable insights from versatility analysis."""
        insights = {}
        
        # Team strength insights
        if team_profile.overall_team_versatility > 0.75:
            insights['strength'] = "Exceptional versatility allows adaptation to any opponent or game situation"
        elif team_profile.overall_team_versatility > 0.6:
            insights['strength'] = "Good versatility provides multiple tactical options"
        else:
            insights['strength'] = "Limited versatility may struggle against diverse opponents"
        
        # Specific versatility insights
        if team_profile.position_flexibility > 0.7:
            insights['position_flexibility'] = "High position flexibility enables creative lineups and matchup advantages"
        
        if team_profile.depth_versatility < 0.4:
            insights['depth_concern'] = "Shallow bench versatility creates vulnerability to injuries and foul trouble"
        
        # Player insights
        most_versatile = max(player_profiles, key=lambda p: p.overall_versatility)
        insights['key_player'] = f"{most_versatile.name} is the key versatility enabler with {most_versatile.overall_versatility:.2f} rating"
        
        # Tournament implications
        if team_profile.tournament_adaptability > 0.7:
            insights['tournament'] = "Strong tournament adaptability suggests March Madness upset potential"
        elif team_profile.tournament_adaptability < 0.4:
            insights['tournament'] = "Low tournament adaptability may lead to early exits against diverse opponents"
        
        return insights

def main():
    """Demo the basketball versatility analysis system."""
    print("College Basketball Versatility Analysis System")
    print("=" * 50)
    
    # Sample player data
    sample_players = [
        {
            'player_id': '1', 'name': 'Marcus Johnson', 'primary_position': 'PG',
            'secondary_positions': ['SG'], 'assists_per_game': 6.5, 'turnovers_per_game': 2.1,
            'three_point_pct': 0.38, 'steals_per_game': 1.8, 'height_inches': 74, 'games_played': 25,
            'positions_played': {'PG': 25, 'SG': 8}, 'minutes_by_position': {'PG': 600, 'SG': 120}
        },
        {
            'player_id': '2', 'name': 'Tyler Williams', 'primary_position': 'SF',
            'secondary_positions': ['SG', 'PF'], 'assists_per_game': 4.2, 'turnovers_per_game': 1.8,
            'three_point_pct': 0.35, 'rebounds_per_game': 7.1, 'height_inches': 79, 'games_played': 27,
            'positions_played': {'SF': 27, 'SG': 12, 'PF': 15}, 'minutes_by_position': {'SF': 540, 'SG': 180, 'PF': 200}
        }
    ]
    
    # Sample team data
    sample_team = {
        'team_id': 'duke', 'team_name': 'Duke Blue Devils', 'games_played': 25,
        'coaching_metrics': {'in_game_adjustment_rating': 0.8, 'scheme_variety_rating': 0.7},
        'offensive_schemes': {'motion_offense': 0.4, 'pick_and_roll': 0.5, 'isolation': 0.2},
        'defensive_schemes': {'man_to_man': 0.7, 'zone_defense': 0.2, 'switching': 0.4}
    }
    
    # Initialize analyzer
    analyzer = BasketballVersatilityAnalyzer()
    
    # Analyze player versatility
    player_profiles = []
    for player_data in sample_players:
        profile = analyzer.analyze_player_versatility(player_data)
        player_profiles.append(profile)
        
        print(f"\nPlayer Analysis: {profile.name}")
        print(f"  Primary Position: {profile.primary_position}")
        print(f"  Secondary Positions: {profile.secondary_positions}")
        print(f"  Overall Versatility: {profile.overall_versatility:.3f}")
        print(f"  Position Flexibility: {profile.position_flexibility:.3f}")
        print(f"  Offensive Versatility: {profile.offensive_versatility:.3f}")
        print(f"  Defensive Versatility: {profile.defensive_versatility:.3f}")
    
    # Analyze team versatility
    team_profile = analyzer.analyze_team_versatility(sample_team, player_profiles)
    
    print(f"\n{'='*30}")
    print(f"Team Analysis: {team_profile.team_name}")
    print(f"  Overall Team Versatility: {team_profile.overall_team_versatility:.3f}")
    print(f"  Position Flexibility: {team_profile.position_flexibility:.3f}")
    print(f"  Depth Versatility: {team_profile.depth_versatility:.3f}")
    print(f"  Coaching Adaptability: {team_profile.coaching_adaptability:.3f}")
    print(f"  Tournament Adaptability: {team_profile.tournament_adaptability:.3f}")
    
    # Generate insights
    insights = analyzer.generate_versatility_insights(team_profile, player_profiles)
    
    print(f"\n{'='*30}")
    print("Versatility Insights:")
    for category, insight in insights.items():
        print(f"  {category.title()}: {insight}")

if __name__ == "__main__":
    main()