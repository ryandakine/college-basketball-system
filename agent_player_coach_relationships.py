#!/usr/bin/env python3
"""
Agent-Player-Coach Relationship Analysis System for College Basketball
Analyzes the complex web of relationships, NIL deals, and financial influences 
that impact team chemistry, performance, and betting outcomes.

Backtests from NIL era start: July 1, 2021
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# NIL Era Constants
NIL_START_DATE = datetime(2021, 7, 1)
TRANSFER_PORTAL_EXPANSION = datetime(2021, 4, 15)

@dataclass
class AgentProfile:
    """Agent representation and influence metrics."""
    agent_id: str
    name: str
    agency: str
    influence_level: float  # 0-1 scale
    client_count: int
    client_roster: List[str]  # Player IDs
    coaching_connections: Dict[str, float]  # Coach ID -> relationship strength
    nil_deal_volume: float  # Total NIL value facilitated
    transfer_portal_activity: int  # Number of transfers facilitated
    recruiting_network_strength: float
    reputation_score: float
    
@dataclass
class PlayerAgentRelationship:
    """Player-Agent relationship dynamics."""
    player_id: str
    agent_id: str
    relationship_start: datetime
    relationship_strength: float  # 0-1
    influence_on_decisions: float  # How much agent influences player decisions
    nil_involvement: float  # Agent's involvement in NIL deals
    transfer_influence: float  # Agent's influence on transfer decisions
    family_dynamics: float  # How agent interacts with player's family
    academic_involvement: float  # Agent's involvement in academic decisions
    
@dataclass
class CoachPlayerDynamics:
    """Coach-Player relationship analysis."""
    player_id: str
    coach_id: str
    relationship_quality: float  # 0-1
    trust_level: float
    playing_time_satisfaction: float
    development_satisfaction: float
    communication_quality: float
    conflict_history: List[Dict]  # Past conflicts/issues
    nil_alignment: float  # How aligned coach and player are on NIL
    future_planning_alignment: float  # NBA/transfer portal discussions
    
@dataclass
class TeamChemistryProfile:
    """Overall team chemistry influenced by relationships."""
    team_id: str
    overall_chemistry_score: float
    agent_influence_score: float  # How much agents affect team
    nil_tension_level: float  # NIL-related tensions
    transfer_portal_risk: float  # Risk of players transferring
    coaching_stability: float
    veteran_leadership_strength: float
    freshmen_integration_quality: float
    playing_time_distribution_satisfaction: float
    
@dataclass
class NILImpactAnalysis:
    """NIL deal impact on team dynamics."""
    team_id: str
    total_nil_value: float
    nil_distribution_equity: float  # How evenly NIL is distributed
    nil_performance_correlation: float  # NIL money vs performance
    nil_distraction_level: float  # How much NIL distracts from basketball
    market_leverage: float  # Team's NIL market advantages
    coaching_nil_management: float  # How well coaches handle NIL

class AgentPlayerCoachAnalyzer:
    """Main analyzer for agent-player-coach relationships in college basketball."""
    
    def __init__(self):
        self.nil_start_date = NIL_START_DATE
        self.relationship_weights = {
            'agent_influence': 0.25,
            'coach_player_dynamics': 0.35,
            'nil_impact': 0.20,
            'team_chemistry': 0.20
        }
        
        # Historical baselines (pre-NIL vs post-NIL)
        self.pre_nil_baselines = {
            'transfer_rate': 0.15,
            'coaching_turnover': 0.12,
            'team_chemistry_variance': 0.08
        }
        
        self.post_nil_metrics = {
            'transfer_rate': 0.35,  # Massive increase post-NIL
            'coaching_turnover': 0.18,
            'team_chemistry_variance': 0.15
        }
    
    def analyze_agent_influence(self, agent_data: Dict) -> AgentProfile:
        """Analyze individual agent's influence and network."""
        logger.info(f"Analyzing agent: {agent_data.get('name', 'Unknown Agent')}")
        
        # Basic agent info
        agent_id = agent_data.get('agent_id', 'unknown')
        name = agent_data.get('name', 'Unknown Agent')
        agency = agent_data.get('agency', 'Independent')
        
        # Calculate influence metrics
        client_count = len(agent_data.get('clients', []))
        influence_level = self._calculate_agent_influence_level(agent_data)
        coaching_connections = self._analyze_coaching_network(agent_data)
        nil_deal_volume = agent_data.get('total_nil_facilitated', 0)
        transfer_activity = len(agent_data.get('transfers_facilitated', []))
        recruiting_strength = self._calculate_recruiting_network_strength(agent_data)
        reputation_score = self._calculate_agent_reputation(agent_data)
        
        return AgentProfile(
            agent_id=agent_id,
            name=name,
            agency=agency,
            influence_level=influence_level,
            client_count=client_count,
            client_roster=agent_data.get('clients', []),
            coaching_connections=coaching_connections,
            nil_deal_volume=nil_deal_volume,
            transfer_portal_activity=transfer_activity,
            recruiting_network_strength=recruiting_strength,
            reputation_score=reputation_score
        )
    
    def _calculate_agent_influence_level(self, agent_data: Dict) -> float:
        """Calculate overall agent influence in college basketball."""
        factors = {
            'client_quality': len([c for c in agent_data.get('clients', []) if agent_data.get('client_ratings', {}).get(c, 0) > 0.7]) / max(1, len(agent_data.get('clients', []))),
            'nil_success_rate': agent_data.get('successful_nil_deals', 0) / max(1, agent_data.get('attempted_nil_deals', 1)),
            'nba_placement_rate': agent_data.get('nba_placements', 0) / max(1, agent_data.get('eligible_clients', 1)),
            'coaching_relationships': len(agent_data.get('coaching_connections', {})) / 100.0,  # Normalize by ~100 major programs
            'media_presence': agent_data.get('media_mentions', 0) / 1000.0,  # Normalize
            'agency_reputation': agent_data.get('agency_power_ranking', 50) / 100.0
        }
        
        # Weight factors
        weights = {'client_quality': 0.3, 'nil_success_rate': 0.25, 'nba_placement_rate': 0.2, 
                  'coaching_relationships': 0.15, 'media_presence': 0.05, 'agency_reputation': 0.05}
        
        influence = sum(factors[factor] * weights[factor] for factor in factors)
        return min(1.0, influence)
    
    def _analyze_coaching_network(self, agent_data: Dict) -> Dict[str, float]:
        """Analyze agent's relationships with coaches."""
        coaching_connections = {}
        raw_connections = agent_data.get('coaching_relationships', {})
        
        for coach_id, relationship_data in raw_connections.items():
            # Calculate relationship strength
            factors = {
                'years_known': min(1.0, relationship_data.get('years_known', 0) / 10.0),
                'successful_collaborations': min(1.0, relationship_data.get('successful_deals', 0) / 5.0),
                'mutual_trust': relationship_data.get('trust_rating', 0.5),
                'recruiting_collaboration': relationship_data.get('recruiting_collaboration', 0.5),
                'conflict_history': 1.0 - relationship_data.get('conflicts', 0) / 10.0
            }
            
            relationship_strength = np.mean(list(factors.values()))
            coaching_connections[coach_id] = max(0.0, min(1.0, relationship_strength))
        
        return coaching_connections
    
    def _calculate_recruiting_network_strength(self, agent_data: Dict) -> float:
        """Calculate strength of agent's recruiting network."""
        network_factors = {
            'aau_connections': len(agent_data.get('aau_connections', [])) / 50.0,  # Normalize
            'high_school_relationships': len(agent_data.get('hs_coach_relationships', [])) / 100.0,
            'international_reach': agent_data.get('international_connections', 0) / 20.0,
            'family_advisor_network': len(agent_data.get('family_connections', [])) / 30.0,
            'social_media_influence': agent_data.get('social_media_reach', 0) / 100000.0
        }
        
        return min(1.0, np.mean(list(network_factors.values())))
    
    def _calculate_agent_reputation(self, agent_data: Dict) -> float:
        """Calculate agent's reputation score."""
        reputation_factors = {
            'client_satisfaction': agent_data.get('avg_client_satisfaction', 0.5),
            'industry_respect': agent_data.get('peer_ratings', 0.5),
            'ethical_record': 1.0 - agent_data.get('violations', 0) / 10.0,
            'success_stories': min(1.0, agent_data.get('success_stories', 0) / 10.0),
            'longevity': min(1.0, agent_data.get('years_in_business', 0) / 15.0)
        }
        
        return np.mean(list(reputation_factors.values()))
    
    def analyze_player_agent_relationship(self, player_data: Dict, agent_data: Dict) -> PlayerAgentRelationship:
        """Analyze specific player-agent relationship dynamics."""
        player_id = player_data.get('player_id', 'unknown')
        agent_id = agent_data.get('agent_id', 'unknown')
        
        # Relationship timeline
        relationship_start = datetime.fromisoformat(player_data.get('agent_start_date', '2021-07-01'))
        
        # Relationship strength factors
        relationship_strength = self._calculate_player_agent_bond(player_data, agent_data)
        decision_influence = self._calculate_agent_decision_influence(player_data, agent_data)
        nil_involvement = self._calculate_nil_involvement(player_data, agent_data)
        transfer_influence = self._calculate_transfer_influence(player_data, agent_data)
        family_dynamics = self._calculate_family_dynamics(player_data, agent_data)
        academic_involvement = player_data.get('agent_academic_involvement', 0.3)
        
        return PlayerAgentRelationship(
            player_id=player_id,
            agent_id=agent_id,
            relationship_start=relationship_start,
            relationship_strength=relationship_strength,
            influence_on_decisions=decision_influence,
            nil_involvement=nil_involvement,
            transfer_influence=transfer_influence,
            family_dynamics=family_dynamics,
            academic_involvement=academic_involvement
        )
    
    def _calculate_player_agent_bond(self, player_data: Dict, agent_data: Dict) -> float:
        """Calculate strength of player-agent relationship."""
        bond_factors = {
            'communication_frequency': player_data.get('agent_contact_frequency', 0.5),
            'trust_level': player_data.get('agent_trust_rating', 0.5),
            'shared_goals': player_data.get('goal_alignment_with_agent', 0.5),
            'personal_connection': player_data.get('personal_relationship_rating', 0.5),
            'professional_satisfaction': player_data.get('agent_performance_satisfaction', 0.5),
            'family_approval': player_data.get('family_approval_of_agent', 0.5)
        }
        
        return np.mean(list(bond_factors.values()))
    
    def _calculate_agent_decision_influence(self, player_data: Dict, agent_data: Dict) -> float:
        """Calculate how much agent influences player decisions."""
        influence_indicators = {
            'college_choice': player_data.get('agent_influenced_college_choice', 0.3),
            'playing_time_expectations': player_data.get('agent_influenced_pt_expectations', 0.4),
            'nba_timeline': player_data.get('agent_influenced_nba_timeline', 0.6),
            'transfer_considerations': player_data.get('agent_influenced_transfer_thoughts', 0.5),
            'nil_decisions': player_data.get('agent_influenced_nil_choices', 0.7),
            'media_interactions': player_data.get('agent_influenced_media', 0.8)
        }
        
        # Weight by player independence level
        independence_factor = player_data.get('player_independence_level', 0.5)
        base_influence = np.mean(list(influence_indicators.values()))
        
        # More independent players = less agent influence
        adjusted_influence = base_influence * (1 - independence_factor * 0.3)
        
        return max(0.0, min(1.0, adjusted_influence))
    
    def _calculate_nil_involvement(self, player_data: Dict, agent_data: Dict) -> float:
        """Calculate agent's involvement in NIL deals."""
        nil_factors = {
            'deal_negotiation': player_data.get('agent_nil_negotiation_involvement', 0.8),
            'opportunity_sourcing': player_data.get('agent_sources_nil_opportunities', 0.7),
            'contract_review': player_data.get('agent_reviews_nil_contracts', 0.9),
            'brand_strategy': player_data.get('agent_brand_strategy_involvement', 0.6),
            'financial_planning': player_data.get('agent_nil_financial_planning', 0.5)
        }
        
        return np.mean(list(nil_factors.values()))
    
    def _calculate_transfer_influence(self, player_data: Dict, agent_data: Dict) -> float:
        """Calculate agent's influence on transfer decisions."""
        transfer_factors = {
            'portal_discussions': player_data.get('agent_discussed_transfer_portal', 0.0),
            'opportunity_research': player_data.get('agent_researches_transfer_options', 0.0),
            'timing_advice': player_data.get('agent_advises_transfer_timing', 0.0),
            'program_evaluation': player_data.get('agent_evaluates_programs', 0.0),
            'family_counseling': player_data.get('agent_counsels_family_on_transfer', 0.0)
        }
        
        # Adjust for player class (upperclassmen more likely to consider transfers)
        class_multiplier = {
            'FR': 0.8,  # Freshmen less likely
            'SO': 1.0,  # Sophomores baseline
            'JR': 1.2,  # Juniors more likely
            'SR': 0.9   # Seniors less likely (graduation)
        }
        
        player_class = player_data.get('class', 'SO')
        multiplier = class_multiplier.get(player_class, 1.0)
        
        base_influence = np.mean(list(transfer_factors.values()))
        return min(1.0, base_influence * multiplier)
    
    def _calculate_family_dynamics(self, player_data: Dict, agent_data: Dict) -> float:
        """Calculate how agent interacts with player's family."""
        family_factors = {
            'family_relationship_quality': player_data.get('agent_family_relationship', 0.5),
            'family_trust': player_data.get('family_trusts_agent', 0.5),
            'family_communication': player_data.get('agent_family_communication_frequency', 0.4),
            'family_decision_inclusion': player_data.get('agent_includes_family_in_decisions', 0.6),
            'family_financial_discussions': player_data.get('agent_discusses_finances_with_family', 0.3)
        }
        
        return np.mean(list(family_factors.values()))
    
    def analyze_coach_player_dynamics(self, player_data: Dict, coach_data: Dict) -> CoachPlayerDynamics:
        """Analyze coach-player relationship dynamics."""
        player_id = player_data.get('player_id', 'unknown')
        coach_id = coach_data.get('coach_id', 'unknown')
        
        # Relationship quality factors
        relationship_quality = self._calculate_coach_player_relationship_quality(player_data, coach_data)
        trust_level = player_data.get('trust_in_coach', 0.5)
        playing_time_satisfaction = player_data.get('playing_time_satisfaction', 0.5)
        development_satisfaction = player_data.get('development_satisfaction', 0.5)
        communication_quality = self._calculate_communication_quality(player_data, coach_data)
        
        # Conflict history
        conflict_history = player_data.get('coach_conflicts', [])
        
        # NIL and future planning alignment
        nil_alignment = self._calculate_nil_alignment(player_data, coach_data)
        future_planning_alignment = self._calculate_future_planning_alignment(player_data, coach_data)
        
        return CoachPlayerDynamics(
            player_id=player_id,
            coach_id=coach_id,
            relationship_quality=relationship_quality,
            trust_level=trust_level,
            playing_time_satisfaction=playing_time_satisfaction,
            development_satisfaction=development_satisfaction,
            communication_quality=communication_quality,
            conflict_history=conflict_history,
            nil_alignment=nil_alignment,
            future_planning_alignment=future_planning_alignment
        )
    
    def _calculate_coach_player_relationship_quality(self, player_data: Dict, coach_data: Dict) -> float:
        """Calculate overall coach-player relationship quality."""
        quality_factors = {
            'mutual_respect': player_data.get('respects_coach', 0.5) * coach_data.get('respects_player', 0.5),
            'communication_openness': player_data.get('open_communication_with_coach', 0.5),
            'shared_basketball_philosophy': player_data.get('aligns_with_coach_philosophy', 0.5),
            'personal_connection': player_data.get('personal_connection_with_coach', 0.4),
            'development_alignment': player_data.get('coach_develops_player_well', 0.5),
            'accountability_acceptance': player_data.get('accepts_coach_accountability', 0.6)
        }
        
        return np.mean(list(quality_factors.values()))
    
    def _calculate_communication_quality(self, player_data: Dict, coach_data: Dict) -> float:
        """Calculate quality of coach-player communication."""
        communication_factors = {
            'frequency': player_data.get('coach_communication_frequency', 0.5),
            'clarity': player_data.get('coach_communication_clarity', 0.5),
            'constructiveness': player_data.get('coach_feedback_constructiveness', 0.5),
            'listening': player_data.get('coach_listens_to_player', 0.5),
            'transparency': player_data.get('coach_transparency', 0.5)
        }
        
        return np.mean(list(communication_factors.values()))
    
    def _calculate_nil_alignment(self, player_data: Dict, coach_data: Dict) -> float:
        """Calculate how aligned coach and player are on NIL approach."""
        nil_factors = {
            'nil_philosophy_alignment': player_data.get('nil_philosophy_matches_coach', 0.5),
            'nil_time_management': player_data.get('coach_helps_manage_nil_time', 0.5),
            'nil_distraction_management': coach_data.get('manages_player_nil_distractions', 0.5),
            'nil_opportunity_support': coach_data.get('supports_player_nil_opportunities', 0.5),
            'nil_team_impact_discussion': player_data.get('coach_discusses_nil_team_impact', 0.4)
        }
        
        return np.mean(list(nil_factors.values()))
    
    def _calculate_future_planning_alignment(self, player_data: Dict, coach_data: Dict) -> float:
        """Calculate alignment on future plans (NBA, transfers, etc.)."""
        planning_factors = {
            'nba_timeline_agreement': player_data.get('coach_agrees_nba_timeline', 0.5),
            'development_plan_alignment': player_data.get('aligns_with_development_plan', 0.5),
            'playing_time_expectations': player_data.get('coach_meets_pt_expectations', 0.5),
            'role_clarity': player_data.get('coach_clear_about_role', 0.5),
            'honest_feedback': player_data.get('coach_honest_about_prospects', 0.5)
        }
        
        return np.mean(list(planning_factors.values()))
    
    def analyze_team_chemistry(self, team_data: Dict, player_relationships: List[Dict], 
                             coach_relationships: List[CoachPlayerDynamics]) -> TeamChemistryProfile:
        """Analyze overall team chemistry influenced by relationships."""
        team_id = team_data.get('team_id', 'unknown')
        
        # Calculate chemistry components
        overall_chemistry = self._calculate_overall_team_chemistry(team_data, player_relationships)
        agent_influence = self._calculate_team_agent_influence(player_relationships)
        nil_tension = self._calculate_nil_tension_level(team_data, player_relationships)
        transfer_risk = self._calculate_transfer_portal_risk(player_relationships, coach_relationships)
        coaching_stability = self._calculate_coaching_stability(team_data, coach_relationships)
        veteran_leadership = self._calculate_veteran_leadership(team_data, player_relationships)
        freshmen_integration = self._calculate_freshmen_integration(team_data, player_relationships)
        playing_time_satisfaction = self._calculate_team_pt_satisfaction(coach_relationships)
        
        return TeamChemistryProfile(
            team_id=team_id,
            overall_chemistry_score=overall_chemistry,
            agent_influence_score=agent_influence,
            nil_tension_level=nil_tension,
            transfer_portal_risk=transfer_risk,
            coaching_stability=coaching_stability,
            veteran_leadership_strength=veteran_leadership,
            freshmen_integration_quality=freshmen_integration,
            playing_time_distribution_satisfaction=playing_time_satisfaction
        )
    
    def _calculate_overall_team_chemistry(self, team_data: Dict, player_relationships: List[Dict]) -> float:
        """Calculate overall team chemistry score."""
        chemistry_indicators = {
            'team_cohesion_rating': team_data.get('team_cohesion', 0.5),
            'locker_room_atmosphere': team_data.get('locker_room_rating', 0.5),
            'on_court_chemistry': team_data.get('on_court_chemistry', 0.5),
            'conflict_frequency': 1.0 - min(1.0, team_data.get('team_conflicts', 0) / 10.0),
            'leadership_quality': team_data.get('team_leadership_rating', 0.5),
            'shared_goals': team_data.get('shared_team_goals', 0.5)
        }
        
        # Adjust for relationship factors
        relationship_adjustment = np.mean([rel.get('relationship_impact_on_team', 0.5) for rel in player_relationships])
        base_chemistry = np.mean(list(chemistry_indicators.values()))
        
        return (base_chemistry * 0.7) + (relationship_adjustment * 0.3)
    
    def _calculate_team_agent_influence(self, player_relationships: List[Dict]) -> float:
        """Calculate how much agents collectively influence the team."""
        if not player_relationships:
            return 0.0
        
        agent_influences = []
        for relationship in player_relationships:
            agent_influence = relationship.get('agent_team_influence', 0.0)
            player_importance = relationship.get('player_importance_to_team', 0.5)
            weighted_influence = agent_influence * player_importance
            agent_influences.append(weighted_influence)
        
        return np.mean(agent_influences)
    
    def _calculate_nil_tension_level(self, team_data: Dict, player_relationships: List[Dict]) -> float:
        """Calculate NIL-related tension within team."""
        nil_factors = {
            'nil_disparity': team_data.get('nil_earning_disparity', 0.0),
            'nil_jealousy': team_data.get('nil_jealousy_incidents', 0),
            'nil_time_conflicts': team_data.get('nil_time_conflicts', 0),
            'nil_media_distractions': team_data.get('nil_media_distractions', 0),
            'nil_team_divide': team_data.get('nil_creates_team_divide', 0.0)
        }
        
        # Normalize conflicts/incidents
        tension_score = (
            nil_factors['nil_disparity'] * 0.3 +
            min(1.0, nil_factors['nil_jealousy'] / 5.0) * 0.2 +
            min(1.0, nil_factors['nil_time_conflicts'] / 10.0) * 0.2 +
            min(1.0, nil_factors['nil_media_distractions'] / 8.0) * 0.15 +
            nil_factors['nil_team_divide'] * 0.15
        )
        
        return min(1.0, tension_score)
    
    def _calculate_transfer_portal_risk(self, player_relationships: List[Dict], 
                                      coach_relationships: List[CoachPlayerDynamics]) -> float:
        """Calculate risk of players entering transfer portal."""
        risk_factors = []
        
        for i, player_rel in enumerate(player_relationships):
            coach_rel = coach_relationships[i] if i < len(coach_relationships) else None
            
            player_risk = 0.0
            
            # Agent influence on transfers
            agent_transfer_influence = player_rel.get('agent_transfer_influence', 0.0)
            player_risk += agent_transfer_influence * 0.3
            
            # Coach relationship quality
            if coach_rel:
                poor_relationship = 1.0 - coach_rel.relationship_quality
                playing_time_dissatisfaction = 1.0 - coach_rel.playing_time_satisfaction
                player_risk += (poor_relationship * 0.25 + playing_time_dissatisfaction * 0.25)
            
            # NIL considerations
            nil_better_elsewhere = player_rel.get('nil_opportunities_elsewhere', 0.0)
            player_risk += nil_better_elsewhere * 0.2
            
            risk_factors.append(min(1.0, player_risk))
        
        return np.mean(risk_factors) if risk_factors else 0.0
    
    def _calculate_coaching_stability(self, team_data: Dict, coach_relationships: List[CoachPlayerDynamics]) -> float:
        """Calculate coaching staff stability."""
        stability_factors = {
            'coach_job_security': team_data.get('coach_job_security', 0.5),
            'player_coach_relationships': np.mean([rel.relationship_quality for rel in coach_relationships]) if coach_relationships else 0.5,
            'coaching_staff_turnover': 1.0 - min(1.0, team_data.get('assistant_turnover', 0) / 3.0),
            'administration_support': team_data.get('admin_support_for_coach', 0.5),
            'recruiting_success': team_data.get('recruiting_class_rating', 0.5)
        }
        
        return np.mean(list(stability_factors.values()))
    
    def _calculate_veteran_leadership(self, team_data: Dict, player_relationships: List[Dict]) -> float:
        """Calculate strength of veteran leadership."""
        leadership_factors = {
            'veteran_player_count': min(1.0, team_data.get('upperclassmen_count', 0) / 8.0),
            'leadership_quality': team_data.get('veteran_leadership_rating', 0.5),
            'mentor_relationships': team_data.get('veteran_mentorship_rating', 0.5),
            'team_culture_strength': team_data.get('team_culture_rating', 0.5),
            'leadership_stability': 1.0 - team_data.get('leadership_turnover_risk', 0.0)
        }
        
        return np.mean(list(leadership_factors.values()))
    
    def _calculate_freshmen_integration(self, team_data: Dict, player_relationships: List[Dict]) -> float:
        """Calculate how well freshmen are integrated."""
        integration_factors = {
            'freshmen_comfort_level': team_data.get('freshmen_comfort_rating', 0.5),
            'veteran_mentorship': team_data.get('veteran_helps_freshmen', 0.5),
            'playing_time_adjustment': team_data.get('freshmen_pt_adjustment', 0.5),
            'academic_integration': team_data.get('freshmen_academic_support', 0.5),
            'social_integration': team_data.get('freshmen_social_integration', 0.5)
        }
        
        return np.mean(list(integration_factors.values()))
    
    def _calculate_team_pt_satisfaction(self, coach_relationships: List[CoachPlayerDynamics]) -> float:
        """Calculate team-wide playing time satisfaction."""
        if not coach_relationships:
            return 0.5
        
        pt_satisfactions = [rel.playing_time_satisfaction for rel in coach_relationships]
        return np.mean(pt_satisfactions)
    
    def analyze_nil_impact(self, team_data: Dict, player_nil_data: List[Dict]) -> NILImpactAnalysis:
        """Analyze NIL impact on team dynamics and performance."""
        team_id = team_data.get('team_id', 'unknown')
        
        # Calculate NIL metrics
        total_nil_value = sum(player.get('nil_value', 0) for player in player_nil_data)
        nil_distribution_equity = self._calculate_nil_distribution_equity(player_nil_data)
        nil_performance_correlation = self._calculate_nil_performance_correlation(player_nil_data)
        nil_distraction_level = self._calculate_nil_distraction_level(team_data, player_nil_data)
        market_leverage = self._calculate_market_leverage(team_data)
        coaching_nil_management = team_data.get('coach_nil_management_rating', 0.5)
        
        return NILImpactAnalysis(
            team_id=team_id,
            total_nil_value=total_nil_value,
            nil_distribution_equity=nil_distribution_equity,
            nil_performance_correlation=nil_performance_correlation,
            nil_distraction_level=nil_distraction_level,
            market_leverage=market_leverage,
            coaching_nil_management=coaching_nil_management
        )
    
    def _calculate_nil_distribution_equity(self, player_nil_data: List[Dict]) -> float:
        """Calculate how equitably NIL money is distributed."""
        nil_values = [player.get('nil_value', 0) for player in player_nil_data]
        
        if not nil_values or max(nil_values) == 0:
            return 1.0  # Perfect equity if no NIL money
        
        # Calculate coefficient of variation (lower = more equitable)
        mean_nil = np.mean(nil_values)
        std_nil = np.std(nil_values)
        
        if mean_nil == 0:
            return 1.0
        
        cv = std_nil / mean_nil
        equity = 1.0 - min(1.0, cv / 2.0)  # Normalize CV to 0-1 scale
        
        return max(0.0, equity)
    
    def _calculate_nil_performance_correlation(self, player_nil_data: List[Dict]) -> float:
        """Calculate correlation between NIL money and performance."""
        nil_values = [player.get('nil_value', 0) for player in player_nil_data]
        performance_ratings = [player.get('performance_rating', 0.5) for player in player_nil_data]
        
        if len(nil_values) < 3:  # Need minimum data for correlation
            return 0.0
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(nil_values, performance_ratings)[0, 1]
        
        # Handle NaN case
        if np.isnan(correlation):
            return 0.0
        
        # Return absolute correlation (strength of relationship)
        return abs(correlation)
    
    def _calculate_nil_distraction_level(self, team_data: Dict, player_nil_data: List[Dict]) -> float:
        """Calculate how much NIL distracts from basketball."""
        distraction_factors = {
            'missed_practices': min(1.0, team_data.get('nil_related_missed_practices', 0) / 10.0),
            'media_time': min(1.0, team_data.get('nil_media_time_weekly_hours', 0) / 20.0),
            'academic_impact': team_data.get('nil_academic_impact_rating', 0.0),
            'team_chemistry_impact': team_data.get('nil_chemistry_impact', 0.0),
            'focus_issues': team_data.get('nil_focus_issues', 0.0)
        }
        
        return np.mean(list(distraction_factors.values()))
    
    def _calculate_market_leverage(self, team_data: Dict) -> float:
        """Calculate team's NIL market advantages."""
        market_factors = {
            'market_size': team_data.get('market_size_rating', 0.5),
            'fan_base_size': team_data.get('fan_base_rating', 0.5),
            'media_coverage': team_data.get('media_coverage_rating', 0.5),
            'corporate_partnerships': team_data.get('corporate_partnership_rating', 0.5),
            'brand_strength': team_data.get('program_brand_strength', 0.5),
            'social_media_reach': team_data.get('social_media_rating', 0.5)
        }
        
        return np.mean(list(market_factors.values()))
    
    def backtest_nil_era_impact(self, historical_data: Dict, start_date: datetime = None) -> Dict[str, Dict]:
        """Backtest relationship impacts from NIL era start."""
        if start_date is None:
            start_date = self.nil_start_date
        
        logger.info(f"Backtesting relationship impacts from {start_date.strftime('%Y-%m-%d')}")
        
        # Split data into pre-NIL and post-NIL periods
        pre_nil_data = {
            season: data for season, data in historical_data.items() 
            if datetime.fromisoformat(data.get('season_start', '2020-11-01')) < start_date
        }
        
        post_nil_data = {
            season: data for season, data in historical_data.items() 
            if datetime.fromisoformat(data.get('season_start', '2022-11-01')) >= start_date
        }
        
        # Analyze changes
        backtest_results = {
            'pre_nil_metrics': self._analyze_era_metrics(pre_nil_data),
            'post_nil_metrics': self._analyze_era_metrics(post_nil_data),
            'nil_impact_analysis': self._calculate_nil_era_changes(pre_nil_data, post_nil_data),
            'betting_implications': self._calculate_betting_implications(pre_nil_data, post_nil_data)
        }
        
        return backtest_results
    
    def _analyze_era_metrics(self, era_data: Dict) -> Dict[str, float]:
        """Analyze key metrics for a specific era."""
        if not era_data:
            return {}
        
        metrics = {
            'avg_transfer_rate': np.mean([data.get('transfer_rate', 0) for data in era_data.values()]),
            'avg_coaching_turnover': np.mean([data.get('coaching_turnover', 0) for data in era_data.values()]),
            'avg_team_chemistry': np.mean([data.get('team_chemistry', 0.5) for data in era_data.values()]),
            'avg_upset_rate': np.mean([data.get('upset_rate', 0.15) for data in era_data.values()]),
            'avg_home_court_advantage': np.mean([data.get('home_court_advantage', 0.06) for data in era_data.values()]),
            'avg_prediction_accuracy': np.mean([data.get('prediction_accuracy', 0.55) for data in era_data.values()])
        }
        
        return metrics
    
    def _calculate_nil_era_changes(self, pre_nil_data: Dict, post_nil_data: Dict) -> Dict[str, float]:
        """Calculate the changes brought by the NIL era."""
        pre_metrics = self._analyze_era_metrics(pre_nil_data)
        post_metrics = self._analyze_era_metrics(post_nil_data)
        
        changes = {}
        for metric in pre_metrics:
            if metric in post_metrics:
                change = post_metrics[metric] - pre_metrics[metric]
                pct_change = (change / pre_metrics[metric] * 100) if pre_metrics[metric] != 0 else 0
                changes[f'{metric}_absolute_change'] = change
                changes[f'{metric}_percent_change'] = pct_change
        
        return changes
    
    def _calculate_betting_implications(self, pre_nil_data: Dict, post_nil_data: Dict) -> Dict[str, str]:
        """Calculate betting strategy implications from NIL era changes."""
        implications = {}
        
        pre_metrics = self._analyze_era_metrics(pre_nil_data)
        post_metrics = self._analyze_era_metrics(post_nil_data)
        
        # Transfer rate implications
        transfer_increase = post_metrics.get('avg_transfer_rate', 0) - pre_metrics.get('avg_transfer_rate', 0)
        if transfer_increase > 0.1:
            implications['roster_volatility'] = "High roster volatility increases upset potential and reduces model reliability"
        
        # Chemistry implications
        chemistry_change = post_metrics.get('avg_team_chemistry', 0.5) - pre_metrics.get('avg_team_chemistry', 0.5)
        if chemistry_change < -0.05:
            implications['team_chemistry'] = "Decreased team chemistry creates more unpredictable outcomes"
        
        # Home court advantage
        hca_change = post_metrics.get('avg_home_court_advantage', 0.06) - pre_metrics.get('avg_home_court_advantage', 0.06)
        if hca_change < -0.01:
            implications['home_court'] = "Reduced home court advantage affects spread betting strategies"
        
        return implications
    
    def generate_relationship_insights(self, team_chemistry: TeamChemistryProfile, 
                                     nil_analysis: NILImpactAnalysis) -> Dict[str, str]:
        """Generate actionable insights from relationship analysis."""
        insights = {}
        
        # Team chemistry insights
        if team_chemistry.overall_chemistry_score > 0.75:
            insights['chemistry_strength'] = "Exceptional team chemistry provides stable performance and reduced upset risk"
        elif team_chemistry.overall_chemistry_score < 0.4:
            insights['chemistry_concern'] = "Poor team chemistry creates volatility and upset vulnerability"
        
        # Agent influence insights
        if team_chemistry.agent_influence_score > 0.6:
            insights['agent_risk'] = "High agent influence increases transfer risk and mid-season disruption potential"
        
        # NIL insights
        if nil_analysis.nil_distraction_level > 0.6:
            insights['nil_distraction'] = "High NIL distraction level may impact performance in crucial games"
        
        if nil_analysis.nil_distribution_equity < 0.3:
            insights['nil_inequality'] = "Severe NIL inequality creates team tension and chemistry issues"
        
        # Transfer portal risk
        if team_chemistry.transfer_portal_risk > 0.7:
            insights['transfer_risk'] = "High transfer portal risk suggests potential mid-season roster changes"
        
        return insights

def main():
    """Demo the agent-player-coach relationship analysis system."""
    print("College Basketball Agent-Player-Coach Relationship Analysis")
    print("=" * 60)
    print(f"Analyzing post-NIL era from: {NIL_START_DATE.strftime('%B %d, %Y')}")
    
    # Sample data for demo
    sample_agent_data = {
        'agent_id': 'agent_001',
        'name': 'Marcus Thompson',
        'agency': 'Elite Sports Management',
        'clients': ['player_001', 'player_002'],
        'total_nil_facilitated': 250000,
        'successful_nil_deals': 8,
        'attempted_nil_deals': 10,
        'nba_placements': 2,
        'eligible_clients': 5,
        'coaching_relationships': {
            'coach_001': {'years_known': 5, 'successful_deals': 3, 'trust_rating': 0.8}
        }
    }
    
    sample_player_data = {
        'player_id': 'player_001',
        'agent_start_date': '2021-08-01',
        'agent_trust_rating': 0.8,
        'agent_influenced_nil_choices': 0.9,
        'class': 'JR',
        'trust_in_coach': 0.7,
        'playing_time_satisfaction': 0.6
    }
    
    sample_coach_data = {
        'coach_id': 'coach_001',
        'respects_player': 0.8,
        'manages_player_nil_distractions': 0.6
    }
    
    sample_team_data = {
        'team_id': 'duke_001',
        'team_cohesion': 0.7,
        'nil_earning_disparity': 0.4,
        'coach_nil_management_rating': 0.8,
        'upperclassmen_count': 4
    }
    
    # Initialize analyzer
    analyzer = AgentPlayerCoachAnalyzer()
    
    try:
        # Analyze agent influence
        print("\n" + "="*30)
        print("Agent Analysis:")
        agent_profile = analyzer.analyze_agent_influence(sample_agent_data)
        print(f"  Agent: {agent_profile.name}")
        print(f"  Influence Level: {agent_profile.influence_level:.3f}")
        print(f"  NIL Deal Volume: ${agent_profile.nil_deal_volume:,}")
        print(f"  Client Count: {agent_profile.client_count}")
        
        # Analyze player-agent relationship
        print("\n" + "="*30)
        print("Player-Agent Relationship:")
        player_agent_rel = analyzer.analyze_player_agent_relationship(sample_player_data, sample_agent_data)
        print(f"  Relationship Strength: {player_agent_rel.relationship_strength:.3f}")
        print(f"  Decision Influence: {player_agent_rel.influence_on_decisions:.3f}")
        print(f"  NIL Involvement: {player_agent_rel.nil_involvement:.3f}")
        print(f"  Transfer Influence: {player_agent_rel.transfer_influence:.3f}")
        
        # Analyze coach-player dynamics
        print("\n" + "="*30)
        print("Coach-Player Dynamics:")
        coach_player_dynamics = analyzer.analyze_coach_player_dynamics(sample_player_data, sample_coach_data)
        print(f"  Relationship Quality: {coach_player_dynamics.relationship_quality:.3f}")
        print(f"  Trust Level: {coach_player_dynamics.trust_level:.3f}")
        print(f"  Playing Time Satisfaction: {coach_player_dynamics.playing_time_satisfaction:.3f}")
        
        # Analyze team chemistry
        print("\n" + "="*30)
        print("Team Chemistry Analysis:")
        team_chemistry = analyzer.analyze_team_chemistry(
            sample_team_data, 
            [{'agent_team_influence': 0.4, 'player_importance_to_team': 0.8}], 
            [coach_player_dynamics]
        )
        print(f"  Overall Chemistry: {team_chemistry.overall_chemistry_score:.3f}")
        print(f"  Agent Influence: {team_chemistry.agent_influence_score:.3f}")
        print(f"  NIL Tension Level: {team_chemistry.nil_tension_level:.3f}")
        print(f"  Transfer Portal Risk: {team_chemistry.transfer_portal_risk:.3f}")
        
        # NIL impact analysis
        print("\n" + "="*30)
        print("NIL Impact Analysis:")
        nil_analysis = analyzer.analyze_nil_impact(
            sample_team_data, 
            [{'nil_value': 50000, 'performance_rating': 0.8}]
        )
        print(f"  Total NIL Value: ${nil_analysis.total_nil_value:,}")
        print(f"  Distribution Equity: {nil_analysis.nil_distribution_equity:.3f}")
        print(f"  Distraction Level: {nil_analysis.nil_distraction_level:.3f}")
        
        # Generate insights
        print("\n" + "="*30)
        print("Strategic Insights:")
        insights = analyzer.generate_relationship_insights(team_chemistry, nil_analysis)
        for category, insight in insights.items():
            print(f"  {category.replace('_', ' ').title()}: {insight}")
            
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        print(f"Analysis failed: {e}")

if __name__ == "__main__":
    main()