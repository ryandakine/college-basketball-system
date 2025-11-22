#!/usr/bin/env python3
"""
MCB Narrative Analyzer - Entertainment & Psychology Model
=========================================================

Analyzes college basketball games through the lens of entertainment value,
TV ratings, and "scripted" narratives that influence outcomes.

Core Principles:
1. Blue Blood Protection - Major programs get favorable treatment on National TV
2. Bubble Push - Teams with TV appeal get help in must-win scenarios
3. March Madness Script - Tournament needs upsets and Cinderella stories
4. Conference Tournament Drama - Desperate teams get the friendly whistle
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class NarrativeAnalysis:
    """Results of narrative analysis"""
    narrative_score: float  # -1.0 to 1.0 (negative favors away, positive favors home)
    confidence: float  # 0.0 to 1.0
    factors: List[str]  # Explanation of narrative factors
    blue_blood_boost: float
    bubble_push_boost: float
    cinderella_boost: float
    drama_boost: float


class MCBNarrativeAnalyzer:
    """Analyzes games through entertainment and psychology lens"""
    
    # Blue Blood programs that drive TV ratings
    BLUE_BLOODS = {
        'Duke', 'North Carolina', 'Kansas', 'Kentucky', 'UConn',
        'UCLA', 'Michigan State', 'Villanova', 'Gonzaga'
    }
    
    # National TV networks (high-profile broadcasts)
    NATIONAL_TV = {
        'ESPN', 'ESPN2', 'CBS', 'FOX', 'FS1', 'TNT', 'TBS'
    }
    
    # Conference networks (moderate profile)
    CONFERENCE_TV = {
        'ACC Network', 'SEC Network', 'Big Ten Network', 'Pac-12 Network'
    }
    
    def __init__(self):
        """Initialize narrative analyzer"""
        logger.info("MCB Narrative Analyzer initialized")
    
    def analyze_narrative(self, game_data: Dict) -> NarrativeAnalysis:
        """
        Analyze game through narrative lens
        
        Args:
            game_data: Dictionary with game information including:
                - home_team, away_team
                - broadcast (network name)
                - is_tournament (bool)
                - month (int, 1-12)
                - home_seed, away_seed (for tournament)
                - home_record, away_record (optional)
                
        Returns:
            NarrativeAnalysis with score and factors
        """
        home_team = game_data.get('home_team', '')
        away_team = game_data.get('away_team', '')
        broadcast = game_data.get('broadcast', '')
        is_tournament = game_data.get('is_tournament', False)
        month = game_data.get('month', datetime.now().month)
        
        factors = []
        total_boost = 0.0
        
        # 1. Blue Blood Protection Protocol
        blue_blood_boost = self._calculate_blue_blood_boost(
            home_team, away_team, broadcast, is_tournament, game_data
        )
        if blue_blood_boost != 0:
            factors.append(self._get_blue_blood_factor(home_team, away_team, blue_blood_boost, broadcast))
        total_boost += blue_blood_boost
        
        # 2. Bubble Push Narrative (Feb/March regular season)
        bubble_push_boost = 0.0
        if month in [2, 3] and not is_tournament:
            bubble_push_boost = self._calculate_bubble_push(home_team, away_team, game_data)
            if bubble_push_boost != 0:
                factors.append(self._get_bubble_factor(home_team, away_team, bubble_push_boost))
            total_boost += bubble_push_boost
        
        # 3. March Madness Cinderella Script
        cinderella_boost = 0.0
        if is_tournament and month == 3:
            cinderella_boost = self._calculate_cinderella_boost(game_data)
            if cinderella_boost != 0:
                factors.append(self._get_cinderella_factor(game_data, cinderella_boost))
            total_boost += cinderella_boost
        
        # 4. Conference Tournament Drama
        drama_boost = 0.0
        if game_data.get('is_conference_tournament', False):
            drama_boost = self._calculate_conference_drama(home_team, away_team, game_data)
            if drama_boost != 0:
                factors.append(self._get_drama_factor(home_team, away_team, drama_boost))
            total_boost += drama_boost
        
        # Calculate final narrative score (-1 to 1, positive favors home)
        narrative_score = max(-1.0, min(1.0, total_boost))
        
        # Confidence based on number of factors and broadcast profile
        confidence = self._calculate_confidence(factors, broadcast, is_tournament)
        
        return NarrativeAnalysis(
            narrative_score=narrative_score,
            confidence=confidence,
            factors=factors,
            blue_blood_boost=blue_blood_boost,
            bubble_push_boost=bubble_push_boost,
            cinderella_boost=cinderella_boost,
            drama_boost=drama_boost
        )
    
    def _calculate_blue_blood_boost(
        self, home_team: str, away_team: str, broadcast: str, 
        is_tournament: bool, game_data: Dict
    ) -> float:
        """Calculate Blue Blood protection boost"""
        home_is_blue = self._is_blue_blood(home_team)
        away_is_blue = self._is_blue_blood(away_team)
        
        # If both or neither are Blue Bloods, no boost
        if home_is_blue == away_is_blue:
            return 0.0
        
        # Base boost for Blue Blood
        base_boost = 0.15
        
        # Amplify on National TV
        if self._is_national_tv(broadcast):
            base_boost *= 1.5
        elif self._is_conference_tv(broadcast):
            base_boost *= 1.2
        
        # Amplify in tournament
        if is_tournament:
            base_boost *= 1.3
        
        # "Get Right" logic - if Blue Blood is struggling
        if home_is_blue:
            recent_losses = game_data.get('home_recent_losses', 0)
            if recent_losses >= 2:
                base_boost *= 1.4  # Needs a win to stay in narrative
        elif away_is_blue:
            recent_losses = game_data.get('away_recent_losses', 0)
            if recent_losses >= 2:
                base_boost *= 1.4
        
        # Return positive if home is Blue Blood, negative if away
        return base_boost if home_is_blue else -base_boost
    
    def _calculate_bubble_push(self, home_team: str, away_team: str, game_data: Dict) -> float:
        """Calculate bubble team boost in must-win scenarios"""
        home_on_bubble = game_data.get('home_on_bubble', False)
        away_on_bubble = game_data.get('away_on_bubble', False)
        
        # If both or neither on bubble, no boost
        if home_on_bubble == away_on_bubble:
            return 0.0
        
        # Base boost for bubble team
        boost = 0.20
        
        # Amplify if opponent is "safe" (already in tournament)
        if home_on_bubble and game_data.get('away_tournament_lock', False):
            boost *= 1.3
        elif away_on_bubble and game_data.get('home_tournament_lock', False):
            boost *= 1.3
        
        # Amplify for high-profile coaches or programs
        if home_on_bubble and self._has_tv_appeal(home_team):
            boost *= 1.2
        elif away_on_bubble and self._has_tv_appeal(away_team):
            boost *= 1.2
        
        return boost if home_on_bubble else -boost
    
    def _calculate_cinderella_boost(self, game_data: Dict) -> float:
        """Calculate Cinderella/upset boost in March Madness"""
        home_seed = game_data.get('home_seed', 0)
        away_seed = game_data.get('away_seed', 0)
        
        if not home_seed or not away_seed:
            return 0.0
        
        # Classic upset scenarios: 12-15 seeds vs 2-5 seeds
        seed_diff = abs(home_seed - away_seed)
        
        # No boost if seeds are close
        if seed_diff < 5:
            return 0.0
        
        # Determine underdog
        underdog_is_home = home_seed > away_seed
        underdog_seed = home_seed if underdog_is_home else away_seed
        favorite_seed = away_seed if underdog_is_home else home_seed
        
        # Classic Cinderella range: 12-15 seed
        if 12 <= underdog_seed <= 15 and 2 <= favorite_seed <= 5:
            boost = 0.25  # Strong Cinderella boost
            
            # Amplify if favorite is vulnerable (recent struggles)
            if underdog_is_home:
                if game_data.get('away_recent_losses', 0) >= 2:
                    boost *= 1.3
            else:
                if game_data.get('home_recent_losses', 0) >= 2:
                    boost *= 1.3
            
            return boost if underdog_is_home else -boost
        
        # Moderate upset potential: 8-11 vs 1-4
        elif 8 <= underdog_seed <= 11 and 1 <= favorite_seed <= 4:
            boost = 0.15
            return boost if underdog_is_home else -boost
        
        return 0.0
    
    def _calculate_conference_drama(self, home_team: str, away_team: str, game_data: Dict) -> float:
        """Calculate conference tournament drama boost"""
        home_needs_bid = game_data.get('home_needs_autobid', False)
        away_needs_bid = game_data.get('away_needs_autobid', False)
        
        # If both or neither need the bid, no boost
        if home_needs_bid == away_needs_bid:
            return 0.0
        
        # Desperate team gets the boost
        boost = 0.18
        
        # Amplify if opponent is already safe
        if home_needs_bid and game_data.get('away_tournament_lock', False):
            boost *= 1.4
        elif away_needs_bid and game_data.get('home_tournament_lock', False):
            boost *= 1.4
        
        return boost if home_needs_bid else -boost
    
    def _calculate_confidence(self, factors: List[str], broadcast: str, is_tournament: bool) -> float:
        """Calculate confidence in narrative analysis"""
        base_confidence = 0.3
        
        # More factors = higher confidence
        base_confidence += len(factors) * 0.15
        
        # National TV increases confidence
        if self._is_national_tv(broadcast):
            base_confidence += 0.2
        elif self._is_conference_tv(broadcast):
            base_confidence += 0.1
        
        # Tournament games have higher narrative influence
        if is_tournament:
            base_confidence += 0.15
        
        return min(1.0, base_confidence)
    
    def _is_blue_blood(self, team: str) -> bool:
        """Check if team is a Blue Blood program"""
        return any(bb in team for bb in self.BLUE_BLOODS)
    
    def _is_national_tv(self, broadcast: str) -> bool:
        """Check if broadcast is National TV"""
        return any(net in broadcast for net in self.NATIONAL_TV)
    
    def _is_conference_tv(self, broadcast: str) -> bool:
        """Check if broadcast is Conference TV"""
        return any(net in broadcast for net in self.CONFERENCE_TV)
    
    def _has_tv_appeal(self, team: str) -> bool:
        """Check if team has high TV appeal (Blue Blood or major conference)"""
        # For now, just check Blue Blood status
        # Could expand to include major conference teams
        return self._is_blue_blood(team)
    
    def _get_blue_blood_factor(self, home: str, away: str, boost: float, broadcast: str) -> str:
        """Generate explanation for Blue Blood factor"""
        blue_blood = home if boost > 0 else away
        return f"Blue Blood Protection: {blue_blood} on {broadcast if broadcast else 'TV'} (+{abs(boost):.2f})"
    
    def _get_bubble_factor(self, home: str, away: str, boost: float) -> str:
        """Generate explanation for Bubble Push factor"""
        bubble_team = home if boost > 0 else away
        return f"Bubble Push: {bubble_team} in must-win scenario (+{abs(boost):.2f})"
    
    def _get_cinderella_factor(self, game_data: Dict, boost: float) -> str:
        """Generate explanation for Cinderella factor"""
        home_seed = game_data.get('home_seed', 0)
        away_seed = game_data.get('away_seed', 0)
        underdog_seed = max(home_seed, away_seed)
        return f"March Madness Script: {underdog_seed}-seed upset potential (+{abs(boost):.2f})"
    
    def _get_drama_factor(self, home: str, away: str, boost: float) -> str:
        """Generate explanation for Conference Drama factor"""
        desperate_team = home if boost > 0 else away
        return f"Conference Drama: {desperate_team} needs auto-bid (+{abs(boost):.2f})"


def main():
    """Test the narrative analyzer"""
    print("MCB Narrative Analyzer Test")
    print("=" * 50)
    
    analyzer = MCBNarrativeAnalyzer()
    
    # Test 1: Duke on National TV
    print("\n1. Duke on ESPN (Blue Blood Protection)")
    game1 = {
        'home_team': 'Duke Blue Devils',
        'away_team': 'Wake Forest Demon Deacons',
        'broadcast': 'ESPN',
        'is_tournament': False,
        'month': 2
    }
    result1 = analyzer.analyze_narrative(game1)
    print(f"   Narrative Score: {result1.narrative_score:+.3f}")
    print(f"   Confidence: {result1.confidence:.2f}")
    for factor in result1.factors:
        print(f"   - {factor}")
    
    # Test 2: Bubble team in March
    print("\n2. Bubble Team Must-Win (Bubble Push)")
    game2 = {
        'home_team': 'St. John\'s Red Storm',
        'away_team': 'Georgetown Hoyas',
        'broadcast': 'FS1',
        'is_tournament': False,
        'month': 3,
        'home_on_bubble': True,
        'away_tournament_lock': True
    }
    result2 = analyzer.analyze_narrative(game2)
    print(f"   Narrative Score: {result2.narrative_score:+.3f}")
    print(f"   Confidence: {result2.confidence:.2f}")
    for factor in result2.factors:
        print(f"   - {factor}")
    
    # Test 3: March Madness upset
    print("\n3. 12-seed vs 5-seed (Cinderella Script)")
    game3 = {
        'home_team': 'Oral Roberts Golden Eagles',
        'away_team': 'Ohio State Buckeyes',
        'broadcast': 'CBS',
        'is_tournament': True,
        'month': 3,
        'home_seed': 12,
        'away_seed': 5
    }
    result3 = analyzer.analyze_narrative(game3)
    print(f"   Narrative Score: {result3.narrative_score:+.3f}")
    print(f"   Confidence: {result3.confidence:.2f}")
    for factor in result3.factors:
        print(f"   - {factor}")
    
    print("\n✅ Narrative Analyzer operational!")


if __name__ == "__main__":
    main()
