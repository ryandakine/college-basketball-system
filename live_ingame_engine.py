#!/usr/bin/env python3
"""
Live In-Game Betting Engine for College Basketball
==================================================

Real-time in-game betting engine featuring:
- Live probability updates every possession
- Momentum shift detection and analysis
- Halftime adjustment modeling
- Timeout impact analysis
- Foul trouble and lineup change impacts
- Real-time line comparison and alerts
- Comeback probability modeling
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import sqlite3
from dataclasses import dataclass, asdict
import asyncio
import aiohttp
import time
from collections import deque
import statistics

@dataclass
class GameState:
    """Current game state"""
    game_id: str
    timestamp: datetime
    
    # Score and time
    home_score: int
    away_score: int
    time_remaining: int  # seconds
    period: int
    
    # Possession info
    possessing_team: str
    shot_clock: int
    
    # Foul situation
    home_team_fouls: int
    away_team_fouls: int
    home_bonus: bool
    away_bonus: bool
    
    # Player status
    home_players_on_court: List[str]
    away_players_on_court: List[str]
    foul_trouble: Dict[str, int]  # player -> foul count
    
    # Game flow
    last_play: str
    momentum_indicator: float  # -1 to 1 (away to home)
    pace_current: float  # possessions per 40 minutes
    
    # Betting context
    opening_spread: float
    current_spread: float
    opening_total: float
    current_total: float

@dataclass
class MomentumShift:
    """Detected momentum shift"""
    timestamp: datetime
    trigger_event: str
    shift_magnitude: float  # -1 to 1
    duration_estimate: int  # estimated duration in minutes
    probability_impact: float  # impact on win probability
    recommended_action: str

@dataclass
class LiveBettingOpportunity:
    """Live betting opportunity"""
    game_id: str
    timestamp: datetime
    bet_type: str
    recommendation: str
    
    # Current market
    current_odds: float
    fair_odds: float
    edge: float
    
    # Model predictions
    live_win_prob: float
    final_score_prediction: Tuple[int, int]
    comeback_probability: float
    
    # Context
    trigger_event: str
    confidence: float
    urgency: str  # Time-sensitive nature
    expected_line_movement: float

class LiveInGameEngine:
    """Real-time in-game betting analysis engine"""
    
    def __init__(self, db_path: str = "live_ingame.db"):
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        
        # Initialize database
        self._init_database()
        
        # Game state tracking
        self.active_games = {}  # game_id -> GameState
        self.momentum_history = {}  # game_id -> deque of momentum readings
        self.play_by_play = {}  # game_id -> list of plays
        
        # Model parameters
        self.MOMENTUM_WINDOW = 10  # plays to analyze for momentum
        self.SIGNIFICANT_RUN_THRESHOLD = 8  # points in a run
        self.TIMEOUT_MOMENTUM_DECAY = 0.5  # momentum decay after timeout
        
        # Betting thresholds
        self.MIN_LIVE_EDGE = 0.06  # 6% minimum edge for live bets
        self.URGENCY_THRESHOLD = 300  # 5 minutes in seconds
        
        # Historical models
        self.comeback_model = self._load_comeback_model()
        self.momentum_model = self._load_momentum_model()
    
    def _init_database(self):
        """Initialize live betting database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Game states
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS game_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT,
                timestamp DATETIME,
                home_score INTEGER,
                away_score INTEGER,
                time_remaining INTEGER,
                period INTEGER,
                possessing_team TEXT,
                momentum_indicator REAL,
                live_home_win_prob REAL,
                current_spread REAL,
                current_total REAL
            )
        ''')
        
        # Play by play
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS play_by_play (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT,
                timestamp DATETIME,
                time_remaining INTEGER,
                period INTEGER,
                play_description TEXT,
                home_score INTEGER,
                away_score INTEGER,
                momentum_impact REAL,
                possession_team TEXT
            )
        ''')
        
        # Live betting opportunities
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS live_opportunities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT,
                timestamp DATETIME,
                bet_type TEXT,
                recommendation TEXT,
                current_odds REAL,
                fair_odds REAL,
                edge REAL,
                live_win_prob REAL,
                trigger_event TEXT,
                confidence REAL,
                urgency TEXT,
                acted_upon BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Momentum shifts
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS momentum_shifts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT,
                timestamp DATETIME,
                trigger_event TEXT,
                shift_magnitude REAL,
                duration_estimate INTEGER,
                probability_impact REAL,
                recommended_action TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_comeback_model(self) -> Dict[str, float]:
        """Load historical comeback probability model"""
        # Simplified comeback model based on score differential and time
        # In reality, this would be trained on historical data
        return {
            'base_rates': {
                # Score differential -> comeback probability by time remaining
                (5, 600): 0.45,   # 5 points down, 10 minutes left
                (10, 600): 0.25,  # 10 points down, 10 minutes left
                (15, 600): 0.12,  # 15 points down, 10 minutes left
                (5, 300): 0.30,   # 5 points down, 5 minutes left
                (10, 300): 0.15,  # 10 points down, 5 minutes left
                (5, 120): 0.20,   # 5 points down, 2 minutes left
                (10, 120): 0.05,  # 10 points down, 2 minutes left
            },
            'momentum_multiplier': 1.5,  # Momentum can increase comeback odds
            'timeout_effect': 1.2,       # Fresh timeout strategy boost
            'foul_trouble_penalty': 0.8  # Key player foul trouble
        }
    
    def _load_momentum_model(self) -> Dict[str, float]:
        """Load momentum detection model"""
        return {
            'run_thresholds': {
                '8-0': 0.3,   # 8-0 run = 0.3 momentum
                '10-0': 0.5,  # 10-0 run = 0.5 momentum
                '12-0': 0.7,  # 12-0 run = 0.7 momentum
                '15-0': 0.9   # 15-0 run = 0.9 momentum
            },
            'play_weights': {
                'three_pointer': 0.15,
                'dunk': 0.12,
                'steal': 0.10,
                'block': 0.08,
                'turnover': -0.10,
                'missed_ft': -0.08,
                'technical_foul': -0.15
            },
            'decay_rate': 0.95,  # Momentum decays 5% per possession
            'timeout_reset': 0.6  # Timeouts reduce momentum to 60%
        }
    
    async def monitor_live_games(self, game_ids: List[str]):
        """Monitor multiple games simultaneously"""
        tasks = []
        for game_id in game_ids:
            tasks.append(self._monitor_single_game(game_id))
        
        await asyncio.gather(*tasks)
    
    async def _monitor_single_game(self, game_id: str):
        """Monitor a single game for live betting opportunities"""
        self.logger.info(f"Starting live monitoring for {game_id}")
        
        # Initialize game tracking
        self.momentum_history[game_id] = deque(maxlen=self.MOMENTUM_WINDOW)
        self.play_by_play[game_id] = []
        
        while True:
            try:
                # Fetch current game state
                game_state = await self._fetch_game_state(game_id)
                
                if not game_state or game_state.time_remaining <= 0:
                    break
                
                # Update game state
                self.active_games[game_id] = game_state
                
                # Detect momentum shifts
                momentum_shift = self._analyze_momentum(game_id, game_state)
                
                # Update live probabilities
                live_probs = self._calculate_live_probabilities(game_state)
                
                # Check for betting opportunities
                opportunities = self._identify_live_opportunities(
                    game_state, live_probs, momentum_shift
                )
                
                # Process opportunities
                for opp in opportunities:
                    self._process_live_opportunity(opp)
                
                # Store game state
                self._store_game_state(game_state, live_probs)
                
                # Wait before next update
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error monitoring {game_id}: {e}")
                await asyncio.sleep(30)
    
    async def _fetch_game_state(self, game_id: str) -> Optional[GameState]:
        """Fetch current game state from data source"""
        # This would integrate with real data sources like ESPN, NCAA, etc.
        # For demo, simulate game state
        
        current_time = datetime.now()
        
        # Simulate game progression
        if game_id not in self.active_games:
            # New game
            return GameState(
                game_id=game_id,
                timestamp=current_time,
                home_score=0,
                away_score=0,
                time_remaining=2400,  # 40 minutes
                period=1,
                possessing_team="home",
                shot_clock=30,
                home_team_fouls=0,
                away_team_fouls=0,
                home_bonus=False,
                away_bonus=False,
                home_players_on_court=["P1", "P2", "P3", "P4", "P5"],
                away_players_on_court=["A1", "A2", "A3", "A4", "A5"],
                foul_trouble={},
                last_play="Game Start",
                momentum_indicator=0.0,
                pace_current=70.0,
                opening_spread=-3.5,
                current_spread=-3.5,
                opening_total=155.5,
                current_total=155.5
            )
        
        # Update existing game (simulate progression)
        prev_state = self.active_games[game_id]
        
        # Simulate score changes and time progression
        time_elapsed = 30  # 30 seconds passed
        new_time = max(0, prev_state.time_remaining - time_elapsed)
        
        # Simulate scoring
        home_pts = np.random.poisson(0.5)  # Average 1 point per minute
        away_pts = np.random.poisson(0.5)
        
        new_home_score = prev_state.home_score + home_pts
        new_away_score = prev_state.away_score + away_pts
        
        return GameState(
            game_id=game_id,
            timestamp=current_time,
            home_score=new_home_score,
            away_score=new_away_score,
            time_remaining=new_time,
            period=min(2, 1 + (2400 - new_time) // 1200),
            possessing_team="away" if prev_state.possessing_team == "home" else "home",
            shot_clock=30,
            home_team_fouls=prev_state.home_team_fouls,
            away_team_fouls=prev_state.away_team_fouls,
            home_bonus=prev_state.home_bonus,
            away_bonus=prev_state.away_bonus,
            home_players_on_court=prev_state.home_players_on_court,
            away_players_on_court=prev_state.away_players_on_court,
            foul_trouble=prev_state.foul_trouble,
            last_play="Field Goal" if home_pts + away_pts > 0 else "Miss",
            momentum_indicator=prev_state.momentum_indicator,
            pace_current=prev_state.pace_current,
            opening_spread=prev_state.opening_spread,
            current_spread=prev_state.current_spread + np.random.normal(0, 0.5),
            opening_total=prev_state.opening_total,
            current_total=prev_state.current_total + np.random.normal(0, 1.0)
        )
    
    def _analyze_momentum(self, game_id: str, game_state: GameState) -> Optional[MomentumShift]:
        """Analyze momentum shifts in the game"""
        
        if game_id not in self.momentum_history:
            return None
        
        # Calculate current momentum
        momentum = self._calculate_momentum(game_state)
        
        # Add to history
        self.momentum_history[game_id].append({
            'timestamp': game_state.timestamp,
            'momentum': momentum,
            'score_diff': game_state.home_score - game_state.away_score,
            'play': game_state.last_play
        })
        
        # Detect significant momentum shifts
        if len(self.momentum_history[game_id]) < 5:
            return None
        
        recent_momentum = [m['momentum'] for m in list(self.momentum_history[game_id])[-5:]]
        momentum_change = recent_momentum[-1] - recent_momentum[0]
        
        # Significant momentum shift detected
        if abs(momentum_change) > 0.3:
            return MomentumShift(
                timestamp=game_state.timestamp,
                trigger_event=game_state.last_play,
                shift_magnitude=momentum_change,
                duration_estimate=3,  # Estimate 3 minutes
                probability_impact=momentum_change * 0.15,  # 15% impact per momentum point
                recommended_action="MONITOR" if abs(momentum_change) < 0.5 else "BET"
            )
        
        return None
    
    def _calculate_momentum(self, game_state: GameState) -> float:
        """Calculate current game momentum"""
        
        # Score differential momentum
        score_diff = game_state.home_score - game_state.away_score
        expected_diff = game_state.opening_spread * (1 - game_state.time_remaining / 2400)
        score_momentum = (score_diff - expected_diff) / 10  # Normalize
        
        # Play-based momentum (simplified)
        play_momentum = 0.0
        if game_state.last_play in self.momentum_model['play_weights']:
            play_momentum = self.momentum_model['play_weights'][game_state.last_play]
            if game_state.possessing_team == "away":
                play_momentum *= -1
        
        # Combined momentum
        total_momentum = score_momentum * 0.7 + play_momentum * 0.3
        
        return max(-1, min(1, total_momentum))
    
    def _calculate_live_probabilities(self, game_state: GameState) -> Dict[str, float]:
        """Calculate live win probabilities and game projections"""
        
        # Time factor
        time_factor = game_state.time_remaining / 2400
        
        # Score differential
        score_diff = game_state.home_score - game_state.away_score
        
        # Base probability from score and time
        if game_state.time_remaining > 0:
            # Using basketball-specific formula
            remaining_possessions = (game_state.time_remaining / 60) * (game_state.pace_current / 40)
            
            # Expected score variance based on remaining possessions
            variance = remaining_possessions * 2.5  # Average points per possession variance
            
            if variance > 0:
                z_score = score_diff / np.sqrt(variance)
                base_win_prob = norm.cdf(z_score)
            else:
                base_win_prob = 1.0 if score_diff > 0 else 0.0
        else:
            base_win_prob = 1.0 if score_diff > 0 else 0.0
        
        # Momentum adjustment
        momentum_adj = game_state.momentum_indicator * 0.10 * time_factor
        live_win_prob = max(0.01, min(0.99, base_win_prob + momentum_adj))
        
        # Comeback probability
        if score_diff < 0:  # Home team behind
            comeback_prob = self._calculate_comeback_probability(game_state)
        else:
            comeback_prob = 1 - self._calculate_comeback_probability(GameState(
                **{**asdict(game_state), 
                   'home_score': game_state.away_score, 
                   'away_score': game_state.home_score}
            ))
        
        # Final score prediction
        remaining_time_ratio = game_state.time_remaining / 2400
        remaining_points_home = (game_state.pace_current / 40) * (game_state.time_remaining / 60) * 1.0
        remaining_points_away = (game_state.pace_current / 40) * (game_state.time_remaining / 60) * 1.0
        
        final_home = game_state.home_score + remaining_points_home
        final_away = game_state.away_score + remaining_points_away
        
        return {
            'live_win_prob': live_win_prob,
            'comeback_prob': comeback_prob,
            'final_home_score': final_home,
            'final_away_score': final_away,
            'final_total': final_home + final_away,
            'final_spread': final_home - final_away
        }
    
    def _calculate_comeback_probability(self, game_state: GameState) -> float:
        """Calculate probability of comeback based on current state"""
        
        score_deficit = game_state.away_score - game_state.home_score
        time_remaining = game_state.time_remaining
        
        if score_deficit <= 0:
            return 0.0
        
        # Base comeback probability from historical model
        base_prob = 0.0
        for (deficit, time), prob in self.comeback_model['base_rates'].items():
            if abs(deficit - score_deficit) <= 2 and abs(time - time_remaining) <= 300:
                base_prob = prob
                break
        
        # Adjustments
        momentum_multiplier = 1.0
        if hasattr(game_state, 'momentum_indicator'):
            if game_state.momentum_indicator > 0:  # Home team has momentum
                momentum_multiplier = self.comeback_model['momentum_multiplier']
        
        # Foul trouble adjustment (simplified)
        foul_multiplier = 1.0
        if len(game_state.foul_trouble) > 2:  # Multiple players in foul trouble
            foul_multiplier = self.comeback_model['foul_trouble_penalty']
        
        comeback_prob = base_prob * momentum_multiplier * foul_multiplier
        
        return min(0.95, max(0.05, comeback_prob))
    
    def _identify_live_opportunities(self, game_state: GameState, live_probs: Dict[str, float], 
                                   momentum_shift: Optional[MomentumShift]) -> List[LiveBettingOpportunity]:
        """Identify live betting opportunities"""
        
        opportunities = []
        
        # Convert current spread to implied probability
        spread_implied_prob = self._spread_to_probability(game_state.current_spread)
        
        # Check spread opportunity
        if abs(live_probs['live_win_prob'] - spread_implied_prob) > self.MIN_LIVE_EDGE:
            edge = live_probs['live_win_prob'] - spread_implied_prob
            
            opportunities.append(LiveBettingOpportunity(
                game_id=game_state.game_id,
                timestamp=game_state.timestamp,
                bet_type="LIVE_SPREAD",
                recommendation="HOME" if edge > 0 else "AWAY",
                current_odds=-110,  # Standard odds
                fair_odds=self._probability_to_odds(live_probs['live_win_prob']),
                edge=abs(edge),
                live_win_prob=live_probs['live_win_prob'],
                final_score_prediction=(int(live_probs['final_home_score']), 
                                      int(live_probs['final_away_score'])),
                comeback_probability=live_probs['comeback_prob'],
                trigger_event=momentum_shift.trigger_event if momentum_shift else "Live Update",
                confidence=0.8 if momentum_shift and momentum_shift.shift_magnitude > 0.5 else 0.6,
                urgency="HIGH" if game_state.time_remaining < self.URGENCY_THRESHOLD else "MEDIUM",
                expected_line_movement=edge * 2  # Expect line to move toward fair value
            ))
        
        # Check total opportunity
        total_implied_prob = 0.5  # Simplified - would use actual total odds
        if abs(live_probs['final_total'] - game_state.current_total) > 3:
            
            opportunities.append(LiveBettingOpportunity(
                game_id=game_state.game_id,
                timestamp=game_state.timestamp,
                bet_type="LIVE_TOTAL",
                recommendation="OVER" if live_probs['final_total'] > game_state.current_total else "UNDER",
                current_odds=-110,
                fair_odds=-110,  # Simplified
                edge=abs(live_probs['final_total'] - game_state.current_total) / game_state.current_total,
                live_win_prob=live_probs['live_win_prob'],
                final_score_prediction=(int(live_probs['final_home_score']), 
                                      int(live_probs['final_away_score'])),
                comeback_probability=live_probs['comeback_prob'],
                trigger_event="Score Projection Update",
                confidence=0.7,
                urgency="MEDIUM",
                expected_line_movement=2.0
            ))
        
        return opportunities
    
    def _spread_to_probability(self, spread: float) -> float:
        """Convert spread to implied win probability"""
        # Basketball-specific conversion
        return 1 / (1 + np.exp(-spread / 3.5))
    
    def _probability_to_odds(self, probability: float) -> float:
        """Convert probability to American odds"""
        if probability >= 0.5:
            return -100 * probability / (1 - probability)
        else:
            return 100 * (1 - probability) / probability
    
    def _process_live_opportunity(self, opportunity: LiveBettingOpportunity):
        """Process and alert on live betting opportunity"""
        
        # Store opportunity
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO live_opportunities 
            (game_id, timestamp, bet_type, recommendation, current_odds, fair_odds, 
             edge, live_win_prob, trigger_event, confidence, urgency)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            opportunity.game_id,
            opportunity.timestamp,
            opportunity.bet_type,
            opportunity.recommendation,
            opportunity.current_odds,
            opportunity.fair_odds,
            opportunity.edge,
            opportunity.live_win_prob,
            opportunity.trigger_event,
            opportunity.confidence,
            opportunity.urgency
        ))
        
        conn.commit()
        conn.close()
        
        # Send alert
        self._send_live_alert(opportunity)
    
    def _send_live_alert(self, opportunity: LiveBettingOpportunity):
        """Send live betting alert"""
        
        urgency_emoji = "🚨" if opportunity.urgency == "HIGH" else "⚡" if opportunity.urgency == "MEDIUM" else "📊"
        
        message = f"""
{urgency_emoji} LIVE BETTING ALERT {urgency_emoji}

Game: {opportunity.game_id}
Bet: {opportunity.bet_type} - {opportunity.recommendation}
Edge: {opportunity.edge:.1%}
Confidence: {opportunity.confidence:.1%}
Live Win Prob: {opportunity.live_win_prob:.1%}
Comeback Prob: {opportunity.comeback_probability:.1%}

Trigger: {opportunity.trigger_event}
Expected Line Movement: {opportunity.expected_line_movement:+.1f}
Urgency: {opportunity.urgency}

Final Score Prediction: {opportunity.final_score_prediction[0]}-{opportunity.final_score_prediction[1]}
        """
        
        print(message)  # Replace with actual alerting system
        self.logger.info(f"Live alert: {opportunity.game_id} - {opportunity.bet_type} - {opportunity.edge:.1%} edge")
    
    def _store_game_state(self, game_state: GameState, live_probs: Dict[str, float]):
        """Store current game state"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO game_states 
            (game_id, timestamp, home_score, away_score, time_remaining, period,
             possessing_team, momentum_indicator, live_home_win_prob, current_spread, current_total)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            game_state.game_id,
            game_state.timestamp,
            game_state.home_score,
            game_state.away_score,
            game_state.time_remaining,
            game_state.period,
            game_state.possessing_team,
            game_state.momentum_indicator,
            live_probs['live_win_prob'],
            game_state.current_spread,
            game_state.current_total
        ))
        
        conn.commit()
        conn.close()
    
    def get_active_opportunities(self) -> List[Dict[str, Any]]:
        """Get current active live opportunities"""
        conn = sqlite3.connect(self.db_path)
        
        # Get opportunities from last 10 minutes that haven't been acted upon
        query = '''
            SELECT * FROM live_opportunities 
            WHERE timestamp >= ? AND acted_upon = FALSE
            ORDER BY timestamp DESC, edge DESC
        '''
        
        cutoff_time = datetime.now() - timedelta(minutes=10)
        df = pd.read_sql_query(query, conn, params=(cutoff_time,))
        conn.close()
        
        return df.to_dict('records')
    
    def mark_opportunity_acted_upon(self, opportunity_id: int):
        """Mark an opportunity as acted upon"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE live_opportunities 
            SET acted_upon = TRUE 
            WHERE id = ?
        ''', (opportunity_id,))
        
        conn.commit()
        conn.close()

# Testing and demonstration
async def demo_live_engine():
    """Demo the live in-game betting engine"""
    engine = LiveInGameEngine()
    
    print("🏀 Live In-Game Betting Engine Demo")
    print("=" * 50)
    
    # Monitor demo games
    demo_games = ["duke_unc_live", "kansas_kentucky_live"]
    
    print(f"Starting live monitoring for {len(demo_games)} games...")
    
    # Run for 2 minutes as demo
    try:
        await asyncio.wait_for(
            engine.monitor_live_games(demo_games),
            timeout=120
        )
    except asyncio.TimeoutError:
        print("Demo completed after 2 minutes")
    
    # Show opportunities found
    opportunities = engine.get_active_opportunities()
    
    print(f"\n📊 Live Opportunities Found: {len(opportunities)}")
    for opp in opportunities[:5]:  # Show top 5
        print(f"• {opp['game_id']}: {opp['bet_type']} - {opp['recommendation']} ({opp['edge']:.1%} edge)")
    
    print("\n✅ Live In-Game Engine ready!")

def main():
    """Main function for testing"""
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_live_engine())

if __name__ == "__main__":
    main()