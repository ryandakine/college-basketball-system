#!/usr/bin/env python3
"""
Advanced Betting Strategy Engine for College Basketball
======================================================

Sophisticated betting engine featuring:
- Kelly Criterion optimization with fractional sizing
- Dynamic bankroll management
- Risk assessment and portfolio theory
- Multi-objective optimization (ROI vs risk)
- Bet correlation analysis
- Advanced stake sizing algorithms
- Performance tracking and adjustment
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import sqlite3
from dataclasses import dataclass, asdict
from scipy.optimize import minimize
from scipy.stats import norm
import math

@dataclass
class BetOpportunity:
    """Individual betting opportunity"""
    game_id: str
    bet_type: str  # "SPREAD", "TOTAL", "MONEYLINE"
    side: str  # "HOME", "AWAY", "OVER", "UNDER"
    
    # Odds and value
    market_odds: float  # e.g., -110
    fair_odds: float    # Our calculated fair value
    implied_prob: float # Market implied probability
    true_prob: float   # Our calculated true probability
    edge: float        # Expected edge (%)
    
    # Confidence metrics
    confidence: float  # Model confidence (0-1)
    kelly_fraction: float  # Optimal Kelly bet size
    
    # Game context
    game_time: datetime
    expires_at: datetime
    sportsbook: str
    
    # Risk metrics
    variance: float
    correlation_risk: float
    liquidity_score: float

@dataclass
class BettingRecommendation:
    """Final betting recommendation with sizing"""
    opportunity: BetOpportunity
    
    # Sizing recommendations
    kelly_size: float
    fractional_kelly: float  # Conservative Kelly (usually 25-50% of full)
    risk_adjusted_size: float
    suggested_units: float
    max_units: float
    
    # Risk analysis
    expected_value: float
    risk_score: float
    sharpe_ratio: float
    max_drawdown_risk: float
    
    # Recommendation level
    action: str  # "STRONG_BET", "BET", "SMALL_BET", "AVOID"
    urgency: str  # "URGENT", "HIGH", "MEDIUM", "LOW"
    
    # Reasoning
    reasoning: List[str]
    risk_factors: List[str]

@dataclass
class BankrollState:
    """Current bankroll state and metrics"""
    current_bankroll: float
    starting_bankroll: float
    peak_bankroll: float
    
    # Performance metrics
    total_return: float
    roi_pct: float
    sharpe_ratio: float
    max_drawdown: float
    current_drawdown: float
    
    # Risk metrics
    volatility: float
    var_95: float  # Value at Risk (95% confidence)
    expected_shortfall: float
    
    # Betting statistics
    total_bets: int
    winning_bets: int
    losing_bets: int
    win_rate: float
    avg_odds: float
    
    # Recent performance (last 30 days)
    recent_roi: float
    recent_win_rate: float
    recent_units_won: float

@dataclass
class RiskParameters:
    """Risk management parameters"""
    max_bet_size: float = 0.05  # 5% of bankroll max
    kelly_fraction: float = 0.25  # 25% of full Kelly
    max_daily_risk: float = 0.10  # 10% daily risk limit
    max_correlation: float = 0.30  # Max correlation between bets
    min_edge_threshold: float = 0.04  # 4% minimum edge
    min_confidence: float = 0.60  # 60% minimum confidence
    max_simultaneous_bets: int = 10
    stop_loss_threshold: float = -0.20  # -20% stop loss

class AdvancedBettingEngine:
    """Advanced betting strategy engine"""
    
    def __init__(self, db_path: str = "betting_strategy.db", 
                 initial_bankroll: float = 10000.0):
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        
        # Initialize bankroll and risk parameters
        self.risk_params = RiskParameters()
        self.bankroll_state = BankrollState(
            current_bankroll=initial_bankroll,
            starting_bankroll=initial_bankroll,
            peak_bankroll=initial_bankroll,
            total_return=0.0,
            roi_pct=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            current_drawdown=0.0,
            volatility=0.0,
            var_95=0.0,
            expected_shortfall=0.0,
            total_bets=0,
            winning_bets=0,
            losing_bets=0,
            win_rate=0.0,
            avg_odds=0.0,
            recent_roi=0.0,
            recent_win_rate=0.0,
            recent_units_won=0.0
        )
        
        # Active bets and correlations
        self.active_bets = []
        self.correlation_matrix = {}
        
        # Initialize database
        self._init_database()
        
        # Load historical performance
        self._load_bankroll_state()
    
    def _init_database(self):
        """Initialize betting strategy database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Bet history
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bet_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                game_id TEXT,
                bet_type TEXT,
                side TEXT,
                stake_amount REAL,
                odds REAL,
                edge_pct REAL,
                kelly_fraction REAL,
                confidence REAL,
                result TEXT,  -- 'WIN', 'LOSS', 'PUSH', 'PENDING'
                profit_loss REAL,
                sportsbook TEXT
            )
        ''')
        
        # Bankroll snapshots
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bankroll_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                bankroll_amount REAL,
                total_return REAL,
                roi_pct REAL,
                max_drawdown REAL,
                current_drawdown REAL,
                sharpe_ratio REAL,
                total_bets INTEGER,
                win_rate REAL
            )
        ''')
        
        # Risk metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                var_95 REAL,
                expected_shortfall REAL,
                volatility REAL,
                correlation_risk REAL,
                max_exposure REAL,
                daily_risk REAL
            )
        ''')
        
        # Strategy performance
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT,
                timestamp DATETIME,
                bets_placed INTEGER,
                total_staked REAL,
                total_profit REAL,
                roi_pct REAL,
                win_rate REAL,
                avg_odds REAL,
                sharpe_ratio REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_bankroll_state(self):
        """Load current bankroll state from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get latest bankroll snapshot
        cursor.execute('''
            SELECT * FROM bankroll_history 
            ORDER BY timestamp DESC LIMIT 1
        ''')
        
        latest = cursor.fetchone()
        if latest:
            self.bankroll_state.current_bankroll = latest[2]
            self.bankroll_state.total_return = latest[3]
            self.bankroll_state.roi_pct = latest[4]
            self.bankroll_state.max_drawdown = latest[5]
            self.bankroll_state.current_drawdown = latest[6]
            self.bankroll_state.sharpe_ratio = latest[7]
            self.bankroll_state.total_bets = latest[8]
            self.bankroll_state.win_rate = latest[9]
        
        # Update peak bankroll
        cursor.execute('SELECT MAX(bankroll_amount) FROM bankroll_history')
        peak = cursor.fetchone()[0]
        if peak:
            self.bankroll_state.peak_bankroll = max(peak, self.bankroll_state.current_bankroll)
        
        conn.close()
    
    def analyze_betting_opportunity(self, market_odds: float, true_probability: float,
                                  confidence: float, game_context: Dict) -> BetOpportunity:
        """Analyze a single betting opportunity"""
        
        # Calculate implied probability from odds
        implied_prob = self._odds_to_probability(market_odds)
        
        # Calculate edge
        edge = true_probability - implied_prob
        
        # Calculate fair odds
        fair_odds = self._probability_to_odds(true_probability)
        
        # Calculate Kelly fraction
        kelly_fraction = self._calculate_kelly_fraction(market_odds, true_probability)
        
        # Estimate variance (simplified)
        variance = true_probability * (1 - true_probability)
        
        return BetOpportunity(
            game_id=game_context.get('game_id', ''),
            bet_type=game_context.get('bet_type', 'SPREAD'),
            side=game_context.get('side', 'HOME'),
            market_odds=market_odds,
            fair_odds=fair_odds,
            implied_prob=implied_prob,
            true_prob=true_probability,
            edge=edge,
            confidence=confidence,
            kelly_fraction=kelly_fraction,
            game_time=game_context.get('game_time', datetime.now() + timedelta(hours=2)),
            expires_at=game_context.get('expires_at', datetime.now() + timedelta(hours=1)),
            sportsbook=game_context.get('sportsbook', 'Unknown'),
            variance=variance,
            correlation_risk=0.0,  # Will be calculated based on other active bets
            liquidity_score=game_context.get('liquidity_score', 0.8)
        )
    
    def generate_betting_recommendation(self, opportunity: BetOpportunity) -> Optional[BettingRecommendation]:
        """Generate comprehensive betting recommendation"""
        
        # Initial filters
        if opportunity.edge < self.risk_params.min_edge_threshold:
            return None
        
        if opportunity.confidence < self.risk_params.min_confidence:
            return None
        
        # Calculate correlation risk with existing positions
        correlation_risk = self._calculate_correlation_risk(opportunity)
        opportunity.correlation_risk = correlation_risk
        
        # Risk-adjusted Kelly sizing
        kelly_size = self._calculate_risk_adjusted_kelly(opportunity)
        
        # Apply fractional Kelly
        fractional_kelly = kelly_size * self.risk_params.kelly_fraction
        
        # Apply additional risk constraints
        risk_adjusted_size = self._apply_risk_constraints(fractional_kelly, opportunity)
        
        # Convert to unit sizing
        suggested_units = risk_adjusted_size * self.bankroll_state.current_bankroll / 100  # Assuming $100 units
        max_units = min(suggested_units * 2, self.bankroll_state.current_bankroll * self.risk_params.max_bet_size / 100)
        
        # Calculate expected value
        expected_value = opportunity.edge * suggested_units
        
        # Risk metrics
        risk_score = self._calculate_risk_score(opportunity, risk_adjusted_size)
        sharpe_ratio = expected_value / (opportunity.variance * suggested_units) if opportunity.variance > 0 else 0
        max_drawdown_risk = self._estimate_drawdown_risk(opportunity, risk_adjusted_size)
        
        # Determine action and urgency
        action = self._determine_action(opportunity, risk_adjusted_size)
        urgency = self._determine_urgency(opportunity)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(opportunity, risk_adjusted_size)
        risk_factors = self._identify_risk_factors(opportunity)
        
        return BettingRecommendation(
            opportunity=opportunity,
            kelly_size=kelly_size,
            fractional_kelly=fractional_kelly,
            risk_adjusted_size=risk_adjusted_size,
            suggested_units=suggested_units,
            max_units=max_units,
            expected_value=expected_value,
            risk_score=risk_score,
            sharpe_ratio=sharpe_ratio,
            max_drawdown_risk=max_drawdown_risk,
            action=action,
            urgency=urgency,
            reasoning=reasoning,
            risk_factors=risk_factors
        )
    
    def optimize_bet_portfolio(self, opportunities: List[BetOpportunity]) -> List[BettingRecommendation]:
        """Optimize entire bet portfolio using modern portfolio theory"""
        
        if not opportunities:
            return []
        
        # Filter opportunities by basic criteria
        filtered_opportunities = [
            opp for opp in opportunities 
            if opp.edge >= self.risk_params.min_edge_threshold and 
               opp.confidence >= self.risk_params.min_confidence
        ]
        
        if not filtered_opportunities:
            return []
        
        # Build expected returns and covariance matrix
        expected_returns = np.array([opp.edge for opp in filtered_opportunities])
        covariance_matrix = self._build_covariance_matrix(filtered_opportunities)
        
        # Portfolio optimization
        optimal_weights = self._optimize_portfolio_weights(expected_returns, covariance_matrix)
        
        # Generate recommendations based on optimal weights
        recommendations = []
        total_risk_budget = self.risk_params.max_daily_risk * self.bankroll_state.current_bankroll
        
        for i, opportunity in enumerate(filtered_opportunities):
            weight = optimal_weights[i]
            if weight > 0.01:  # Only recommend if weight > 1%
                
                # Calculate position size based on portfolio weight
                position_size = weight * total_risk_budget
                units = position_size / 100  # Convert to units
                
                recommendation = BettingRecommendation(
                    opportunity=opportunity,
                    kelly_size=opportunity.kelly_fraction,
                    fractional_kelly=opportunity.kelly_fraction * self.risk_params.kelly_fraction,
                    risk_adjusted_size=weight * 100,  # As percentage
                    suggested_units=units,
                    max_units=units * 1.5,
                    expected_value=opportunity.edge * units,
                    risk_score=self._calculate_risk_score(opportunity, weight * 100),
                    sharpe_ratio=opportunity.edge / opportunity.variance if opportunity.variance > 0 else 0,
                    max_drawdown_risk=weight * 0.1,  # Simplified
                    action=self._determine_action(opportunity, weight * 100),
                    urgency=self._determine_urgency(opportunity),
                    reasoning=[f"Portfolio optimized weight: {weight:.2%}"],
                    risk_factors=self._identify_risk_factors(opportunity)
                )
                
                recommendations.append(recommendation)
        
        return recommendations
    
    def _odds_to_probability(self, odds: float) -> float:
        """Convert American odds to implied probability"""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)
    
    def _probability_to_odds(self, probability: float) -> float:
        """Convert probability to American odds"""
        if probability >= 0.5:
            return -100 * probability / (1 - probability)
        else:
            return 100 * (1 - probability) / probability
    
    def _calculate_kelly_fraction(self, odds: float, true_probability: float) -> float:
        """Calculate optimal Kelly fraction"""
        # Convert odds to decimal
        if odds > 0:
            decimal_odds = (odds / 100) + 1
        else:
            decimal_odds = (100 / abs(odds)) + 1
        
        # Kelly formula: f = (bp - q) / b
        # where b = decimal odds - 1, p = true probability, q = 1 - p
        b = decimal_odds - 1
        p = true_probability
        q = 1 - p
        
        kelly = (b * p - q) / b
        return max(0, kelly)  # Don't bet if Kelly is negative
    
    def _calculate_correlation_risk(self, opportunity: BetOpportunity) -> float:
        """Calculate correlation risk with existing positions"""
        if not self.active_bets:
            return 0.0
        
        total_correlation = 0.0
        for active_bet in self.active_bets:
            # Simplified correlation calculation
            # Same game = high correlation
            if active_bet['game_id'] == opportunity.game_id:
                total_correlation += 0.8
            # Same team = medium correlation  
            elif any(team in active_bet['teams'] for team in opportunity.game_id.split('_')):
                total_correlation += 0.3
            # Same day = low correlation
            elif abs((active_bet['game_time'] - opportunity.game_time).days) == 0:
                total_correlation += 0.1
        
        return min(total_correlation, 1.0)
    
    def _calculate_risk_adjusted_kelly(self, opportunity: BetOpportunity) -> float:
        """Calculate risk-adjusted Kelly sizing"""
        base_kelly = opportunity.kelly_fraction
        
        # Adjust for confidence
        confidence_adjustment = opportunity.confidence
        
        # Adjust for correlation risk
        correlation_adjustment = 1.0 - (opportunity.correlation_risk * 0.5)
        
        # Adjust for liquidity
        liquidity_adjustment = opportunity.liquidity_score
        
        risk_adjusted = base_kelly * confidence_adjustment * correlation_adjustment * liquidity_adjustment
        
        return risk_adjusted
    
    def _apply_risk_constraints(self, kelly_size: float, opportunity: BetOpportunity) -> float:
        """Apply additional risk management constraints"""
        
        # Maximum bet size constraint
        constrained_size = min(kelly_size, self.risk_params.max_bet_size)
        
        # Daily risk constraint
        current_daily_exposure = self._calculate_current_daily_exposure()
        remaining_daily_budget = self.risk_params.max_daily_risk - current_daily_exposure
        constrained_size = min(constrained_size, remaining_daily_budget)
        
        # Maximum simultaneous bets constraint
        if len(self.active_bets) >= self.risk_params.max_simultaneous_bets:
            constrained_size *= 0.5  # Reduce size if at limit
        
        # Drawdown protection
        if self.bankroll_state.current_drawdown > 0.1:  # If in 10%+ drawdown
            constrained_size *= 0.7  # Reduce sizing
        
        return max(0, constrained_size)
    
    def _build_covariance_matrix(self, opportunities: List[BetOpportunity]) -> np.ndarray:
        """Build covariance matrix for portfolio optimization"""
        n = len(opportunities)
        covariance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    # Diagonal: variance
                    covariance_matrix[i][j] = opportunities[i].variance
                else:
                    # Off-diagonal: covariance (simplified)
                    correlation = self._estimate_correlation(opportunities[i], opportunities[j])
                    covariance = correlation * np.sqrt(opportunities[i].variance * opportunities[j].variance)
                    covariance_matrix[i][j] = covariance
        
        return covariance_matrix
    
    def _estimate_correlation(self, opp1: BetOpportunity, opp2: BetOpportunity) -> float:
        """Estimate correlation between two betting opportunities"""
        # Same game
        if opp1.game_id == opp2.game_id:
            return 0.8
        
        # Same teams
        teams1 = set(opp1.game_id.split('_')[:2])
        teams2 = set(opp2.game_id.split('_')[:2])
        if teams1 & teams2:  # Common team
            return 0.4
        
        # Same day (simplified - assuming game_id contains date info)
        return 0.1
    
    def _optimize_portfolio_weights(self, expected_returns: np.ndarray, 
                                  covariance_matrix: np.ndarray) -> np.ndarray:
        """Optimize portfolio weights using mean-variance optimization"""
        n = len(expected_returns)
        
        # Objective function: maximize Sharpe ratio
        def objective(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            if portfolio_volatility == 0:
                return 0
            
            # Negative because we minimize
            return -(portfolio_return / portfolio_volatility)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
        ]
        
        # Bounds: each weight between 0 and max_bet_size
        bounds = [(0, self.risk_params.max_bet_size) for _ in range(n)]
        
        # Initial guess: equal weights
        initial_guess = np.ones(n) / n
        
        # Optimize
        result = minimize(
            objective, 
            initial_guess, 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            return result.x
        else:
            # Fallback: equal weights
            return np.ones(n) / n
    
    def _calculate_current_daily_exposure(self) -> float:
        """Calculate current daily risk exposure"""
        today = datetime.now().date()
        daily_exposure = 0.0
        
        for bet in self.active_bets:
            if bet['game_time'].date() == today:
                daily_exposure += bet['stake_pct']
        
        return daily_exposure
    
    def _calculate_risk_score(self, opportunity: BetOpportunity, size_pct: float) -> float:
        """Calculate comprehensive risk score (0-10)"""
        risk_components = {
            'size_risk': size_pct / self.risk_params.max_bet_size,  # Size relative to max
            'confidence_risk': 1 - opportunity.confidence,  # Lower confidence = higher risk
            'edge_risk': max(0, (0.1 - opportunity.edge) / 0.06),  # Risk if edge is small
            'correlation_risk': opportunity.correlation_risk,
            'time_risk': max(0, 1 - (opportunity.expires_at - datetime.now()).total_seconds() / 3600),  # Time pressure
            'liquidity_risk': 1 - opportunity.liquidity_score
        }
        
        # Weighted average
        weights = {
            'size_risk': 0.25,
            'confidence_risk': 0.20,
            'edge_risk': 0.20,
            'correlation_risk': 0.15,
            'time_risk': 0.10,
            'liquidity_risk': 0.10
        }
        
        risk_score = sum(risk_components[component] * weights[component] 
                        for component in risk_components)
        
        return min(10, risk_score * 10)  # Scale to 0-10
    
    def _estimate_drawdown_risk(self, opportunity: BetOpportunity, size_pct: float) -> float:
        """Estimate potential contribution to maximum drawdown"""
        # Simplified: assume worst case is losing the entire bet
        worst_case_loss = size_pct
        
        # Adjust for probability of loss
        loss_probability = 1 - opportunity.true_prob
        
        expected_drawdown_contribution = worst_case_loss * loss_probability
        
        return expected_drawdown_contribution
    
    def _determine_action(self, opportunity: BetOpportunity, size_pct: float) -> str:
        """Determine betting action level"""
        if opportunity.edge >= 0.08 and opportunity.confidence >= 0.8 and size_pct >= 0.03:
            return "STRONG_BET"
        elif opportunity.edge >= 0.06 and opportunity.confidence >= 0.7 and size_pct >= 0.02:
            return "BET"
        elif opportunity.edge >= 0.04 and opportunity.confidence >= 0.6 and size_pct >= 0.01:
            return "SMALL_BET"
        else:
            return "AVOID"
    
    def _determine_urgency(self, opportunity: BetOpportunity) -> str:
        """Determine urgency level"""
        time_to_expiry = (opportunity.expires_at - datetime.now()).total_seconds() / 3600
        
        if time_to_expiry <= 0.5:  # 30 minutes
            return "URGENT"
        elif time_to_expiry <= 2:  # 2 hours
            return "HIGH"
        elif time_to_expiry <= 6:  # 6 hours
            return "MEDIUM"
        else:
            return "LOW"
    
    def _generate_reasoning(self, opportunity: BetOpportunity, size_pct: float) -> List[str]:
        """Generate human-readable reasoning for the recommendation"""
        reasoning = []
        
        reasoning.append(f"Edge: {opportunity.edge:.1%} (Market: {opportunity.implied_prob:.1%}, Model: {opportunity.true_prob:.1%})")
        reasoning.append(f"Confidence: {opportunity.confidence:.1%}")
        reasoning.append(f"Kelly sizing: {opportunity.kelly_fraction:.1%}, Risk-adjusted: {size_pct:.1%}")
        
        if opportunity.correlation_risk > 0.3:
            reasoning.append(f"High correlation risk ({opportunity.correlation_risk:.1%}) with existing positions")
        
        if opportunity.confidence < 0.7:
            reasoning.append("Lower confidence due to model uncertainty")
        
        return reasoning
    
    def _identify_risk_factors(self, opportunity: BetOpportunity) -> List[str]:
        """Identify key risk factors"""
        risk_factors = []
        
        if opportunity.edge < 0.06:
            risk_factors.append("Small edge - vulnerable to line movement")
        
        if opportunity.confidence < 0.7:
            risk_factors.append("Model confidence below 70%")
        
        if opportunity.correlation_risk > 0.4:
            risk_factors.append("High correlation with existing positions")
        
        if opportunity.liquidity_score < 0.6:
            risk_factors.append("Limited market liquidity")
        
        time_to_expiry = (opportunity.expires_at - datetime.now()).total_seconds() / 3600
        if time_to_expiry < 1:
            risk_factors.append("Time pressure - limited window to place bet")
        
        if self.bankroll_state.current_drawdown > 0.1:
            risk_factors.append("Currently in drawdown - consider reduced sizing")
        
        return risk_factors
    
    def record_bet(self, recommendation: BettingRecommendation, actual_stake: float):
        """Record a placed bet"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO bet_history 
            (timestamp, game_id, bet_type, side, stake_amount, odds, edge_pct, 
             kelly_fraction, confidence, result, profit_loss, sportsbook)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(),
            recommendation.opportunity.game_id,
            recommendation.opportunity.bet_type,
            recommendation.opportunity.side,
            actual_stake,
            recommendation.opportunity.market_odds,
            recommendation.opportunity.edge,
            recommendation.opportunity.kelly_fraction,
            recommendation.opportunity.confidence,
            'PENDING',
            0.0,
            recommendation.opportunity.sportsbook
        ))
        
        # Add to active bets
        self.active_bets.append({
            'game_id': recommendation.opportunity.game_id,
            'stake_amount': actual_stake,
            'stake_pct': actual_stake / self.bankroll_state.current_bankroll,
            'game_time': recommendation.opportunity.game_time,
            'teams': recommendation.opportunity.game_id.split('_')[:2]
        })
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Recorded bet: {recommendation.opportunity.game_id} - ${actual_stake:.2f}")
    
    def update_bet_result(self, game_id: str, result: str, profit_loss: float):
        """Update bet result and recalculate bankroll"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Update bet history
        cursor.execute('''
            UPDATE bet_history 
            SET result = ?, profit_loss = ?
            WHERE game_id = ? AND result = 'PENDING'
        ''', (result, profit_loss, game_id))
        
        # Update bankroll
        self.bankroll_state.current_bankroll += profit_loss
        
        # Update statistics
        self._update_performance_metrics()
        
        # Remove from active bets
        self.active_bets = [bet for bet in self.active_bets if bet['game_id'] != game_id]
        
        # Save bankroll snapshot
        self._save_bankroll_snapshot()
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Updated bet result: {game_id} - {result} - P&L: ${profit_loss:.2f}")
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all completed bets
        cursor.execute('''
            SELECT result, profit_loss, stake_amount 
            FROM bet_history 
            WHERE result IN ('WIN', 'LOSS', 'PUSH')
        ''')
        
        bets = cursor.fetchall()
        
        if bets:
            total_profit = sum(bet[1] for bet in bets)
            total_staked = sum(bet[2] for bet in bets)
            
            self.bankroll_state.total_return = total_profit
            self.bankroll_state.roi_pct = (total_profit / total_staked) if total_staked > 0 else 0
            
            self.bankroll_state.total_bets = len(bets)
            self.bankroll_state.winning_bets = sum(1 for bet in bets if bet[0] == 'WIN')
            self.bankroll_state.losing_bets = sum(1 for bet in bets if bet[0] == 'LOSS')
            self.bankroll_state.win_rate = self.bankroll_state.winning_bets / len(bets)
            
            # Calculate drawdown
            self.bankroll_state.peak_bankroll = max(self.bankroll_state.peak_bankroll, 
                                                   self.bankroll_state.current_bankroll)
            self.bankroll_state.current_drawdown = (
                (self.bankroll_state.peak_bankroll - self.bankroll_state.current_bankroll) / 
                self.bankroll_state.peak_bankroll
            )
            self.bankroll_state.max_drawdown = max(self.bankroll_state.max_drawdown,
                                                  self.bankroll_state.current_drawdown)
        
        conn.close()
    
    def _save_bankroll_snapshot(self):
        """Save current bankroll snapshot"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO bankroll_history 
            (timestamp, bankroll_amount, total_return, roi_pct, max_drawdown, 
             current_drawdown, sharpe_ratio, total_bets, win_rate)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(),
            self.bankroll_state.current_bankroll,
            self.bankroll_state.total_return,
            self.bankroll_state.roi_pct,
            self.bankroll_state.max_drawdown,
            self.bankroll_state.current_drawdown,
            self.bankroll_state.sharpe_ratio,
            self.bankroll_state.total_bets,
            self.bankroll_state.win_rate
        ))
        
        conn.commit()
        conn.close()

# Testing and demonstration
def test_betting_engine():
    """Test the advanced betting engine"""
    engine = AdvancedBettingEngine(initial_bankroll=10000.0)
    
    print("💰 Advanced Betting Engine Demo")
    print("=" * 50)
    
    # Create sample betting opportunities
    opportunities = []
    
    # High-confidence opportunity
    opp1 = engine.analyze_betting_opportunity(
        market_odds=-110,
        true_probability=0.60,
        confidence=0.85,
        game_context={
            'game_id': 'duke_unc_20241201',
            'bet_type': 'SPREAD',
            'side': 'HOME',
            'game_time': datetime.now() + timedelta(hours=3),
            'expires_at': datetime.now() + timedelta(hours=1),
            'sportsbook': 'DraftKings',
            'liquidity_score': 0.9
        }
    )
    opportunities.append(opp1)
    
    # Medium-confidence opportunity
    opp2 = engine.analyze_betting_opportunity(
        market_odds=+150,
        true_probability=0.45,
        confidence=0.75,
        game_context={
            'game_id': 'kansas_kentucky_20241201',
            'bet_type': 'MONEYLINE',
            'side': 'AWAY',
            'game_time': datetime.now() + timedelta(hours=5),
            'expires_at': datetime.now() + timedelta(hours=2),
            'sportsbook': 'FanDuel',
            'liquidity_score': 0.8
        }
    )
    opportunities.append(opp2)
    
    print(f"Analyzed {len(opportunities)} opportunities:")
    for i, opp in enumerate(opportunities, 1):
        print(f"{i}. {opp.game_id}: Edge {opp.edge:.1%}, Kelly {opp.kelly_fraction:.1%}")
    
    # Generate individual recommendations
    print(f"\n📊 Individual Recommendations:")
    for opp in opportunities:
        rec = engine.generate_betting_recommendation(opp)
        if rec:
            print(f"{opp.game_id}: {rec.action} - {rec.suggested_units:.1f} units ({rec.expected_value:+.2f} EV)")
    
    # Portfolio optimization
    print(f"\n🎯 Portfolio Optimization:")
    portfolio_recs = engine.optimize_bet_portfolio(opportunities)
    
    for rec in portfolio_recs:
        print(f"{rec.opportunity.game_id}: {rec.suggested_units:.1f} units (Risk Score: {rec.risk_score:.1f})")
    
    # Bankroll status
    print(f"\n💼 Bankroll Status:")
    print(f"Current: ${engine.bankroll_state.current_bankroll:,.2f}")
    print(f"ROI: {engine.bankroll_state.roi_pct:.1%}")
    print(f"Max Drawdown: {engine.bankroll_state.max_drawdown:.1%}")
    print(f"Win Rate: {engine.bankroll_state.win_rate:.1%}")
    
    print("\n✅ Advanced Betting Engine operational!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_betting_engine()