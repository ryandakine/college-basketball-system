#!/usr/bin/env python3
"""
Paper Trading System for College Basketball Betting
==================================================

Virtual money management system to test betting strategies:
- Simulated bankroll management
- Bet tracking and result recording
- Performance analytics and reporting
- Risk management with fake money
- Strategy backtesting capabilities
- Portfolio tracking across different bet types
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import sqlite3
from dataclasses import dataclass, asdict
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns

class BetType(Enum):
    SPREAD = "spread"
    MONEYLINE = "moneyline"  
    TOTAL = "total"
    PLAYER_PROP = "player_prop"
    TEAM_PROP = "team_prop"
    LIVE_BET = "live_bet"
    FUTURES = "futures"

class BetStatus(Enum):
    PENDING = "pending"
    WON = "won"
    LOST = "lost"
    PUSH = "push"
    CANCELLED = "cancelled"

@dataclass
class PaperBet:
    """Individual paper bet record"""
    bet_id: str
    timestamp: datetime
    
    # Bet details
    bet_type: BetType
    game_id: str
    description: str
    side: str  # Team name or OVER/UNDER
    
    # Financial details
    stake: float
    odds: float  # American odds
    decimal_odds: float
    implied_probability: float
    
    # Analysis
    predicted_probability: float
    edge: float
    kelly_fraction: float
    confidence: float
    
    # Tracking
    status: BetStatus
    result_value: Optional[float] = None  # Actual outcome value
    payout: float = 0.0  # Amount won/lost
    
    # Strategy info
    strategy: str = "general"
    unit_size: float = 1.0
    
    # Settlement
    settled_date: Optional[datetime] = None
    notes: str = ""

@dataclass
class BankrollSnapshot:
    """Bankroll state at a point in time"""
    timestamp: datetime
    balance: float
    total_staked: float
    total_won: float
    total_lost: float
    net_profit: float
    roi: float
    win_rate: float
    pending_stakes: float
    num_bets: int
    avg_odds: float
    sharpe_ratio: float

class PaperTradingSystem:
    """Paper trading system for testing betting strategies"""
    
    def __init__(self, initial_bankroll: float = 10000.0, db_path: str = "paper_trading.db"):
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path
        
        # Bankroll management
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.max_bet_percentage = 0.05  # Max 5% of bankroll per bet
        self.min_edge_required = 0.03   # 3% minimum edge
        
        # Performance tracking
        self.bets_history = []
        self.bankroll_history = []
        
        # Risk management
        self.daily_loss_limit = initial_bankroll * 0.10  # 10% daily loss limit
        self.max_exposure = initial_bankroll * 0.25      # 25% max pending exposure
        
        # Strategy tracking
        self.strategy_performance = {}
        
        # Initialize database
        self._init_database()
        
        # Load existing state
        self._load_state()
    
    def _init_database(self):
        """Initialize paper trading database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Paper bets table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS paper_bets (
                bet_id TEXT PRIMARY KEY,
                timestamp DATETIME,
                bet_type TEXT,
                game_id TEXT,
                description TEXT,
                side TEXT,
                stake REAL,
                odds REAL,
                decimal_odds REAL,
                implied_probability REAL,
                predicted_probability REAL,
                edge REAL,
                kelly_fraction REAL,
                confidence REAL,
                status TEXT,
                result_value REAL,
                payout REAL,
                strategy TEXT,
                unit_size REAL,
                settled_date DATETIME,
                notes TEXT
            )
        ''')
        
        # Bankroll snapshots
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS bankroll_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                balance REAL,
                total_staked REAL,
                total_won REAL,
                total_lost REAL,
                net_profit REAL,
                roi REAL,
                win_rate REAL,
                pending_stakes REAL,
                num_bets INTEGER,
                avg_odds REAL,
                sharpe_ratio REAL
            )
        ''')
        
        # Strategy performance
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT,
                timestamp DATETIME,
                num_bets INTEGER,
                total_staked REAL,
                net_profit REAL,
                roi REAL,
                win_rate REAL,
                avg_edge REAL,
                sharpe_ratio REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_state(self):
        """Load existing paper trading state"""
        conn = sqlite3.connect(self.db_path)
        
        # Load bets
        bets_df = pd.read_sql_query('''
            SELECT * FROM paper_bets ORDER BY timestamp
        ''', conn)
        
        self.bets_history = []
        for _, row in bets_df.iterrows():
            bet = PaperBet(
                bet_id=row['bet_id'],
                timestamp=pd.to_datetime(row['timestamp']),
                bet_type=BetType(row['bet_type']),
                game_id=row['game_id'],
                description=row['description'],
                side=row['side'],
                stake=row['stake'],
                odds=row['odds'],
                decimal_odds=row['decimal_odds'],
                implied_probability=row['implied_probability'],
                predicted_probability=row['predicted_probability'],
                edge=row['edge'],
                kelly_fraction=row['kelly_fraction'],
                confidence=row['confidence'],
                status=BetStatus(row['status']),
                result_value=row['result_value'],
                payout=row['payout'],
                strategy=row['strategy'],
                unit_size=row['unit_size'],
                settled_date=pd.to_datetime(row['settled_date']) if row['settled_date'] else None,
                notes=row['notes'] or ""
            )
            self.bets_history.append(bet)
        
        # Calculate current bankroll
        self._recalculate_bankroll()
        
        conn.close()
    
    def _recalculate_bankroll(self):
        """Recalculate current bankroll from bet history"""
        settled_profit = sum(bet.payout for bet in self.bets_history 
                           if bet.status in [BetStatus.WON, BetStatus.LOST, BetStatus.PUSH])
        self.current_bankroll = self.initial_bankroll + settled_profit
    
    def _american_to_decimal_odds(self, american_odds: float) -> float:
        """Convert American odds to decimal odds"""
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1
    
    def _odds_to_probability(self, american_odds: float) -> float:
        """Convert American odds to implied probability"""
        if american_odds > 0:
            return 100 / (american_odds + 100)
        else:
            return abs(american_odds) / (abs(american_odds) + 100)
    
    def calculate_kelly_sizing(self, predicted_prob: float, odds: float) -> float:
        """Calculate Kelly criterion bet size"""
        decimal_odds = self._american_to_decimal_odds(odds)
        kelly = (predicted_prob * decimal_odds - 1) / (decimal_odds - 1)
        return max(0, min(0.25, kelly))  # Cap at 25% of bankroll
    
    def place_paper_bet(self, bet_type: BetType, game_id: str, description: str,
                       side: str, odds: float, predicted_probability: float,
                       strategy: str = "general", max_stake: Optional[float] = None) -> Optional[str]:
        """Place a paper bet"""
        
        # Calculate bet metrics
        decimal_odds = self._american_to_decimal_odds(odds)
        implied_prob = self._odds_to_probability(odds)
        edge = predicted_probability - implied_prob
        
        # Check minimum edge requirement
        if edge < self.min_edge_required:
            self.logger.info(f"Bet rejected: Edge {edge:.1%} below minimum {self.min_edge_required:.1%}")
            return None
        
        # Calculate Kelly sizing
        kelly_fraction = self.calculate_kelly_sizing(predicted_probability, odds)
        
        # Determine stake
        max_allowed_stake = self.current_bankroll * self.max_bet_percentage
        kelly_stake = self.current_bankroll * kelly_fraction
        
        if max_stake:
            stake = min(max_stake, max_allowed_stake, kelly_stake)
        else:
            stake = min(max_allowed_stake, kelly_stake)
        
        # Check exposure limits
        pending_exposure = sum(bet.stake for bet in self.bets_history 
                             if bet.status == BetStatus.PENDING)
        
        if pending_exposure + stake > self.max_exposure:
            self.logger.warning(f"Bet rejected: Would exceed max exposure limit")
            return None
        
        # Create bet
        bet_id = f"{game_id}_{bet_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        paper_bet = PaperBet(
            bet_id=bet_id,
            timestamp=datetime.now(),
            bet_type=bet_type,
            game_id=game_id,
            description=description,
            side=side,
            stake=stake,
            odds=odds,
            decimal_odds=decimal_odds,
            implied_probability=implied_prob,
            predicted_probability=predicted_probability,
            edge=edge,
            kelly_fraction=kelly_fraction,
            confidence=min(0.95, max(0.5, 0.5 + edge * 2)),  # Confidence based on edge
            status=BetStatus.PENDING,
            strategy=strategy,
            unit_size=stake / (self.current_bankroll * 0.01)  # Units as % of bankroll
        )
        
        # Store bet
        self.bets_history.append(paper_bet)
        self._save_bet(paper_bet)
        
        self.logger.info(f"Paper bet placed: {description} | ${stake:.2f} @ {odds:+d} | Edge: {edge:.1%}")
        
        return bet_id
    
    def settle_bet(self, bet_id: str, result_value: float, won: bool, 
                  actual_odds: Optional[float] = None) -> bool:
        """Settle a paper bet with actual result"""
        
        bet = next((b for b in self.bets_history if b.bet_id == bet_id), None)
        if not bet or bet.status != BetStatus.PENDING:
            self.logger.error(f"Bet {bet_id} not found or already settled")
            return False
        
        # Update bet result
        bet.result_value = result_value
        bet.settled_date = datetime.now()
        
        # Calculate payout
        if won:
            bet.status = BetStatus.WON
            # Use actual closing odds if provided
            odds_used = actual_odds if actual_odds else bet.odds
            if odds_used > 0:
                bet.payout = bet.stake * (odds_used / 100)
            else:
                bet.payout = bet.stake * (100 / abs(odds_used))
        else:
            bet.status = BetStatus.LOST
            bet.payout = -bet.stake
        
        # Update database
        self._update_bet(bet)
        self._recalculate_bankroll()
        
        # Log result
        result_str = "WON" if won else "LOST"
        self.logger.info(f"Bet settled: {bet.description} {result_str} | P&L: ${bet.payout:+.2f}")
        
        # Update performance tracking
        self._update_strategy_performance(bet.strategy)
        
        return True
    
    def push_bet(self, bet_id: str) -> bool:
        """Mark bet as push (no winner)"""
        bet = next((b for b in self.bets_history if b.bet_id == bet_id), None)
        if not bet or bet.status != BetStatus.PENDING:
            return False
        
        bet.status = BetStatus.PUSH
        bet.payout = 0.0
        bet.settled_date = datetime.now()
        
        self._update_bet(bet)
        self.logger.info(f"Bet pushed: {bet.description}")
        
        return True
    
    def cancel_bet(self, bet_id: str, reason: str = "") -> bool:
        """Cancel a pending bet"""
        bet = next((b for b in self.bets_history if b.bet_id == bet_id), None)
        if not bet or bet.status != BetStatus.PENDING:
            return False
        
        bet.status = BetStatus.CANCELLED
        bet.payout = 0.0
        bet.settled_date = datetime.now()
        bet.notes = reason
        
        self._update_bet(bet)
        self.logger.info(f"Bet cancelled: {bet.description} - {reason}")
        
        return True
    
    def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get performance summary for specified period"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_bets = [bet for bet in self.bets_history 
                      if bet.timestamp >= cutoff_date and 
                      bet.status in [BetStatus.WON, BetStatus.LOST, BetStatus.PUSH]]
        
        if not recent_bets:
            return {"error": "No settled bets in specified period"}
        
        # Basic metrics
        total_bets = len(recent_bets)
        won_bets = len([b for b in recent_bets if b.status == BetStatus.WON])
        lost_bets = len([b for b in recent_bets if b.status == BetStatus.LOST])
        push_bets = len([b for b in recent_bets if b.status == BetStatus.PUSH])
        
        total_staked = sum(bet.stake for bet in recent_bets)
        total_profit = sum(bet.payout for bet in recent_bets)
        
        win_rate = won_bets / (total_bets - push_bets) if (total_bets - push_bets) > 0 else 0
        roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
        
        # Average metrics
        avg_stake = total_staked / total_bets if total_bets > 0 else 0
        avg_odds = np.mean([bet.decimal_odds for bet in recent_bets])
        avg_edge = np.mean([bet.edge for bet in recent_bets])
        
        # Risk metrics
        returns = [bet.payout / bet.stake for bet in recent_bets if bet.stake > 0]
        sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) > 0 else 0
        
        # Longest streaks
        win_streak = self._calculate_longest_streak(recent_bets, BetStatus.WON)
        loss_streak = self._calculate_longest_streak(recent_bets, BetStatus.LOST)
        
        return {
            "period_days": days,
            "total_bets": total_bets,
            "won_bets": won_bets,
            "lost_bets": lost_bets,
            "push_bets": push_bets,
            "win_rate": win_rate,
            "total_staked": total_staked,
            "total_profit": total_profit,
            "roi": roi,
            "avg_stake": avg_stake,
            "avg_odds": avg_odds,
            "avg_edge": avg_edge,
            "sharpe_ratio": sharpe_ratio,
            "longest_win_streak": win_streak,
            "longest_loss_streak": loss_streak,
            "current_bankroll": self.current_bankroll,
            "bankroll_change": total_profit,
            "bankroll_change_pct": (total_profit / self.initial_bankroll) * 100
        }
    
    def _calculate_longest_streak(self, bets: List[PaperBet], status: BetStatus) -> int:
        """Calculate longest streak of specified status"""
        max_streak = 0
        current_streak = 0
        
        for bet in sorted(bets, key=lambda x: x.timestamp):
            if bet.status == status:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def get_strategy_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance breakdown by strategy"""
        
        strategy_stats = {}
        
        for strategy in set(bet.strategy for bet in self.bets_history):
            strategy_bets = [bet for bet in self.bets_history 
                           if bet.strategy == strategy and 
                           bet.status in [BetStatus.WON, BetStatus.LOST, BetStatus.PUSH]]
            
            if not strategy_bets:
                continue
            
            total_bets = len(strategy_bets)
            won_bets = len([b for b in strategy_bets if b.status == BetStatus.WON])
            total_staked = sum(bet.stake for bet in strategy_bets)
            total_profit = sum(bet.payout for bet in strategy_bets)
            
            win_rate = won_bets / total_bets if total_bets > 0 else 0
            roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
            avg_edge = np.mean([bet.edge for bet in strategy_bets])
            
            strategy_stats[strategy] = {
                "total_bets": total_bets,
                "win_rate": win_rate,
                "total_staked": total_staked,
                "total_profit": total_profit,
                "roi": roi,
                "avg_edge": avg_edge
            }
        
        return strategy_stats
    
    def get_pending_bets(self) -> List[PaperBet]:
        """Get all pending bets"""
        return [bet for bet in self.bets_history if bet.status == BetStatus.PENDING]
    
    def get_daily_pnl(self, days: int = 30) -> pd.DataFrame:
        """Get daily P&L over specified period"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        settled_bets = [bet for bet in self.bets_history 
                       if bet.settled_date and bet.settled_date >= cutoff_date and
                       bet.status in [BetStatus.WON, BetStatus.LOST, BetStatus.PUSH]]
        
        # Group by date
        daily_data = {}
        for bet in settled_bets:
            date = bet.settled_date.date()
            if date not in daily_data:
                daily_data[date] = {"profit": 0, "stakes": 0, "bets": 0}
            
            daily_data[date]["profit"] += bet.payout
            daily_data[date]["stakes"] += bet.stake
            daily_data[date]["bets"] += 1
        
        # Create DataFrame
        df = pd.DataFrame([
            {
                "date": date,
                "daily_profit": data["profit"],
                "daily_stakes": data["stakes"],
                "daily_bets": data["bets"],
                "daily_roi": (data["profit"] / data["stakes"]) * 100 if data["stakes"] > 0 else 0
            }
            for date, data in daily_data.items()
        ])
        
        if not df.empty:
            df = df.sort_values('date')
            df['cumulative_profit'] = df['daily_profit'].cumsum()
        
        return df
    
    def _save_bet(self, bet: PaperBet):
        """Save bet to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO paper_bets VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            bet.bet_id, bet.timestamp, bet.bet_type.value, bet.game_id, bet.description,
            bet.side, bet.stake, bet.odds, bet.decimal_odds, bet.implied_probability,
            bet.predicted_probability, bet.edge, bet.kelly_fraction, bet.confidence,
            bet.status.value, bet.result_value, bet.payout, bet.strategy, bet.unit_size,
            bet.settled_date, bet.notes
        ))
        
        conn.commit()
        conn.close()
    
    def _update_bet(self, bet: PaperBet):
        """Update existing bet in database"""
        self._save_bet(bet)  # Same as save due to INSERT OR REPLACE
    
    def _update_strategy_performance(self, strategy: str):
        """Update strategy performance metrics"""
        # This would calculate and store updated strategy metrics
        pass
    
    def create_performance_report(self, filename: Optional[str] = None) -> str:
        """Create a comprehensive performance report"""
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("PAPER TRADING PERFORMANCE REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Initial Bankroll: ${self.initial_bankroll:,.2f}")
        report_lines.append(f"Current Bankroll: ${self.current_bankroll:,.2f}")
        report_lines.append(f"Net P&L: ${self.current_bankroll - self.initial_bankroll:+,.2f}")
        report_lines.append("")
        
        # Overall performance
        summary = self.get_performance_summary(days=9999)  # All time
        if "error" not in summary:
            report_lines.append("OVERALL PERFORMANCE:")
            report_lines.append(f"  Total Bets: {summary['total_bets']}")
            report_lines.append(f"  Win Rate: {summary['win_rate']:.1%}")
            report_lines.append(f"  ROI: {summary['roi']:+.2f}%")
            report_lines.append(f"  Total Staked: ${summary['total_staked']:,.2f}")
            report_lines.append(f"  Average Edge: {summary['avg_edge']:+.1%}")
            report_lines.append(f"  Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
            report_lines.append("")
        
        # Strategy breakdown
        strategy_stats = self.get_strategy_performance()
        if strategy_stats:
            report_lines.append("PERFORMANCE BY STRATEGY:")
            for strategy, stats in strategy_stats.items():
                report_lines.append(f"  {strategy.upper()}:")
                report_lines.append(f"    Bets: {stats['total_bets']}")
                report_lines.append(f"    Win Rate: {stats['win_rate']:.1%}")
                report_lines.append(f"    ROI: {stats['roi']:+.2f}%")
                report_lines.append(f"    Profit: ${stats['total_profit']:+,.2f}")
                report_lines.append("")
        
        # Pending bets
        pending = self.get_pending_bets()
        if pending:
            report_lines.append("PENDING BETS:")
            total_exposure = 0
            for bet in pending:
                report_lines.append(f"  {bet.description} | ${bet.stake:.2f} @ {bet.odds:+d}")
                total_exposure += bet.stake
            report_lines.append(f"  Total Exposure: ${total_exposure:.2f}")
            report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        if filename:
            with open(filename, 'w') as f:
                f.write(report_text)
        
        return report_text

# Demo and testing functions
def demo_paper_trading():
    """Demonstrate the paper trading system"""
    
    print("💰 Paper Trading System Demo")
    print("=" * 50)
    
    # Initialize system with $10,000
    trader = PaperTradingSystem(initial_bankroll=10000.0)
    print(f"Starting Bankroll: ${trader.current_bankroll:,.2f}")
    
    # Place some sample bets
    print("\n📊 Placing Sample Bets...")
    
    # Spread bet
    bet1 = trader.place_paper_bet(
        bet_type=BetType.SPREAD,
        game_id="duke_unc_123",
        description="Duke -5.5 vs UNC",
        side="Duke -5.5",
        odds=-110,
        predicted_probability=0.58,
        strategy="spread_model"
    )
    
    # Player prop
    bet2 = trader.place_paper_bet(
        bet_type=BetType.PLAYER_PROP,
        game_id="duke_unc_123", 
        description="Player Points Over 15.5",
        side="OVER",
        odds=-115,
        predicted_probability=0.62,
        strategy="prop_model"
    )
    
    # Total bet
    bet3 = trader.place_paper_bet(
        bet_type=BetType.TOTAL,
        game_id="kentucky_kansas_456",
        description="Total Over 145.5",
        side="OVER",
        odds=105,
        predicted_probability=0.55,
        strategy="totals_model"
    )
    
    print(f"Placed {len([b for b in [bet1, bet2, bet3] if b])} bets")
    
    # Show pending bets
    pending = trader.get_pending_bets()
    print(f"\n🕒 Pending Bets ({len(pending)}):")
    for bet in pending:
        print(f"  {bet.description} | ${bet.stake:.2f} @ {bet.odds:+d}")
    
    # Simulate settling some bets
    print("\n🎯 Settling Bets...")
    if bet1:
        trader.settle_bet(bet1, result_value=6.0, won=True)  # Duke won by 6
    
    if bet2:
        trader.settle_bet(bet2, result_value=14.0, won=False)  # Player scored 14
    
    if bet3:
        trader.settle_bet(bet3, result_value=148.0, won=True)  # Total was 148
    
    # Show performance
    print(f"\n📈 Performance Summary:")
    summary = trader.get_performance_summary(days=1)
    if "error" not in summary:
        print(f"  Current Bankroll: ${summary['current_bankroll']:,.2f}")
        print(f"  Net P&L: ${summary['total_profit']:+,.2f}")
        print(f"  ROI: {summary['roi']:+.1f}%")
        print(f"  Win Rate: {summary['win_rate']:.1%}")
        print(f"  Total Bets: {summary['total_bets']}")
    
    # Create report
    print("\n📋 Generating Performance Report...")
    report = trader.create_performance_report()
    print(report)
    
    print("\n✅ Paper Trading System Ready!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo_paper_trading()