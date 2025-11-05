#!/usr/bin/env python3
"""
Integrated Prop Trading System
=============================

Combines the prop betting engine with paper trading system for complete
testing and validation of prop betting strategies with fake money.

Features:
- Automated prop opportunity detection and paper betting
- Real-time performance tracking
- Strategy validation with fake money
- Risk management and Kelly sizing
- Comprehensive reporting and analytics
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import numpy as np

# Import our modules
from prop_betting_engine import PropBettingEngine, PropRecommendation
from paper_trading_system import PaperTradingSystem, BetType, BetStatus

class IntegratedPropTrader:
    """Integrated system combining prop analysis with paper trading"""
    
    def __init__(self, initial_bankroll: float = 10000.0):
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.prop_engine = PropBettingEngine()
        self.paper_trader = PaperTradingSystem(initial_bankroll=initial_bankroll)
        
        # Trading parameters
        self.auto_bet_threshold = 0.10  # 10% minimum edge for auto-betting
        self.max_props_per_game = 5     # Limit props per game
        self.correlation_limit = 0.3    # Max correlation between props
        
        # Performance tracking
        self.session_stats = {
            'opportunities_found': 0,
            'bets_placed': 0,
            'avg_edge': 0.0,
            'session_profit': 0.0
        }
    
    async def analyze_and_bet_game(self, game_id: str, home_team: str, 
                                  away_team: str, game_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a game and place paper bets on valuable props"""
        
        self.logger.info(f"Analyzing game: {home_team} vs {away_team}")
        
        # Load player profiles if not already loaded
        teams = [home_team, away_team]
        await self.prop_engine.load_player_profiles(teams)
        
        # Find prop opportunities
        prop_opportunities = await self.prop_engine.analyze_prop_opportunities(
            game_id, home_team, away_team, game_context
        )
        
        self.session_stats['opportunities_found'] += len(prop_opportunities)
        
        if not prop_opportunities:
            return {"message": "No valuable prop opportunities found", "bets_placed": 0}
        
        # Generate recommendations
        recommendations = self.prop_engine.generate_prop_recommendations(prop_opportunities)
        
        # Filter and place bets
        bets_placed = []
        game_exposure = 0.0
        
        for rec in recommendations[:self.max_props_per_game]:  # Limit per game
            prop = rec.prop_bet
            edge = max(prop.over_edge, prop.under_edge)
            
            # Skip if edge below auto-bet threshold
            if edge < self.auto_bet_threshold:
                continue
            
            # Check correlation limits (simplified)
            if self._check_correlation_risk(prop, bets_placed):
                continue
            
            # Convert prop type to paper trading bet type
            paper_bet_type = self._convert_prop_to_bet_type(prop.prop_type)
            
            # Place paper bet
            bet_id = self.paper_trader.place_paper_bet(
                bet_type=paper_bet_type,
                game_id=game_id,
                description=f"{prop.player_name or prop.team} {prop.prop_type.replace('_', ' ').title()} {prop.best_bet} {prop.line}",
                side=prop.best_bet,
                odds=prop.over_odds if prop.best_bet == 'OVER' else prop.under_odds,
                predicted_probability=prop.over_probability if prop.best_bet == 'OVER' else prop.under_probability,
                strategy=f"prop_{prop.prop_type}"
            )
            
            if bet_id:
                bets_placed.append({
                    'bet_id': bet_id,
                    'prop': prop,
                    'recommendation': rec
                })
                game_exposure += rec.unit_size * self.paper_trader.current_bankroll
                self.session_stats['bets_placed'] += 1
        
        # Calculate session stats
        if bets_placed:
            avg_edge = np.mean([max(bet['prop'].over_edge, bet['prop'].under_edge) 
                               for bet in bets_placed])
            self.session_stats['avg_edge'] = avg_edge
        
        self.logger.info(f"Placed {len(bets_placed)} paper bets with ${game_exposure:.2f} exposure")
        
        return {
            "game_id": game_id,
            "opportunities_found": len(prop_opportunities),
            "bets_placed": len(bets_placed),
            "total_exposure": game_exposure,
            "avg_edge": avg_edge if bets_placed else 0,
            "bet_details": [
                {
                    "description": bet['prop'].player_name or bet['prop'].team,
                    "prop_type": bet['prop'].prop_type,
                    "line": bet['prop'].line,
                    "side": bet['prop'].best_bet,
                    "edge": max(bet['prop'].over_edge, bet['prop'].under_edge),
                    "stake": bet['recommendation'].unit_size * self.paper_trader.current_bankroll
                }
                for bet in bets_placed
            ]
        }
    
    def _convert_prop_to_bet_type(self, prop_type: str) -> BetType:
        """Convert prop type to paper trading bet type"""
        if prop_type.startswith('player_'):
            return BetType.PLAYER_PROP
        elif prop_type.startswith('team_'):
            return BetType.TEAM_PROP
        else:
            return BetType.PLAYER_PROP  # Default
    
    def _check_correlation_risk(self, prop, existing_bets: List[Dict]) -> bool:
        """Check if prop is too correlated with existing bets"""
        # Simplified correlation check
        # In practice, would calculate actual correlations
        
        same_game_props = len([bet for bet in existing_bets 
                              if bet['prop'].game_id == prop.game_id])
        
        return same_game_props >= 3  # Max 3 props per game
    
    async def simulate_bet_settlement(self, bet_id: str, simulate_realistic: bool = True) -> bool:
        """Simulate settling a bet with realistic outcomes"""
        
        # Find the bet
        bet = next((b for b in self.paper_trader.bets_history 
                   if b.bet_id == bet_id and b.status == BetStatus.PENDING), None)
        
        if not bet:
            return False
        
        # Simulate realistic outcome based on predicted probability
        if simulate_realistic:
            # Use the predicted probability to determine win/loss
            random_outcome = np.random.random()
            won = random_outcome < bet.predicted_probability
        else:
            # 50/50 random for testing
            won = np.random.random() < 0.5
        
        # Simulate a result value (for props)
        if bet.bet_type in [BetType.PLAYER_PROP, BetType.TEAM_PROP]:
            # Simulate actual statistical outcome
            if bet.side == 'OVER':
                if won:
                    result_value = bet.implied_probability + np.random.uniform(0.5, 3.0)
                else:
                    result_value = bet.implied_probability - np.random.uniform(0.1, 2.0)
            else:  # UNDER
                if won:
                    result_value = bet.implied_probability - np.random.uniform(0.1, 2.0) 
                else:
                    result_value = bet.implied_probability + np.random.uniform(0.5, 3.0)
        else:
            result_value = 1.0 if won else 0.0
        
        # Settle the bet
        success = self.paper_trader.settle_bet(bet_id, result_value, won)
        
        if success:
            # Update session stats
            payout = bet.payout if hasattr(bet, 'payout') else 0
            self.session_stats['session_profit'] += payout
        
        return success
    
    def get_prop_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary focused on prop betting"""
        
        # Get overall paper trading performance
        paper_summary = self.paper_trader.get_performance_summary(days=30)
        
        # Get prop-specific metrics
        prop_bets = [bet for bet in self.paper_trader.bets_history 
                    if bet.bet_type in [BetType.PLAYER_PROP, BetType.TEAM_PROP]]
        
        if not prop_bets:
            return {"error": "No prop bets found"}
        
        # Prop-specific calculations
        settled_props = [bet for bet in prop_bets 
                        if bet.status in [BetStatus.WON, BetStatus.LOST]]
        
        prop_metrics = {}
        
        if settled_props:
            prop_win_rate = len([b for b in settled_props if b.status == BetStatus.WON]) / len(settled_props)
            prop_roi = sum(bet.payout for bet in settled_props) / sum(bet.stake for bet in settled_props) * 100
            avg_prop_edge = np.mean([bet.edge for bet in settled_props])
            
            # Break down by prop type
            prop_types = {}
            for bet in settled_props:
                strategy = bet.strategy
                if strategy not in prop_types:
                    prop_types[strategy] = {'bets': 0, 'wins': 0, 'profit': 0, 'stakes': 0}
                
                prop_types[strategy]['bets'] += 1
                if bet.status == BetStatus.WON:
                    prop_types[strategy]['wins'] += 1
                prop_types[strategy]['profit'] += bet.payout
                prop_types[strategy]['stakes'] += bet.stake
            
            prop_metrics = {
                "total_prop_bets": len(settled_props),
                "prop_win_rate": prop_win_rate,
                "prop_roi": prop_roi,
                "avg_prop_edge": avg_prop_edge,
                "prop_breakdown": {
                    prop_type: {
                        "bets": stats['bets'],
                        "win_rate": stats['wins'] / stats['bets'] if stats['bets'] > 0 else 0,
                        "roi": stats['profit'] / stats['stakes'] * 100 if stats['stakes'] > 0 else 0,
                        "profit": stats['profit']
                    }
                    for prop_type, stats in prop_types.items()
                }
            }
        
        return {
            **paper_summary,
            **prop_metrics,
            "session_stats": self.session_stats,
            "pending_props": len([b for b in prop_bets if b.status == BetStatus.PENDING])
        }
    
    def create_prop_trading_report(self) -> str:
        """Create comprehensive prop trading report"""
        
        summary = self.get_prop_performance_summary()
        
        report = []
        report.append("=" * 70)
        report.append("PROP BETTING PAPER TRADING REPORT")
        report.append("=" * 70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Current Bankroll: ${self.paper_trader.current_bankroll:,.2f}")
        report.append("")
        
        if "error" not in summary:
            report.append("PROP BETTING PERFORMANCE:")
            report.append(f"  Total Prop Bets: {summary.get('total_prop_bets', 0)}")
            report.append(f"  Prop Win Rate: {summary.get('prop_win_rate', 0):.1%}")
            report.append(f"  Prop ROI: {summary.get('prop_roi', 0):+.1f}%")
            report.append(f"  Average Edge: {summary.get('avg_prop_edge', 0):+.1%}")
            report.append("")
            
            # Prop type breakdown
            prop_breakdown = summary.get('prop_breakdown', {})
            if prop_breakdown:
                report.append("PERFORMANCE BY PROP TYPE:")
                for prop_type, stats in prop_breakdown.items():
                    report.append(f"  {prop_type.upper().replace('_', ' ')}:")
                    report.append(f"    Bets: {stats['bets']}")
                    report.append(f"    Win Rate: {stats['win_rate']:.1%}")
                    report.append(f"    ROI: {stats['roi']:+.1f}%")
                    report.append(f"    Profit: ${stats['profit']:+.2f}")
                    report.append("")
        
        # Session statistics
        session = summary.get('session_stats', {})
        if session:
            report.append("SESSION STATISTICS:")
            report.append(f"  Opportunities Found: {session.get('opportunities_found', 0)}")
            report.append(f"  Bets Placed: {session.get('bets_placed', 0)}")
            report.append(f"  Average Edge: {session.get('avg_edge', 0):+.1%}")
            report.append(f"  Session P&L: ${session.get('session_profit', 0):+.2f}")
            report.append("")
        
        return "\n".join(report)

# Demo function
async def demo_integrated_prop_trading():
    """Demo the integrated prop trading system"""
    
    print("🎯💰 Integrated Prop Trading System Demo")
    print("=" * 60)
    
    # Initialize with $25,000 fake bankroll
    trader = IntegratedPropTrader(initial_bankroll=25000.0)
    print(f"Starting Bankroll: ${trader.paper_trader.current_bankroll:,.2f}")
    
    # Simulate analyzing multiple games
    games = [
        {
            'game_id': 'duke_unc_20241201',
            'home_team': 'Duke', 
            'away_team': 'UNC',
            'context': {
                'expected_pace': 72,
                'home_offensive_efficiency': 110,
                'away_offensive_efficiency': 105,
                'home_expected_points': 82,
                'away_expected_points': 78
            }
        },
        {
            'game_id': 'kansas_uk_20241201',
            'home_team': 'Kansas',
            'away_team': 'Kentucky', 
            'context': {
                'expected_pace': 68,
                'home_offensive_efficiency': 108,
                'away_offensive_efficiency': 103,
                'home_expected_points': 79,
                'away_expected_points': 76
            }
        }
    ]
    
    print(f"\n🔍 Analyzing {len(games)} games for prop opportunities...")
    
    all_results = []
    for game in games:
        result = await trader.analyze_and_bet_game(
            game['game_id'],
            game['home_team'],
            game['away_team'], 
            game['context']
        )
        all_results.append(result)
        
        print(f"\n{game['home_team']} vs {game['away_team']}:")
        print(f"  Opportunities: {result['opportunities_found']}")
        print(f"  Bets Placed: {result['bets_placed']}")
        print(f"  Total Exposure: ${result['total_exposure']:.2f}")
        print(f"  Avg Edge: {result['avg_edge']:.1%}")
    
    # Simulate settling some bets  
    print(f"\n🎲 Simulating bet settlements...")
    pending_bets = trader.paper_trader.get_pending_bets()
    
    settled_count = 0
    for bet in pending_bets[:6]:  # Settle first 6 bets
        success = await trader.simulate_bet_settlement(bet.bet_id, simulate_realistic=True)
        if success:
            settled_count += 1
    
    print(f"Settled {settled_count} bets with realistic outcomes")
    
    # Show performance summary
    print(f"\n📊 Prop Trading Performance:")
    summary = trader.get_prop_performance_summary()
    
    if "error" not in summary:
        print(f"  Current Bankroll: ${summary['current_bankroll']:,.2f}")
        print(f"  Net P&L: ${summary['total_profit']:+,.2f}")
        print(f"  Overall ROI: {summary['roi']:+.1f}%")
        print(f"  Prop Win Rate: {summary.get('prop_win_rate', 0):.1%}")
        print(f"  Total Prop Bets: {summary.get('total_prop_bets', 0)}")
    
    # Generate detailed report
    print(f"\n📋 Generating Comprehensive Report...")
    report = trader.create_prop_trading_report()
    print(report)
    
    print("✅ Integrated Prop Trading System Ready for Testing!")

def main():
    """Main function"""
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demo_integrated_prop_trading())

if __name__ == "__main__":
    main()