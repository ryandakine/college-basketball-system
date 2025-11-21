#!/usr/bin/env python3
"""
Paper Trading System for College Basketball
============================================

Simulates betting with fake money to validate the system before going live.

Features:
- Track simulated bets and performance
- Daily predictions for upcoming games
- Performance dashboard
- Bankroll tracking
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaperTradingSystem:
    """
    Paper trading system for basketball betting simulation
    """

    def __init__(self, initial_bankroll: float = 1000, data_file: str = "paper_trading_data.json"):
        self.data_file = Path(data_file)
        self.initial_bankroll = initial_bankroll

        # Load existing data or initialize
        if self.data_file.exists():
            self.load_data()
        else:
            self.data = {
                'bankroll': initial_bankroll,
                'initial_bankroll': initial_bankroll,
                'bets': [],
                'pending_bets': [],
                'daily_results': [],
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat()
            }
            self.save_data()

    def load_data(self):
        """Load paper trading data from file"""
        with open(self.data_file, 'r') as f:
            self.data = json.load(f)
        logger.info(f"Loaded paper trading data: ${self.data['bankroll']:.2f} bankroll")

    def save_data(self):
        """Save paper trading data to file"""
        self.data['last_updated'] = datetime.now().isoformat()
        with open(self.data_file, 'w') as f:
            json.dump(self.data, f, indent=2)

    def place_bet(self,
                  game_id: str,
                  home_team: str,
                  away_team: str,
                  pick: str,  # 'HOME' or 'AWAY'
                  confidence: float,
                  spread_prediction: float,
                  odds: float = 1.91,  # -110 default
                  bet_amount: float = None):
        """
        Place a simulated bet

        Args:
            game_id: Unique game identifier
            home_team: Home team name
            away_team: Away team name
            pick: 'HOME' or 'AWAY'
            confidence: Model confidence (0-1)
            spread_prediction: Predicted spread
            odds: Decimal odds (default -110 = 1.91)
            bet_amount: Bet size (default: Kelly sizing)
        """
        # Calculate bet size if not provided
        if bet_amount is None:
            bet_amount = self._calculate_bet_size(confidence, odds)

        if bet_amount > self.data['bankroll']:
            bet_amount = self.data['bankroll'] * 0.05  # Max 5%

        if bet_amount < 5:
            logger.warning("Bet too small, skipping")
            return None

        bet = {
            'id': f"BET-{len(self.data['bets']) + len(self.data['pending_bets']) + 1:04d}",
            'game_id': game_id,
            'home_team': home_team,
            'away_team': away_team,
            'pick': pick,
            'confidence': confidence,
            'spread_prediction': spread_prediction,
            'odds': odds,
            'bet_amount': round(bet_amount, 2),
            'potential_profit': round(bet_amount * (odds - 1), 2),
            'placed_at': datetime.now().isoformat(),
            'status': 'pending'
        }

        self.data['pending_bets'].append(bet)
        self.save_data()

        logger.info(f"📝 Placed bet: {bet['id']}")
        logger.info(f"   {pick} ({home_team} vs {away_team})")
        logger.info(f"   Amount: ${bet_amount:.2f} | Confidence: {confidence*100:.1f}%")

        return bet

    def resolve_bet(self, game_id: str, winner: str, home_score: int = None, away_score: int = None):
        """
        Resolve a pending bet with the actual result

        Args:
            game_id: Game identifier
            winner: 'HOME' or 'AWAY'
            home_score: Final home score
            away_score: Final away score
        """
        # Find the pending bet
        bet_idx = None
        for i, bet in enumerate(self.data['pending_bets']):
            if bet['game_id'] == game_id:
                bet_idx = i
                break

        if bet_idx is None:
            logger.warning(f"No pending bet found for game {game_id}")
            return None

        bet = self.data['pending_bets'].pop(bet_idx)

        # Determine outcome
        bet_won = (bet['pick'] == winner)

        if bet_won:
            profit = bet['bet_amount'] * (bet['odds'] - 1)
            self.data['bankroll'] += profit
            bet['status'] = 'won'
            bet['profit'] = round(profit, 2)
            logger.info(f"✅ WON: {bet['id']} | +${profit:.2f}")
        else:
            self.data['bankroll'] -= bet['bet_amount']
            bet['status'] = 'lost'
            bet['profit'] = -bet['bet_amount']
            logger.info(f"❌ LOST: {bet['id']} | -${bet['bet_amount']:.2f}")

        bet['resolved_at'] = datetime.now().isoformat()
        bet['actual_winner'] = winner
        if home_score is not None:
            bet['home_score'] = home_score
            bet['away_score'] = away_score
            bet['actual_spread'] = home_score - away_score

        self.data['bets'].append(bet)
        self.save_data()

        return bet

    def _calculate_bet_size(self, confidence: float, odds: float) -> float:
        """Calculate bet size using Kelly Criterion"""
        implied_prob = 1 / odds
        edge = confidence - implied_prob

        if edge < 0.02:  # Minimum 2% edge
            return 0

        b = odds - 1
        kelly = (b * confidence - (1 - confidence)) / b
        kelly *= 0.25  # Quarter Kelly

        # Cap at 5% of bankroll
        kelly = min(kelly, 0.05)

        return self.data['bankroll'] * kelly

    def get_stats(self) -> dict:
        """Get paper trading statistics"""
        if not self.data['bets']:
            return {
                'total_bets': 0,
                'pending_bets': len(self.data['pending_bets']),
                'bankroll': self.data['bankroll'],
                'initial_bankroll': self.data['initial_bankroll'],
                'profit': 0,
                'roi': 0,
                'total_return': 0
            }

        wins = sum(1 for b in self.data['bets'] if b['status'] == 'won')
        losses = sum(1 for b in self.data['bets'] if b['status'] == 'lost')
        total_staked = sum(b['bet_amount'] for b in self.data['bets'])
        total_profit = sum(b['profit'] for b in self.data['bets'])

        return {
            'total_bets': len(self.data['bets']),
            'wins': wins,
            'losses': losses,
            'win_rate': wins / len(self.data['bets']) if self.data['bets'] else 0,
            'pending_bets': len(self.data['pending_bets']),
            'bankroll': self.data['bankroll'],
            'initial_bankroll': self.data['initial_bankroll'],
            'profit': total_profit,
            'total_staked': total_staked,
            'roi': (total_profit / total_staked * 100) if total_staked > 0 else 0,
            'total_return': ((self.data['bankroll'] - self.data['initial_bankroll']) /
                           self.data['initial_bankroll'] * 100)
        }

    def print_dashboard(self):
        """Print paper trading dashboard"""
        stats = self.get_stats()

        print("\n" + "="*60)
        print("📊 PAPER TRADING DASHBOARD")
        print("="*60)

        print(f"\n💰 BANKROLL:")
        print(f"   Starting: ${stats['initial_bankroll']:,.2f}")
        print(f"   Current:  ${stats['bankroll']:,.2f}")
        profit_color = "+" if stats['profit'] >= 0 else ""
        print(f"   Profit:   ${profit_color}{stats['profit']:,.2f}")
        print(f"   Return:   {profit_color}{stats['total_return']:.1f}%")

        if stats['total_bets'] > 0:
            print(f"\n📈 PERFORMANCE:")
            print(f"   Total Bets: {stats['total_bets']}")
            print(f"   Wins/Losses: {stats['wins']}/{stats['losses']}")
            print(f"   Win Rate: {stats['win_rate']*100:.1f}%")
            print(f"   ROI: {stats['roi']:.1f}%")

        if stats['pending_bets'] > 0:
            print(f"\n⏳ PENDING BETS: {stats['pending_bets']}")
            for bet in self.data['pending_bets']:
                # Get the pick detail (team name) if available
                pick_team = bet['home_team'] if bet['pick'] == 'HOME' else bet['away_team']
                print(f"   {bet['id']}: {pick_team} ({bet['pick']})")
                print(f"      Match: {bet['home_team']} vs {bet['away_team']}")
                print(f"      Wager: ${bet['bet_amount']:.2f} @ {bet['odds']:.2f} (Conf: {bet.get('confidence', 0)*100:.1f}%)")

        # Recent bets
        if self.data['bets']:
            print(f"\n📋 RECENT BETS (last 5):")
            for bet in self.data['bets'][-5:]:
                status = "✅" if bet['status'] == 'won' else "❌"
                print(f"   {status} {bet['id']}: {bet['pick']} | ${bet['profit']:+.2f}")

        print("\n" + "="*60 + "\n")

    def reset(self, initial_bankroll: float = None):
        """Reset paper trading account"""
        if initial_bankroll is None:
            initial_bankroll = self.initial_bankroll

        self.data = {
            'bankroll': initial_bankroll,
            'initial_bankroll': initial_bankroll,
            'bets': [],
            'pending_bets': [],
            'daily_results': [],
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        }
        self.save_data()
        logger.info(f"🔄 Paper trading reset with ${initial_bankroll:.2f}")


def demo():
    """Demo the paper trading system"""
    print("\n🏀 Paper Trading System Demo\n")

    # Initialize with $1000
    pt = PaperTradingSystem(initial_bankroll=1000)

    # Simulate some bets
    print("Placing sample bets...\n")

    # Bet 1: High confidence home pick
    pt.place_bet(
        game_id="GAME001",
        home_team="Duke",
        away_team="UNC",
        pick="HOME",
        confidence=0.72,
        spread_prediction=5.5
    )

    # Bet 2: Away pick
    pt.place_bet(
        game_id="GAME002",
        home_team="Kentucky",
        away_team="Kansas",
        pick="AWAY",
        confidence=0.68,
        spread_prediction=-3.2
    )

    # Resolve bets
    print("\nResolving bets...\n")
    pt.resolve_bet("GAME001", "HOME", 78, 72)  # Duke wins
    pt.resolve_bet("GAME002", "HOME", 85, 80)  # Kansas loses (Kentucky wins)

    # Show dashboard
    pt.print_dashboard()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="🏀 Paper Trading System")
    parser.add_argument('--demo', action='store_true', help='Run demo')
    parser.add_argument('--dashboard', action='store_true', help='Show dashboard')
    parser.add_argument('--reset', action='store_true', help='Reset account')
    parser.add_argument('--bankroll', type=float, default=1000, help='Initial bankroll')

    args = parser.parse_args()

    if args.demo:
        demo()
    elif args.dashboard:
        pt = PaperTradingSystem()
        pt.print_dashboard()
    elif args.reset:
        pt = PaperTradingSystem(initial_bankroll=args.bankroll)
        pt.reset(args.bankroll)
        print(f"✅ Reset with ${args.bankroll:.2f}")
    else:
        # Default: show dashboard or create new
        pt = PaperTradingSystem(initial_bankroll=args.bankroll)
        pt.print_dashboard()
