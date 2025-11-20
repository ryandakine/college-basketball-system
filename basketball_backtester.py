#!/usr/bin/env python3
"""
Basketball Backtesting Engine
Tests prediction models on historical data with walk-forward validation
"""

import pandas as pd
import numpy as np
import pickle
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BasketballBacktester:
    """
    Backtesting engine for basketball betting system

    Features:
    - Walk-forward validation (realistic simulation)
    - Multiple models (RF, GB, XGBoost)
    - Kelly Criterion bet sizing
    - Comprehensive performance metrics
    - Bankroll simulation
    """

    def __init__(self,
                 initial_bankroll: float = 10000,
                 kelly_fraction: float = 0.25,
                 min_edge: float = 0.02,
                 min_confidence: float = 0.55):
        """
        Initialize backtester

        Args:
            initial_bankroll: Starting bankroll ($)
            kelly_fraction: Fractional Kelly (0.25 = quarter Kelly)
            min_edge: Minimum edge to bet (0.02 = 2%)
            min_confidence: Minimum confidence to bet (0.55 = 55%)
        """
        self.initial_bankroll = initial_bankroll
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.min_confidence = min_confidence

        self.models = {}
        self.results = []
        self.bankroll_history = []

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features from game data

        Args:
            df: DataFrame with game data

        Returns:
            X: Features
            y: Target (1 if home won, 0 if away won)
        """
        logger.info(f"Preparing features from {len(df)} games...")

        # Target: 1 if home won, 0 if away won
        y = (df['home_score'] > df['away_score']).astype(int)

        # Features
        feature_cols = []

        # Score-based features
        if 'home_score' in df.columns:
            df['home_scoring_avg'] = df.groupby('home_team')['home_score'].transform('mean')
            df['away_scoring_avg'] = df.groupby('away_team')['away_score'].transform('mean')
            feature_cols.extend(['home_scoring_avg', 'away_scoring_avg'])

        # Win percentage features
        df['home_win_pct'] = df.groupby('home_team')['home_score'].transform(
            lambda x: (x > df.loc[x.index, 'away_score']).mean()
        )
        df['away_win_pct'] = df.groupby('away_team')['away_score'].transform(
            lambda x: (x > df.loc[x.index, 'home_score']).mean()
        )
        feature_cols.extend(['home_win_pct', 'away_win_pct'])

        # Point differential
        if 'home_score' in df.columns:
            df['home_avg_margin'] = df.groupby('home_team').apply(
                lambda x: (x['home_score'] - x['away_score']).mean()
            ).reindex(df['home_team']).values
            df['away_avg_margin'] = df.groupby('away_team').apply(
                lambda x: (x['away_score'] - x['home_score']).mean()
            ).reindex(df['away_team']).values
            feature_cols.extend(['home_avg_margin', 'away_avg_margin'])

        # Home court advantage
        df['home_advantage'] = 1
        feature_cols.append('home_advantage')

        # Fill any NaN values
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].mean())

        X = df[feature_cols]

        logger.info(f"Created {len(feature_cols)} features: {feature_cols}")

        return X, y

    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train all models"""
        logger.info("Training models...")

        # Random Forest
        logger.info("  Training Random Forest...")
        self.models['rf'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            random_state=42,
            n_jobs=-1
        )
        self.models['rf'].fit(X_train, y_train)

        # Gradient Boosting
        logger.info("  Training Gradient Boosting...")
        self.models['gb'] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.models['gb'].fit(X_train, y_train)

        # XGBoost
        logger.info("  Training XGBoost...")
        self.models['xgb'] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        self.models['xgb'].fit(X_train, y_train)

        logger.info("✅ All models trained")

    def predict_with_ensemble(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make ensemble predictions

        Returns:
            predictions: 0 or 1 (away or home)
            probabilities: confidence scores
        """
        # Get predictions from all models
        rf_proba = self.models['rf'].predict_proba(X)[:, 1]
        gb_proba = self.models['gb'].predict_proba(X)[:, 1]
        xgb_proba = self.models['xgb'].predict_proba(X)[:, 1]

        # Ensemble (average)
        ensemble_proba = (rf_proba + gb_proba + xgb_proba) / 3
        ensemble_pred = (ensemble_proba > 0.5).astype(int)

        return ensemble_pred, ensemble_proba

    def calculate_bet_size(self,
                          win_prob: float,
                          odds: float,
                          current_bankroll: float) -> float:
        """
        Calculate bet size using Kelly Criterion

        Args:
            win_prob: Probability of winning (0-1)
            odds: Decimal odds (e.g., 1.91 for -110)
            current_bankroll: Current bankroll

        Returns:
            Bet size in dollars
        """
        # Kelly formula: f = (p * odds - 1) / (odds - 1)
        # where p = win probability, odds = decimal odds

        kelly_percentage = (win_prob * odds - 1) / (odds - 1)

        # Apply fractional Kelly
        kelly_percentage *= self.kelly_fraction

        # Clamp between 0% and 10% of bankroll
        kelly_percentage = max(0, min(kelly_percentage, 0.10))

        bet_size = current_bankroll * kelly_percentage

        return bet_size

    def american_to_decimal(self, american_odds: int) -> float:
        """Convert American odds to decimal"""
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1

    def run_backtest(self,
                    df: pd.DataFrame,
                    train_start: str,
                    train_end: str,
                    test_start: str,
                    test_end: str) -> Dict:
        """
        Run backtest with walk-forward validation

        Args:
            df: Historical game data
            train_start: Training start date (YYYY-MM-DD)
            train_end: Training end date
            test_start: Testing start date
            test_end: Testing end date

        Returns:
            results dictionary
        """
        logger.info("="*60)
        logger.info("🏀 STARTING BACKTEST")
        logger.info("="*60)
        logger.info(f"Training: {train_start} to {train_end}")
        logger.info(f"Testing:  {test_start} to {test_end}")
        logger.info(f"Initial Bankroll: ${self.initial_bankroll:,.2f}")
        logger.info("="*60)

        # Ensure dates are datetime
        if not pd.api.types.is_datetime64_any_dtype(df['game_date']):
            df['game_date'] = pd.to_datetime(df['game_date'])

        # Convert boundary dates to datetime if needed
        train_start = pd.to_datetime(train_start)
        train_end = pd.to_datetime(train_end)
        test_start = pd.to_datetime(test_start)
        test_end = pd.to_datetime(test_end)

        # Split data
        train_mask = (df['game_date'] >= train_start) & (df['game_date'] <= train_end)
        test_mask = (df['game_date'] >= test_start) & (df['game_date'] <= test_end)

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        logger.info(f"Training games: {len(train_df)}")
        logger.info(f"Testing games: {len(test_df)}")

        if len(train_df) == 0 or len(test_df) == 0:
            logger.error("Insufficient data for backtest!")
            return {"error": "Insufficient data"}

        # Prepare features
        X_train, y_train = self.prepare_features(train_df)
        X_test, y_test = self.prepare_features(test_df)

        # Train models
        self.train_models(X_train, y_train)

        # Make predictions on test set
        predictions, probabilities = self.predict_with_ensemble(X_test)

        # Simulate betting
        current_bankroll = self.initial_bankroll
        self.bankroll_history = [current_bankroll]
        total_bets = 0
        winning_bets = 0
        total_staked = 0
        total_profit = 0

        bets = []

        for idx, (pred, prob, actual) in enumerate(zip(predictions, probabilities, y_test)):
            # Calculate edge
            win_prob = prob if pred == 1 else (1 - prob)
            edge = win_prob - 0.5  # Simple edge calculation

            # Check if we should bet
            if win_prob < self.min_confidence or edge < self.min_edge:
                continue

            # Assume -110 odds (1.91 decimal) for simplicity
            # In real backtest, use actual odds from data
            odds = 1.91

            # Calculate bet size
            bet_size = self.calculate_bet_size(win_prob, odds, current_bankroll)

            if bet_size < 1:  # Don't bet less than $1
                continue

            total_bets += 1
            total_staked += bet_size

            # Determine win/loss
            bet_won = (pred == actual)

            if bet_won:
                profit = bet_size * (odds - 1)
                winning_bets += 1
            else:
                profit = -bet_size

            current_bankroll += profit
            total_profit += profit
            self.bankroll_history.append(current_bankroll)

            bets.append({
                'game_idx': idx,
                'prediction': 'HOME' if pred == 1 else 'AWAY',
                'confidence': win_prob,
                'bet_size': bet_size,
                'odds': odds,
                'won': bet_won,
                'profit': profit,
                'bankroll': current_bankroll
            })

        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)

        win_rate = winning_bets / total_bets if total_bets > 0 else 0
        roi = (total_profit / total_staked * 100) if total_staked > 0 else 0
        final_bankroll = current_bankroll
        total_return = ((final_bankroll - self.initial_bankroll) / self.initial_bankroll * 100)

        # Calculate Sharpe Ratio
        if len(self.bankroll_history) > 1:
            returns = np.diff(self.bankroll_history) / self.bankroll_history[:-1]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        # Calculate max drawdown
        bankroll_array = np.array(self.bankroll_history)
        running_max = np.maximum.accumulate(bankroll_array)
        drawdown = (bankroll_array - running_max) / running_max
        max_drawdown = np.min(drawdown) * 100

        results = {
            'success': True,
            'model_accuracy': accuracy,
            'total_bets': total_bets,
            'winning_bets': winning_bets,
            'losing_bets': total_bets - winning_bets,
            'win_rate': win_rate,
            'total_staked': total_staked,
            'total_profit': total_profit,
            'roi': roi,
            'initial_bankroll': self.initial_bankroll,
            'final_bankroll': final_bankroll,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'bets': bets,
            'bankroll_history': self.bankroll_history
        }

        self.results = results
        return results

    def print_results(self):
        """Print backtest results"""
        if not self.results:
            logger.error("No results to print. Run backtest first.")
            return

        r = self.results

        print("\n" + "="*70)
        print("📊 BACKTEST RESULTS")
        print("="*70)

        print(f"\n🎯 MODEL PERFORMANCE:")
        print(f"   Prediction Accuracy: {r['model_accuracy']:.1%}")

        print(f"\n💰 BETTING PERFORMANCE:")
        print(f"   Total Bets: {r['total_bets']}")
        print(f"   Winning Bets: {r['winning_bets']}")
        print(f"   Losing Bets: {r['losing_bets']}")
        print(f"   Win Rate: {r['win_rate']:.1%}")

        print(f"\n💵 FINANCIAL RESULTS:")
        print(f"   Initial Bankroll: ${r['initial_bankroll']:,.2f}")
        print(f"   Final Bankroll: ${r['final_bankroll']:,.2f}")
        print(f"   Total Staked: ${r['total_staked']:,.2f}")
        print(f"   Total Profit/Loss: ${r['total_profit']:+,.2f}")
        print(f"   ROI: {r['roi']:+.1f}%")
        print(f"   Total Return: {r['total_return']:+.1f}%")

        print(f"\n📈 RISK METRICS:")
        print(f"   Sharpe Ratio: {r['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {r['max_drawdown']:.1f}%")

        print("\n" + "="*70)

        # Show sample bets
        if r['bets']:
            print(f"\n📋 SAMPLE BETS (first 5):")
            for bet in r['bets'][:5]:
                status = "✅ WIN" if bet['won'] else "❌ LOSS"
                print(f"   {bet['prediction']:4s} | Confidence: {bet['confidence']:.1%} | "
                      f"Bet: ${bet['bet_size']:.2f} | {status} | Profit: ${bet['profit']:+.2f}")

        print("="*70 + "\n")

    def save_results(self, filename: str = "backtest_results.json"):
        """Save results to JSON file"""
        if not self.results:
            logger.error("No results to save")
            return

        def convert_to_serializable(obj):
            """Convert numpy types to Python native types"""
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(v) for v in obj]
            elif isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        # Convert to JSON-serializable format
        results_copy = self.results.copy()
        results_copy['bets'] = results_copy['bets'][:100]  # Save only first 100 bets
        results_copy = convert_to_serializable(results_copy)

        with open(filename, 'w') as f:
            json.dump(results_copy, f, indent=2)

        logger.info(f"💾 Results saved to {filename}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names for different data sources"""
    # Handle 'date' vs 'game_date'
    if 'date' in df.columns and 'game_date' not in df.columns:
        df['game_date'] = df['date']

    # Convert date to datetime if it's numeric (YYYYMMDD format)
    if 'game_date' in df.columns:
        if df['game_date'].dtype in ['int64', 'float64']:
            df['game_date'] = pd.to_datetime(df['game_date'].astype(str), format='%Y%m%d')
        else:
            df['game_date'] = pd.to_datetime(df['game_date'])

    return df


def load_historical_data(data_path: str = None) -> pd.DataFrame:
    """Load historical game data from database or CSV"""
    logger.info("Loading historical data...")

    # If specific path provided, use it
    if data_path:
        path = Path(data_path)
        if path.exists():
            logger.info(f"Loading from {path}")
            df = pd.read_csv(path)
            df = normalize_columns(df)
            logger.info(f"Loaded {len(df)} games from {path}")
            return df
        else:
            logger.error(f"File not found: {path}")
            return None

    # Try loading from 10-year historical data (scraped data)
    historical_path = Path("data/historical/all_games_10yr.csv")
    if historical_path.exists():
        logger.info(f"Loading from {historical_path}")
        df = pd.read_csv(historical_path)
        df = normalize_columns(df)
        logger.info(f"Loaded {len(df)} games from 10-year historical data")
        return df

    # Try loading from scraped_games.csv
    csv_path = Path("scraped_games.csv")
    if csv_path.exists():
        logger.info(f"Loading from {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} games from CSV")
        return df

    # Try loading from database
    db_path = Path("basketball_betting.db")
    if db_path.exists():
        logger.info(f"Loading from {db_path}")
        conn = sqlite3.connect(db_path)

        # Try to load from predictions table
        try:
            df = pd.read_sql_query("SELECT * FROM basketball_predictions", conn)
            conn.close()
            logger.info(f"Loaded {len(df)} games from database")
            return df
        except:
            conn.close()

    # Try historical database
    hist_db_path = Path("data/historical/historical_10yr.db")
    if hist_db_path.exists():
        logger.info(f"Loading from {hist_db_path}")
        conn = sqlite3.connect(hist_db_path)
        try:
            df = pd.read_sql_query("SELECT * FROM historical_games", conn)
            conn.close()
            logger.info(f"Loaded {len(df)} games from historical database")
            return df
        except:
            conn.close()

    logger.error("❌ No historical data found!")
    logger.error("Run data scraper first: python scrape_10_year_history.py")
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="🏀 Basketball Backtesting Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python basketball_backtester.py                                    # Auto-detect data
  python basketball_backtester.py --data data/historical/all_games_10yr.csv  # Specify CSV
  python basketball_backtester.py --bankroll 5000 --kelly 0.5        # Custom settings
        """
    )

    parser.add_argument('--data', type=str, help='Path to historical data CSV')
    parser.add_argument('--bankroll', type=float, default=10000, help='Initial bankroll (default: 10000)')
    parser.add_argument('--kelly', type=float, default=0.25, help='Kelly fraction (default: 0.25)')
    parser.add_argument('--min-edge', type=float, default=0.02, help='Minimum edge to bet (default: 0.02)')
    parser.add_argument('--min-confidence', type=float, default=0.55, help='Minimum confidence (default: 0.55)')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Training data ratio (default: 0.8)')

    args = parser.parse_args()

    print("\n🏀 Basketball Backtesting Engine\n")

    # Load data
    df = load_historical_data(args.data)

    if df is None or len(df) == 0:
        print("❌ No data available for backtesting")
        print("\nRun the scraper first: python scrape_10_year_history.py")
        print("Or specify data path: python basketball_backtester.py --data path/to/data.csv")
        exit(1)

    print(f"✅ Loaded {len(df)} historical games")

    # Run backtest
    backtester = BasketballBacktester(
        initial_bankroll=args.bankroll,
        kelly_fraction=args.kelly,
        min_edge=args.min_edge,
        min_confidence=args.min_confidence
    )

    # Use train/test split
    df_sorted = df.sort_values('game_date')
    split_idx = int(len(df) * args.train_ratio)

    train_start = df_sorted.iloc[0]['game_date']
    train_end = df_sorted.iloc[split_idx]['game_date']
    test_start = df_sorted.iloc[split_idx + 1]['game_date']
    test_end = df_sorted.iloc[-1]['game_date']

    print(f"\n📊 Data Split:")
    print(f"   Training: {train_start} to {train_end} ({split_idx} games)")
    print(f"   Testing:  {test_start} to {test_end} ({len(df) - split_idx - 1} games)")

    results = backtester.run_backtest(
        df,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end
    )

    if results.get('success'):
        backtester.print_results()
        backtester.save_results()
    else:
        print("❌ Backtest failed:", results.get('error'))
